"""Training and prediction at a cutoff (scheme 4.4).

v3 splits the old run_model into: train_at_cutoff (train exactly N rounds on
d<=cutoff rows), predict_28_days (full 28-day prediction, recursive fills day
by day), evaluate_round_candidates (prefix-equivalence selection: one training
to the candidate cap, then one full 28-day prediction per candidate k via
model.predict(..., num_iteration=k)) and run_model (deliverable retrain at the
stage train_cutoff with the rounds selected earlier)."""
import gc
import json
import os
import resource
import sys
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import F, FILES, selection_root, features_root, stage_root, stage_spec, artifact_id, complete, checkpoint, atomic_csv, atomic_json, event

# Operational smoke-time tolerance for the prefix-equivalence premise; the
# unit test uses tighter tolerances on synthetic data with one thread.
PREFIX_EQUIVALENCE_LIMITS={'nonrecursive':1e-4,'recursive':1e-3}

def load_frame(c,cutoff,store,mode):
    dest=features_root(c,cutoff)/store
    assert complete(dest/'complete.json',artifact_id(c,'features',cutoff,store,'cache'),FILES)
    return read_feature_tables(dest,mode)

def split_masks(frame,cutoff,horizon):
    """Training rows d<=cutoff vs forecast rows cutoff<d<=cutoff+horizon; the
    two masks must never intersect (scheme 4.4)."""
    train=frame.d.le(cutoff)
    forecast=frame.d.gt(cutoff)&frame.d.le(cutoff+horizon)
    assert not (train&forecast).any()
    return train,forecast



def predict_28_days(model,frame,features,cut,mode,threads,num_iteration=None):
    """Full 28-day prediction for cut+1..cut+28 aligned id x F1..F28; the
    recursive mode re-fills day by day from its own predictions only."""
    if mode=='recursive':
        return predict_recursive(model,frame.loc[frame.d>cut-100],features,cut,threads,num_iteration)
    rows=frame.loc[frame.d>cut].copy()
    rows['prediction']=np.asarray(model.predict(rows[features],num_threads=threads,num_iteration=num_iteration),dtype=np.float64)
    wide=rows.pivot(index='id',columns='d',values='prediction')
    assert wide.columns.tolist()==list(range(cut+1,cut+29))
    wide.columns=F
    return wide.reset_index()

def validate_prediction(output,c):
    expected=c.get('smoke_items',3049)
    assert output.id.nunique()==expected
    values=output[F].to_numpy()
    assert np.isfinite(values).all() and (values>=0).all()
    return output

def train_at_cutoff(c,cutoff,store,mode,num_boost_round,dest):
    """Train exactly num_boost_round iterations on d<=cutoff rows of the
    cutoff-specific cache; dest receives the reusable binary dataset. Early
    stopping is unreachable in v1 (scheme §5): enabled=true is rejected."""
    if c.get('early_stopping',{}).get('enabled'):
        raise NotImplementedError('early_stopping.enabled=true is reserved for future scouting/calibration experiments (scheme §5); v1 selection and delivery never early-stop.')
    frame,features=load_frame(c,cutoff,store,mode)
    train_mask,forecast_mask=split_masks(frame,cutoff,c['horizon'])
    assert frame.loc[train_mask,'sales'].notna().all()
    assert frame.loc[forecast_mask,'sales'].isna().all()
    assert frame.loc[train_mask,'d'].max()==cutoff
    assert frame.loc[forecast_mask,'d'].min()==cutoff+1
    recent=frame.loc[frame.d>cutoff-100].copy()
    params={**c['params'],**c['mode_params'][mode],'num_threads':c['threads']}
    cache=dest/'dataset'; cache.mkdir(parents=True,exist_ok=True)
    cache_fp=artifact_id(c,'dataset',cutoff,store,kind='train',mode=mode)
    train_rows=int(train_mask.sum())
    schema=dict(features=features,params=params,train_first_day=c.get('first_day',1),train_last_day=cutoff,train_rows=train_rows)
    if complete(cache/'complete.json',cache_fp,['train.bin','schema.json']):
        assert json.loads((cache/'schema.json').read_text())==schema
        dataset=lgb.Dataset(str(cache/'train.bin'),params=params,free_raw_data=True)
        del frame; gc.collect()
        dataset.construct()
    else:
        train=frame.loc[train_mask,features]
        labels=frame.loc[train_mask,'sales'].to_numpy(dtype=np.float32)
        del frame; gc.collect()
        dataset=lgb.Dataset(train,label=labels,params=params,free_raw_data=True)
        dataset.construct()
        del train,labels; gc.collect()
        temporary=cache/f'train.bin.tmp-{os.getpid()}'
        dataset.save_binary(str(temporary)); os.replace(temporary,cache/'train.bin')
        atomic_json(schema,cache/'schema.json')
        checkpoint(cache/'complete.json',cache_fp,['train.bin','schema.json'])
    started=time.monotonic()
    event('training_started',cutoff=cutoff,store=store,mode=mode,rows=train_rows,features=len(features),rounds=num_boost_round)
    def progress(env):
        if (env.iteration+1)%100==0:
            event('iteration',cutoff=cutoff,store=store,mode=mode,iteration=env.iteration+1,seconds=time.monotonic()-started)
    estimator=lgb.train(params,dataset,num_boost_round=num_boost_round,callbacks=[progress])
    return estimator,recent,features,params,train_rows,time.monotonic()-started

def evaluate_round_candidates(c,select_cutoff,store,mode):
    """Prefix-equivalence path (scheme 4.4): one training per selection-stage x
    store x mode up to the candidate cap, then a full 28-day prediction per
    candidate k via model.predict(..., num_iteration=k) (recursive candidates
    also refill day by day). Candidate products never read official-window
    labels. The model file is deleted once the checkpoint is written; only the
    explicit audit flag keeps it (scheme §9)."""
    dest=selection_root(c,select_cutoff)/mode/store; dest.mkdir(parents=True,exist_ok=True)
    candidates=sorted(c['round_candidates']); cap=candidates[-1]
    products=[f'predictions_{k}.csv' for k in candidates]+['timings.json','metadata.json']
    fp=artifact_id(c,'selection',select_cutoff,store,kind='candidates',mode=mode,rounds=candidates)
    if complete(dest/'complete.json',fp,products):
        event('candidates_verified',select_cutoff=select_cutoff,store=store,mode=mode); return
    started=time.monotonic()
    trained_fp=artifact_id(c,'selection',select_cutoff,store,kind='selection_model',mode=mode,rounds=cap)
    train_seconds=None
    if complete(dest/'trained.json',trained_fp,['model.txt']):
        estimator=lgb.Booster(model_file=str(dest/'model.txt'))
        frame,features=load_frame(c,select_cutoff,store,mode)
        recent=frame.loc[frame.d>select_cutoff-100].copy(); del frame; gc.collect()
        event('selection_model_resumed',select_cutoff=select_cutoff,store=store,mode=mode)
    else:
        estimator,recent,features,params,train_rows,train_seconds=train_at_cutoff(c,select_cutoff,store,mode,cap,dest)
        estimator.save_model(str(dest/'model.txt.tmp')); os.replace(dest/'model.txt.tmp',dest/'model.txt')
        checkpoint(dest/'trained.json',trained_fp,['model.txt'],cap=cap)
    predict_seconds={k:None for k in candidates}
    for k in candidates:
        began=time.monotonic()
        output=validate_prediction(predict_28_days(estimator,recent,features,select_cutoff,mode,c['threads'],num_iteration=k),c)
        atomic_csv(output,dest/f'predictions_{k}.csv')
        predict_seconds[k]=time.monotonic()-began
        event('candidate_predicted',select_cutoff=select_cutoff,store=store,mode=mode,rounds=k,seconds=predict_seconds[k])
    atomic_json(dict(cap=cap,candidates=candidates,train_seconds=train_seconds,
                     predict_seconds={str(k):predict_seconds[k] for k in candidates}),dest/'timings.json')
    atomic_json(dict(select_cutoff=select_cutoff,store=store,mode=mode,cap=cap,candidates=candidates,
                     train_rows=None if train_seconds is None else train_rows,
                     predict_days=[select_cutoff+1,select_cutoff+28],
                     peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
                     prefix_equivalence=True,note='one training to the cap; candidates scored via model.predict(num_iteration=k)'),dest/'metadata.json')
    checkpoint(dest/'complete.json',fp,products)
    if c.get('keep_selection_models'):
        event('selection_model_kept_for_audit',select_cutoff=select_cutoff,store=store,mode=mode)
    else:
        for path in (dest/'model.txt',dest/'trained.json'):
            if path.exists(): path.unlink()
        event('selection_model_deleted',select_cutoff=select_cutoff,store=store,mode=mode)
    event('candidates_complete',select_cutoff=select_cutoff,store=store,mode=mode,seconds=time.monotonic()-started)

def run_model(c,stage,store,mode,rounds):
    """Deliverable retrain at the stage train_cutoff with the ALREADY selected
    rounds (selection lives only in development/test; final inherits test's —
    scheme 4.7). Under prefix_fallback the model trains to the candidate cap
    and delivers via num_iteration=rounds so selection and delivery share one
    prefix semantics (scheme 4.4 兜底分支)."""
    spec=stage_spec(c,stage); cutoff=spec['train_cutoff']
    dest=stage_root(c,stage)/mode/store; dest.mkdir(parents=True,exist_ok=True)
    fallback=bool(c.get('prefix_fallback')); cap=max(c['round_candidates'])
    trained_rounds=cap if fallback else int(rounds)
    num_iteration=int(rounds) if fallback else None
    products=['model.txt','predictions.csv','importance.csv','metadata.json']
    fp=artifact_id(c,stage,cutoff,store,kind='model',mode=mode,rounds=int(rounds))
    if complete(dest/'complete.json',fp,products):
        event('model_verified',stage=stage,store=store,mode=mode); return
    started=time.monotonic()
    estimator,recent,features,params,train_rows,train_seconds=train_at_cutoff(c,cutoff,store,mode,trained_rounds,dest)
    predict_start=time.monotonic()
    output=validate_prediction(predict_28_days(estimator,recent,features,cutoff,mode,c['threads'],num_iteration=num_iteration),c)
    estimator.save_model(str(dest/'model.txt.tmp')); os.replace(dest/'model.txt.tmp',dest/'model.txt')
    atomic_csv(output,dest/'predictions.csv')
    atomic_csv(pd.DataFrame({'feature':features,'gain':estimator.feature_importance('gain',iteration=num_iteration),
                             'split':estimator.feature_importance('split',iteration=num_iteration)}).sort_values('gain',ascending=False),dest/'importance.csv')
    atomic_json(dict(features=features,params=params,rounds=int(rounds),trained_rounds=trained_rounds,
                     predict_num_iteration=num_iteration,prefix_fallback=fallback,train_rows=train_rows,
                     train_first_day=c.get('first_day',1),train_last_day=cutoff,predict_days=[cutoff+1,cutoff+28],
                     train_seconds=train_seconds,predict_seconds=time.monotonic()-predict_start,
                     total_seconds=time.monotonic()-started,
                     peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)),dest/'metadata.json')
    checkpoint(dest/'complete.json',fp,products)
    event('model_complete',stage=stage,store=store,mode=mode,seconds=time.monotonic()-started,train_seconds=train_seconds)

def equivalence_probe(c,cutoff,store):
    """Smoke-time verification of the prefix-equivalence premise (scheme 4.4):
    predict with the FIRST candidate via num_iteration=k from a cap-round
    model, then train exactly k rounds and predict again; compare. A gap
    beyond tolerance means the premise fails and the fallback branch must be
    decided NOW (enable prefix_fallback), never after a full run."""
    candidates=sorted(c['round_candidates']); cap=candidates[-1]; k=candidates[0]
    results={}
    for mode in c['modes']:
        dest=selection_root(c,cutoff)/mode/store
        prefix_model,recent,features,_,_,_=train_at_cutoff(c,cutoff,store,mode,cap,dest)
        prefix_pred=predict_28_days(prefix_model,recent,features,cutoff,mode,c['threads'],num_iteration=k)
        del prefix_model; gc.collect()
        fresh_model,recent2,features2,_,_,_=train_at_cutoff(c,cutoff,store,mode,k,dest)
        fresh_pred=predict_28_days(fresh_model,recent2,features2,cutoff,mode,c['threads'],num_iteration=None)
        del fresh_model; gc.collect()
        gap=float(np.max(np.abs(prefix_pred[F].to_numpy()-fresh_pred[F].to_numpy())))
        results[mode]=gap
        event('prefix_equivalence_probe',cutoff=cutoff,store=store,mode=mode,candidate=k,cap=cap,max_abs_diff=gap)
    for mode,gap in results.items():
        limit=PREFIX_EQUIVALENCE_LIMITS[mode]
        assert gap<=limit,(f'Prefix equivalence exceeded tolerance for {mode}: {gap} > {limit}. '
                           'Enable prefix_fallback in config.json (train to cap, deliver via num_iteration=k) and re-run smoke; '
                           'the config change alters the fingerprint, so delete the experiments/<experiment> directory '
                           '(or use a new experiment name) before rerunning (scheme 4.4 兜底分支).')
    event('prefix_equivalence_verified',cutoff=cutoff,store=store,gaps=results)
    return results

from m5_shared.model_inputs import MEAN,REMOVE,ROLL,read_feature_tables,predict_recursive as _predict_recursive
predict_recursive=_predict_recursive
