import gc
import json
import os
import resource
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import F, FILES, stage_root, artifact_id, complete, checkpoint, atomic_csv, atomic_json, event


def load_frame(c,stage,store,mode):
    dest=stage_root(c,stage)/'cache'/store
    assert complete(dest/'complete.json',artifact_id(c,stage,store,'cache'),FILES)
    return read_feature_tables(dest,mode)

def predict_recursive(model,frame,features,cut,threads):
    return _predict_recursive(model,frame,features,cut,threads)

def run_model(c,stage,store,mode):
    dest=stage_root(c,stage)/mode/store; dest.mkdir(parents=True,exist_ok=True)
    fp=artifact_id(c,stage,store,mode); products=['model.txt','predictions.csv','importance.csv','metadata.json']
    if complete(dest/'complete.json',fp,products):
        event('model_verified',stage=stage,store=store,mode=mode); return
    started=time.monotonic(); cut=c['stages'][stage]
    frame,features=load_frame(c,stage,store,mode)
    train_mask=frame.d.le(cut); forecast_mask=frame.d.gt(cut)&frame.d.le(cut+28)
    assert not (train_mask & forecast_mask).any()
    assert frame.loc[train_mask,'sales'].notna().all()
    assert frame.loc[forecast_mask,'sales'].isna().all()
    assert frame.loc[train_mask,'d'].max()==cut
    assert frame.loc[forecast_mask,'d'].min()==cut+1
    # No overlapping LightGBM valid_set. Holdout is scored only after 28-day inference.
    params={**c['params'],**c['mode_params'][mode],'num_threads':c['threads']}
    train_rows=int(train_mask.sum())
    future=frame.loc[frame.d>cut-100].copy()
    cache=dest/'cache'; cache.mkdir(exist_ok=True)
    cache_fp=artifact_id(c,stage,store,mode+'_training_cache')
    schema=dict(features=features,params=params,train_first_day=1,train_last_day=cut,train_rows=train_rows)
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
    train_start=time.monotonic()
    event('training_started',stage=stage,store=store,mode=mode,rows=train_rows,features=len(features),rounds=c['rounds'])
    def progress(env):
        if (env.iteration+1)%100==0:
            event('iteration',stage=stage,store=store,mode=mode,iteration=env.iteration+1,seconds=time.monotonic()-train_start)
    estimator=lgb.train(params,dataset,num_boost_round=c['rounds'],callbacks=[progress])
    train_seconds=time.monotonic()-train_start
    estimator.save_model(str(dest/'model.txt.tmp')); os.replace(dest/'model.txt.tmp',dest/'model.txt')
    del dataset; gc.collect()
    predict_start=time.monotonic()
    if mode=='recursive': output=predict_recursive(estimator,future,features,cut,c['threads'])
    else:
        rows=future.loc[future.d>cut].copy()
        rows['prediction']=np.asarray(estimator.predict(rows[features],num_threads=c['threads']),dtype=np.float64)
        wide=rows.pivot(index='id',columns='d',values='prediction')
        assert wide.columns.tolist()==list(range(cut+1,cut+29))
        wide.columns=F; output=wide.reset_index()
    assert output.id.nunique()==c.get('smoke_items',3049)
    assert np.isfinite(output[F].to_numpy()).all() and (output[F].to_numpy()>=0).all()
    atomic_csv(output,dest/'predictions.csv')
    atomic_csv(pd.DataFrame({'feature':features,'gain':estimator.feature_importance('gain'),'split':estimator.feature_importance('split')}).sort_values('gain',ascending=False),dest/'importance.csv')
    atomic_json(dict(features=features,params=params,rounds=c['rounds'],train_rows=train_rows,
                     train_first_day=1,train_last_day=cut,predict_days=[cut+1,cut+28],
                     train_seconds=train_seconds,predict_seconds=time.monotonic()-predict_start,
                     total_seconds=time.monotonic()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),dest/'metadata.json')
    checkpoint(dest/'complete.json',fp,products)
    event('model_complete',stage=stage,store=store,mode=mode,seconds=time.monotonic()-started,train_seconds=train_seconds)

from m5_shared.model_inputs import MEAN,REMOVE,ROLL,read_feature_tables,predict_recursive as _predict_recursive
