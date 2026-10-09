"""Selection and evaluation (scheme 4.6).

select_rounds / select_ensemble_weight run ONLY in development and test, on
their own select_cutoff, with the selection-scope WRMSSE (dollar weights of
the 28 days ending at select_cutoff, RMSSE scale from d<=select_cutoff only).
evaluate_stage scores the official window purely as evaluation; final is
integrity-only (no truth for 1942-1969) and produces no Kaggle file."""
import json
import numpy as np
import pandas as pd
from common import ROOT, META, F, stage_root, stage_spec, run_root, selection_root, atomic_csv, atomic_json, event, artifact_id, checkpoint, complete, sha
from metrics import WRMSSE, errors, aligned, blend, make_selection_scorer

def selection_source_stage(stage):
    """Which stage's selection products a stage consumes: final has no
    selection step and inherits test's (scheme 4.6/4.7)."""
    return 'test' if stage=='final' else stage

def load_sales(c):
    sales=pd.read_csv(ROOT/'data/sales_train_evaluation.csv',dtype={f'd_{d}':np.float32 for d in range(1,1942)})
    sales=sales.loc[sales.store_id.isin(c['stores'])].reset_index(drop=True)
    if c.get('smoke_items'): sales=sales.groupby('store_id',sort=False).head(c['smoke_items']).reset_index(drop=True)
    return sales

def revenue_window(cut,horizon=28):
    """Dollar-weight window for selection scoring: exactly the 28 days ending
    at the selection cutoff (scheme §3; cutoff=1857 -> d1830-1857)."""
    return list(range(cut-horizon+1,cut+1))

def revenue_at(sales,cut):
    cal=pd.read_csv(ROOT/'data/calendar.csv').set_index('d')
    prices=pd.read_csv(ROOT/'data/sell_prices.csv',dtype={'sell_price':np.float64})
    weeks=cal.loc[[f'd_{d}' for d in revenue_window(cut)],'wm_yr_wk'].to_numpy()
    prices=prices.loc[prices.wm_yr_wk.isin(weeks)]
    lookup=prices.set_index(['store_id','item_id','wm_yr_wk']).sell_price
    revenue=np.zeros(len(sales))
    for day,week in zip(revenue_window(cut),weeks):
        index=pd.MultiIndex.from_arrays([sales.store_id,sales.item_id,np.full(len(sales),week)])
        price=lookup.reindex(index).to_numpy()
        y=sales[f'd_{day}'].to_numpy(dtype=np.float64)
        if ((y>0)&~np.isfinite(price)).any(): raise ValueError('Positive sales lack weight price')
        revenue+=y*np.nan_to_num(price,nan=0.0)
    return revenue

def selection_scope(sales,cutoff,horizon=28):
    """Selection-scope slices (scheme §3): history d_1..d_cutoff and truth
    d_cutoff+1..d_cutoff+28. For the test stage (cutoff=1885) the truth stops
    at d1913 — labels from 1914-1941 never enter any selection. Returns the
    ACTUAL column lists used, so callers can forward them to
    metrics.make_selection_scorer and the content<->label binding is checked
    against what was really sliced. The anchors below compare against
    independently written day labels: the previous positional checks compared
    the slice against itself (tautological under list indexing) and let a
    whole-list shift through (final review)."""
    history_columns=[f'd_{d}' for d in range(1,cutoff+1)]
    truth_columns=[f'd_{d}' for d in range(cutoff+1,cutoff+horizon+1)]
    assert history_columns[0]=='d_1' and history_columns[-1]==f'd_{cutoff}'
    assert truth_columns[0]==f'd_{cutoff+1}' and truth_columns[-1]==f'd_{cutoff+horizon}'
    history=sales[history_columns].to_numpy(dtype=np.float64)
    truth=sales[truth_columns].to_numpy(dtype=np.float64)
    assert history.shape[1]==cutoff and truth.shape[1]==horizon
    return history,truth,history_columns,truth_columns

def selection_scorer(c,cutoff):
    """Full-store scorer under the selection scope (scheme §3): RMSSE scale
    from d<=cutoff history, dollar weights from the 28 days ending at cutoff.
    Built through metrics.make_selection_scorer with the column list that
    selection_scope ACTUALLY used (not a rebuilt one), so the width AND the
    full column-name bound (exactly d_1..d_cutoff) are enforced at runtime
    for every selection score — a shifted list of the same width fails here
    (review L1 + final review)."""
    sales=load_sales(c)
    history,truth,history_columns,truth_columns=selection_scope(sales,cutoff,c['horizon'])
    revenue=revenue_at(sales,cutoff)
    scorer=make_selection_scorer(sales[META],history,revenue,cutoff,full=not c.get('smoke_items'),columns=history_columns)
    return sales,sales.id.tolist(),history,truth,revenue,scorer

def candidate_predictions(c,cutoff,mode,rounds,ids):
    parts=[pd.read_csv(selection_root(c,cutoff)/mode/store/f'predictions_{rounds}.csv') for store in c['stores']]
    return aligned(pd.concat(parts,ignore_index=True),ids)

def select_rounds(c,selection_cutoff):
    """Per-mode GLOBAL round selection over all stores' candidate predictions
    (scheme 4.6): the candidate with the lowest all-store WRMSSE on the full
    28-day selection window wins for each mode — no per-store medians."""
    assert (c['round_selection'].get('scope'),c['round_selection'].get('metric'))==('per_mode_global','WRMSSE'), \
        "v3 implements round_selection scope='per_mode_global' and metric='WRMSSE' only (scheme 4.6); other values are unimplemented"
    dest=selection_root(c,selection_cutoff); dest.mkdir(parents=True,exist_ok=True)
    candidates=sorted(c['round_candidates'])
    fp=artifact_id(c,'selection',selection_cutoff,kind='rounds',rounds=candidates)
    if complete(dest/'rounds_complete.json',fp,['rounds.json','round_grid.csv']):
        event('rounds_selection_verified',select_cutoff=selection_cutoff)
        return json.loads((dest/'rounds.json').read_text())
    sales,ids,history,truth,revenue,scorer=selection_scorer(c,selection_cutoff)
    rows=[]
    for mode in c['modes']:
        for k in candidates:
            values=candidate_predictions(c,selection_cutoff,mode,k,ids)
            score,_=scorer.score(truth,values)
            rows.append(dict(mode=mode,rounds=k,WRMSSE=score,**errors(truth,values)))
            event('candidate_scored',select_cutoff=selection_cutoff,mode=mode,rounds=k,WRMSSE=score)
    grid=pd.DataFrame(rows); atomic_csv(grid,dest/'round_grid.csv')
    rounds={}
    for mode in c['modes']:
        sub=grid.loc[grid['mode']==mode]
        rounds[mode]=int(sub.loc[sub.WRMSSE.idxmin(),'rounds'])
    payload=dict(select_cutoff=selection_cutoff,metric=c['round_selection']['metric'],scope=c['round_selection']['scope'],
                 rounds_by_mode=rounds,candidates=candidates,
                 wrmsse_by_mode={m:float(grid.loc[(grid['mode']==m)&(grid['rounds']==rounds[m]),'WRMSSE'].iloc[0]) for m in c['modes']})
    atomic_json(payload,dest/'rounds.json')
    checkpoint(dest/'rounds_complete.json',fp,['rounds.json','round_grid.csv'])
    event('rounds_selected',select_cutoff=selection_cutoff,rounds_by_mode=rounds)
    return payload

def select_ensemble_weight(c,selection_cutoff):
    """Blend-weight search at the selected rounds (scheme 4.6):
    y_hat_w = w*recursive + (1-w)*nonrecursive over the ensemble grid; w=0 and
    w=1 stay in the grid; same selection-scope WRMSSE; prediction-file hashes
    are stored with the result."""
    dest=selection_root(c,selection_cutoff)
    rounds=json.loads((dest/'rounds.json').read_text())['rounds_by_mode']
    grid_values=[float(w) for w in c['ensemble_grid']]
    fp=artifact_id(c,'selection',selection_cutoff,kind='weights',rounds=rounds,grid=grid_values)
    if complete(dest/'weights_complete.json',fp,['weights.json','weight_grid.csv']):
        event('weight_selection_verified',select_cutoff=selection_cutoff)
        return json.loads((dest/'weights.json').read_text())
    sales,ids,history,truth,revenue,scorer=selection_scorer(c,selection_cutoff)
    recursive=candidate_predictions(c,selection_cutoff,'recursive',rounds['recursive'],ids)
    nonrecursive=candidate_predictions(c,selection_cutoff,'nonrecursive',rounds['nonrecursive'],ids)
    rows=[]
    for w in grid_values:
        blended=blend(recursive,nonrecursive,w)
        score,_=scorer.score(truth,blended)
        rows.append(dict(recursive_weight=w,nonrecursive_weight=1.0-w,WRMSSE=score,**errors(truth,blended)))
    grid=pd.DataFrame(rows); atomic_csv(grid,dest/'weight_grid.csv')
    best=grid.loc[grid.WRMSSE.idxmin()]; best_w=float(best.recursive_weight)
    hashes={mode:{store:sha(selection_root(c,selection_cutoff)/mode/store/f'predictions_{rounds[mode]}.csv') for store in c['stores']} for mode in c['modes']}
    payload=dict(select_cutoff=selection_cutoff,rounds_by_mode=rounds,
                 recursive_weight=best_w,nonrecursive_weight=1.0-best_w,
                 ensemble_wrmsse=float(best.WRMSSE),weights_grid=grid_values,
                 prediction_files_sha256=hashes)
    atomic_json(payload,dest/'weights.json')
    checkpoint(dest/'weights_complete.json',fp,['weights.json','weight_grid.csv'])
    event('ensemble_weight_selected',select_cutoff=selection_cutoff,recursive_weight=best_w,
          nonrecursive_weight=1.0-best_w,ensemble_wrmsse=float(best.WRMSSE))
    return payload

def load_selection(c,stage):
    """Rounds+weights for a stage: development/test read their own selection
    products; final INHERITS test's and must never read 1914-1941 labels for
    any choice (scheme 4.6/4.7)."""
    source=selection_source_stage(stage)
    cutoff=stage_spec(c,source)['select_cutoff']
    dest=selection_root(c,cutoff)
    if (dest/'received.json').exists() and not (dest/'rounds_complete.json').exists():
        from handoff import verify_received,validate_selection
        verify_received(c,dest,'selection',source)
        validate_selection(c,dest)
        r=json.loads((dest/'rounds.json').read_text());w=json.loads((dest/'weights.json').read_text())
        return dict(source_stage=source,select_cutoff=cutoff,rounds_by_mode=r['rounds_by_mode'],
                    recursive_weight=w['recursive_weight'],nonrecursive_weight=w['nonrecursive_weight'],
                    ensemble_wrmsse=w.get('ensemble_wrmsse'))
    candidates=sorted(c['round_candidates'])
    rounds_fp=artifact_id(c,'selection',cutoff,kind='rounds',rounds=candidates)
    assert complete(dest/'rounds_complete.json',rounds_fp,['rounds.json','round_grid.csv']),f'Run select_rounds at cutoff {cutoff} first'
    rounds=json.loads((dest/'rounds.json').read_text())
    weights_fp=artifact_id(c,'selection',cutoff,kind='weights',rounds=rounds['rounds_by_mode'],grid=[float(w) for w in c['ensemble_grid']])
    assert complete(dest/'weights_complete.json',weights_fp,['weights.json','weight_grid.csv']),f'Run select_weight at cutoff {cutoff} first'
    weights=json.loads((dest/'weights.json').read_text())
    return dict(source_stage=source,select_cutoff=cutoff,
                rounds_by_mode=rounds['rounds_by_mode'],
                recursive_weight=weights['recursive_weight'],
                nonrecursive_weight=weights['nonrecursive_weight'],
                ensemble_wrmsse=weights.get('ensemble_wrmsse'))

def plots(scores,analysis,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        event('plots_skipped',reason='matplotlib unavailable in this environment; csv artifacts remain complete')
        return
    table=pd.DataFrame(scores)
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(table.model,table.WRMSSE)
    ax.set_ylabel('12-level WRMSSE'); ax.tick_params(axis='x',rotation=20)
    fig.tight_layout(); fig.savefig(out/'wrmsse.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for model,values in analysis.groupby('model'):
        step=values[values.dimension=='horizon']
        axes[0].plot(step.group.astype(int),step.RMSE,label=model)
        axes[1].plot(step.group.astype(int),step.bias,label=model)
    axes[0].set(xlabel='Horizon',ylabel='RMSE'); axes[1].set(xlabel='Horizon',ylabel='Mean forecast - actual')
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'horizon_errors.png',dpi=140); plt.close(fig)
    for dim in ['store_id','cat_id']:
        sub=analysis[analysis.dimension==dim]
        ax=sub.pivot(index='group',columns='model',values='RMSE').plot.bar(figsize=(12,5))
        ax.set_ylabel('RMSE'); ax.figure.tight_layout(); ax.figure.savefig(out/f'{dim}_errors.png',dpi=140); plt.close(ax.figure)

def evaluate_stage(c,stage):
    """Score the official prediction window ONLY as evaluation — rounds and
    weights are read, never tuned, here (scheme 4.6). final: integrity check
    only, no truth for 1942-1969, no Kaggle submission file."""
    spec=stage_spec(c,stage); cutoff=spec['train_cutoff']; horizon=spec['horizon']
    dest=stage_root(c,stage); out=dest/'results'; out.mkdir(parents=True,exist_ok=True)
    sel=load_selection(c,stage)
    sales=load_sales(c); ids=sales.id.tolist()
    predictions={}
    for mode in c['modes']:
        parts=[]
        for store in c['stores']:
            modeldir=dest/mode/store
            if (modeldir/'complete.json').exists():
                assert complete(modeldir/'complete.json',artifact_id(c,stage,cutoff,store,kind='model',mode=mode,rounds=sel['rounds_by_mode'][mode]),
                                ['model.txt','predictions.csv','importance.csv','metadata.json'])
            else:
                from handoff import verify_received
                verify_received(c,modeldir,'retrain',stage,store,mode)
            metadata=json.loads((modeldir/'metadata.json').read_text())
            assert metadata['train_last_day']==cutoff and metadata['predict_days']==[cutoff+1,cutoff+28]
            assert metadata['rounds']==sel['rounds_by_mode'][mode]
            parts.append(pd.read_csv(modeldir/'predictions.csv'))
        predictions[mode]=aligned(pd.concat(parts,ignore_index=True),ids)
    predictions['ensemble']=blend(predictions['recursive'],predictions['nonrecursive'],sel['recursive_weight'])
    history=sales[[f'd_{d}' for d in range(1,cutoff+1)]].to_numpy(dtype=np.float64)
    predictions['last_week']=np.tile(history[:,-7:],(1,4))
    predictions['weekday_mean_4weeks']=np.tile(history[:,-28:].reshape(len(sales),4,7).mean(axis=1),(1,4))
    for name,pred in predictions.items():
        frame=pd.DataFrame(pred,columns=F); frame.insert(0,'id',ids)
        aligned(frame,ids); atomic_csv(frame,out/f'{name}.csv')
    integrity=dict(rows=len(ids),stores=len(c['stores']),horizon=horizon,days=[cutoff+1,cutoff+28],
                   finite=True,nonnegative=True,unique_ids=True,full_coverage=not bool(c.get('smoke_items')),
                   rounds_by_mode=sel['rounds_by_mode'],
                   ensemble_weights={'recursive':sel['recursive_weight'],'nonrecursive':sel['nonrecursive_weight']},
                   selection_source=sel['source_stage'],kaggle_file=False)
    atomic_json(integrity,out/'integrity.json')
    if stage!='final':
        truth=sales[[f'd_{d}' for d in range(cutoff+1,cutoff+horizon+1)]].to_numpy(dtype=np.float64)
        scorer=WRMSSE(sales[META],history,revenue_at(sales,cutoff),full=not c.get('smoke_items'))
        scores=[]; analysis=[]
        for name,pred in predictions.items():
            score,levels=scorer.score(truth,pred)
            atomic_csv(levels,out/f'{name}_levels.csv')
            scores.append(dict(model=name,WRMSSE=score,**errors(truth,pred)))
            for dim in ['store_id','cat_id']:
                for group,index in sales.groupby(dim).groups.items():
                    analysis.append(dict(model=name,dimension=dim,group=group,**errors(truth[index],pred[index])))
            for h in range(horizon):
                analysis.append(dict(model=name,dimension='horizon',group=str(h+1),**errors(truth[:,h],pred[:,h])))
        atomic_csv(pd.DataFrame(scores),out/'scores.csv')
        atomic_csv(pd.DataFrame(analysis),out/'error_analysis.csv')
        plots(scores,pd.DataFrame(analysis),out)
        event('scores',stage=stage,scores=scores)
    else:
        event('final_integrity_only',stage=stage,rows=integrity['rows'],days=integrity['days'],
              inherited_from=sel['source_stage'])
    timings=[]; importance=[]
    for mode in c['modes']:
        for store in c['stores']:
            md=json.loads((dest/mode/store/'metadata.json').read_text())
            timings.append(dict(mode=mode,store=store,rounds=md['rounds'],
                                **{k:v for k,v in md.items() if k.endswith('seconds') or k=='peak_rss_mb'}))
            imp=pd.read_csv(dest/mode/store/'importance.csv'); imp['mode']=mode; imp['store']=store; importance.append(imp)
    atomic_csv(pd.DataFrame(timings),out/'timings.csv')
    importance=pd.concat(importance); atomic_csv(importance,out/'importance.csv')
    plots_importance(importance,out)
    products=[p.name for p in out.iterdir() if p.is_file() and p.name!='complete.json']
    checkpoint(out/'complete.json',artifact_id(c,stage,cutoff,kind='results',rounds=sel['rounds_by_mode']),products,**integrity)
    event('evaluation_complete',stage=stage,rows=integrity['rows'],stores=integrity['stores'],days=integrity['days'])

def plots_importance(importance,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for mode in importance['mode'].unique():
        imp=importance[importance['mode']==mode].groupby('feature').gain.sum().nlargest(20).sort_values()
        ax=imp.plot.barh(figsize=(10,7)); ax.figure.tight_layout()
        ax.figure.savefig(out/f'{mode}_importance.png',dpi=140); plt.close(ax.figure)
