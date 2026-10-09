import json
import numpy as np
import pandas as pd
from common import ROOT, META, F, stage_root, run_root, atomic_csv, atomic_json, event, artifact_id, checkpoint, complete
from metrics import WRMSSE, errors, aligned

def revenue_at(sales,cut):
    cal=pd.read_csv(ROOT/'data/calendar.csv').set_index('d')
    prices=pd.read_csv(ROOT/'data/sell_prices.csv',dtype={'sell_price':np.float64})
    weeks=cal.loc[[f'd_{d}' for d in range(cut-27,cut+1)],'wm_yr_wk'].to_numpy()
    prices=prices.loc[prices.wm_yr_wk.isin(weeks)]
    lookup=prices.set_index(['store_id','item_id','wm_yr_wk']).sell_price
    revenue=np.zeros(len(sales))
    for day,week in zip(range(cut-27,cut+1),weeks):
        index=pd.MultiIndex.from_arrays([sales.store_id,sales.item_id,np.full(len(sales),week)])
        price=lookup.reindex(index).to_numpy()
        y=sales[f'd_{day}'].to_numpy(dtype=np.float64)
        if ((y>0)&~np.isfinite(price)).any(): raise ValueError('Positive sales lack weight price')
        revenue+=y*np.nan_to_num(price,nan=0.0)
    return revenue

def plot_report(rows,analysis,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows)
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

def evaluate(c,stage):
    dest=stage_root(c,stage); out=dest/'results'; out.mkdir(parents=True,exist_ok=True)
    cut=c['stages'][stage]
    sales=pd.read_csv(ROOT/'data/sales_train_evaluation.csv',dtype={f'd_{d}':np.float32 for d in range(1,1942)})
    sales=sales.loc[sales.store_id.isin(c['stores'])].reset_index(drop=True)
    if c.get('smoke_items'): sales=sales.groupby('store_id',sort=False).head(c['smoke_items']).reset_index(drop=True)
    ids=sales.id.tolist(); predictions={}
    for mode in c['modes']:
        parts=[]
        for store in c['stores']:
            modeldir=dest/mode/store
            assert complete(modeldir/'complete.json',artifact_id(c,stage,store,mode),['model.txt','predictions.csv','importance.csv','metadata.json'])
            metadata=json.loads((modeldir/'metadata.json').read_text())
            assert metadata['train_last_day']==cut and metadata['predict_days']==[cut+1,cut+28]
            parts.append(pd.read_csv(modeldir/'predictions.csv'))
        predictions[mode]=aligned(pd.concat(parts,ignore_index=True),ids)
    predictions['ensemble']=sum(c['ensemble'][name]*predictions[name] for name in c['modes'])
    history=sales[[f'd_{d}' for d in range(1,cut+1)]].to_numpy(dtype=np.float64)
    predictions['last_week']=np.tile(history[:,-7:],(1,4))
    predictions['weekday_mean_4weeks']=np.tile(history[:,-28:].reshape(len(sales),4,7).mean(axis=1),(1,4))
    for name,pred in predictions.items():
        frame=pd.DataFrame(pred,columns=F); frame.insert(0,'id',ids)
        aligned(frame,ids); atomic_csv(frame,out/f'{name}.csv')
    integrity=dict(rows=len(ids),stores=len(c['stores']),horizon=28,days=[cut+1,cut+28],
                   finite=True,nonnegative=True,unique_ids=True,full_coverage=not bool(c.get('smoke_items')))
    atomic_json(integrity,out/'integrity.json')
    if stage!='final':
        truth=sales[[f'd_{d}' for d in range(cut+1,cut+29)]].to_numpy(dtype=np.float64)
        scorer=WRMSSE(sales[META],history,revenue_at(sales,cut),full=not c.get('smoke_items'))
        scores=[]; analysis=[]
        for name,pred in predictions.items():
            score,levels=scorer.score(truth,pred)
            atomic_csv(levels,out/f'{name}_levels.csv')
            scores.append(dict(model=name,WRMSSE=score,**errors(truth,pred)))
            for dim in ['store_id','cat_id']:
                for group,index in sales.groupby(dim).groups.items():
                    analysis.append(dict(model=name,dimension=dim,group=group,**errors(truth[index],pred[index])))
            for h in range(28):
                analysis.append(dict(model=name,dimension='horizon',group=str(h+1),**errors(truth[:,h],pred[:,h])))
        atomic_csv(pd.DataFrame(scores),out/'scores.csv'); details=pd.DataFrame(analysis)
        atomic_csv(details,out/'error_analysis.csv'); plot_report(scores,details,out)
        event('scores',stage=stage,scores=scores)
    timings=[]; importance=[]
    for mode in c['modes']:
        for store in c['stores']:
            md=json.loads((dest/mode/store/'metadata.json').read_text())
            timings.append(dict(mode=mode,store=store,**{k:v for k,v in md.items() if k.endswith('seconds') or k=='peak_rss_mb'}))
            imp=pd.read_csv(dest/mode/store/'importance.csv'); imp['mode']=mode; imp['store']=store; importance.append(imp)
    atomic_csv(pd.DataFrame(timings),out/'timings.csv')
    importance=pd.concat(importance); atomic_csv(importance,out/'importance.csv')
    import matplotlib.pyplot as plt
    for mode in c['modes']:
        imp=importance[importance['mode']==mode].groupby('feature').gain.sum().nlargest(20).sort_values()
        ax=imp.plot.barh(figsize=(10,7)); ax.figure.tight_layout(); ax.figure.savefig(out/f'{mode}_importance.png',dpi=140); plt.close(ax.figure)
    products=[p.name for p in out.iterdir() if p.is_file() and p.name!='complete.json']
    checkpoint(out/'complete.json',artifact_id(c,stage,kind='results'),products,**integrity)
    event('evaluation_complete',stage=stage,**integrity)

def kaggle(c):
    assert not c.get('smoke_items')
    final=stage_root(c,'final')/'results'; test=stage_root(c,'test')/'results'
    validation=pd.read_csv(test/'ensemble.csv')
    validation['id']=validation.id.str.replace('_evaluation','_validation',regex=False)
    evaluation=pd.read_csv(final/'ensemble.csv')
    both=pd.concat([validation,evaluation],ignore_index=True)
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    values=aligned(both,sample.id.tolist())
    assert len(sample)==60980
    result=pd.DataFrame(values,columns=F); result.insert(0,'id',sample.id)
    atomic_csv(result,final/'kaggle_ensemble.csv')
    atomic_json({'validation':{'train_end':1913,'days':[1914,1941],'content':'genuine out-of-sample test-window equal blend'},
                 'evaluation':{'train_end':1941,'days':[1942,1969],'content':'final full-history equal blend'},
                 'rows':60980,'submitted':False},final/'kaggle_semantics.json')
