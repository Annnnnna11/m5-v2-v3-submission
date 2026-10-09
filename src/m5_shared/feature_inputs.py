import importlib.util
import numpy as np
import pandas as pd
from common import ROOT,META
def legacy(cutoff):
    spec=importlib.util.spec_from_file_location('base_features',ROOT/'src/1_preprocessing_by_store.py')
    p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    p.END_TRAIN=cutoff; p.TOTAL_DAYS=cutoff+28; p.ENCODING_END=cutoff
    return p

def inputs(c, cutoff):
    p=legacy(cutoff)
    cal=pd.read_csv(ROOT/'data/calendar.csv')
    cal['d_num']=cal.d.str.removeprefix('d_').astype(np.int16)
    cal=cal.loc[cal.d_num<=cutoff+28].copy()
    meta=pd.read_csv(ROOT/'data/sales_train_evaluation.csv', usecols=META, dtype='string')
    voc=p.make_vocabs(meta,cal)
    dtype={k:pd.CategoricalDtype(voc[k]) for k in META}
    days=[f'd_{d}' for d in range(1,cutoff+1)]
    dtype.update({d:np.float32 for d in days})
    sales=pd.read_csv(ROOT/'data/sales_train_evaluation.csv', usecols=META+days, dtype=dtype)
    prices=pd.read_csv(ROOT/'data/sell_prices.csv', dtype={'store_id':dtype['store_id'],'item_id':dtype['item_id'],'wm_yr_wk':np.int16,'sell_price':np.float32})
    prices=prices.loc[prices.wm_yr_wk.isin(cal.wm_yr_wk.unique())].sort_values(['store_id','item_id','wm_yr_wk']).reset_index(drop=True)
    assert not prices.duplicated(['store_id','item_id','wm_yr_wk']).any()
    release=prices.groupby(['store_id','item_id'],observed=True).wm_yr_wk.min().rename('release').reset_index()
    for k in ['event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']:
        cal[k]=p.as_global_category(cal[k],voc[k])
    ordinals={id:i for i,id in enumerate(meta.id)}
    return p,sales,prices,cal,voc,release,ordinals

def encoding_stats(p,sales,cal,release,cutoff):
    # Sufficient statistics over all stores, without a 59M-row long table.
    # Match the original release filtering; use only d<=this origin.
    rows=sales[META].merge(release,on=['store_id','item_id'],how='left',validate='one_to_one',sort=False)
    assert rows.id.astype(str).tolist()==sales.id.astype(str).tolist()
    y=sales[[f'd_{d}' for d in range(1,cutoff+1)]].to_numpy(dtype=np.float64)
    weeks=cal.set_index('d_num').loc[np.arange(1,cutoff+1),'wm_yr_wk'].to_numpy()
    mask=weeks[None,:]>=rows.release.to_numpy()[:,None]
    rows['count']=mask.sum(axis=1)
    y[~mask]=0
    rows['total']=y.sum(axis=1); rows['sumsq']=np.einsum('ij,ij->i',y,y)
    del y,mask
    stats={}
    for keys in p.ENCODING_GROUPS:
        s=rows.groupby(keys,observed=True)[['count','total','sumsq']].sum().reset_index()
        s['mean']=s.total/s['count'].replace(0,np.nan)
        var=(s.sumsq-s.total**2/s['count'].replace(0,np.nan)).clip(lower=0)/(s['count']-1).replace(0,np.nan)
        s['std']=np.sqrt(var).where(s['count']>=2)
        stats[tuple(keys)]=s[[*keys,'count','mean','std']]
    return stats
