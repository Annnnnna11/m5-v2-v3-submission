import gc
import numpy as np
import pandas as pd
from common import F,FILES
MEAN={'recursive':['enc_cat_id_mean','enc_cat_id_std','enc_dept_id_mean','enc_dept_id_std','enc_item_id_mean','enc_item_id_std'], 'nonrecursive':['enc_store_id_dept_id_mean','enc_store_id_dept_id_std','enc_item_id_state_id_mean','enc_item_id_state_id_std']}
REMOVE={'id','state_id','store_id','date','wm_yr_wk','d','sales'}
ROLL=[(s,w) for s in [1,7,14] for w in [7,14,30,60]]
def read_feature_tables(dest,mode):
    base=pd.read_pickle(dest/FILES[0]); reference=base.index
    for name in FILES[1:]:
        table=pd.read_pickle(dest/name)
        assert table.index.equals(reference)
        assert table[['id','d']].equals(base[['id','d']])
        cols=[col for col in table if col not in {'id','d','sales'}]
        if name==FILES[4]: cols=MEAN[mode]
        if mode=='nonrecursive': cols=[col for col in cols if '_tmp_' not in col]
        for col in cols:
            assert col not in base
            base[col]=table[col]
        del table; gc.collect()
    features=[col for col in base if col not in REMOVE]
    return base,features

def predict_recursive(model,frame,features,cut,threads,num_iteration=None):
    history=frame.loc[frame.d>cut-100,['id','d','sales']].copy()
    history['sales']=history.sales.astype(np.float32)
    history.loc[history.d>cut,'sales']=np.float32(np.nan)
    assert history.loc[history.d>cut,'sales'].isna().all()
    future=frame.loc[frame.d>cut,['id','d',*features]].copy()
    # Pure positional history matrix keyed by explicit id and day. Each day recomputes
    # short rolling features from observed history plus earlier predictions only.
    ids=pd.Index(future.id.astype(str).drop_duplicates())
    days=np.arange(cut-99,cut+29)
    wide=history.assign(id=history.id.astype(str)).pivot(index='id',columns='d',values='sales').reindex(index=ids,columns=days)
    values=wide.to_numpy(dtype=np.float32)
    result=np.empty((len(ids),28),dtype=np.float64)
    for h in range(28):
        day=cut+h+1; position=100+h
        rows=future.loc[future.d==day].copy()
        order=ids.get_indexer(rows.id.astype(str)); assert (order>=0).all()
        for shift,window in ROLL:
            start=position-shift-window+1; end=position-shift+1
            # np.mean deliberately requires the entire window, matching pandas min_periods.
            rows[f'rolling_mean_tmp_{shift}_{window}']=values[order,start:end].mean(axis=1,dtype=np.float64).astype(np.float32)
        pred=np.asarray(model.predict(rows[features],num_threads=threads,num_iteration=num_iteration),dtype=np.float64)
        assert np.isfinite(pred).all() and (pred>=0).all()
        values[order,position]=pred.astype(np.float32)
        result[order,h]=pred
    return pd.DataFrame(result,index=ids,columns=F).rename_axis('id').reset_index()
