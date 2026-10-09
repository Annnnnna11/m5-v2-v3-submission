"""Reuse original feature definitions, with origin-specific data and global encodings.

Every cutoff owns a private five-table cache under features/cutoff_<d>/<store>;
caches are never shared across cutoffs (scheme 4.3)."""
import gc
import importlib.util
import time
import numpy as np
import pandas as pd
from common import ROOT, META, FILES, features_root, artifact_id, complete, checkpoint, event, atomic_json







def prepare(c,cutoff,store):
    dest=features_root(c,cutoff)/store; dest.mkdir(parents=True,exist_ok=True)
    fp=artifact_id(c,'features',cutoff,store,'cache')
    if complete(dest/'complete.json',fp,FILES):
        event('cache_verified',cutoff=cutoff,store=store); return
    started=time.monotonic()
    p,sales,prices,cal,voc,release,ordinals=inputs(c,cutoff)
    stats=encoding_stats(p,sales,cal,release,cutoff)
    if c.get('smoke_items'):
        chosen=sales.loc[sales.store_id==store,'id'].iloc[:c['smoke_items']]
        sales=sales.loc[sales.id.isin(chosen)].copy()
    grid=p.build_base_grid(store,sales,cal,voc,ordinals,int(release.release.min()),release)
    del sales; gc.collect()
    assert grid.index.is_unique and grid.index.is_monotonic_increasing
    assert grid.loc[grid.d>cutoff,'sales'].isna().all()
    assert grid.loc[grid.d<=cutoff,'sales'].notna().all()
    future=grid.loc[grid.d>cutoff]
    assert future.groupby('id',observed=True).size().eq(28).all()
    expected=c.get('smoke_items',3049)
    assert future.id.nunique()==expected
    p.atomic_pickle(grid,dest/FILES[0])
    for name,build in [
        (FILES[1],lambda:p.build_price_table(grid,p.make_price_features(store,prices,cal),cal)),
        (FILES[2],lambda:p.build_calendar_table(grid,cal,voc)),
        (FILES[3],lambda:p.build_lag_table(grid)),
        (FILES[4],lambda:p.build_encoding_table(grid,stats))]:
        frame=build()
        assert frame.index.equals(grid.index)
        assert frame[['id','d']].equals(grid[['id','d']])
        p.atomic_pickle(frame,dest/name)
        del frame; gc.collect()
    atomic_json(voc,dest/'categories.json')
    checkpoint(dest/'complete.json',fp,FILES,cutoff=cutoff,encoding_end=cutoff,first_day=1,last_day=cutoff+28,
               predict_days=[cutoff+1,cutoff+28],rows=len(grid),series=expected,
               encoding_stores=10,seconds=time.monotonic()-started,
               price_week_max=int(prices.wm_yr_wk.max()))
    event('features_complete',cutoff=cutoff,store=store,seconds=time.monotonic()-started)

from m5_shared.feature_inputs import legacy, inputs as _inputs, encoding_stats as _encoding_stats
inputs=_inputs
encoding_stats=_encoding_stats
