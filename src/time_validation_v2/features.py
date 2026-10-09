"""Reuse original feature definitions, with origin-specific data and global encodings."""
import gc
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from common import ROOT, META, FILES, stage_root, artifact_id, complete, checkpoint, event, atomic_json



def inputs(c, stage):
    return _inputs(c,c['stages'][stage])

def encoding_stats(p,sales,cal,release,cut):
    return _encoding_stats(p,sales,cal,release,cut)

def prepare(c,stage,store):
    dest=stage_root(c,stage)/'cache'/store; dest.mkdir(parents=True,exist_ok=True)
    fp=artifact_id(c,stage,store,'cache')
    if complete(dest/'complete.json',fp,FILES):
        event('cache_verified',stage=stage,store=store); return
    started=time.monotonic(); cut=c['stages'][stage]
    p,sales,prices,cal,voc,release,ordinals=inputs(c,stage)
    stats=encoding_stats(p,sales,cal,release,cut)
    if c.get('smoke_items'):
        chosen=sales.loc[sales.store_id==store,'id'].iloc[:c['smoke_items']]
        sales=sales.loc[sales.id.isin(chosen)].copy()
    grid=p.build_base_grid(store,sales,cal,voc,ordinals,int(release.release.min()),release)
    del sales; gc.collect()
    assert grid.index.is_unique and grid.index.is_monotonic_increasing
    assert grid.loc[grid.d>cut,'sales'].isna().all()
    assert grid.loc[grid.d<=cut,'sales'].notna().all()
    future=grid.loc[grid.d>cut]
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
    checkpoint(dest/'complete.json',fp,FILES,cutoff=cut,first_day=1,last_day=cut+28,
               rows=len(grid),series=expected,encoding_end=cut,encoding_stores=10,
               seconds=time.monotonic()-started,price_week_max=int(prices.wm_yr_wk.max()))
    event('features_complete',stage=stage,store=store,seconds=time.monotonic()-started)

from m5_shared.feature_inputs import legacy, inputs as _inputs, encoding_stats as _encoding_stats
