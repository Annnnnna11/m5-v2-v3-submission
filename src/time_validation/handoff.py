"""Verified prediction handoffs; foreign checkpoints are never rewritten."""
import importlib.metadata
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,HERE,META,F,sha,digest,atomic_json,manifest,run_root,selection_root,stage_root,stage_spec,artifact_id,complete

CORE_PACKAGES=['lightgbm','numpy','pandas','scipy','scikit-learn']

def plan():
    return json.loads((HERE/'two_machine_plan.json').read_text())

def contract(c):
    m=json.loads((run_root(c)/'manifest.json').read_text())
    semantic={k:v for k,v in c.items() if k not in ['threads','experiment']}
    return dict(plan_id=plan()['plan_id'],task_plan=plan(),config=semantic,code=m['code'],inputs=m['inputs'],
                core_versions={k:importlib.metadata.version(k) for k in CORE_PACKAGES})

def expected_ids(c,store):
    m=pd.read_csv(ROOT/'data/sales_train_evaluation.csv',usecols=['id','store_id'],dtype=str)
    ids=m.loc[m.store_id==store,'id'].tolist()
    return ids[:c['smoke_items']] if c.get('smoke_items') else ids

def validate_prediction(path,ids):
    p=pd.read_csv(path,dtype={'id':str})
    assert p.columns.tolist()==['id',*F],f'Wrong columns: {path}'
    assert len(p)==len(ids) and p.id.is_unique and set(p.id)==set(ids),f'Wrong ID coverage: {path}'
    a=p[F].to_numpy(dtype=np.float64)
    assert np.isfinite(a).all() and (a>=0).all(),f'Invalid prediction values: {path}'

def cutoff(c,stage,phase):
    spec=stage_spec(c,stage)
    return spec['select_cutoff'] if phase=='candidates' else spec['train_cutoff']

def folder(c,stage,phase,mode,store):
    return (selection_root(c,cutoff(c,stage,phase)) if phase=='candidates' else stage_root(c,stage))/mode/store

def products(c,phase):
    if phase=='candidates':
        return [f'predictions_{k}.csv' for k in c['round_candidates']]+['metadata.json','timings.json','feature_contract.json']
    return ['predictions.csv','importance.csv','metadata.json','feature_contract.json']

def paths(c,stage,phase,stores,modes):
    if phase=='selection':
        scut=stage_spec(c,stage)['select_cutoff'];assert scut is not None
        return [str((selection_root(c,scut)/n).relative_to(run_root(c))) for n in ['rounds.json','round_grid.csv','weights.json','weight_grid.csv']]
    if phase=='gate':return [f'gates/{stage}.json']
    return [str((folder(c,stage,phase,m,s)/n).relative_to(run_root(c))) for s in stores for m in modes for n in products(c,phase)]

def valid_relative(name):
    p=Path(name)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,'Unsafe path'

def verify_received(c,dest,phase,stage,store=None,mode=None):
    r=json.loads((dest/'received.json').read_text())
    assert r['contract']==contract(c),'Received computation contract changed'
    assert r['phase']==phase and r['stage']==stage
    if store is not None:assert r['store']==store and r['mode']==mode
    for name,expected in r['files'].items():
        valid_relative(name);assert sha(dest/name)==expected,f'Imported artifact changed: {dest/name}'
    return r

def verify_store(c,stage,phase,mode,store):
    d=folder(c,stage,phase,mode,store);cut=cutoff(c,stage,phase)
    if (d/'complete.json').exists():
        if phase=='candidates':
            names=[f'predictions_{k}.csv' for k in c['round_candidates']]+['metadata.json','timings.json']
            fp=artifact_id(c,'selection',cut,store,kind='candidates',mode=mode,rounds=sorted(c['round_candidates']))
        else:
            from evaluate import load_selection
            k=load_selection(c,stage)['rounds_by_mode'][mode]
            names=['model.txt','predictions.csv','importance.csv','metadata.json']
            fp=artifact_id(c,stage,cut,store,kind='model',mode=mode,rounds=k)
        assert complete(d/'complete.json',fp,names),f'Incomplete local task: {d}'
    else:verify_received(c,d,phase,stage,store,mode)

def feature_contract(c,stage,phase,mode,store):
    from common import features_root
    d=folder(c,stage,phase,mode,store);cut=cutoff(c,stage,phase)
    schema=json.loads((d/'dataset/schema.json').read_text())
    vocab=json.loads((features_root(c,cut)/store/'categories.json').read_text())
    params={k:v for k,v in schema['params'].items() if k!='num_threads'}
    atomic_json(dict(cutoff=cut,encoding_end=cut,encoding_stores=10,features=schema['features'],
                     train_rows=schema['train_rows'],params=params,categories_digest=digest(vocab)),d/'feature_contract.json')

def expected_vocabulary(c,cut):
    from features import legacy
    p=legacy(cut)
    cal=pd.read_csv(ROOT/'data/calendar.csv');cal['d_num']=cal.d.str.removeprefix('d_').astype(np.int16)
    cal=cal.loc[cal.d_num<=cut+c['horizon']]
    meta=pd.read_csv(ROOT/'data/sales_train_evaluation.csv',usecols=META,dtype='string')
    return digest(p.make_vocabs(meta,cal))

def reference_features(c,cut,mode):
    # A local task at this cutoff establishes the actual ordered feature schema.
    # Imported prediction-only bundles never contain dataset/schema.json.
    for p in sorted(run_root(c).rglob('schema.json')):
        if p.parent.name!='dataset' or p.parent.parent.parent.name!=mode:continue
        schema=json.loads(p.read_text())
        if schema['train_last_day']!=cut:continue
        store=p.parent.parent.name
        fp=artifact_id(c,'dataset',cut,store,kind='train',mode=mode)
        if complete(p.parent/'complete.json',fp,['train.bin','schema.json']):return schema['features']
    raise AssertionError(f'Finish one local {mode} task at cutoff {cut} before importing remote predictions')

def validate_selection(c,d):
    r=json.loads((d/'rounds.json').read_text());w=json.loads((d/'weights.json').read_text())
    assert set(r['rounds_by_mode'])==set(c['modes'])
    assert all(k in c['round_candidates'] for k in r['rounds_by_mode'].values())
    assert w['rounds_by_mode']==r['rounds_by_mode']
    a,b=w['recursive_weight'],w['nonrecursive_weight']
    assert 0<=a<=1 and 0<=b<=1 and abs(a+b-1)<1e-12
    assert any(abs(a-v)<1e-12 for v in c['ensemble_grid'])

def export_bundle(c,node,stage,phase,dest,stores,modes,takeover=False):
    dest=Path(dest);assert not dest.exists(),'Use a new bundle directory; no overwrite'
    if phase in ['candidates','retrain']:
        for s in stores:
            for m in modes:
                verify_store(c,stage,phase,m,s);feature_contract(c,stage,phase,m,s)
    names=paths(c,stage,phase,stores,modes)
    desc=dict(version=1,node=node,stage=stage,phase=phase,stores=stores,modes=modes,takeover=takeover,
              contract=contract(c),producer_manifest=json.loads((run_root(c)/'manifest.json').read_text()),files={})
    dest.mkdir(parents=True)
    for name in names:
        src=run_root(c)/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target);desc['files'][name]=sha(target)
    atomic_json(desc,dest/'bundle.json');return desc

def validate_bundle(c,bundle):
    bundle=Path(bundle);r=json.loads((bundle/'bundle.json').read_text())
    assert r['version']==1 and r['contract']==contract(c),'Code/data/config/core versions do not match'
    producer=r['producer_manifest']
    assert producer['code']==r['contract']['code'] and producer['inputs']==r['contract']['inputs']
    assert {k:v for k,v in producer['config'].items() if k not in ['threads','experiment']}==r['contract']['config']
    assert r['node'] in plan()['nodes'] and r['stage'] in c['stages']
    assert r['phase'] in ['candidates','retrain','selection','gate']
    assert r['stores'] and len(r['stores'])==len(set(r['stores']))
    assert set(r['stores'])<=set(c['stores']) and set(r['modes'])<=set(c['modes'])
    if r['phase'] in ['gate','selection']:assert r['node']==plan()['coordinator']
    elif not r.get('takeover'):assert set(r['stores'])<=set(plan()['nodes'][r['node']]['stores'])
    else:assert r['node']==plan()['coordinator']
    expected=paths(c,r['stage'],r['phase'],r['stores'],r['modes'])
    assert set(r['files'])==set(expected),'Missing or unexpected bundle entries'
    for name,expected_hash in r['files'].items():
        valid_relative(name);assert sha(bundle/name)==expected_hash,f'Bundle corrupted: {name}'
    if r['phase']=='selection':
        d=bundle/Path(expected[0]).parent;validate_selection(c,d)
        cut=stage_spec(c,r['stage'])['select_cutoff']
        assert json.loads((d/'rounds.json').read_text())['select_cutoff']==cut
        assert json.loads((d/'weights.json').read_text())['select_cutoff']==cut
    elif r['phase']=='gate':
        g=json.loads((bundle/expected[0]).read_text())
        assert g['stage']==r['stage'] and g['previous_stage']=={'test':'development','final':'test'}[r['stage']]
        assert g['contract']==contract(c) and g['previous_complete_sha256']
    else:
        cut=cutoff(c,r['stage'],r['phase']);vocab=expected_vocabulary(c,cut)
        ordered={m:reference_features(c,cut,m) for m in r['modes']}
        for s in r['stores']:
            ids=expected_ids(c,s)
            for m in r['modes']:
                d=bundle/folder(c,r['stage'],r['phase'],m,s).relative_to(run_root(c))
                fc=json.loads((d/'feature_contract.json').read_text())
                assert fc['cutoff']==cut and fc['encoding_end']==cut and fc['encoding_stores']==10
                assert fc['categories_digest']==vocab,'Category mapping mismatch'
                assert fc['params']=={**c['params'],**c['mode_params'][m]},'Model parameter mismatch'
                assert fc['features'] and len(fc['features'])==len(set(fc['features']))
                assert fc['features']==ordered[m],'Ordered feature schema mismatch'
                md=json.loads((d/'metadata.json').read_text())
                assert md['predict_days']==[cut+1,cut+c['horizon']]
                if r['phase']=='candidates':
                    assert md['select_cutoff']==cut and md['candidates']==sorted(c['round_candidates'])
                    for k in c['round_candidates']:validate_prediction(d/f'predictions_{k}.csv',ids)
                else:
                    from evaluate import load_selection
                    chosen=load_selection(c,r['stage'])['rounds_by_mode'][m]
                    assert md['train_last_day']==cut and md['rounds']==chosen
                    assert md['features']==fc['features'] and md['train_rows']==fc['train_rows']
                    assert {k:v for k,v in md['params'].items() if k!='num_threads'}==fc['params']
                    assert md['params']['num_threads']==producer['config']['threads']
                    validate_prediction(d/'predictions.csv',ids)
                    imp=pd.read_csv(d/'importance.csv')
                    assert set(imp.feature)==set(fc['features']) and imp.feature.is_unique
    return r

def import_bundle(c,bundle):
    bundle=Path(bundle);r=validate_bundle(c,bundle)
    # Validate ALL entries and conflicts before copying any file. Identical replay is safe.
    for name,h in r['files'].items():
        target=run_root(c)/name
        if target.exists():assert sha(target)==h,f'Conflicting existing result: {target}'
    dirs={}
    for name,h in r['files'].items():
        target=run_root(c)/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(bundle/name,target)
        dirs.setdefault(target.parent,{})[target.name]=h
    for d,entries in dirs.items():
        receipt=dict(contract=r['contract'],stage=r['stage'],phase=r['phase'],files=entries,
                     source_node=r['node'],source_manifest=r['producer_manifest'],bundle_sha256=sha(bundle/'bundle.json'))
        if r['phase'] in ['candidates','retrain']:receipt.update(mode=d.parent.name,store=d.name)
        if r['phase']=='gate':
            atomic_json(receipt,d/(r['stage']+'.received.json'))
        else:atomic_json(receipt,d/'received.json')
    return r
