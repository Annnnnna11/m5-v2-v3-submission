#!/usr/bin/env python
"""Explicit phase barriers, worker-local checkpoints, verified prediction exchange."""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT,HERE,config,ensure_run,run_root,stage_root,selection_root,stage_spec,atomic_json,event,sha,artifact_id,complete
from handoff import plan,contract,folder,verify_store,verify_received,export_bundle,import_bundle

def setup(node,smoke):
    os.environ['M5_COLLAB_NODE']=node
    os.environ['M5_PYTHON']=sys.executable
    if smoke:os.environ['M5_COLLAB_SMOKE']='1'
    else:os.environ.pop('M5_COLLAB_SMOKE',None)
    c=config();ensure_run(c)
    return c

def selected(c,stage):
    from evaluate import load_selection
    return load_selection(c,stage)

def stage_done(c,stage):
    out=stage_root(c,stage)/'results';p=out/'complete.json'
    assert p.exists(),f'Coordinator must score {stage} first'
    r=json.loads(p.read_text());sel=selected(c,stage);cut=stage_spec(c,stage)['train_cutoff']
    assert complete(p,artifact_id(c,stage,cut,kind='results',rounds=sel['rounds_by_mode']),list(r['artifacts']))
    return p

def require_gate(c,node,stage):
    if stage=='development':return
    previous={'test':'development','final':'test'}[stage]
    if node=='windows':stage_done(c,previous);return
    g=run_root(c)/'gates'/f'{stage}.json';r=g.with_suffix('.received.json')
    assert g.exists() and r.exists(),f'Import coordinator {stage} gate bundle first'
    gate=json.loads(g.read_text());receipt=json.loads(r.read_text())
    assert receipt['contract']==contract(c) and receipt['source_node']=='windows'
    assert receipt['files'][g.name]==sha(g)
    assert gate['stage']==stage and gate['previous_stage']==previous and gate['contract']==contract(c)

def may_start(profile,action,smoke):
    if smoke or not profile.get('stop_at'):return True
    deadline=dt.datetime.fromisoformat(profile['stop_at'])
    estimate=profile['admission_seconds'][action]
    remaining=(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining<estimate+300:
        event('deadline_admission_stopped',remaining_seconds=remaining,required_seconds=estimate+300,
              note='No new process launched; a running task is never killed automatically.')
        return False
    return True

def child(c,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(c['threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-u',str(HERE/'run.py'),*args]
    event('worker_child',command=command,log=str(log))
    with log.open('a') as stream:subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def worker(c,args,stores,modes):
    require_gate(c,args.node,args.stage)
    spec=stage_spec(c,args.stage)
    assert args.phase!='candidates' or spec['select_cutoff'] is not None
    if args.phase=='retrain':selected(c,args.stage)
    profile=plan()['nodes'][args.node]
    state=dict(state='running',node=args.node,stage=args.stage,phase=args.phase,threads=c['threads'],started=time.time(),completed=[])
    status=run_root(c)/f'worker_{args.stage}_{args.phase}.json'
    logs=run_root(c)/'worker_logs'/args.stage/args.phase
    lock=run_root(c)/'worker.lock'
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(state,status)
        try:
            cut=spec['select_cutoff'] if args.phase=='candidates' else spec['train_cutoff']
            for store in stores:
                if not may_start(profile,'prepare',args.smoke):
                    state.update(state='stopped_before_deadline',finished=time.time());atomic_json(state,status);return
                child(c,['prepare','--cutoff',str(cut),'--store',store],logs/f'{store}_prepare.log')
                for mode in modes:
                    # Resume a completed local task without reserving time for another training.
                    d=folder(c,args.stage,args.phase,mode,store)
                    if (d/'complete.json').exists():
                        verify_store(c,args.stage,args.phase,mode,store)
                        state['completed'].append(dict(store=store,mode=mode,resumed=True));atomic_json(state,status);continue
                    if not may_start(profile,mode,args.smoke):
                        state.update(state='stopped_before_deadline',finished=time.time());atomic_json(state,status);return
                    action=['train_candidates','--cutoff',str(cut)] if args.phase=='candidates' else ['model','--stage',args.stage]
                    child(c,[*action,'--store',store,'--mode',mode],logs/f'{store}_{mode}.log')
                    state['completed'].append(dict(store=store,mode=mode));atomic_json(state,status)
            state.update(state='complete',finished=time.time());atomic_json(state,status)
        except BaseException as e:
            state.update(state='failed',error=repr(e),finished=time.time());atomic_json(state,status);raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['show','worker','export','import','select','score','gate'])
    p.add_argument('--node',choices=['windows','mac'],required=True)
    p.add_argument('--stage',choices=['development','test','final'],default='development')
    p.add_argument('--phase',choices=['candidates','retrain','selection','gate'],default='candidates')
    p.add_argument('--stores',nargs='+');p.add_argument('--modes',nargs='+',choices=['nonrecursive','recursive'])
    p.add_argument('--bundle');p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    p.add_argument('--takeover',action='store_true',help='Windows explicitly takes over unfinished Mac stores')
    a=p.parse_args();c=setup(a.node,a.smoke)
    stores=a.stores or [s for s in plan()['nodes'][a.node]['stores'] if s in c['stores']]
    modes=a.modes or c['modes']
    assert stores and set(stores)<=set(c['stores']) and len(stores)==len(set(stores))
    if a.action in ['worker','export'] and a.phase in ['candidates','retrain']:
        if a.takeover:assert a.node=='windows' and set(stores)<=set(plan()['nodes']['mac']['stores'])
        else:assert set(stores)<=set(plan()['nodes'][a.node]['stores'])
    if a.action=='show':
        print(json.dumps(dict(plan=plan(),local_root=str(run_root(c)),threads=c['threads'],stores=stores),indent=2));return
    if a.action=='worker':
        assert a.phase in ['candidates','retrain'];worker(c,a,stores,modes)
    elif a.action=='export':
        assert a.output
        if a.phase=='selection':assert a.node=='windows';selected(c,a.stage)
        if a.phase=='gate':assert a.node=='windows' and a.stage!='development'
        print(json.dumps(export_bundle(c,a.node,a.stage,a.phase,a.output,stores,modes,a.takeover),indent=2))
    elif a.action=='import':
        assert a.bundle;received=import_bundle(c,a.bundle)
        event('bundle_imported',source_node=received['node'],stage=received['stage'],phase=received['phase'])
    elif a.action=='select':
        assert a.node=='windows' and a.stage!='final';require_gate(c,a.node,a.stage)
        for store in c['stores']:
            for mode in c['modes']:verify_store(c,a.stage,'candidates',mode,store)
        cut=stage_spec(c,a.stage)['select_cutoff']
        from evaluate import select_rounds,select_ensemble_weight
        select_rounds(c,cut);select_ensemble_weight(c,cut)
    elif a.action=='score':
        assert a.node=='windows';require_gate(c,a.node,a.stage)
        for store in c['stores']:
            for mode in c['modes']:verify_store(c,a.stage,'retrain',mode,store)
        from evaluate import evaluate_stage
        from run import verify_preserved,report
        evaluate_stage(c,a.stage);verify_preserved();report(c)
    elif a.action=='gate':
        assert a.node=='windows' and a.stage!='development'
        previous={'test':'development','final':'test'}[a.stage]
        done=stage_done(c,previous)
        atomic_json(dict(stage=a.stage,previous_stage=previous,contract=contract(c),
                         previous_complete_sha256=sha(done),issued=time.time()),run_root(c)/'gates'/f'{a.stage}.json')
        event('next_stage_authorized',stage=a.stage)

if __name__=='__main__':main()
