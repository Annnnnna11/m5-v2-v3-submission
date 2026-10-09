#!/usr/bin/env python
"""Serial, resumable driver. Each store/mode runs in a fresh low-memory process."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from common import ROOT, HERE, config, run_root, stage_root, ensure_run, manifest, atomic_json, event, sha, interpreter

def child(c,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[interpreter(),'-u',str(HERE/'run.py'),*args]
    if c.get('smoke_items'): command+=['--smoke']
    env=os.environ.copy(); env.update(OMP_NUM_THREADS=str(c['threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    event('process_start',command=command,log=str(log))
    with log.open('a') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def report(c):
    import csv
    root=run_root(c)
    lines=['# M5 时间验证运行结果','',
           '各阶段只有全部门店完成训练、预测和检查后才生成正式评分。预测偏差定义为预测减真实销量。',
           '开发期、最后测试期、最终预测期分别为 d1886–1913、d1914–1941、d1942–1969。','']
    for name in c['stages']:
        results=stage_root(c,name)/'results'
        lines += [f'## {name}','']
        path=results/'scores.csv'
        if path.exists():
            rows=list(csv.DictReader(path.open()))
            lines+=['|模型|WRMSSE|MAE|RMSE|Bias|','|---|---:|---:|---:|---:|']
            for r in rows: lines+=['|'+r['model']+'|'+'|'.join(f'{float(r[k]):.6f}' for k in ['WRMSSE','MAE','RMSE','bias'])+'|']
            lines+=['',f'完整明细、误差图和特征重要性：`{results.relative_to(root)}`','']
        elif (results/'integrity.json').exists():
            lines += ['已生成预测并完成完整性检查；最终窗口没有真实销量，不报告准确度。','']
        else: lines += ['尚未完成。','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def verify_preserved():
    from m5_shared.runtime import verify_baseline
    verify_baseline(ROOT)

def stage(c,name):
    dest=stage_root(c,name)
    atomic_json(dict(state='running',stage=name,pid=os.getpid(),started=time.time()),run_root(c)/'status.json')
    for store in c['stores']:
        child(c,['prepare','--stage',name,'--store',store],dest/'logs'/f'{store}_features.log')
        for mode in c['modes']:
            atomic_json(dict(state='running',stage=name,store=store,mode=mode,pid=os.getpid(),updated=time.time()),run_root(c)/'status.json')
            child(c,['model','--stage',name,'--store',store,'--mode',mode],dest/'logs'/f'{store}_{mode}.log')
    child(c,['evaluate','--stage',name],dest/'logs'/'evaluate.log')

def orchestrate(c,requested):
    root=run_root(c); root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ensure_run(c); verify_preserved()
        try:
            stages=list(c['stages']) if requested=='all' else [requested]
            for name in stages:
                if name in ['test','final']:
                    frozen=root/'frozen_after_development.json'
                    assert frozen.exists(), 'Finish development and freeze configuration before test/final'
                    assert json.loads(frozen.read_text())['fingerprint']==manifest(c)
                stage(c,name)
                report(c)
                if name=='development':
                    # Predetermined no-tuning protocol: retain original two models and equal blend.
                    scores=stage_root(c,name)/'results/scores.csv'
                    atomic_json(dict(fingerprint=manifest(c),selection=c['selection'],
                                     development_scores_sha256=sha(scores),ensemble=c['ensemble'],
                                     frozen_at=time.time()),root/'frozen_after_development.json')
                verify_preserved()
            if requested=='all' and not c.get('smoke_items'):
                from evaluate import kaggle
                kaggle(c)
            atomic_json(dict(state='complete',stages=stages,finished=time.time()),root/'status.json')
        except BaseException as exc:
            atomic_json(dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),time=time.time()),root/'status.json')
            raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['all','stage','prepare','model','evaluate','check'])
    parser.add_argument('--stage',choices=['development','test','final'])
    parser.add_argument('--store'); parser.add_argument('--mode',choices=['recursive','nonrecursive'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); c=config(args.smoke)
    if args.action in ['all','stage']:
        orchestrate(c,'all' if args.action=='all' else args.stage)
    elif args.action=='check':
        subprocess.run([sys.executable,str(HERE/'tests.py')],check=True,cwd=ROOT)
    else:
        assert args.stage
        # Children share the immutable run manifest; the supervisor verifies content before launch.
        if args.action=='prepare':
            from features import prepare
            prepare(c,args.stage,args.store)
        elif args.action=='model':
            from model import run_model
            run_model(c,args.stage,args.store,args.mode)
        else:
            from evaluate import evaluate
            evaluate(c,args.stage)

if __name__=='__main__': main()
