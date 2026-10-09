#!/usr/bin/env python
"""Serial, resumable driver for scheme 10.2 v3 (prefix-equivalence selection).

development/test: prepare@select_cutoff -> train_candidates (one training to
the cap, one full 28-day prediction per candidate) -> select_rounds ->
select_weight -> prepare@train_cutoff -> retrain with selected rounds ->
predict -> evaluate. final: NO selection — inherits test's selection, retrains
on d<=1941 and predicts 1942-1969 with integrity check only (scheme 4.7)."""
import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from common import ROOT, HERE, config, run_root, stage_root, stage_spec, features_root, selection_root, ensure_run, manifest, atomic_json, event, sha, interpreter

STAGES=['development','test','final']

def child(c,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[interpreter(),'-u',str(HERE/'run.py'),*args]
    if c.get('smoke_items'): command+=['--smoke']
    env=os.environ.copy(); env.update(OMP_NUM_THREADS=str(c['threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    event('process_start',command=command,log=str(log))
    with log.open('a') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def verify_preserved():
    from m5_shared.runtime import verify_baseline
    verify_baseline(ROOT)

def note(c,**state):
    payload=dict(state='running',pid=os.getpid(),updated=time.time()); payload.update(state)
    atomic_json(payload,run_root(c)/'status.json')

def stage_completed(c,name):
    return (stage_root(c,name)/'results'/'complete.json').exists()

def cleanup_decisions(c,completed):
    """Which select_cutoff feature caches may be deleted now (scheme 4.3): a
    select_cutoff cache dies only when EVERY stage referencing that cutoff (in
    select or train role) has completed; cutoff_1885's consumers are
    development-retrain and test-selection, so it dies after test. Train_cutoff
    caches are never cleaned."""
    decisions=[]
    for stage,name_cut in c['stages'].items():
        cut=name_cut.get('select_cutoff')
        if cut is None or stage not in completed: continue
        consumers=[s for s in c['stages'] if cut in (c['stages'][s].get('select_cutoff'),c['stages'][s]['train_cutoff'])]
        if all(s in completed for s in consumers): decisions.append(cut)
    return decisions

def apply_cleanup(c,completed):
    for cut in cleanup_decisions(c,completed):
        directory=features_root(c,cut)
        if directory.exists():
            shutil.rmtree(directory)
            event('select_cache_cleaned',cutoff=cut,note='all consumers finished; regenerable on demand')
        for mode in c['modes']:
            for store in c['stores']:
                dataset=selection_root(c,cut)/mode/store/'dataset'
                if dataset.exists():
                    shutil.rmtree(dataset)
                    event('selection_dataset_cleaned',cutoff=cut,store=store,mode=mode)
    event('cleanup_applied',completed=completed,decisions=cleanup_decisions(c,completed))

def run_stage(c,name):
    spec=stage_spec(c,name); scut=spec['select_cutoff']; tcut=spec['train_cutoff']
    logs=stage_root(c,name)/'logs'
    if scut is not None:
        for store in c['stores']:
            note(c,stage=name,step='prepare_selection',cutoff=scut,store=store)
            child(c,['prepare','--cutoff',str(scut),'--store',store],logs/f'{store}_features_select.log')
            for mode in c['modes']:
                note(c,stage=name,step='train_candidates',cutoff=scut,store=store,mode=mode)
                child(c,['train_candidates','--cutoff',str(scut),'--store',store,'--mode',mode],logs/f'{store}_{mode}_candidates.log')
        if c.get('smoke_items') and name=='development':
            # The prefix-equivalence premise must be decided in smoke, before
            # any full run (scheme 4.4 兜底分支).
            child(c,['equivalence_check','--cutoff',str(scut),'--store',c['stores'][0]],logs/'equivalence_check.log')
        note(c,stage=name,step='select_rounds',cutoff=scut)
        child(c,['select_rounds','--cutoff',str(scut)],logs/'select_rounds.log')
        note(c,stage=name,step='select_weight',cutoff=scut)
        child(c,['select_weight','--cutoff',str(scut)],logs/'select_weight.log')
    for store in c['stores']:
        note(c,stage=name,step='prepare_train',cutoff=tcut,store=store)
        child(c,['prepare','--cutoff',str(tcut),'--store',store],logs/f'{store}_features_train.log')
        for mode in c['modes']:
            note(c,stage=name,step='retrain',cutoff=tcut,store=store,mode=mode)
            child(c,['model','--stage',name,'--store',store,'--mode',mode],logs/f'{store}_{mode}.log')
    note(c,stage=name,step='evaluate',cutoff=tcut)
    child(c,['evaluate','--stage',name],logs/'evaluate.log')

def report(c):
    root=run_root(c)
    lines=['# M5 时间验证 v3 运行结果','',
           '方案 A：development 与 test 各自独立选择轮数与融合权重；final 不选参，沿用 test 的选择结果。',
           '选择阶段 WRMSSE 的美元权重取 select_cutoff 前 28 天、RMSSE 尺度只用 d<=select_cutoff 历史。',
           '开发期/测试期/最终预测期分别为 d1886-1913、d1914-1941、d1942-1969。','']
    for name in c['stages']:
        spec=stage_spec(c,name); out=stage_root(c,name)/'results'
        lines.append(f'## {name}（select_cutoff={spec["select_cutoff"]}, train_cutoff={spec["train_cutoff"]}）')
        lines.append('')
        if spec['select_cutoff'] is not None:
            sel=selection_root(c,spec['select_cutoff'])
            if (sel/'rounds.json').exists() and (sel/'weights.json').exists():
                rounds=json.loads((sel/'rounds.json').read_text())['rounds_by_mode']
                weights=json.loads((sel/'weights.json').read_text())
                lines.append(f'- 选中轮数：{rounds}')
                lines.append(f'- 融合权重：recursive={weights["recursive_weight"]}, nonrecursive={weights["nonrecursive_weight"]}（选择窗 ensemble WRMSSE={weights["ensemble_wrmsse"]:.6f}）')
                lines.append('')
        elif name=='final':
            lines.append('- 无选参步骤：轮数与融合权重继承自 test 阶段的选择结果。')
            lines.append('')
        path=out/'scores.csv'
        if path.exists():
            rows=list(csv.DictReader(path.open()))
            lines+=['|模型|WRMSSE|MAE|RMSE|Bias|','|---|---:|---:|---:|---:|']
            for r in rows: lines+=['|'+r['model']+'|'+'|'.join(f'{float(r[k]):.6f}' for k in ['WRMSSE','MAE','RMSE','bias'])+'|']
            lines+=['',f'完整明细、误差图和特征重要性：`{out.relative_to(root)}`','']
        elif (out/'integrity.json').exists():
            integrity=json.loads((out/'integrity.json').read_text())
            lines+=[f'- 完整性检查通过：{integrity["rows"]} 条序列、{integrity["horizon"]} 天、无 NaN、无负数、无重复 ID。',
                    '- 最终窗口没有真实销量，不报告准确度；不生成 Kaggle 提交文件。','']
        else: lines+=['尚未完成。','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event('report_updated',path=str(root/'REPORT.md'))

def changelog_note(c,stage):
    """Runtime per-stage entry under the current experiment; smoke skips it."""
    if c.get('smoke_items'): return
    path=run_root(c)/'CHANGELOG.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    fp=manifest(c)
    if path.exists() and fp in path.read_text(): return
    out=stage_root(c,stage)/'results'
    if stage=='final':
        integrity=json.loads((out/'integrity.json').read_text())
        summary=(f'final 阶段完成：继承 test 选择结果，用 d<=1941 重训并预测 1942-1969，'
                 f'{integrity["rows"]} 条序列通过完整性检查（无真值不评分，不生成 Kaggle 提交文件）')
    else:
        sel=selection_root(c,stage_spec(c,stage)['select_cutoff'])
        rounds=json.loads((sel/'rounds.json').read_text())['rounds_by_mode']
        weights=json.loads((sel/'weights.json').read_text())
        rows=[r for r in csv.DictReader((out/'scores.csv').open()) if r['model']=='ensemble']
        score=f'正式窗 ensemble WRMSSE={float(rows[0]["WRMSSE"]):.6f}' if rows else '正式窗评分未生成'
        summary=(f'{stage} 阶段完成：select_cutoff={stage_spec(c,stage)["select_cutoff"]} 独立选择，'
                 f'轮数 {rounds}，融合权重 recursive={weights["recursive_weight"]}，{score}')
    with path.open('a') as handle:
        handle.write(f'- {time.strftime("%Y-%m-%d")} 实验 `{c["experiment"]}` {summary}（指纹 `{fp}`）。\n')
    event('changelog_updated',stage=stage)

def orchestrate(c,requested):
    root=run_root(c); root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ensure_run(c); verify_preserved()
        try:
            stages=STAGES if requested=='all' else [requested]
            for name in stages:
                if name=='final':
                    from evaluate import load_selection
                    load_selection(c,'final')  # asserts test's selection products exist and match
                run_stage(c,name)
                report(c); changelog_note(c,name)
                verify_preserved()
            # Deferred one-shot cleanup (review M1): select_cutoff caches stay on
            # disk until EVERY requested stage has succeeded, so an interrupted
            # `all` rerun resumes through cache_verified instead of rebuilding
            # hours of features; only a fully successful run frees them now
            # (smoke included, so the cleanup path is exercised every smoke).
            apply_cleanup(c,[s for s in STAGES if stage_completed(c,s)])
            atomic_json(dict(state='complete',stages=stages,finished=time.time()),root/'status.json')
        except BaseException as exc:
            atomic_json(dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),time=time.time()),root/'status.json')
            raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['all','stage','prepare','train_candidates','select_rounds','select_weight',
                                          'model','retrain','evaluate','equivalence_check','cleanup','check'])
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--cutoff',type=int)
    parser.add_argument('--store'); parser.add_argument('--mode',choices=['recursive','nonrecursive'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); c=config(args.smoke)
    if c['experiment'].startswith('time_validation_v3_two_machine') and args.action in ['all','stage']:
        raise RuntimeError('Use collaborate.py phase commands for this two-machine experiment; run.py all/stage would bypass task ownership and handoff barriers.')
    if args.action in ['all','stage']:
        orchestrate(c,'all' if args.action=='all' else args.stage)
    elif args.action=='check':
        subprocess.run([sys.executable,str(HERE/'tests.py')],check=True,cwd=ROOT)
    elif args.action=='cleanup':
        completed=[s for s in STAGES if stage_completed(c,s)]
        apply_cleanup(c,completed)
    else:
        # Children share the immutable run manifest; the supervisor verifies content before launch.
        if args.action=='prepare':
            assert args.cutoff and args.store
            from features import prepare
            prepare(c,args.cutoff,args.store)
        elif args.action=='train_candidates':
            assert args.cutoff and args.store and args.mode
            from model import evaluate_round_candidates
            evaluate_round_candidates(c,args.cutoff,args.store,args.mode)
        elif args.action=='equivalence_check':
            assert args.cutoff and args.store
            from model import equivalence_probe
            equivalence_probe(c,args.cutoff,args.store)
        elif args.action=='select_rounds':
            assert args.cutoff
            from evaluate import select_rounds
            select_rounds(c,args.cutoff)
        elif args.action=='select_weight':
            assert args.cutoff
            from evaluate import select_ensemble_weight
            select_ensemble_weight(c,args.cutoff)
        elif args.action in ['model','retrain']:
            assert args.stage and args.store and args.mode
            from evaluate import load_selection
            from model import run_model
            rounds=load_selection(c,args.stage)['rounds_by_mode'][args.mode]
            run_model(c,args.stage,args.store,args.mode,rounds)
        else:
            assert args.stage
            from evaluate import evaluate_stage
            evaluate_stage(c,args.stage)

if __name__=='__main__': main()
