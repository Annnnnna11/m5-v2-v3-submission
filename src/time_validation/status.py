#!/usr/bin/env python
"""Pipeline flags and progress (scheme 4.8): selection flags apply only to
development/test; final shows the rounds and weights inherited from test."""
import argparse
import json
from common import config, run_root, stage_root, stage_spec, features_root, selection_root

parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args(); c=config(args.smoke); root=run_root(c)

status=root/'status.json'
print(status.read_text() if status.exists() else 'Not started')
for name in c['stages']:
    spec=stage_spec(c,name); scut=spec['select_cutoff']; tcut=spec['train_cutoff']
    print(f'\n== {name}: select_cutoff={scut} train_cutoff={tcut}')
    flags={}
    if scut is not None:
        flags['features_selection_complete']=all((features_root(c,scut)/s/'complete.json').exists() for s in c['stores'])
        sel=selection_root(c,scut)
        flags['round_selection_complete']=(sel/'rounds_complete.json').exists()
        flags['ensemble_selection_complete']=(sel/'weights_complete.json').exists()
    flags['features_train_complete']=all((features_root(c,tcut)/s/'complete.json').exists() for s in c['stores'])
    flags['final_models_complete']=all((stage_root(c,name)/m/s/'complete.json').exists() for m in c['modes'] for s in c['stores'])
    flags['stage_evaluation_complete']=(stage_root(c,name)/'results'/'complete.json').exists()
    for key,value in flags.items(): print(f'  {key}: {value}')
    if scut is not None:
        sel=selection_root(c,scut)
        if (sel/'rounds.json').exists():
            payload=json.loads((sel/'rounds.json').read_text())
            print('  candidates:',payload['candidates'])
            print('  selected rounds:',payload['rounds_by_mode'])
        if (sel/'weights.json').exists():
            w=json.loads((sel/'weights.json').read_text())
            print(f'  selected weight: recursive={w["recursive_weight"]} (selection-window ensemble WRMSSE {w.get("ensemble_wrmsse"):.6f})')
    elif name=='final':
        tsel=selection_root(c,stage_spec(c,'test')['select_cutoff'])
        if (tsel/'rounds.json').exists() and (tsel/'weights.json').exists():
            rounds=json.loads((tsel/'rounds.json').read_text())['rounds_by_mode']
            weights=json.loads((tsel/'weights.json').read_text())
            print('  继承自 test 的轮数:',rounds)
            print(f'  继承自 test 的融合权重: recursive={weights["recursive_weight"]}')
models=sum((stage_root(c,name)/m/s/'complete.json').exists() for name in c['stages'] for m in c['modes'] for s in c['stores'])
total=len(c['stages'])*len(c['modes'])*len(c['stores'])
print(f'\nFinal model/prediction checkpoints: {models}/{total}')
remaining=[]
for name in c['stages']:
    spec=stage_spec(c,name)
    if spec['select_cutoff'] is not None and not (selection_root(c,spec['select_cutoff'])/'weights_complete.json').exists():
        remaining.append(f'{name}: selection')
    if not all((stage_root(c,name)/m/s/'complete.json').exists() for m in c['modes'] for s in c['stores']):
        remaining.append(f'{name}: retrain')
    if not (stage_root(c,name)/'results'/'complete.json').exists():
        remaining.append(f'{name}: evaluation')
print('Remaining:', '; '.join(remaining) if remaining else 'none')
for name in c['stages']:
    p=stage_root(c,name)/'results/scores.csv'
    if p.exists(): print(name+'\n'+p.read_text())
