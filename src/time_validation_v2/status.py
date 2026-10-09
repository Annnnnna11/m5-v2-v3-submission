#!/usr/bin/env python
import json
import time
from datetime import datetime
from common import config, run_root, stage_root

c=config(); root=run_root(c)
status=root/'status.json'
print(status.read_text() if status.exists() else 'Not started')
completed=[]
for stage in c['stages']:
    for p in (stage_root(c,stage)).glob('*/*/metadata.json'):
        if (p.parent/'complete.json').exists():
            m=json.loads(p.read_text()); completed.append((stage,p.parent.parent.name,p.parent.name,m))
print(f'Completed model/prediction checkpoints: {len(completed)}/60')
for mode in c['modes']:
    values=[r[3]['total_seconds'] for r in completed if r[1]==mode]
    if values:
        remaining=30-len(values)
        print(f'{mode}: {len(values)}/30, average {sum(values)/len(values):.0f}s, remaining model work ~{remaining*sum(values)/len(values)/3600:.1f}h')
if all(any(r[1]==m for r in completed) for m in c['modes']):
    seconds=sum((30-sum(r[1]==m for r in completed))*sum(r[3]['total_seconds'] for r in completed if r[1]==m)/sum(r[1]==m for r in completed) for m in c['modes'])
    # Include measured cache preparation time; this is an estimate, not a promise.
    caches=[json.loads(p.read_text())['seconds'] for p in root.glob('*/cache/*/complete.json')]
    seconds+=(30-len(caches))*sum(caches)/len(caches) if caches else 0
    print('Projected finish (local, excludes unknown delays):',datetime.fromtimestamp(time.time()+seconds).isoformat(timespec='minutes'))
for stage in c['stages']:
    p=stage_root(c,stage)/'results/scores.csv'
    if p.exists(): print(stage+'\n'+p.read_text())
