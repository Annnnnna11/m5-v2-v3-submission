from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from m5_shared.runtime import sha,digest,atomic_json,atomic_csv,event,complete,checkpoint
from m5_shared.runtime import interpreter as _interpreter
META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
F = [f'F{i}' for i in range(1, 29)]
FILES = ['grid_part_1.pkl','grid_part_2.pkl','grid_part_3.pkl','lags_df_28.pkl','mean_encoding_df.pkl']











def config(smoke=False):
    c = json.loads(Path(os.environ.get('M5_CONFIG',str(HERE/'config.json'))).read_text(encoding='utf-8'))
    if smoke:
        c.update(experiment='smoke_v3', rounds=5, stores=['CA_1'], smoke_items=48)
    return c

def run_root(c):
    return ROOT/'experiments'/c['experiment']

def stage_root(c, stage):
    return run_root(c)/f'{stage}_d{c["stages"][stage]}'

def identity(c):
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))}
    code['src/1_preprocessing_by_store.py'] = sha(ROOT/'src/1_preprocessing_by_store.py')
    code.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/m5_shared').glob('*.py'))})
    raw = {name: sha(ROOT/'data'/name) for name in ['calendar.csv','sell_prices.csv','sales_train_evaluation.csv','sales_train_validation.csv','sample_submission.csv']}
    deps = subprocess.check_output([interpreter(), '-m','pip','freeze'], text=True)
    return dict(config=c, code=code, inputs=raw, dependencies=deps,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())

def ensure_run(c):
    p = run_root(c)/'manifest.json'; actual = identity(c)
    # Commit ID is provenance; content hashes determine compatibility across doc-only commits.
    key = digest({k:v for k,v in actual.items() if k!='git_commit'})
    actual['fingerprint'] = key
    if p.exists():
        old = json.loads(p.read_text())
        if old['fingerprint'] != key:
            raise RuntimeError('Run input/config/code/dependencies changed. Use a new experiment name; no silent reuse.')
    else: atomic_json(actual,p)
    return key

def manifest(c):
    return json.loads((run_root(c)/'manifest.json').read_text())['fingerprint']

def artifact_id(c,stage,store=None,kind=None):
    return digest(dict(run=manifest(c),stage=stage,cutoff=c['stages'][stage],store=store,kind=kind))





def interpreter():
    return _interpreter(ROOT)
