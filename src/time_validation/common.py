from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from m5_shared.runtime import sha,digest,atomic_json,atomic_csv,event,complete,checkpoint
from m5_shared.runtime import interpreter as _interpreter
META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
F = [f'F{i}' for i in range(1, 29)]
FILES = ['grid_part_1.pkl','grid_part_2.pkl','grid_part_3.pkl','lags_df_28.pkl','mean_encoding_df.pkl']
# Feature/policy version bound into every artifact fingerprint (scheme 4.2):
# bump when feature definitions or selection-scope semantics change.
FEATURES_VERSION = 'v3_prefix_selection'











def interpreter():
    return _interpreter(ROOT)

def config(smoke=False):
    c = json.loads(Path(os.environ.get('M5_CONFIG',str(HERE/'config.json'))).read_text(encoding='utf-8'))
    if smoke:
        # smoke_tv3 (NOT smoke_v3): v2 smoke runs left experiments/smoke_v3 behind in
        # the WSL environment; reusing that name would hit the manifest fingerprint
        # wall on the first --smoke run. The name lives here, not in config.json,
        # so a config edit cannot accidentally collide with it either (review H1).
        c.update(experiment='smoke_tv3', round_candidates=[10,20,30], stores=['CA_1'], smoke_items=48)
    node=os.environ.get('M5_COLLAB_NODE')
    if node:
        allocation=json.loads((HERE/'two_machine_plan.json').read_text())
        c['threads']=allocation['nodes'][node]['threads']
        c['experiment']='time_validation_v3_two_machine_'+node
    if os.environ.get('M5_COLLAB_SMOKE')=='1':
        c.update(experiment='two_machine_smoke_verified_'+(node or 'windows'),round_candidates=[10,20,30],
                 stores=['CA_1','CA_3'],smoke_items=48)
    return c

def stage_spec(c, stage):
    """Select cutoff, retrain cutoff, prediction start and horizon for a stage.

    final has no selection step: select_cutoff is None and callers must skip
    selection sub-steps for it (scheme 4.2)."""
    spec=c['stages'][stage]; horizon=c['horizon']
    return dict(stage=stage, select_cutoff=spec.get('select_cutoff'), train_cutoff=spec['train_cutoff'], horizon=horizon)

def run_root(c):
    return ROOT/'experiments'/c['experiment']

def stage_root(c, stage):
    """Stage directory. Only final results live directly under it; features are
    shared per cutoff under features/cutoff_<d> (scheme 4.3)."""
    return run_root(c)/stage

def features_root(c, cutoff):
    return run_root(c)/'features'/f'cutoff_{cutoff}'

def selection_root(c, cutoff):
    return run_root(c)/'selection'/f'cutoff_{cutoff}'

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
            raise RuntimeError('Run input/config/code/dependencies changed. Use a new experiment name '
                               '(or delete the experiments/<name> directory) before rerunning; no silent reuse.')
    else: atomic_json(actual,p)
    return key

def manifest(c):
    return json.loads((run_root(c)/'manifest.json').read_text())['fingerprint']

def artifact_id(c,stage,cutoff,store=None,kind=None,mode=None,rounds=None,grid=None):
    """Fingerprint binding stage, cutoff, store, mode, candidate rounds, blend
    grid and feature/policy version, so no product from one cutoff can be
    mistaken for another (scheme 4.2, 4.4, 4.6)."""
    return digest(dict(run=manifest(c),stage=stage,cutoff=cutoff,store=store,kind=kind,
                       mode=mode,rounds=rounds,grid=grid,features_version=FEATURES_VERSION,
                       price_calendar_policy=c.get('price_calendar_policy'),
                       zero_scale_policy=c.get('zero_scale_policy')))




