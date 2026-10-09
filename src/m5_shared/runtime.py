from pathlib import Path
import hashlib,json,os,sys,time
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_csv(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    value.to_csv(tmp, index=False); os.replace(tmp, path)

def event(name, **kwargs):
    print(json.dumps(dict(event=name, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs), default=str), flush=True)

def complete(path, fingerprint, files):
    path = Path(path)
    if not path.exists(): return False
    value=json.loads(path.read_text())
    if value['fingerprint'] != fingerprint: raise RuntimeError(f'Incompatible checkpoint: {path}')
    for name in files:
        p=path.parent/name
        if not p.exists() or value['artifacts'].get(name)!=sha(p): return False
    return True

def checkpoint(path, fingerprint, files, **extra):
    path=Path(path)
    atomic_json(dict(fingerprint=fingerprint, artifacts={name:sha(path.parent/name) for name in files}, **extra), path)

def interpreter(root):
    if os.environ.get('M5_PYTHON'):
        return os.environ['M5_PYTHON']
    root=Path(root)
    for candidate in [root/'.venv/bin/python',root/'.venv/Scripts/python.exe']:
        if candidate.exists():return str(candidate)
    return sys.executable

def verify_baseline(root):
    root=Path(root)
    value=os.environ.get('M5_BASELINE_MANIFEST')
    path=Path(value).expanduser() if value else root/'docs/baseline_CA_1_manifest.json'
    if not path.is_absolute():path=root/path
    if not path.exists():
        if value:raise FileNotFoundError(path)
        event('original_baseline_not_supplied',note='Fresh course repository contains no old baseline artifacts; no preservation manifest required.')
        return
    baseline=json.loads(path.read_text(encoding='utf-8'))
    for name,expected in baseline['artifacts'].items():
        assert sha(root/name)==expected['sha256'],f'Original artifact changed: {name}'
    event('original_artifacts_verified',files=len(baseline['artifacts']))
