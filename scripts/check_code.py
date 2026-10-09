"""Syntax, import, config and synthetic tests only. Never train or read M5 CSVs."""
import argparse,ast,importlib,inspect,json,os,pathlib,subprocess,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]

def check_version(version):
    folder=ROOT/'src'/('time_validation_v2' if version=='v2' else 'time_validation')
    sys.path.insert(0,str(folder))
    for name in ['common','features','model','metrics','evaluate','run']:importlib.import_module(name)
    if version=='v3':
        for name in ['handoff','collaborate']:importlib.import_module(name)
    import common,features,model,metrics
    c=common.config()
    assert c['horizon']==28 and c['first_day']==1 and len(c['stores'])==10
    assert c['params']['learning_rate']==.015 and c['params']['tweedie_variance_power']==1.1
    assert c['mode_params']['recursive']['seed']==42 and c['mode_params']['nonrecursive']['seed']==1995
    if version=='v2':
        assert c['rounds']==3000 and c['threads']==4 and c['stages']==dict(development=1885,test=1913,final=1941)
        assert c['ensemble']==dict(recursive=.5,nonrecursive=.5)
    else:
        assert c['threads']==12 and c['round_candidates']==[500,1000,1500,2000,2500,3000]
        assert c['ensemble_grid']==[i/10 for i in range(11)]
        assert common.stage_spec(c,'development')['select_cutoff']==1857
        assert common.stage_spec(c,'test')['select_cutoff']==1885
        assert common.stage_spec(c,'final')['select_cutoff'] is None
    assert common.ROOT==ROOT and pathlib.Path(common.interpreter()).is_file()
    # Existing retained unit tests use synthetic inputs; training equivalence test was excluded.
    import unittest
    tests=importlib.import_module('tests')
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful()
    print(json.dumps({'version':version,'imports':'passed','configuration':'passed','synthetic_tests':result.testsRun,'training_executed':False}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['v2','v3'])
    args=parser.parse_args()
    if args.version:return check_version(args.version)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MPLBACKEND']='Agg'
    for key in ['M5_CONFIG','M5_COLLAB_NODE','M5_COLLAB_SMOKE','M5_RUNTIME_THREADS','M5_PYTHON','PYTHONPATH']:env.pop(key,None)
    for v in ['v2','v3']:
        subprocess.run([sys.executable,str(pathlib.Path(__file__)),'--version',v],env=env,cwd=ROOT,check=True)
        folder='time_validation_v2' if v=='v2' else 'time_validation'
        subprocess.run([sys.executable,str(ROOT/'src'/folder/'run.py'),'--help'],env=env,cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    for script in ['scripts/make_submission.py','src/check_raw_inputs.py']:
        subprocess.run([sys.executable,str(ROOT/script),'--help'],env=env,cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    sys.path.insert(0,str(ROOT/'scripts'))
    import pandas as pd,numpy as np
    from make_submission import assemble
    columns=['id',*[f'F{i}' for i in range(1,29)]]
    a=pd.DataFrame([['x_evaluation',*[.25]*28]],columns=columns);b=pd.DataFrame([['x_evaluation',*[.75]*28]],columns=columns)
    sample=pd.DataFrame([['x_evaluation',*[0]*28],['x_validation',*[0]*28]],columns=columns)
    result=assemble(a,b,sample,expected_rows=2)
    assert result.id.tolist()==sample.id.tolist() and result.F1.tolist()==[.75,.25]
    print(json.dumps({'syntax_files':len(files),'help_commands':'passed','tiny_submission_alignment':'passed','full_training_run_verified':False}))

if __name__=='__main__':main()
