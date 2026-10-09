"""Assemble genuine Public and final forecasts; never train, tune or submit online."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def assemble(validation,evaluation,sample,expected_rows=60980):
    import numpy as np
    import pandas as pd
    columns=['id',*[f'F{i}' for i in range(1,29)]]
    for frame in [validation,evaluation,sample]:
        assert frame.columns.tolist()==columns
        assert frame.id.is_unique
    assert validation.id.str.endswith('_evaluation').all()
    assert evaluation.id.str.endswith('_evaluation').all()
    assert set(validation.id)==set(evaluation.id)
    validation=validation.copy()
    validation['id']=validation.id.str.replace('_evaluation','_validation',regex=False)
    both=pd.concat([validation,evaluation],ignore_index=True)
    assert len(sample)==expected_rows and both.id.is_unique and set(both.id)==set(sample.id)
    result=both.set_index('id').loc[sample.id].reset_index()
    values=result.iloc[:,1:].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values>=0).all()
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['v2','v3'],required=True)
    parser.add_argument('--output',type=Path,help='Default: final/results/kaggle_ensemble.csv under this version experiment')
    args=parser.parse_args()
    folder=ROOT/'src'/('time_validation_v2' if args.version=='v2' else 'time_validation')
    sys.path.insert(0,str(folder))
    import pandas as pd
    common=importlib.import_module('common');c=common.config()
    assert not c.get('smoke_items') and c['horizon']==28
    if args.version=='v3':
        selection=importlib.import_module('evaluate').load_selection(c,'final')
        weights={m:selection[m+'_weight'] for m in c['modes']}
    else:weights=c['ensemble']
    test=common.stage_root(c,'test')/'results';final=common.stage_root(c,'final')/'results'
    for folder,days in [(test,[1914,1941]),(final,[1942,1969])]:
        integrity=json.loads((folder/'integrity.json').read_text())
        assert integrity['days']==days and integrity['rows']==30490 and integrity['full_coverage']
    result=assemble(pd.read_csv(test/'ensemble.csv'),pd.read_csv(final/'ensemble.csv'),pd.read_csv(ROOT/'data/sample_submission.csv'))
    output=args.output or final/'kaggle_ensemble.csv'
    common.atomic_csv(result,output)
    common.atomic_json({'version':args.version,'rows':60980,'weights':weights,
        'validation':{'train_end':1913,'days':[1914,1941],'content':'genuine out-of-sample Public-window ensemble'},
        'evaluation':{'train_end':1941,'days':[1942,1969],'content':'final full-history ensemble'},
        'submitted':False},output.with_suffix('.semantics.json'))
    print(output)

if __name__=='__main__':main()
