"""Original chunked M5 audit; no training or preprocessing."""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",default=str(Path(__file__).resolve().parents[1]/"data"))
    parser.add_argument("--output",default=str(Path(__file__).resolve().parents[1]/"experiments/raw_input_audit.json"))
    args=parser.parse_args()
    import json
    import pandas as pd
    import numpy as np
    root=Path(args.data_dir).resolve().parent
    cols=[f'd_{d}' for d in range(1,1914)]
    dtype={k:np.int16 for k in cols}
    a=pd.read_csv(Path(args.data_dir)/'sales_train_validation.csv',chunksize=512,dtype=dtype)
    b=pd.read_csv(Path(args.data_dir)/'sales_train_evaluation.csv',chunksize=512,dtype={**dtype,**{f'd_{d}':np.int16 for d in range(1914,1942)}})
    rows=0; ids=[]; stores={}
    from itertools import zip_longest
    for old,new in zip_longest(a,b):
        assert old is not None and new is not None
        assert old.id.str.replace('_validation','_evaluation',regex=False).tolist()==new.id.tolist()
        np.testing.assert_array_equal(old[cols].to_numpy(),new[cols].to_numpy())
        assert (new[[c for c in new if c.startswith('d_')]].to_numpy()>=0).all()
        rows+=len(new); ids+=new.id.tolist()
        for store,count in new.store_id.value_counts().items(): stores[store]=stores.get(store,0)+int(count)
    assert rows==30490 and len(set(ids))==30490 and len(stores)==10 and set(stores.values())=={3049}
    sample=pd.read_csv(Path(args.data_dir)/'sample_submission.csv')
    assert sample.columns.tolist()==['id',*[f'F{i}' for i in range(1,29)]]
    assert len(sample)==60980 and sample.id.is_unique
    assert set(sample.loc[sample.id.str.endswith('_evaluation'),'id'])==set(ids)
    assert set(sample.loc[sample.id.str.endswith('_validation'),'id'].str.replace('_validation','_evaluation',regex=False))==set(ids)
    calendar=pd.read_csv(Path(args.data_dir)/'calendar.csv')
    assert calendar.d.tolist()==[f'd_{d}' for d in range(1,len(calendar)+1)]
    assert len(calendar)>=1969
    assert (pd.to_datetime(calendar.date).diff().dropna()==pd.Timedelta(days=1)).all()
    result={'pass':True,'series':rows,'stores':stores,'validation_evaluation_shared_history_exact_match':True,
            'sample_submission_ids_exact_match':True,'calendar_days':len(calendar),
            'calendar_contiguous':True,'sales_nonnegative':True}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
