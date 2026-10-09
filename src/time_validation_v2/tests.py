import unittest
import warnings
import numpy as np
import pandas as pd
from common import F
from features import legacy
from metrics import LEVELS, WRMSSE, scaling, rmsse, aligned
from model import predict_recursive, ROLL

def reference(meta,history,truth,pred,revenue):
    # Deliberately independent: pandas groupby, scalar trimming and Python loops.
    # Follows Nixtla M5Evaluation.aggregate_levels / evaluate (source provenance in docs).
    scores=[]
    for keys in LEVELS:
        groups=meta.groupby(keys,sort=True).indices.values() if keys else [np.arange(len(meta))]
        total=0
        for idx in groups:
            y=history[idx].sum(axis=0)
            nz=np.flatnonzero(y)
            y=y[nz[0]:] if len(nz) else y
            scale=np.mean([(float(y[j])-float(y[j-1]))**2 for j in range(1,len(y))]) if len(y)>1 else 0
            error=pred[idx].sum(axis=0)-truth[idx].sum(axis=0)
            mse=sum(float(v)**2 for v in error)/len(error)
            weight=float(revenue[idx].sum()/revenue.sum())
            if weight:
                total+=weight*(np.sqrt(mse/scale) if scale>0 else (0 if mse==0 else np.inf))
        scores.append(total)
    return np.mean(scores),scores

def metadata(n=8):
    return pd.DataFrame({'id':[f'i{i}' for i in range(n)],'state_id':[f's{i%2}' for i in range(n)],
                         'store_id':[f't{i%4}' for i in range(n)],'cat_id':[f'c{i%2}' for i in range(n)],
                         'dept_id':[f'd{i%3}' for i in range(n)],'item_id':[f'p{i//2}' for i in range(n)]})

class Checks(unittest.TestCase):
    def test_resume_rejects_wrong_scope_or_corruption(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from common import checkpoint,complete,digest
        with TemporaryDirectory() as directory:
            path=Path(directory); (path/'data').write_text('valid')
            first=digest({'cutoff':1885,'store':'CA_1','mode':'recursive'})
            wrong=digest({'cutoff':1913,'store':'CA_1','mode':'recursive'})
            checkpoint(path/'complete.json',first,['data'])
            self.assertTrue(complete(path/'complete.json',first,['data']))
            with self.assertRaises(RuntimeError): complete(path/'complete.json',wrong,['data'])
            (path/'data').write_text('corrupt')
            self.assertFalse(complete(path/'complete.json',first,['data']))
    def test_hand_calculation(self):
        y=np.array([[0.,0,1,2,3]])
        self.assertEqual(scaling(y)[0],1)
        score,levels=WRMSSE(metadata(1),y,np.array([9.])).score(np.array([[4.,5.]]),np.array([[5.,6.]]))
        self.assertEqual(score,1); self.assertTrue((levels.wrmsse==1).all())
    def test_independent_reference(self):
        rng=np.random.default_rng(47); meta=metadata()
        y=rng.poisson(2,(8,80)).astype(float); y[:,:5]=0
        truth=rng.poisson(2,(8,28)); pred=rng.uniform(0,4,(8,28)); dollars=np.arange(1,9,dtype=float)
        score,levels=WRMSSE(meta,y,dollars).score(truth,pred)
        expected,per_level=reference(meta,y,truth,pred,dollars)
        np.testing.assert_allclose(levels.wrmsse,per_level,rtol=1e-12)
        self.assertAlmostEqual(score,expected,12)
    def test_edge_cases(self):
        np.testing.assert_array_equal(scaling(np.array([[0,0,0],[0,0,2],[2,2,2],[0,1,3]])),[0,0,0,4])
        np.testing.assert_array_equal(rmsse(np.array([0,1,4]),np.array([0,0,4])),[0,np.inf,1])
        with self.assertRaises(ValueError): WRMSSE(metadata(1),np.ones((1,4)),np.zeros(1))
        # An unweighted undefined series must not poison a valid score.
        score,_=WRMSSE(metadata(2),np.array([[0,1,2],[0,0,0]]),np.array([1.,0.])).score(np.array([[2.],[0.]]),np.array([[2.],[3.]]))
        self.assertTrue(np.isfinite(score))
    def test_alignment(self):
        frame=pd.DataFrame(np.ones((2,28)),columns=F); frame.insert(0,'id',['b','a'])
        frame.loc[0,F]=2
        np.testing.assert_array_equal(aligned(frame,['a','b'])[:,0],[1,2])
        frame.loc[1,'id']='b'
        with self.assertRaises(AssertionError): aligned(frame,['a','b'])
    def test_grouped_lags(self):
        p=legacy(200)
        grid=pd.DataFrame({'id':np.repeat(['a','b'],228),'d':np.tile(np.arange(1,229),2),
                           'sales':np.r_[np.arange(228),1000+np.arange(228)].astype(np.float32)})
        grid.loc[grid.d>200,'sales']=np.float32(np.nan)
        lag=p.build_lag_table(grid)
        for i in [200,227,428,455]:
            self.assertEqual(lag.loc[i,'sales_lag_28'],grid.loc[i-28,'sales'])
            self.assertAlmostEqual(lag.loc[i,'rolling_mean_7'],grid.loc[i-34:i-28,'sales'].mean())
        self.assertTrue(lag.loc[228:254,'sales_lag_28'].isna().all())
    def test_recursive_mask_and_float_feedback(self):
        cut=110; days=np.arange(11,139)
        frame=pd.DataFrame({'id':['a']*len(days),'d':days,'sales':np.full(len(days),2,dtype=np.float32)})
        frame.loc[frame.d>cut,'sales']=np.float32(999999)  # poisonous holdout labels
        features=[f'rolling_mean_tmp_{s}_{w}' for s,w in ROLL]
        for f in features: frame[f]=np.float32(np.nan)
        class Model:
            def predict(self,rows,**kwargs):
                return rows['rolling_mean_tmp_1_7'].to_numpy(dtype=float)+0.125
        with warnings.catch_warnings():
            warnings.simplefilter('error',FutureWarning)
            result=predict_recursive(Model(),frame,features,cut,1)
        self.assertEqual(result.F1.iloc[0],2.125)
        self.assertGreater(result.F2.iloc[0],result.F1.iloc[0])
        self.assertLess(result.F28.iloc[0],10)
    def test_global_encoding_sufficient_statistics(self):
        from features import encoding_stats
        p=legacy(3); p.ENCODING_GROUPS=[['item_id'],['store_id','dept_id']]
        sales=pd.DataFrame({'id':['a','b'],'item_id':['x','x'],'dept_id':['d','d'],'cat_id':['c','c'],
                            'store_id':['A','B'],'state_id':['S','S'],'d_1':[0.,1.],'d_2':[2.,3.],'d_3':[4.,5.]})
        cal=pd.DataFrame({'d_num':[1,2,3],'wm_yr_wk':[1,2,3]})
        release=pd.DataFrame({'store_id':['A','B'],'item_id':['x','x'],'release':[2,1]})
        stats=encoding_stats(p,sales,cal,release,3)[('item_id',)].iloc[0]
        self.assertEqual(stats['count'],5)
        self.assertEqual(stats['mean'],3)
        self.assertAlmostEqual(stats['std'],np.std([2,4,1,3,5],ddof=1))

if __name__=='__main__': unittest.main(verbosity=2)
