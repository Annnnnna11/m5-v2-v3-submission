import unittest
import warnings
import numpy as np
import pandas as pd
from common import F, config, stage_spec
from features import legacy, encoding_stats
from metrics import LEVELS, WRMSSE, scaling, rmsse, aligned, blend, selection_wrmsse
from model import predict_recursive, predict_28_days, split_masks, ROLL

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

def synthetic_frame(seed=7,ids=6,cut=120):
    """Small frame shaped like a real cutoff cache: history d<=cut with real
    sales, future cut<d<=cut+28 masked NaN, feature columns for both modes."""
    rng=np.random.default_rng(seed)
    rows=[]
    for i,identifier in enumerate([f'i{j}' for j in range(ids)]):
        for d in range(1,cut+29):
            value=float(max(0,rng.poisson(2+i))) if d<=cut else np.nan
            rows.append((identifier,d,value))
    frame=pd.DataFrame(rows,columns=['id','d','sales'])
    frame['x1']=(frame.d%7).astype(np.float32)
    frame['x2']=(frame.id.str[1:].astype(int)*0.5).astype(np.float32)
    for shift,window in ROLL: frame[f'rolling_mean_tmp_{shift}_{window}']=np.float32(np.nan)
    features=['x1','x2']+[f'rolling_mean_tmp_{shift}_{window}' for shift,window in ROLL]
    return frame,features

def tiny_params(mode):
    """Config-like params scaled down for synthetic data; one thread for
    determinism (scheme 4.9 prefix-equivalence test protocol)."""
    return {'objective':'tweedie','tweedie_variance_power':1.1,'learning_rate':0.05,
            'num_leaves':15,'min_data_in_leaf':5,'feature_fraction':0.8,
            'subsample':0.8,'subsample_freq':1,'metric':'rmse',
            'seed':42 if mode=='recursive' else 1995,'verbosity':-1,
            'force_col_wise':True,'num_threads':1}

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
        p=legacy(3); p.ENCODING_GROUPS=[['item_id'],['store_id','dept_id']]
        sales=pd.DataFrame({'id':['a','b'],'item_id':['x','x'],'dept_id':['d','d'],'cat_id':['c','c'],
                            'store_id':['A','B'],'state_id':['S','S'],'d_1':[0.,1.],'d_2':[2.,3.],'d_3':[4.,5.]})
        cal=pd.DataFrame({'d_num':[1,2,3],'wm_yr_wk':[1,2,3]})
        release=pd.DataFrame({'store_id':['A','B'],'item_id':['x','x'],'release':[2,1]})
        stats=encoding_stats(p,sales,cal,release,3)[('item_id',)].iloc[0]
        self.assertEqual(stats['count'],5)
        self.assertEqual(stats['mean'],3)
        self.assertAlmostEqual(stats['std'],np.std([2,4,1,3,5],ddof=1))
    # --- v3 additions (scheme 4.9) ---
    def test_config_v3(self):
        c=config()
        self.assertEqual(c['experiment'],'time_validation_v3_submission')
        for legacy_key in ('selection','ensemble','rounds'):  # old-scheme keys must be gone
            self.assertNotIn(legacy_key,c)
        self.assertEqual(c['round_candidates'],sorted(c['round_candidates']))
        self.assertGreater(min(c['round_candidates']),0)
        grid=c['ensemble_grid']
        self.assertEqual(grid[0],0.0); self.assertEqual(grid[-1],1.0); self.assertEqual(grid,sorted(grid))
        self.assertFalse(c['early_stopping']['enabled'])
        self.assertEqual(set(c['stages']),{'development','test','final'})
    def test_stage_spec_and_final_inheritance(self):
        from evaluate import selection_source_stage
        c=config()
        self.assertEqual(stage_spec(c,'development'),dict(stage='development',select_cutoff=1857,train_cutoff=1885,horizon=28))
        self.assertEqual(stage_spec(c,'test')['select_cutoff'],1885)
        self.assertIsNone(stage_spec(c,'final')['select_cutoff'])  # final has no selection step
        self.assertEqual(stage_spec(c,'final')['train_cutoff'],1941)
        self.assertEqual(selection_source_stage('final'),'test')
        self.assertEqual(selection_source_stage('test'),'test')
        self.assertEqual(selection_source_stage('development'),'development')
    def test_split_masks_disjoint(self):
        frame=pd.DataFrame({'d':[1,50,100,101,128,129,200]})
        train,forecast=split_masks(frame,100,28)
        self.assertEqual(train.tolist(),[True,True,True,False,False,False,False])
        self.assertEqual(forecast.tolist(),[False,False,False,True,True,False,False])
        self.assertFalse((train&forecast).any())
    def test_base_grid_masks_future(self):
        p=legacy(6); cut=6
        sales=pd.DataFrame({'id':['A_X_1','A_X_2'],'item_id':['X','X'],'dept_id':['D','D'],'cat_id':['C','C'],
                            'store_id':['A','A'],'state_id':['S','S'],
                            'd_1':[1.,0.],'d_2':[2.,1.],'d_3':[0.,2.],'d_4':[3.,0.],'d_5':[1.,1.],'d_6':[2.,3.]})
        cal=pd.DataFrame({'d_num':list(range(1,cut+29)),'wm_yr_wk':[1+(d-1)//7 for d in range(1,cut+29)]})
        voc={'id':['A_X_1','A_X_2'],'item_id':['X'],'dept_id':['D'],'cat_id':['C'],'store_id':['A'],'state_id':['S']}
        release=pd.DataFrame({'store_id':['A'],'item_id':['X'],'release':[1]})
        grid=p.build_base_grid('A',sales,cal,voc,{'A_X_1':0,'A_X_2':1},1,release)
        self.assertTrue(grid.loc[grid.d>cut,'sales'].isna().all())   # future truth fully masked
        self.assertTrue(grid.loc[grid.d<=cut,'sales'].notna().all())
        self.assertEqual(int(grid.d.max()),cut+28)
        sizes=grid.groupby('id',observed=True).size()
        self.assertTrue(sizes.eq(cut+28).all())                       # exactly 28 future days per id
    def test_encoding_ignores_days_after_cutoff(self):
        p=legacy(4); p.ENCODING_GROUPS=[['item_id']]
        sales=pd.DataFrame({'id':['a','b'],'item_id':['x','x'],'dept_id':['d','d'],'cat_id':['c','c'],
                            'store_id':['A','B'],'state_id':['S','S'],
                            'd_1':[1.,2.],'d_2':[5.,3.],'d_3':[1000.,1000.],'d_4':[1000.,1000.]})
        cal=pd.DataFrame({'d_num':[1,2,3,4],'wm_yr_wk':[1,1,2,2]})
        release=pd.DataFrame({'store_id':['A','B'],'item_id':['x','x'],'release':[1,1]})
        stats=encoding_stats(p,sales,cal,release,2)[('item_id',)].iloc[0]
        self.assertEqual(stats['count'],4)
        self.assertAlmostEqual(stats['mean'],(1+5+2+3)/4)             # d_3/d_4 excluded
        self.assertAlmostEqual(stats['std'],np.std([1,5,2,3],ddof=1))
    def test_selection_scope_bounds(self):
        from evaluate import selection_scope, revenue_window
        days=40
        sales=pd.DataFrame({'id':[f'i{i}' for i in range(3)],
                            **{f'd_{d}':[float(d)]*3 for d in range(1,days+1)}})
        history,truth,hcols,tcols=selection_scope(sales,12,28)
        self.assertEqual(history.shape,(3,12)); self.assertEqual(truth.shape,(3,28))
        self.assertTrue((history[:,0]==1).all() and (history[:,-1]==12).all())
        self.assertTrue((truth[:,0]==13).all() and (truth[:,-1]==40).all())
        # Returned lists are the ACTUAL slices used (final review): first/last
        # labels anchor the whole day range.
        self.assertEqual(hcols[0],'d_1'); self.assertEqual(hcols[-1],'d_12')
        self.assertEqual(tcols[0],'d_13'); self.assertEqual(tcols[-1],'d_40')
        # Selection-window boundaries: test-stage selection truth stops at
        # 1913; dollar weights cover only the 28 days ending at the cutoff.
        self.assertEqual(1885+28,1913)
        self.assertEqual(revenue_window(1857),list(range(1830,1858)))
        self.assertEqual(revenue_window(1885)[-1],1885)
        self.assertNotIn(1914,revenue_window(1885))
    def test_shifted_column_list_rejected(self):
        # Final review: a whole-list day-column shift (same width) previously
        # slipped past BOTH guards — selection_scope compared a slice against
        # itself (tautological) and selection_scorer forwarded a REBUILT list,
        # so the factory never saw what was actually sliced. Now the actual
        # lists are returned and forwarded, and the shift must fail.
        from unittest.mock import patch
        from metrics import make_selection_scorer
        import evaluate as ev
        meta=metadata(2); days=40
        sales=meta.copy()
        for d in range(1,days+1): sales[f'd_{d}']=[float(d)]*2
        cutoff=12; revenue=np.array([1.,2.])
        shifted=[f'd_{d}' for d in range(3,cutoff+3)]   # d_3..d_14: same width, wrong days
        history=sales[shifted].to_numpy(dtype=np.float64)
        with self.assertRaises(AssertionError):         # factory with the ACTUAL list
            make_selection_scorer(meta,history,revenue,cutoff,columns=shifted)
        correct=[f'd_{d}' for d in range(1,cutoff+1)]
        make_selection_scorer(meta,sales[correct].to_numpy(dtype=np.float64),
                              revenue,cutoff,columns=correct)   # correct content passes
        # End-to-end: a drifted selection_scope now fails inside selection_scorer
        truth=sales[[f'd_{d}' for d in range(cutoff+1,cutoff+29)]].to_numpy(dtype=np.float64)
        truth_cols=[f'd_{d}' for d in range(cutoff+1,cutoff+29)]
        with patch.object(ev,'selection_scope',return_value=(history,truth,shifted,truth_cols)), \
             patch.object(ev,'load_sales',return_value=sales), \
             patch.object(ev,'revenue_at',return_value=revenue):
            with self.assertRaises(AssertionError):
                ev.selection_scorer({'horizon':28,'smoke_items':48},cutoff)
    def test_selection_scoring_covers_full_28_days(self):
        meta=metadata(2)
        history=np.array([[0.,1,2,3],[0,1,2,3]])  # cutoff=4
        truth=np.full((2,28),2.0); revenue=np.array([1.,2.])
        score,_=selection_wrmsse(meta,history,revenue,truth,truth.copy(),4)
        self.assertEqual(score,0)
        pred=truth.copy(); pred[:,-1]+=1.0        # change only day 28
        changed,_=selection_wrmsse(meta,history,revenue,truth,pred,4)
        self.assertGreater(changed,0)             # day 28 is inside the score
        with self.assertRaises(AssertionError):   # history must be bounded by cutoff
            selection_wrmsse(meta,np.zeros((2,5)),revenue,truth,truth,4)
    def test_blend(self):
        a=np.array([[1.,2.]]); b=np.array([[3.,4.]])
        np.testing.assert_allclose(blend(a,b,0.0),b)
        np.testing.assert_allclose(blend(a,b,1.0),a)
        np.testing.assert_allclose(blend(a,b,0.25),0.25*a+0.75*b)
        for w in (-0.1,1.1):
            with self.assertRaises(AssertionError): blend(a,b,w)
    def test_select_cache_cleanup_decisions(self):
        from run import cleanup_decisions
        c=config()
        self.assertEqual(cleanup_decisions(c,[]),[])
        self.assertEqual(cleanup_decisions(c,['development']),[1857])
        self.assertEqual(cleanup_decisions(c,['test']),[])       # 1885 still serves dev retrain
        self.assertEqual(cleanup_decisions(c,['development','test']),[1857,1885])
        self.assertEqual(cleanup_decisions(c,['development','test','final']),[1857,1885])
        self.assertEqual(cleanup_decisions(c,['final']),[])
    def test_artifact_id_binds_grid(self):
        # Regression: select_ensemble_weight/load_selection pass grid=... into
        # artifact_id; the keyword must be accepted (an earlier smoke run died
        # on TypeError here) and a changed blend grid or rounds must change the
        # fingerprint (scheme 4.6).
        from unittest.mock import patch
        import common
        c=config()
        rounds={'recursive':3000,'nonrecursive':2500}
        with patch.object(common,'manifest',return_value='run-fingerprint'):
            base=common.artifact_id(c,'selection',1857,kind='weights',rounds=rounds,grid=[0.0,0.5,1.0])
            same=common.artifact_id(c,'selection',1857,kind='weights',rounds=rounds,grid=[0.0,0.5,1.0])
            grid_changed=common.artifact_id(c,'selection',1857,kind='weights',rounds=rounds,grid=[0.0,0.25,1.0])
            rounds_changed=common.artifact_id(c,'selection',1857,kind='weights',rounds={'recursive':2000,'nonrecursive':2500},grid=[0.0,0.5,1.0])
            self.assertEqual(base,same)
            self.assertNotEqual(base,grid_changed)
            self.assertNotEqual(base,rounds_changed)

if __name__=='__main__': unittest.main(verbosity=2)
