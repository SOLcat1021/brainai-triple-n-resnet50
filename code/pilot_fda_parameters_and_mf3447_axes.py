"""FDA-inspired local amplitude/phase/dilation parameter pilot and MF3447 concept links."""
from pathlib import Path
import sys, h5py
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); B=P/'proxy_bank_10ms_all_windows'; R=P/'results'; C=P/'code'; OUT=R/'pilot_fda_parameters_2026-08-30'; OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(C))
from run_trin_closed_loop_validation import fixed_folds
from plot_orthogonal_temporal_modes_2d import unit_vector
FROZEN=R/'orthogonal_temporal_modes_2d_high_consistency_2026-08-27'; ROIS=['V1','V2','V4','posterior IT','middle IT','anterior IT']
def corr(a,b): return float(pearsonr(a,b).statistic) if np.std(a)>1e-9 and np.std(b)>1e-9 else np.nan
def basis(template,times):
    mu=gaussian_filter1d(template.astype(float),.75); der=np.gradient(mu,times); e=mu*mu; center=np.sum(times*e)/max(e.sum(),1e-9)
    raw=np.column_stack([mu,-der,-(times-center)*der-.5*mu]); return raw/np.linalg.norm(raw,axis=0,keepdims=True)
def load(roi,units,resp,base,tid):
    neural=(np.asarray(resp[tid][:,:,units],np.float32)-np.asarray(resp[base][:,:,units],np.float32).mean(0)[None]).transpose(2,1,0)
    with h5py.File(B/f'{roi.lower().replace(" ","_")}_proxy_bank.h5','r') as h:
        lookup={int(v):i for i,v in enumerate(h['unit_global'][:])}; loc=np.array([lookup[int(u)] for u in units]); order=np.argsort(loc); pred=np.empty((len(base)+len(tid),1000,len(units)),np.float32); idx=np.r_[base,tid]
        for k,w in enumerate(idx): pred[k,:,order]=np.asarray(h['oof_prediction'][w,:,loc[order]]).T
    return neural,(pred[len(base):]-pred[:len(base)].mean(0)[None]).transpose(2,1,0)
def main():
    meta=pd.read_csv(B/'unit_metadata.csv'); frozen=pd.read_csv(FROZEN/'unit_orthogonal_GTD_weights_and_modes.csv'); resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r'); w=np.load(B/'windows_10ms.npy').astype(float); baseidx=np.flatnonzero(w[:,1]<0); tid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); times=w[tid].mean(1)
    rows=[]
    for roi in ROIS:
        q=frozen[frozen.roi.eq(roi)].sort_values('unit_global'); units=q.unit_global.to_numpy(int); neural,proxy=load(roi,units,resp,baseidx,tid)
        for ui,u in enumerate(q.itertuples(index=False)):
            vals={k:[] for k in ['A','tau','dilation','residual']}; preds={k:[] for k in vals}
            for train,test in fixed_folds(1000):
                z=basis(neural[ui,train].mean(0),times); target=(neural[ui]-neural[ui,train].mean(0))@z; pred=(proxy[ui]-neural[ui,train].mean(0))@z
                for k,n in enumerate(['A','tau','dilation']): vals[n].append(target[test,k]); preds[n].append(pred[test,k])
                # Residual shape magnitude after removing the three FDA tangent directions.
                tm=neural[ui,train].mean(0); vals['residual'].append(np.linalg.norm((neural[ui,test]-tm)-target[test]@z.T,axis=1)); preds['residual'].append(np.linalg.norm((proxy[ui,test]-tm)-pred[test]@z.T,axis=1))
            for n in vals: rows.append({'unit_global':int(u.unit_global),'roi':roi,'metric':n,'proxy_r':corr(np.concatenate(vals[n]),np.concatenate(preds[n]))})
        print(roi,flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/'unit_fda_proxy_r.csv',index=False); summary=d.groupby('metric').proxy_r.agg(['count','median','mean',lambda x:(x>0).mean()]).reset_index(); summary.columns=['metric','n_units','median_r','mean_r','positive_fraction']; summary.to_csv(OUT/'fda_parameter_predictability_summary.csv',index=False)
    # MF3447: use already generated candidate-circle head/wall score and observed G/T/D, append FDA parameter analogues from full curve.
    mf=R/'mf_head_wall_rf_gtd_pilot_2026-08-28'; zscores=np.load(mf/'head_wall_rf_scores_and_GTD.npz'); concepts=[str(x) for x in zscores['concepts']]; units=zscores['units'].astype(int); ui=int(np.flatnonzero(units==3447)[0])
    raw=np.asarray(resp[:,:,3447],np.float32); curve=(raw[tid]-raw[baseidx].mean(0)[None]).T; z=basis(curve.mean(0),times); coef=(curve-curve.mean(0))@z
    # Existing file has concept scores in the first three columns; detect score columns robustly.
    if zscores['image_scores'].shape[-1]==1000:
        for ci,concept in enumerate(concepts):
            col=zscores['image_scores'][ci,ui]
            out=[]
            for k,n in enumerate(['A','tau','dilation']): out.append({'concept':concept,'metric':n,'r':corr(col,coef[:,k])})
            pd.DataFrame(out).to_csv(OUT/f'mf3447_{concept}_fda_correlations.csv',index=False)
    print(summary.to_string(index=False))
if __name__=='__main__': main()
