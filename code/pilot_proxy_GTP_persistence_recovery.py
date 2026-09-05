"""Pilot: can existing temporal proxies recover a G/T-conditioned persistence score P?"""
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24')
CODE=P/'code'; sys.path.insert(0,str(CODE))
from plot_orthogonal_temporal_modes_2d import raw_directions, unit_vector
from run_trin_closed_loop_validation import fixed_folds
BANK=P/'proxy_bank_10ms_all_windows'; OUT=P/'results'/'pilot_proxy_GTP_persistence_2026-08-30'; OUT.mkdir(parents=True,exist_ok=True)
FROZEN=P/'results'/'orthogonal_temporal_modes_2d_high_consistency_2026-08-27'
ROIS=['V1','V2','V4','posterior IT','middle IT','anterior IT']; SEED=20260830

def slug(x): return x.lower().replace(' ','_')
def corr(x,y):
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<3: return np.nan
    x=x-x.mean(); y=y-y.mean(); den=np.sqrt((x*x).sum()*(y*y).sum())
    return float((x*y).sum()/den) if den>1e-12 else np.nan

def load(roi,units,responses,base,tid):
    raw=np.asarray(responses[:,:,units],np.float32); neural=(raw[tid]-raw[base].mean(axis=0)[None]).transpose(2,1,0)
    with h5py.File(BANK/f'{slug(roi)}_proxy_bank.h5','r') as h:
        lookup={int(v):i for i,v in enumerate(h['unit_global'][:])}; local=np.array([lookup[int(u)] for u in units]); order=np.argsort(local); sl=local[order]
        take=np.r_[base,tid]; pred=np.empty((len(take),1000,len(units)),np.float32)
        for oi,w in enumerate(take): pred[oi,:,order]=np.asarray(h['oof_prediction'][w,:,sl]).T
    proxy=(pred[len(base):]-pred[:len(base)].mean(axis=0)[None]).transpose(2,1,0)
    return neural,proxy

def gtp_basis(template,times):
    raw,_=raw_directions(template,times)
    if raw is None: return None
    g=raw[:,0]
    # Preserve the physical gain direction; make timing and persistence conditional on predecessors.
    t0=raw[:,1]-g*(g@raw[:,1]); t=unit_vector(t0)
    # Predefined contrast: positive late residual response, negative early response.
    p0=np.where(times>=130.,1.,np.where(times<=120.,-1.,0.)).astype(float)
    p0-=p0.mean(); X=np.column_stack([g,t]); p=unit_vector(p0-X@np.linalg.lstsq(X,p0,rcond=None)[0])
    if t is None or p is None: return None
    return np.column_stack([g,t,p])

def main():
    frozen=pd.read_csv(FROZEN/'unit_orthogonal_GTD_weights_and_modes.csv').sort_values(['roi','unit_global'])
    responses=np.load(BANK/'trin_all_units_10ms_responses.npy',mmap_mode='r'); w=np.load(BANK/'windows_10ms.npy').astype(float)
    base=np.flatnonzero(w[:,1]<0); tid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); times=w[tid].mean(1)
    rows=[]; coupling=[]
    for roi in ROIS:
        q=frozen[frozen.roi.eq(roi)].sort_values('unit_global'); units=q.unit_global.to_numpy(int); neural,proxy=load(roi,units,responses,base,tid)
        for ui,u in enumerate(q.itertuples(index=False)):
            pooled={x:[[],[]] for x in ['G','T','P']}
            for fold,(train,test) in enumerate(fixed_folds(1000)):
                basis=gtp_basis(neural[ui,train].mean(axis=0),times)
                if basis is None: continue
                # Directions and targets are learned only from training images; proxy remains OOF throughout.
                target=(neural[ui,test]-neural[ui,train].mean(axis=0))@basis
                pred=(proxy[ui,test]-neural[ui,train].mean(axis=0))@basis
                for k,name in enumerate(['G','T','P']): pooled[name][0].append(target[:,k]); pooled[name][1].append(pred[:,k])
                coupling.append({'unit_global':int(u.unit_global),'roi':roi,'fold':fold,'cos_G_T':float(basis[:,0]@basis[:,1]),'cos_G_P':float(basis[:,0]@basis[:,2]),'cos_T_P':float(basis[:,1]@basis[:,2])})
            for name,(a,b) in pooled.items():
                a=np.concatenate(a); b=np.concatenate(b)
                rows.append({'unit_global':int(u.unit_global),'roi':roi,'monkey':u.monkey,'session':u.session,'metric':name,'n_images':len(a),'proxy_r':corr(a,b)})
        print(roi,flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/'unit_GTP_proxy_recovery.csv',index=False)
    pd.DataFrame(coupling).to_csv(OUT/'fold_GTP_basis_audit.csv',index=False)
    session=d.groupby(['roi','monkey','session','metric'],as_index=False).proxy_r.median()
    session.to_csv(OUT/'session_GTP_proxy_recovery.csv',index=False)
    summary=d.groupby('metric').proxy_r.agg(['count','median','mean',lambda x:(x>0).mean()]).reset_index(); summary.columns=['metric','n_units','median_unit_r','mean_unit_r','positive_fraction']
    ss=session.groupby('metric').proxy_r.agg(['count','median','mean']).reset_index(); ss.columns=['metric','n_session_groups','median_session_r','mean_session_r']; summary=summary.merge(ss,on='metric')
    # Within-unit paired comparison establishes whether P is systematically more/less recoverable than G/T.
    wide=d.pivot(index='unit_global',columns='metric',values='proxy_r')
    comps=[]
    for ref in ['G','T']:
        delta=wide['P']-wide[ref]; stat,p=wilcoxon(delta.dropna()); comps.append({'comparison':f'P minus {ref}','median_delta_r':float(delta.median()),'wilcoxon_p':float(p),'n':int(delta.notna().sum())})
    summary.to_csv(OUT/'GTP_proxy_recovery_summary.csv',index=False); pd.DataFrame(comps).to_csv(OUT/'P_vs_GT_paired_comparisons.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.5)); arrays=[d[d.metric.eq(m)].proxy_r.to_numpy() for m in ['G','T','P']]; vp=ax.violinplot(arrays,showmedians=True,showextrema=False)
    for body,color in zip(vp['bodies'],['#3b75af','#df8f29','#4d9a65']): body.set_facecolor(color); body.set_alpha(.55)
    ax.axhline(0,color='0.4',lw=.8); ax.set(xticks=[1,2,3],xticklabels=['G','T','P'],ylabel='Held-out image Pearson r',title='Existing proxy recovery: G/T-conditioned persistence pilot'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'GTP_proxy_recovery_violin.png',dpi=200); plt.close(fig)
    print(summary.to_string(index=False)); print(pd.DataFrame(comps).to_string(index=False))
if __name__=='__main__': main()
