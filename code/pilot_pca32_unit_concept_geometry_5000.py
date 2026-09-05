"""Exploratory PCA32 geometry of 10 TVSD units and 500 res4 concepts."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent; sys.path.insert(0,str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as tvsd

ROOT=Path(r'D:\Coding\BrainAI'); PROJECT=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
FEATURES=ROOT/'cache/tvsd/original_fwrf_resnet50/spatial_full_14'
CAV=PROJECT/'tcav_broden500/broden500_native_channel_cav_bank.npz'
OUT=PROJECT/'results/pilot_pca32_unit_concept_geometry_5000'; N=5000; SEED=20260904
WINDOW_STARTS=(70,110,150); N_UNITS=10

def stats(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    return {'n':int(v.size),'min':float(v.min()),'max':float(v.max()),'p5':float(np.percentile(v,5)),'median':float(np.median(v)),'p95':float(np.percentile(v,95))}
def pair(a,b):
    a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(1e-12); b=b/np.linalg.norm(b,axis=1,keepdims=True).clip(1e-12)
    c=np.clip(a@b.T,-1,1); return c,1-c,np.sqrt(np.maximum(0,2-2*c))
def pooled(path,field):
    m=np.load(path,mmap_mode='r'); out=np.empty((len(m),m.shape[1]),np.float32)
    for a in range(0,len(m),256):
        b=min(a+256,len(m)); out[a:b]=np.einsum('nchw,hw->nc',np.asarray(m[a:b],np.float32),field)
    return out
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    y,_,chosen,rois,_=tvsd.load_targets(); y=y[:,[1,3,5],:N_UNITS].astype(np.float32)
    xs,ys,sg=tvsd.candidates(); j=int(np.argmin((xs**2+ys**2)+(sg-8)**2)); field=tvsd.gaussian_mass_stack(xs[j:j+1],ys[j:j+1],sg[j:j+1],14)[0]
    X=pooled(FEATURES/'train_res4_f16.npy',field); rng=np.random.default_rng(SEED); idx=np.sort(rng.choice(len(X),N,replace=False)); mu=X[idx].mean(0); sd=X[idx].std(0).clip(1e-5); Xs=(X[idx]-mu)/sd
    _,_,V=np.linalg.svd(Xs,full_matrices=False); V=V[:32].T; Z=Xs@V
    # Independent time-window unit axes in the same fixed PCA coordinates.
    axes=[]; names=[]
    for wi,start in enumerate(WINDOW_STARTS):
        zc=Z-Z.mean(0); yc=y[idx,wi]-y[idx,wi].mean(0); g=zc.T@zc; lam=.05*np.trace(g)/32
        beta=np.linalg.solve(g+np.eye(32,dtype=np.float32)*lam,zc.T@yc).T
        axes.append(beta); names += [f'unit_{int(chosen[u])}_{rois[u]}_{start}ms' for u in range(N_UNITS)]
    U=np.concatenate(axes); # 30 x 32
    cav=np.load(CAV,allow_pickle=True); concepts=cav['concepts'].astype(str); nodes=cav['nodes'].astype(str); off=cav['offsets'].astype(int); full=cav['cav_full'].astype(np.float32)
    ni=int(np.flatnonzero(nodes=='res4_b6')[0]); a,b=off[ni],off[ni+1]; C=(full[:,a:b]/sd[None])@V; C=C/np.linalg.norm(C,axis=1,keepdims=True).clip(1e-12)
    U=U/np.linalg.norm(U,axis=1,keepdims=True).clip(1e-12)
    rows=[]
    def add(kind,vals):
        s=stats(vals); s['comparison']=kind; rows.append(s)
    iu=np.triu_indices(len(U),1); ic=np.triu_indices(len(C),1)
    cosuu=U@U.T; coscc=C@C.T; cosuc=U@C.T
    add('unit_unit_all_30_axes',1-cosuu[iu]); add('concept_concept_500_axes',1-coscc[ic]); add('unit_concept_30x500',1-cosuc.ravel())
    # Same-unit temporal distances, explicitly separate from all unit pairs.
    temporal=[]
    for u in range(N_UNITS):
        for p in range(3):
            for q in range(p+1,3): temporal.append(1-float(U[p*N_UNITS+u]@U[q*N_UNITS+u]))
    add('same_unit_time_window_pairs',temporal)
    pd.DataFrame(rows).to_csv(OUT/'distance_summary.csv',index=False)
    # Save the full pairwise matrices with labels for follow-up analyses.
    pd.DataFrame(1-cosuu,index=names,columns=names).to_csv(OUT/'unit_unit_cosine_distance.csv')
    pd.DataFrame(1-coscc,index=concepts,columns=concepts).to_csv(OUT/'concept_concept_cosine_distance.csv')
    pd.DataFrame(1-cosuc,index=names,columns=concepts).to_csv(OUT/'unit_concept_cosine_distance.csv')
    print(pd.DataFrame(rows).set_index('comparison').to_string())
    print('unit globals:',chosen[:N_UNITS].tolist(),'rois:',rois[:N_UNITS].tolist())
    print('saved',OUT)
if __name__=='__main__': main()
