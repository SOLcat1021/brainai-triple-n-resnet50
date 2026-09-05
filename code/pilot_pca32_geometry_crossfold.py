"""Cross-fold stability audit for the 5000-image PCA32 unit/concept geometry."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parent; sys.path.insert(0,str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as tvsd

ROOT=Path(r'D:\Coding\BrainAI'); PROJECT=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
FEATURES=ROOT/'cache/tvsd/original_fwrf_resnet50/spatial_full_14'; CAV=PROJECT/'tcav_broden500/broden500_native_channel_cav_bank.npz'
OUT=PROJECT/'results/pilot_pca32_geometry_5000_crossfold'; N=5000; REPS=5; SEED=20260904; WINDOWS=(1,3,5); N_UNITS=10
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)).clip(1e-12))
def pooled(path,field):
 m=np.load(path,mmap_mode='r'); out=np.empty((len(m),m.shape[1]),np.float32)
 for a in range(0,len(m),256):
  b=min(a+256,len(m)); out[a:b]=np.einsum('nchw,hw->nc',np.asarray(m[a:b],np.float32),field)
 return out
def fit(x,y):
 xc=x-x.mean(0); g=xc.T@xc; lam=.05*np.trace(g)/g.shape[0]
 return np.linalg.solve(g+np.eye(g.shape[0],dtype=np.float32)*lam,xc.T@(y-y.mean(0))).T
def main():
 OUT.mkdir(parents=True,exist_ok=True); y,_,chosen,rois,_=tvsd.load_targets(); y=y[:,WINDOWS,:N_UNITS].astype(np.float32)
 xs,ys,sg=tvsd.candidates(); j=int(np.argmin((xs**2+ys**2)+(sg-8)**2)); field=tvsd.gaussian_mass_stack(xs[j:j+1],ys[j:j+1],sg[j:j+1],14)[0]
 X=pooled(FEATURES/'train_res4_f16.npy',field); cav=np.load(CAV,allow_pickle=True); concepts=cav['concepts'].astype(str); nodes=cav['nodes'].astype(str); off=cav['offsets'].astype(int); full=cav['cav_full'].astype(np.float32); ni=int(np.flatnonzero(nodes=='res4_b6')[0]); a,b=off[ni],off[ni+1]; Cnative=full[:,a:b]
 rng=np.random.default_rng(SEED); folds=[]; fold_axes=[]; fold_concepts=[]
 for rep in range(REPS):
  idx=np.sort(rng.choice(len(X),N,replace=False)); mu=X[idx].mean(0); sd=X[idx].std(0).clip(1e-5); z=(X[idx]-mu)/sd
  _,_,V=np.linalg.svd(z,full_matrices=False); V=V[:32].T; Z=z@V
  ax=np.stack([fit(Z,y[idx,w]) for w in range(3)])
  # Convert PCA coefficients and CAVs back to common native coordinates.
  axes_native=np.einsum('wud,fd->wuf',ax,V)/sd[None,None,:]
  cav_native=(Cnative/sd[None])@V@V.T
  fold_axes.append(axes_native); fold_concepts.append(cav_native)
  folds.append({'rep':rep,'n_images':N,'pca_dim':32,'unit_axis_norm_median':float(np.median(np.linalg.norm(axes_native,axis=2)))})
 pair_rows=[]
 for i in range(REPS):
  for j in range(i+1,REPS):
   for w,start in enumerate((70,110,150)):
    for u in range(N_UNITS): pair_rows.append({'rep_a':i,'rep_b':j,'kind':'unit_axis','window_ms':start,'unit_global':int(chosen[u]),'cosine':cos(fold_axes[i][w,u],fold_axes[j][w,u])})
   for k,name in enumerate(concepts): pair_rows.append({'rep_a':i,'rep_b':j,'kind':'concept_axis','window_ms':'res4','unit_global':name,'cosine':cos(fold_concepts[i][k],fold_concepts[j][k])})
 pd.DataFrame(folds).to_csv(OUT/'fold_manifest.csv',index=False); d=pd.DataFrame(pair_rows); d.to_csv(OUT/'pairwise_axis_cosines.csv',index=False)
 print(d.groupby('kind').cosine.agg(['count','min','median','max','mean']).to_string()); print('unit by window'); print(d[d.kind=='unit_axis'].groupby('window_ms').cosine.agg(['count','min','median','max']).to_string()); print('saved',OUT)
if __name__=='__main__': main()
