"""Small TVSD pilot: independent 25x25x8 fwRF and PCA32 per time window."""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import torch

CODE=Path(__file__).resolve().parent; sys.path.insert(0,str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as base
from extract_trin_resnet50_modelspace import gaussian_mass_stack

ROOT=Path(r'D:\Coding\BrainAI'); PROJECT=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
PCA8=ROOT/'cache/tvsd/original_fwrf_resnet50/spatial_pca8_14'; FULL=ROOT/'cache/tvsd/original_fwrf_resnet50/spatial_full_14'
OUT=PROJECT/'results/pilot_tvsd_independent_rf_pca32_full'; N=5000; FOLDS=5; SEED=20260904; WINDOWS=(1,3,5); N_UNITS=10
def cosine(a,b): return np.sum(a*b,-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).clip(1e-12)
def fit_ridge(x,y):
 xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym; g=xc.T@xc; lam=.05*np.trace(g)/g.shape[0]
 return np.linalg.solve(g+np.eye(g.shape[1],dtype=np.float32)*lam,xc.T@yc).T,xm,ym
def pool(path,fields,idx):
 m=np.load(path,mmap_mode='r'); out=np.empty((len(idx),len(fields),m.shape[1]),np.float32)
 for a in range(0,len(idx),128):
  b=min(a+128,len(idx)); z=np.asarray(m[idx[a:b]],np.float32); out[a:b]=np.einsum('nchw,ghw->ngc',z,fields)
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); y,yt,chosen,rois,_=base.load_targets(); y=y[:,WINDOWS,:N_UNITS].astype(np.float32); n=len(y)
 rng=np.random.default_rng(SEED); sample=np.sort(rng.choice(n,N,replace=False)); foldid=np.arange(N)%FOLDS
 xs=np.linspace(-10+20/25/2,10-20/25/2,25).astype(np.float32); ys=xs.copy(); sig=np.exp(np.linspace(np.log(.7),np.log(8),8)).astype(np.float32); xx,yy,ss=np.meshgrid(xs,ys,sig,indexing='ij'); fields=gaussian_mass_stack(xx.ravel(),yy.ravel(),ss.ravel(),14)
 maps8=np.load(PCA8/'train_maps_f16.npy',mmap_mode='r'); mapsfull=FULL/'train_res4_f16.npy'; chosen_all=[]; fold_axes=[]; rows=[]
 for fi in range(FOLDS):
  tr=sample[foldid!=fi]; te=sample[foldid==fi]; f=np.asarray(maps8[tr],np.float32); f=f.reshape(len(tr),f.shape[1],-1); yc=y[tr]
  fm=f-f.mean(0); ym=yc-yc.mean(0); cov=np.einsum('ncp,nwt->cpwt',fm,ym); ff=np.einsum('ncp,ncq->cpq',fm,fm); vy=np.sum(ym*ym,0).clip(1e-8); vx=np.einsum('gp,cpq,gq->gc',fields.reshape(len(fields),-1),ff,fields.reshape(len(fields),-1)).clip(1e-8); num=np.einsum('gp,cpwt->gcwt',fields.reshape(len(fields),-1),cov); energy=(num*num/(vx[:,:,None,None]*vy[None,None,:,:])).sum(1); sel=energy.argmax(0); chosen_all.append(sel)
  for wi in range(3):
   for u in range(N_UNITS):
    g=int(sel[wi,u]); z=pool(mapsfull,fields[g:g+1],tr)[:,0,:]; mu=z.mean(0); sd=z.std(0).clip(1e-5); zz=(z-mu)/sd; _,_,V=np.linalg.svd(zz,full_matrices=False); V=V[:32].T; Z=zz@V; beta,xm,ym=fit_ridge(Z,y[tr,wi,u]); axis=(V@beta)/sd; testz=pool(mapsfull,fields[g:g+1],te)[:,0,:]; pred=((testz-mu)/sd@V-xm)@beta+ym; r=np.corrcoef(pred,y[te,wi,u])[0,1] if len(te)>2 else np.nan; fold_axes.append({'fold':fi,'window':(70,110,150)[wi],'unit':int(chosen[u]),'axis':axis,'rf_id':g,'test_r':float(r)})
  print('fold',fi+1,'done',flush=True)
 axes=fold_axes; out=[]
 # Keep restored axes in the common original res4 channel space for valid comparisons.
 axis_arr=np.stack([x['axis'] for x in axes]).reshape(FOLDS,3,N_UNITS,-1)
 np.save(OUT/'axes_1024.npy', axis_arr.astype(np.float32))
 rf_arr=np.array([x['rf_id'] for x in axes],dtype=np.int32).reshape(FOLDS,3,N_UNITS)
 np.save(OUT/'rf_ids.npy', rf_arr)
 for wi,start in enumerate((70,110,150)):
  for u in range(N_UNITS):
   a=[x['axis'] for x in axes if x['window']==start and x['unit']==int(chosen[u])]; vals=[cosine(a[i][None],a[j][None])[0] for i in range(FOLDS) for j in range(i+1,FOLDS)]; rf=[x['rf_id'] for x in axes if x['window']==start and x['unit']==int(chosen[u])]; out.append({'kind':'unit','window_ms':start,'unit':int(chosen[u]),'axis_crossfold_median':float(np.median(vals)),'rf_unique_fraction':len(set(rf))/len(rf),'test_r_median':float(np.median([x['test_r'] for x in axes if x['window']==start and x['unit']==int(chosen[u])]))})
 d=pd.DataFrame(out)
 # Within-window fold jitter and within-fold cross-window change.
 rows=[]
 for u in range(N_UNITS):
  for wi,wa in enumerate((70,110,150)):
   for wj,wb in enumerate((70,110,150)):
    if wj<=wi: continue
    change=[]
    for fi in range(FOLDS): change.append(1-float(cosine(axis_arr[fi,wi,u][None],axis_arr[fi,wj,u][None])[0]))
    jit=[]
    for w in (wi,wj):
     q=[1-float(cosine(axis_arr[i,w,u][None],axis_arr[j,w,u][None])[0]) for i in range(FOLDS) for j in range(i+1,FOLDS)]
     jit.append(np.median(q))
    rows.append({'kind':'cross_window','unit':int(chosen[u]),'window_a_ms':wa,'window_b_ms':wb,'axis_distance_median':float(np.median(change)),'fold_jitter_median':float(np.mean(jit)),'change_to_jitter':float(np.median(change)/(np.mean(jit)+1e-8))})
 pd.DataFrame(rows).to_csv(OUT/'cross_window_vs_fold_jitter.csv',index=False)
 d.to_csv(OUT/'unit_stability_summary.csv',index=False); pd.DataFrame([{'fold':x['fold'],'window_ms':x['window'],'unit':x['unit'],'rf_id':x['rf_id'],'test_r':x['test_r']} for x in axes]).to_csv(OUT/'fold_detail.csv',index=False); print(d.groupby('window_ms').agg(axis_cosine=('axis_crossfold_median','median'),rf_unique_fraction=('rf_unique_fraction','median'),test_r=('test_r_median','median')).to_string()); print(pd.DataFrame(rows).groupby(['window_a_ms','window_b_ms']).agg(distance=('axis_distance_median','median'),jitter=('fold_jitter_median','median'),ratio=('change_to_jitter','median')).to_string()); print('saved',OUT)
if __name__=='__main__': main()
