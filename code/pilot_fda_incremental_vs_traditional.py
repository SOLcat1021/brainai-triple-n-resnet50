"""Test whether FDA coordinates add held-out curve reconstruction beyond scalar PSTH summaries."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); B=P/'proxy_bank_10ms_all_windows'; R=P/'results'; C=P/'code'; FROZEN=R/'orthogonal_temporal_modes_2d_high_consistency_2026-08-27'; OUT=R/'pilot_fda_incremental_vs_traditional_2026-08-30'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(C)); from run_trin_closed_loop_validation import fixed_folds
from plot_orthogonal_temporal_modes_2d import raw_directions
ROIS=['V1','V2','V4','posterior IT','middle IT','anterior IT']
def corr(a,b):
 a=np.asarray(a); b=np.asarray(b); return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-9 and np.std(b)>1e-9 else np.nan
def scalars(y,t):
 sm=gaussian_filter1d(y,.75,mode='nearest'); pos=np.maximum(sm,0); mass=pos.sum()
 peak=int(np.argmax(sm)); center=float(np.average(t,weights=pos)) if mass>1e-9 else float(t.mean())
 width=float(np.sqrt(np.average((t-center)**2,weights=pos))) if mass>1e-9 else np.nan
 return np.array([float(sm[peak]),float(t[peak]),float(mass),width])
def main():
 frozen=pd.read_csv(FROZEN/'unit_orthogonal_GTD_weights_and_modes.csv').sort_values('unit_global'); units=frozen.unit_global.to_numpy(int); resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r'); w=np.load(B/'windows_10ms.npy').astype(float); base=np.flatnonzero(w[:,1]<0); tid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); t=w[tid].mean(1); raw=np.asarray(resp[tid][:,:,units],np.float32); curves=(raw-np.asarray(resp[base][:,:,units],np.float32).mean(0)[None]).transpose(2,1,0)
 rows=[]; residual_rows=[]
 for ui,u in enumerate(frozen.itertuples(index=False)):
  fold_r0=[]; fold_r1=[]; fold_r2=[]; fold_delta=[]; fold_f_resid=[]
  y=curves[ui]
  for train,test in fixed_folds(1000):
   template=y[train].mean(0); physical,_=raw_directions(template,t)
   if physical is None: continue
   F=(y-template)@physical
   S=np.vstack([scalars(y[i],t) for i in range(1000)])
   good=np.isfinite(S).all(1)&np.isfinite(F).all(1); train=train[good[train]]; test=test[good[test]]
   if len(train)<20 or len(test)<10: continue
   # Standardize using training images only.
   sm=S[train].mean(0); ss=np.maximum(S[train].std(0),1e-9); fm=F[train].mean(0); fs=np.maximum(F[train].std(0),1e-9)
   Sz=(S-sm)/ss; Fz=(F-fm)/fs
   X0=np.column_stack([np.ones(len(train)),Sz[train]]); X1=np.column_stack([X0,Fz[train]]); Xt0=np.column_stack([np.ones(len(test)),Sz[test]]); Xt1=np.column_stack([Xt0,Fz[test]])
   b0=np.linalg.lstsq(X0,y[train],rcond=None)[0]; b1=np.linalg.lstsq(X1,y[train],rcond=None)[0]
   p0=Xt0@b0; p1=Xt1@b1
   yy=y[test]
   r0=corr(yy.ravel(),p0.ravel()); r1=corr(yy.ravel(),p1.ravel()); r2=r2_score(yy.ravel(),p0.ravel()); r3=r2_score(yy.ravel(),p1.ravel())
   # Unique FDA residual: F after training-only linear removal of traditional scalar space.
   bf=np.linalg.lstsq(X0,F[train],rcond=None)[0]; Fres=F-X0@bf if False else Fz-np.column_stack([np.ones(len(Fz)),Sz])@np.linalg.lstsq(np.column_stack([np.ones(len(train)),Sz[train]]),Fz[train],rcond=None)[0]
   fold_r0.append(r0); fold_r1.append(r1); fold_r2.append(r2); fold_delta.append(r3-r2); fold_f_resid.append(np.nanmean(np.var(Fres[test],axis=0)))
  rows.append({'unit_global':int(u.unit_global),'roi':str(u.roi),'r_curve_traditional':np.nanmean(fold_r0),'r_curve_fda_augmented':np.nanmean(fold_r1),'r2_traditional':np.nanmean(fold_r2),'r2_fda_augmented':np.nanmean([x for x in fold_r2]) + np.nanmean(fold_delta),'delta_r2_fda':np.nanmean(fold_delta),'fda_residual_variance':np.nanmean(fold_f_resid)})
  if ui%100==0: print(ui,flush=True)
 d=pd.DataFrame(rows); d.to_csv(OUT/'unit_incremental_fda_vs_traditional.csv',index=False); summ=d.groupby('roi').agg(n_units=('unit_global','size'),median_r_traditional=('r_curve_traditional','median'),median_r_fda_augmented=('r_curve_fda_augmented','median'),median_delta_r2=('delta_r2_fda','median'),positive_delta_fraction=('delta_r2_fda',lambda x:float((x>0).mean())),median_fda_residual_variance=('fda_residual_variance','median')).reset_index(); summ.to_csv(OUT/'roi_incremental_fda_summary.csv',index=False); overall=pd.DataFrame([{'n_units':len(d),'median_r_traditional':d.r_curve_traditional.median(),'median_r_fda_augmented':d.r_curve_fda_augmented.median(),'median_delta_r2':d.delta_r2_fda.median(),'positive_delta_fraction':float((d.delta_r2_fda>0).mean()),'median_fda_residual_variance':d.fda_residual_variance.median()}]); overall.to_csv(OUT/'overall_incremental_fda_summary.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(d.r2_traditional,d.r2_fda_augmented,s=16,alpha=.45,c='#4585a5',edgecolors='none'); lo=min(d.r2_traditional.min(),d.r2_fda_augmented.min()); hi=max(d.r2_traditional.max(),d.r2_fda_augmented.max()); ax.plot([lo,hi],[lo,hi],'--',color='#777'); ax.set(xlabel='Held-out curve R²: traditional scalars',ylabel='Held-out curve R²: + FDA coordinates',title='Does FDA add curve information?'); fig.tight_layout(); fig.savefig(OUT/'traditional_vs_fda_curve_r2.png',dpi=220); plt.close(fig); print(summ.to_string(index=False)); print(overall.to_string(index=False))
if __name__=='__main__': main()
