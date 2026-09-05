"""Exploratory cross-area semantic-to-dynamics map; deliberately outside the frozen framework."""
from pathlib import Path
import numpy as np, pandas as pd
from plot_orthogonal_temporal_modes_2d import raw_directions

P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; OLD=P/'proxy_bank_10ms_all_windows'; C=R/'circle5000_axis500_2026-08-29'; OUT=R/'exploratory_cross_area_semantic_G_T_2026-08-29'; OUT.mkdir(parents=True,exist_ok=True)
AREAS=['V1','V2','V4','MF','MO','CLC']; NIMG,NAX,NC=1000,500,5000

def batch_corr(x,y):
 x=x-x.mean(1,keepdims=True); y=y-y.mean(1,keepdims=True); den=np.sqrt((x*x).sum(1,keepdims=True)*(y*y).sum(1,keepdims=True).T); return (x@y.T)/np.maximum(den,1e-12)
def choose_circle(p,c):
 gx=(p[0]+10)/(20/14)-.5; gy=(-p[1]+10)/(20/14)-.5; gr=p[2]/(20/14); d=np.hypot(c.center_x_grid-gx,c.center_y_grid-gy); ok=d+gr<=c.radius_grid+1e-9
 if not ok.any(): return int(np.argmin(d+np.abs(c.radius_grid-gr)))
 m=ok & np.isclose(c.radius_grid,c.loc[ok,'radius_grid'].min()); return int(c.loc[m].iloc[np.argmin(d[m])].circle_id)
def main():
 meta=pd.read_csv(R/'all_unit_manifest.csv'); meta['target_local']=meta.groupby('roi',sort=False).cumcount(); meta=meta[meta.native_area.isin(AREAS)].copy(); meta['analysis_area']=meta.native_area
 fields=pd.read_csv(R/'all_fold_spatial_and_depth_parameters.csv'); circles=pd.read_csv(C/'circle_manifest_5000.csv'); mm=np.memmap(C/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(NAX,NC,NIMG)); axes=pd.read_csv(C/'axis_manifest_500.csv'); resp=np.load(OLD/'trin_all_units_10ms_responses.npy',mmap_mode='r'); wins=np.load(OLD/'windows_10ms.npy').astype(int); base=np.flatnonzero(wins[:,1]<0); tid=np.flatnonzero((wins[:,0]>=70)&(wins[:,1]<=189)); times=wins[tid].mean(1).astype(float)
 circle_by_unit=[]; params=[]
 for u in meta.itertuples(index=False):
  q=fields[(fields.roi==u.roi)&(fields.target_local==u.target_local)]; p=q[['center_x','center_y','sigma_space']].mean().to_numpy(float); params.append(p); circle_by_unit.append(choose_circle(p,circles))
 meta['circle_id']=circle_by_unit; np.save(OUT/'unit_representative_circle_ids.npy',np.asarray(circle_by_unit,np.int32)); pd.DataFrame(params,columns=['center_x','center_y','sigma_space']).to_csv(OUT/'unit_representative_rf_parameters.csv',index=False)
 all_rows=[]; area_medians=[]
 for area in AREAS:
  idx=np.flatnonzero(meta.analysis_area.to_numpy()==area); units=meta.iloc[idx]; yG=[]; yT=[]
  for g in units.unit_global.to_numpy(int):
   raw=np.asarray(resp[:,:,g],np.float32); template=raw[tid].mean(1); physical,_=raw_directions(template,times); basecurve=raw[base].mean(0); rel=(raw[tid]-basecurve[None]).T; coef=rel@physical; yG.append(coef[:,0]); yT.append(coef[:,1])
  YG=np.asarray(yG); YT=np.asarray(yT); circles_used=np.asarray(circle_by_unit)[idx]; corrG=np.empty((NAX,len(idx)),np.float32); corrT=np.empty_like(corrG)
  for cid in np.unique(circles_used):
   local=np.flatnonzero(circles_used==cid); S=np.asarray(mm[:,int(cid),:],np.float32); corrG[:,local]=batch_corr(S,YG[local]); corrT[:,local]=batch_corr(S,YT[local])
  area_medians.append(pd.DataFrame({'area':area,'axis_id':np.arange(NAX),'concept':axes.concept.astype(str),'layer':axes.layer.astype(str),'median_r_G':np.nanmedian(corrG,1),'median_r_T':np.nanmedian(corrT,1),'mean_r_G':np.nanmean(corrG,1),'mean_r_T':np.nanmean(corrT,1),'median_abs_r_G':np.nanmedian(np.abs(corrG),1),'median_abs_r_T':np.nanmedian(np.abs(corrT),1)}))
  for j,row in enumerate(units.itertuples(index=False)):
   for name,M in [('G',corrG),('T',corrT)]:
    order=np.argsort(np.abs(M[:,j]))[::-1][:10]
    for rank,ai in enumerate(order,1): all_rows.append({'area':area,'unit_global':int(row.unit_global),'proxy_oof_r':float(row.repeat_mean_oof_r),'window_start_ms':int(row.window_start_ms),'window_end_ms':int(row.window_end_ms),'circle_id':int(circles_used[j]),'direction':name,'axis_id':int(ai),'concept':str(axes.iloc[ai].concept),'layer':str(axes.iloc[ai].layer),'pearson_r':float(M[ai,j]),'abs_rank':rank})
  print(area,len(idx),flush=True)
 top=pd.DataFrame(all_rows); med=pd.concat(area_medians,ignore_index=True); top.to_csv(OUT/'top10_axes_per_unit_G_T.csv',index=False); med.to_csv(OUT/'area_axis_summary_G_T.csv',index=False); meta.to_csv(OUT/'exploratory_unit_manifest.csv',index=False)
 # compact area-level strongest axes by median absolute association
 best=med.sort_values(['area','median_abs_r_G'],ascending=[True,False]).groupby('area',sort=False).head(20); best.to_csv(OUT/'area_top20_axes_by_median_abs_r_G.csv',index=False)
 print(best[['area','concept','layer','median_r_G','median_r_T','median_abs_r_G','median_abs_r_T']].to_string(index=False))
if __name__=='__main__': main()
