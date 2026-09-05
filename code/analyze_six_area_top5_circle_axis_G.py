from pathlib import Path
import numpy as np,pandas as pd,h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plot_orthogonal_temporal_modes_2d import raw_directions,symmetric_orthogonalize
from train_64d_high_quality_proxy_batch import gaussian_fields

P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; C=R/'circle5000_axis500_2026-08-29'; OLD=P/'proxy_bank_10ms_all_windows'; OUT=R/'six_area_top5_circle_axis_G_2026-08-29'; OUT.mkdir(parents=True,exist_ok=True)
AREAS=['V1','V2','V4','MF','MO','CLC']; NIMG=1000; NAX=500; NC=5000; GRID=14

def corr(x,y): return float(pearsonr(np.asarray(x,float),np.asarray(y,float)).statistic)
def main():
 meta=pd.read_csv(R/'all_unit_best_window_results.csv'); resp=np.load(OLD/'trin_all_units_10ms_responses.npy',mmap_mode='r'); wins=np.load(OLD/'windows_10ms.npy').astype(int); base=np.flatnonzero(wins[:,1]<0); tid=np.flatnonzero((wins[:,0]>=70)&(wins[:,1]<=189)); times=wins[tid].mean(1).astype(float)
 circ=pd.read_csv(C/'circle_manifest_5000.csv'); masks=np.load(C/'circle_masks_5000_f32.npy'); axes=pd.read_csv(C/'axis_manifest_500.csv'); mm=np.memmap(C/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(NAX,NC,NIMG))
 fields=pd.read_csv(R/'all_fold_spatial_and_depth_parameters.csv')
 # establish local index within each roi as used by final model
 meta['target_local']=meta.groupby('roi',sort=False).cumcount()
 chosen=[]
 for area in AREAS:
  q=meta[meta.native_area.eq(area)].sort_values('repeat_mean_oof_r',ascending=False).head(5).copy(); q['analysis_area']=area; chosen.append(q)
 units=pd.concat(chosen,ignore_index=True)
 rows=[]; allr=[]
 for u in units.itertuples(index=False):
  area=str(u.analysis_area); roi=str(u.roi); local=int(u.target_local); fs=fields[(fields.roi==roi)&(fields.target_local==local)].sort_values('fold'); p=fs[['center_x','center_y','sigma_space']].mean().to_numpy(float)
  gx=(p[0]+10)/(20/GRID)-.5; gy=(-p[1]+10)/(20/GRID)-.5; gr=p[2]/(20/GRID); d=np.hypot(circ.center_x_grid-gx,circ.center_y_grid-gy); ok=d+gr<=circ.radius_grid+1e-9
  cid=int(circ.loc[ok & np.isclose(circ.radius_grid,circ.loc[ok,'radius_grid'].min())].iloc[np.argmin(d[ok & np.isclose(circ.radius_grid,circ.loc[ok,'radius_grid'].min())])].circle_id) if ok.any() else int(np.argmin(d+np.abs(circ.radius_grid-gr)))
  raw=np.asarray(resp[:, :, int(u.unit_global)],np.float32); template=raw[tid].mean(1); physical,_=raw_directions(template,times); basis,_,_=symmetric_orthogonalize(physical); G=(raw[tid]-raw[base].mean(0)[None]).T @ basis[:,0]
  ai_scores=np.empty(NAX,float)
  for ai in range(NAX): ai_scores[ai]=corr(mm[ai,cid],G)
  order=np.argsort(np.abs(ai_scores))[::-1]; top=order[:10]
  for rank,ai in enumerate(top,1): rows.append({'area':area,'roi':roi,'unit_global':int(u.unit_global),'window_start_ms':int(u.window_start_ms),'window_end_ms':int(u.window_end_ms),'proxy_oof_r':float(u.repeat_mean_oof_r),'circle_id':cid,'axis_id':int(ai),'concept':str(axes.iloc[ai].concept),'axis_layer':str(axes.iloc[ai].layer),'pearson_r_G':float(ai_scores[ai]),'abs_r_rank':rank})
  allr.append((area,int(u.unit_global),ai_scores, cid))
 out=pd.DataFrame(rows); out.to_csv(OUT/'top10_axes_per_unit.csv',index=False); top1=out[out.abs_r_rank==1].copy(); top1.to_csv(OUT/'top_axis_per_unit.csv',index=False)
 # heatmap of top signed r
 fig,ax=plt.subplots(figsize=(12,5.5),constrained_layout=True); pivot=top1.pivot(index='area',columns='unit_global',values='pearson_r_G'); im=ax.imshow(pivot.to_numpy(float),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto'); ax.set_xticks(range(len(pivot.columns)),[str(x) for x in pivot.columns],rotation=45,ha='right'); ax.set_yticks(range(len(pivot.index)),pivot.index); ax.set_xlabel('Selected unit (top 5 per native area)'); ax.set_ylabel('Native area'); ax.set_title('Top concept-axis correlation with observed G (geometry-selected circle)'); fig.colorbar(im,ax=ax,label='Pearson r'); fig.savefig(OUT/'01_top_axis_r_heatmap.png',dpi=300,bbox_inches='tight'); plt.close(fig)
 print(top1[['area','unit_global','proxy_oof_r','circle_id','concept','axis_layer','pearson_r_G']].to_string(index=False))
if __name__=='__main__': main()
