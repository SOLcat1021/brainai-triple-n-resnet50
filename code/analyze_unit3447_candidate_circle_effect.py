from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from train_64d_high_quality_proxy_batch import gaussian_fields

PROJECT=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); PILOT=PROJECT/'results'/'mf_head_wall_rf_gtd_pilot_2026-08-28'; CIRC=PROJECT/'results'/'circle5000_axis500_2026-08-29'; BANK=PROJECT/'proxy_bank_64d_high_quality_pilot_113_2026-08-26'; UNIT=3447; GRID=14

def corr(x,y): return float(pearsonr(np.asarray(x,float),np.asarray(y,float)).statistic)
def main():
 old=np.load(PILOT/'head_wall_rf_scores_and_GTD.npz',allow_pickle=True); ui=old['units'].astype(int).tolist().index(UNIT); y=old['observed_gtd'][ui,:,0].astype(float)
 maps=np.load(PILOT/'head_wall_native_cav_maps_f32.npy').astype(float); circles=pd.read_csv(CIRC/'circle_manifest_5000.csv'); axes=pd.read_csv(CIRC/'axis_manifest_500.csv'); mm=np.memmap(CIRC/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(500,5000,1000))
 selected=pd.read_csv(BANK/'selected_units.csv'); mid=selected[selected.roi.eq('middle IT')].reset_index(drop=True); local=int(mid.index[mid.unit_global.eq(UNIT)][0]); row=mid.iloc[local]
 with h5py.File(BANK/'middle_it_proxy_bank_64d.h5','r') as h:
  w=h['windows_ms'][:]; wi=int(np.flatnonzero((w[:,0]==row.window_start_ms)&(w[:,1]==row.window_end_ms))[0]); p=np.asarray(h['spatial_parameters'][wi,local],float)
 dpix=20/GRID; gx=(p[0]+10)/dpix-.5; gy=(-p[1]+10)/dpix-.5; gr=p[2]/dpix
 d=np.hypot(circles.center_x_grid-gx,circles.center_y_grid-gy); contains=d+gr<=circles.radius_grid+1e-9
 if not contains.any(): raise RuntimeError('No candidate circle contains continuous RF')
 minr=circles.loc[contains,'radius_grid'].min(); eligible=circles[contains & np.isclose(circles.radius_grid,minr)].copy(); cid=int(eligible.iloc[np.argmin(d[eligible.index])].circle_id)
 field=gaussian_fields(p[None].astype(np.float32))[0].reshape(GRID,GRID); field/=field.sum()
 rows=[]; series={}
 for ci,c in enumerate(('head','wall')):
  ai=int(axes.index[axes.concept.eq(c)][0]); full=maps[ci].mean((1,2)); continuous=np.einsum('nhw,hw->n',maps[ci],field); discrete=np.asarray(mm[ai,cid],np.float32)
  series[c]=(full,continuous,discrete)
  for method,x in zip(('full_image','continuous_RF','geometry_selected_circle'),series[c]): rows.append({'concept':c,'method':method,'pearson_r':corr(x,y),'circle_id':cid if method.endswith('circle') else np.nan})
 summary=pd.DataFrame(rows); summary.to_csv(PILOT/'unit3447_geometry_selected_circle_summary.csv',index=False)
 pd.DataFrame([{'unit_global':UNIT,'window_start_ms':int(row.window_start_ms),'window_end_ms':int(row.window_end_ms),'continuous_center_x_grid':gx,'continuous_center_y_grid':gy,'continuous_radius_grid_sigma':gr,'selected_circle_id':cid,**circles.iloc[cid].to_dict(),'center_distance_grid':float(d[cid]),'containment_margin_grid':float(circles.iloc[cid].radius_grid-d[cid]-gr)}]).to_csv(PILOT/'unit3447_geometry_selected_circle.csv',index=False)
 fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 for ci,c in enumerate(('head','wall')):
  for mi,(x,lab) in enumerate(zip(series[c],('Full image','Continuous fwRF','Geometry-selected circle'))):
   z=(x-x.mean())/x.std(ddof=1); slope,inter=np.polyfit(z,y,1); g=np.linspace(z.min(),z.max(),200); rr=corr(x,y)
   ax[ci,mi].scatter(z,y,s=8,alpha=.22,color=('#2878A9' if c=='head' else '#C43B2B'),edgecolors='none',rasterized=True); ax[ci,mi].plot(g,inter+slope*g,color='#222',lw=1.7); ax[ci,mi].set_title(f'{c} | {lab}\nr={rr:+.3f}',fontweight='bold'); ax[ci,mi].set(xlabel=f'{c} score (SD)',ylabel='Observed relative G'); ax[ci,mi].grid(alpha=.22); ax[ci,mi].set_axisbelow(True)
 fig.suptitle(f'MF unit 3447: circle {cid} selected only by RF containment geometry',fontweight='bold'); fig.savefig(PILOT/'09_unit3447_geometry_selected_circle_vs_observed_G.png',dpi=300,bbox_inches='tight'); plt.close(fig)
 print('RF grid',gx,gy,gr,'circle',cid,circles.iloc[cid].to_dict(),'distance',d[cid]); print(summary.to_string(index=False))
if __name__=='__main__': main()
