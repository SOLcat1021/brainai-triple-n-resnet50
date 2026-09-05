from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plot_orthogonal_temporal_modes_2d import raw_directions
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; O=R/'exploratory_cross_area_semantic_G_T_2026-08-29'; OLD=P/'proxy_bank_10ms_all_windows'; AREAS=['V1','V2','V4','MF','MO','CLC']
def main():
 med=pd.read_csv(O/'area_axis_summary_G_T.csv'); meta=pd.read_csv(O/'exploratory_unit_manifest.csv'); resp=np.load(OLD/'trin_all_units_10ms_responses.npy',mmap_mode='r'); w=np.load(OLD/'windows_10ms.npy').astype(int); base=np.flatnonzero(w[:,1]<0); tid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); times=w[tid].mean(1).astype(float)
 rows=[]
 for u in meta.itertuples(index=False):
  raw=np.asarray(resp[:,:,int(u.unit_global)],np.float32); physical,_=raw_directions(raw[tid].mean(1),times); coef=(raw[tid]-raw[base].mean(0)[None]).T@physical; rows.append({'area':u.analysis_area,'unit_global':int(u.unit_global),'G_T_image_r':float(pearsonr(coef[:,0],coef[:,1]).statistic),'raw_direction_cos_G_T':float(physical[:,0]@physical[:,1])})
 unit=pd.DataFrame(rows); unit.to_csv(O/'unit_G_T_coupling.csv',index=False)
 area=[]
 for a in AREAS:
  q=med[med.area.eq(a)]; u=unit[unit.area.eq(a)]; area.append({'area':a,'n_units':len(u),'median_unit_G_T_image_r':u.G_T_image_r.median(),'median_abs_unit_G_T_image_r':u.G_T_image_r.abs().median(),'axis_profile_r_G_vs_T':pearsonr(q.median_r_G,q.median_r_T).statistic,'median_abs_axis_r_G':q.median_abs_r_G.median(),'median_abs_axis_r_T':q.median_abs_r_T.median()})
 area=pd.DataFrame(area); area.to_csv(O/'area_G_T_coupling_summary.csv',index=False)
 # top five area-level axes separately for G and T
 tops=[]
 for a in AREAS:
  q=med[med.area.eq(a)]
  for d in ('G','T'):
   z=q.nlargest(5,f'median_abs_r_{d}')
   for rank,x in enumerate(z.itertuples(),1): tops.append({'area':a,'direction':d,'rank':rank,'concept':x.concept,'layer':x.layer,'median_r':getattr(x,f'median_r_{d}'),'median_abs_r':getattr(x,f'median_abs_r_{d}')})
 top=pd.DataFrame(tops); top.to_csv(O/'area_top5_axes_G_and_T.csv',index=False)
 fig,ax=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
 x=np.arange(len(area)); ax[0].bar(x-.18,area.median_abs_axis_r_G,.36,label='G'); ax[0].bar(x+.18,area.median_abs_axis_r_T,.36,label='T'); ax[0].set_xticks(x,area.area); ax[0].set(ylabel='Median |r| across 500 axes',title='Population semantic association strength'); ax[0].legend(); ax[0].grid(axis='y',alpha=.2)
 ax[1].bar(x-.18,area.median_unit_G_T_image_r,.36,label='Unit G-T image correlation'); ax[1].bar(x+.18,area.axis_profile_r_G_vs_T,.36,label='Axis-profile G-T correlation'); ax[1].axhline(0,color='#444',lw=.8); ax[1].set_xticks(x,area.area); ax[1].set(ylabel='Pearson r',title='How separable are raw G and T?'); ax[1].legend(fontsize=8); ax[1].grid(axis='y',alpha=.2)
 fig.savefig(O/'01_cross_area_G_T_exploration.png',dpi=300,bbox_inches='tight'); plt.close(fig)
 print(area.to_string(index=False)); print('\nTOPS\n',top.to_string(index=False))
if __name__=='__main__': main()
