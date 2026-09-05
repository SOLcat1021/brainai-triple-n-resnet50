from pathlib import Path
import numpy as np,pandas as pd,h5py
from scipy.stats import pearsonr
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; B=P/'proxy_bank_10ms_all_windows'; C=R/'circle5000_axis500_2026-08-29'
W=(70,110,150); WI={w:(w+20)//10 for w in W}; NAX,NC,NIMG=500,5000,1000; GRID=14
def corr(a,b):
 a=np.asarray(a,float); b=np.asarray(b,float); a-=a.mean(); b-=b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum()); return (a*b).sum()/d if d>1e-12 else np.nan
def main():
 axes=pd.read_csv(C/'axis_manifest_500.csv'); circ=pd.read_csv(C/'circle_manifest_5000.csv'); score=np.memmap(C/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(NAX,NC,NIMG)); resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r'); meta=pd.read_csv(R/'all_unit_best_window_results.csv')
 with h5py.File(B/'middle_it_proxy_bank.h5','r') as h: ids=h['unit_global'][:].astype(int); sp=h['spatial_parameters'][:]; oo=h['oof_r'][:]
 lut={int(u):i for i,u in enumerate(ids)}; rows=[]
 for area in ('MF','MO','CLC'):
  q=meta[meta.native_area.eq(area)].copy(); q=q[q.unit_global.isin(ids)].copy(); q['min_r']=[float(np.min([oo[WI[w],lut[int(u)]] for w in W])) for u in q.unit_global]; q=q[q.min_r>=.4].sort_values(['min_r','unit_global'],ascending=[False,True]).head(3)
  print(f'\n{area}: {len(q)} units pass all-window r>=0.4; showing top by minimum r')
  for u in q.itertuples(index=False):
   local=lut[int(u.unit_global)]; profiles={}
   for w in W:
    wi=WI[w]; p=sp[wi,local].astype(float); gx=(p[0]+10)/(20/GRID)-.5; gy=(p[1]*0+(-p[1]+10))/(20/GRID)-.5; gr=p[2]/(20/GRID); d=np.hypot(circ.center_x_grid-gx,circ.center_y_grid-gy); ok=d+gr<=circ.radius_grid+1e-9; cid=int(np.flatnonzero(ok & np.isclose(circ.radius_grid,circ.loc[ok,'radius_grid'].min()))[np.argmin(d[ok & np.isclose(circ.radius_grid,circ.loc[ok,'radius_grid'].min())])] if ok.any() else np.argmin(d+np.abs(circ.radius_grid-gr)))
    rs=np.array([corr(score[a,cid],resp[wi,:,int(u.unit_global)]) for a in range(NAX)]); top=np.argsort(np.abs(rs))[::-1][:5]; profiles[w]=(set(top),top,rs)
   print(f' unit {int(u.unit_global)} min_r={float(u.min_r):.3f} oof={[round(float(oo[WI[w],local]),3) for w in W]}')
   for w in W:
    _,top,rs=profiles[w]; print('  ',w,', '.join(f"{axes.iloc[a].concept}({rs[a]:+.3f})" for a in top))
   for a,b in ((70,110),(110,150),(70,150)):
    overlap=len(profiles[a][0]&profiles[b][0]); print(f'   overlap {a}-{b}: {overlap}/5')
 print('\n探索结束：只统计三个窗口均可靠的 unit；未写入结果文件。')
if __name__=='__main__': main()
