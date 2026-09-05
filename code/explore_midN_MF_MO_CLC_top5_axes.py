from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.stats import pearsonr

P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; B=P/'proxy_bank_10ms_all_windows'
C=R/'circle5000_axis500_2026-08-29'; NAX,NC,NIMG=500,5000,1000; GRID=14
WINS=(70,110,150); WIDX={w:(w+20)//10 for w in WINS}

def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); x=x-x.mean(); y=y-y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/d) if d>1e-12 else np.nan
def main():
    axes=pd.read_csv(C/'axis_manifest_500.csv'); circ=pd.read_csv(C/'circle_manifest_5000.csv')
    masks=np.load(C/'circle_masks_5000_f32.npy',mmap_mode='r')
    score=np.memmap(C/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(NAX,NC,NIMG))
    resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r')
    meta=pd.read_csv(R/'all_unit_best_window_results.csv')
    rows=[]
    for area in ('MF','MO','CLC'):
        q=meta[meta.native_area.eq(area)].copy(); q['mean3']=q.groupby('unit_global')['repeat_mean_oof_r'].transform('mean')
        chosen=q.sort_values(['mean3','unit_global'],ascending=[False,True]).drop_duplicates('unit_global').head(3)
        # local index in middle-IT bank follows unit_global order stored in H5
        with h5py.File(B/'middle_it_proxy_bank.h5','r') as h:
            ids=h['unit_global'][:].astype(int); sp=h['spatial_parameters'][:]; oo=h['oof_r'][:]; windows=h['windows_ms'][:]
        lut={int(u):i for i,u in enumerate(ids)}
        for u in chosen.itertuples(index=False):
            local=lut[int(u.unit_global)]
            print(f'\n{area} unit {int(u.unit_global)} (quality mean={float(u.mean3):.3f})')
            for w in WINS:
                wi=WIDX[w]; p=sp[wi,local].astype(float); gx=(p[0]+10)/(20/GRID)-.5; gy=(-p[1]+10)/(20/GRID)-.5; gr=p[2]/(20/GRID)
                d=np.hypot(circ.center_x_grid-gx,circ.center_y_grid-gy)
                ok=(d+gr<=circ.radius_grid+1e-9)
                if ok.any():
                    rmin=circ.loc[ok,'radius_grid'].min(); cand=np.flatnonzero(ok & np.isclose(circ.radius_grid,rmin)); cid=int(cand[np.argmin(d[cand])])
                else: cid=int(np.argmin(d+np.abs(circ.radius_grid-gr)))
                y=np.asarray(resp[wi,:,int(u.unit_global)],float)
                rs=np.asarray([corr(score[a,cid],y) for a in range(NAX)])
                top=np.argsort(np.abs(rs))[::-1][:5]
                print(f'  {w} ms circle={cid}: '+ ' | '.join(f"{axes.iloc[a].concept} ({rs[a]:+.3f})" for a in top))
                for rank,a in enumerate(top,1): rows.append({'area':area,'unit_global':int(u.unit_global),'window_ms':w,'circle_id':cid,'rank':rank,'axis_id':int(a),'concept':str(axes.iloc[a].concept),'layer':str(axes.iloc[a].layer),'pearson_r':float(rs[a])})
    print('\nDone; exploratory output only, no files written.')
if __name__=='__main__': main()
