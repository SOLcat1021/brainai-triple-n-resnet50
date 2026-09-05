"""Balanced-session sensitivity analysis for exploratory Triple-N cluster/G-T result."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import kruskal, f as f_dist

OUT=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24\results\exploratory_independent_tripleN_cluster_GT_2026-08-30')
AREAS=['V1','V2','V4','MF','MO','CLC']; CLS=['fast-transient','delayed-peak','delayed-sustained']

def nested_f(y, xr, xf):
    br=np.linalg.lstsq(xr,y,rcond=None)[0]; bf=np.linalg.lstsq(xf,y,rcond=None)[0]
    ssr=np.sum((y-xr@br)**2); ssf=np.sum((y-xf@bf)**2); dfn=xf.shape[1]-xr.shape[1]; dfd=len(y)-xf.shape[1]
    F=max(0,(ssr-ssf)/dfn/(ssf/max(dfd,1))); return F,float(f_dist.sf(F,dfn,dfd)),dfn,dfd

def main():
    s=pd.read_csv(OUT/'session_area_cluster_medians.csv')
    # Keep only session-area cells that contain all three global clusters with >=5 units each.
    wide=s.pivot_table(index=['session','monkey','native_area'],columns='cluster_name',values='n_units',aggfunc='first').reindex(columns=CLS)
    keep=wide.index[(wide>=5).all(axis=1)]
    q=s.set_index(['session','monkey','native_area']).loc[keep].reset_index()
    q.to_csv(OUT/'session_area_cluster_medians_balanced_nge5.csv',index=False)
    rows=[]
    for area,g in q.groupby('native_area'):
        for metric in ['GT','GT_resid']:
            groups=[g[g.cluster_name==c][metric].to_numpy() for c in CLS]
            h,p=kruskal(*groups); rows.append({'area':area,'metric':metric,'H':h,'p':p,'n_session_area_rows':len(g),'n_sessions':g.session.nunique()})
    result=pd.DataFrame(rows); result.to_csv(OUT/'session_balanced_nge5_cluster_tests.csv',index=False)
    # Interaction with fixed area/cluster indicators on balanced session medians.
    a=pd.get_dummies(pd.Categorical(q.native_area,categories=AREAS),drop_first=True,dtype=float).to_numpy()
    c=pd.get_dummies(pd.Categorical(q.cluster_name,categories=CLS),drop_first=True,dtype=float).to_numpy()
    inter=np.einsum('ij,ik->ijk',a,c).reshape(len(q),-1); xr=np.column_stack([np.ones(len(q)),a,c]); xf=np.column_stack([xr,inter])
    F,p,dfn,dfd=nested_f(q.GT.to_numpy(),xr,xf)
    audit={'minimum_units_per_session_area_cluster':5,'session_area_cells':int(len(keep)),'n_rows':int(len(q)),'interaction_F':F,'interaction_p':p,'df_num':dfn,'df_den':dfd}
    (OUT/'session_balanced_nge5_interaction.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(result.to_string(index=False)); print(audit)
if __name__=='__main__': main()
