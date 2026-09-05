"""Exploratory split-image validation of global Triple-N PSTH clusters versus G/T coupling."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kruskal, f as f_dist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24')
R,B,CODE=P/'results',P/'proxy_bank_10ms_all_windows',P/'code'
OUT=R/'exploratory_independent_tripleN_cluster_GT_2026-08-30'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(CODE))
from plot_orthogonal_temporal_modes_2d import raw_directions
SEED=20260830
AREAS=['V1','V2','V4','MF','MO','CLC']

def f_test(full, reduced):
    """Nested least-squares F test; rows are session x area x cluster medians."""
    y=full['y']; Xf=full['X']; Xr=reduced['X']
    bf=np.linalg.lstsq(Xf,y,rcond=None)[0]; br=np.linalg.lstsq(Xr,y,rcond=None)[0]
    ssf=np.sum((y-Xf@bf)**2); ssr=np.sum((y-Xr@br)**2)
    dfn=Xf.shape[1]-Xr.shape[1]; dfd=len(y)-Xf.shape[1]
    F=max(0.,(ssr-ssf)/max(dfn,1)/max(ssf/max(dfd,1),1e-12))
    return F, float(f_dist.sf(F,dfn,dfd)), dfn,dfd

def main():
    meta=pd.read_csv(R/'all_unit_manifest.csv')
    resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r')
    w=np.load(B/'windows_10ms.npy').astype(float)
    # Fixed image split: discovery creates labels/templates; validation is never used for either.
    rng=np.random.default_rng(SEED); discovery=np.sort(rng.choice(1000,500,replace=False)); valid=np.setdiff1d(np.arange(1000),discovery)
    np.savez(OUT/'image_split.npz', discovery=discovery, validation=valid, seed=SEED)
    psth_idx=np.flatnonzero((w[:,0]>=0)&(w[:,1]<=189)); psth_t=w[psth_idx].mean(1)
    # All-unit global clustering from discovery-image average PSTHs only.
    x=np.asarray(resp[psth_idx][:,discovery,:].mean(axis=1),np.float32).T
    x=(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)
    km=KMeans(n_clusters=3,n_init=50,random_state=SEED,algorithm='lloyd'); lab=km.fit_predict(x); cen=km.cluster_centers_
    peak=cen.argmax(1); posarea=np.maximum(cen,0).sum(1); order=np.argsort(peak)
    names={order[0]:'fast-transient',order[1]:'delayed-peak',order[2]:'delayed-sustained'}
    if len(set(peak))<3:
        order=np.argsort(posarea); names={order[0]:'fast-transient',order[1]:'delayed-peak',order[2]:'delayed-sustained'}
    sample=rng.choice(len(x),5000,replace=False); sil=float(silhouette_score(x[sample],lab[sample]))
    labels=pd.DataFrame({'unit_global':meta.unit_global.to_numpy(int),'cluster_id':lab,'cluster_name':[names[i] for i in lab]})
    labels.to_csv(OUT/'unit_global_clusters_discovery_images.csv',index=False)
    # G/T directions use only discovery image templates in the reliable window.
    baseidx=np.flatnonzero(w[:,1]<0); gtid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); gt_t=w[gtid].mean(1)
    rows=[]
    for j,u in enumerate(meta.itertuples(index=False)):
        raw=np.asarray(resp[:,:,int(u.unit_global)],np.float32)
        base_disc=float(raw[baseidx][:,discovery].mean())
        # Mean evoked discovery PSTH is the frozen direction template.
        template=raw[gtid][:,discovery].mean(axis=1)-base_disc
        basis,center=raw_directions(template,gt_t)
        if basis is None: continue
        # Validation images only: each curve baseline-corrected by its own pre-stimulus response.
        curves=(raw[gtid][:,valid]-raw[baseidx][:,valid].mean(axis=0)[None]).T
        coeff=curves@basis; g,t=coeff[:,0],coeff[:,1]
        gt_r=float(np.corrcoef(g,t)[0,1]) if g.std()>1e-10 and t.std()>1e-10 else np.nan
        smooth=gaussian_filter1d(template,.75,mode='nearest'); positive=np.maximum(smooth,0)
        amplitude=float(template.mean()); peak_ms=float(gt_t[int(np.argmax(smooth))])
        # Positive-response temporal width, 0 when the template has no positive mass.
        width=float(np.sqrt(np.average((gt_t-np.average(gt_t,weights=positive))**2,weights=positive))) if positive.sum()>1e-10 else np.nan
        rows.append({'unit_global':int(u.unit_global),'cluster_id':int(lab[j]),'cluster_name':names[lab[j]],'native_area':u.native_area,'session':u.session,'monkey':u.monkey,'G_T_image_r_validation':gt_r,'G_T_direction_cos_discovery':float(basis[:,0]@basis[:,1]),'peak_latency_ms_discovery':peak_ms,'mean_evoked_discovery':amplitude,'temporal_width_ms_discovery':width})
    d=pd.DataFrame(rows).merge(meta[['unit_global','released_independent_consistency']],on='unit_global',how='left')
    d.to_csv(OUT/'unit_validation_GT_and_traditional_metrics.csv',index=False)
    # Unit-level raw and residualized tests. Residualization explicitly does not use cluster labels.
    cov=['peak_latency_ms_discovery','mean_evoked_discovery','temporal_width_ms_discovery']
    good=d.dropna(subset=['G_T_image_r_validation']+cov).copy(); Z=good[cov].to_numpy(float); Z=(Z-Z.mean(0))/np.maximum(Z.std(0),1e-12); Z=np.column_stack([np.ones(len(Z)),Z])
    good['GT_resid_traditional']=good.G_T_image_r_validation.to_numpy()-Z@np.linalg.lstsq(Z,good.G_T_image_r_validation.to_numpy(),rcond=None)[0]
    # Sensitivity only: also remove template-defined G/T cosine, the most direct geometric overlap.
    Z2=np.column_stack([Z,(good.G_T_direction_cos_discovery-good.G_T_direction_cos_discovery.mean())/good.G_T_direction_cos_discovery.std()])
    good['GT_resid_traditional_and_cosine']=good.G_T_image_r_validation.to_numpy()-Z2@np.linalg.lstsq(Z2,good.G_T_image_r_validation.to_numpy(),rcond=None)[0]
    good.to_csv(OUT/'unit_validation_GT_residualized.csv',index=False)
    results=[]
    for area,q in good[good.native_area.isin(AREAS)].groupby('native_area'):
        for metric in ['G_T_image_r_validation','GT_resid_traditional','GT_resid_traditional_and_cosine']:
            groups=[v[metric].to_numpy() for _,v in q.groupby('cluster_name')]
            h,p=kruskal(*groups); results.append({'level':'unit','area':area,'metric':metric,'H':h,'p':p,'n':len(q)})
    # Session x area x cluster medians: each session supplies one independent summary per category.
    sess=good[good.native_area.isin(AREAS)].groupby(['session','monkey','native_area','cluster_name'],as_index=False).agg(GT=('G_T_image_r_validation','median'),GT_resid=('GT_resid_traditional','median'),n_units=('unit_global','size'))
    sess.to_csv(OUT/'session_area_cluster_medians.csv',index=False)
    for area,q in sess.groupby('native_area'):
        groups=[v.GT.to_numpy() for _,v in q.groupby('cluster_name') if len(v)>=3]
        if len(groups)==3:
            h,p=kruskal(*groups); results.append({'level':'session_median','area':area,'metric':'G_T_image_r_validation','H':h,'p':p,'n':len(q)})
            groups=[v.GT_resid.to_numpy() for _,v in q.groupby('cluster_name') if len(v)>=3]; h,p=kruskal(*groups); results.append({'level':'session_median','area':area,'metric':'GT_resid_traditional','H':h,'p':p,'n':len(q)})
    test=pd.DataFrame(results); test.to_csv(OUT/'cluster_GT_tests_unit_and_session.csv',index=False)
    # Cluster x area interaction on session medians, using full vs additive least-squares model.
    q=sess.copy(); q['area']=pd.Categorical(q.native_area,categories=AREAS); q['cluster']=pd.Categorical(q.cluster_name,categories=['fast-transient','delayed-peak','delayed-sustained']); q=q.dropna()
    A=pd.get_dummies(q.area,drop_first=True,dtype=float).to_numpy(); K=pd.get_dummies(q.cluster,drop_first=True,dtype=float).to_numpy(); inter=np.einsum('ij,ik->ijk',A,K).reshape(len(q),-1)
    red={'y':q.GT.to_numpy(),'X':np.column_stack([np.ones(len(q)),A,K])}; full={'y':q.GT.to_numpy(),'X':np.column_stack([red['X'],inter])}; F,p,dfn,dfd=f_test(full,red)
    interaction={'metric':'session-median G/T','F':F,'p':p,'df_num':dfn,'df_den':dfd,'n_rows':len(q)}
    (OUT/'session_cluster_by_area_interaction.json').write_text(json.dumps(interaction,indent=2),encoding='utf-8')
    # Compact figures: centers and session-median distributions for MF/MO/CLC.
    fig,axs=plt.subplots(1,2,figsize=(13,4.5)); colors={'fast-transient':'#2878b5','delayed-peak':'#e58e26','delayed-sustained':'#59a14f'}
    for i in range(3): axs[0].plot(psth_t,cen[i],lw=2.5,color=colors[names[i]],label=names[i])
    axs[0].axvspan(0,150,color='0.93',zorder=-1); axs[0].set(xlabel='Time (ms)',ylabel='Discovery PSTH (z)',title=f'Global k=3 clusters, discovery half\nSilhouette={sil:.3f}'); axs[0].legend(frameon=False)
    shown=['MF','MO','CLC']; positions=[]; vals=[]; cols=[]; labs=[]; k=1
    for area in shown:
        for cl in ['fast-transient','delayed-peak','delayed-sustained']:
            v=sess[(sess.native_area==area)&(sess.cluster_name==cl)].GT.dropna().to_numpy()
            if len(v): positions.append(k); vals.append(v); cols.append(colors[cl]); labs.append(f'{area}\n{cl.split("-")[0]}'); k+=1
        k+=.6
    bp=axs[1].boxplot(vals,positions=positions,widths=.65,patch_artist=True,showfliers=False)
    for box,c in zip(bp['boxes'],cols): box.set_facecolor(c); box.set_alpha(.75)
    axs[1].axhline(0,color='0.4',lw=.8); axs[1].set_xticks(positions,labs,rotation=35,ha='right'); axs[1].set_ylabel('Median validation-image corr(G,T)'); axs[1].set_title('Independent validation, session medians')
    fig.tight_layout(); fig.savefig(OUT/'independent_cluster_GT_controls.png',dpi=200); plt.close(fig)
    print('silhouette',sil); print(test.to_string(index=False)); print('interaction',interaction)

if __name__=='__main__': main()
