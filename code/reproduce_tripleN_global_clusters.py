"""Reproduce Triple-N-style global PSTH k-means clusters on available 0-189 ms data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import kruskal, f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

P = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
R, B, CODE = P/'results', P/'proxy_bank_10ms_all_windows', P/'code'
OUT = R/'tripleN_global_kmeans_0_189ms_2026-08-29'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))
from plot_orthogonal_temporal_modes_2d import raw_directions

def main():
    meta = pd.read_csv(R/'all_unit_manifest.csv')
    resp = np.load(B/'trin_all_units_10ms_responses.npy', mmap_mode='r')
    windows = np.load(B/'windows_10ms.npy').astype(float)
    # Triple-N uses post-onset PSTH; use all available post-onset bins here.
    tid = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    times = windows[tid].mean(1)
    # Average over images to obtain one PSTH per unit, then z-score per unit.
    psth = np.asarray(resp[tid].mean(axis=1), dtype=np.float32).T
    psth -= psth.mean(axis=1, keepdims=True)
    psth /= np.maximum(psth.std(axis=1, keepdims=True), 1e-6)
    km = KMeans(n_clusters=3, n_init=50, random_state=20260829, algorithm='lloyd')
    labels = km.fit_predict(psth)
    # Silhouette on a deterministic representative sample keeps the audit tractable.
    rng = np.random.default_rng(20260829)
    sample = rng.choice(len(psth), size=min(5000, len(psth)), replace=False)
    sil = silhouette_score(psth[sample], labels[sample], metric='euclidean')
    centers = km.cluster_centers_
    # Name clusters by center timing/shape, not arbitrary k-means IDs.
    peak = centers.argmax(axis=1)
    duration = np.sum(np.maximum(centers, 0), axis=1)
    order = np.argsort(peak)
    names = {}
    names[order[0]] = 'fast-transient'
    names[order[1]] = 'delayed-peak'
    names[order[2]] = 'delayed-sustained'
    # If peaks are too close, the broadest/most sustained center is explicitly retained.
    if len(set(peak)) < 3:
        order2 = np.argsort(duration)
        names[order2[0]] = 'fast-transient'
        names[order2[1]] = 'delayed-peak'
        names[order2[2]] = 'delayed-sustained'
    out = meta[['unit_global','roi','native_area','session','monkey','released_independent_consistency','repeat_mean_oof_r']].copy()
    out['cluster_id'] = labels
    out['cluster_name'] = [names[x] for x in labels]
    out.to_csv(OUT/'unit_tripleN_global_cluster.csv', index=False)
    pd.DataFrame(centers, columns=[f'{x:.1f}ms' for x in times]).assign(cluster_id=np.arange(3), cluster_name=[names[i] for i in range(3)]).to_csv(OUT/'global_cluster_centers.csv', index=False)
    pd.DataFrame({'cluster_id':list(range(3)), 'cluster_name':[names[i] for i in range(3)], 'n_units':[(labels==i).sum() for i in range(3)], 'peak_ms':times[peak], 'silhouette_sample':sil}).to_csv(OUT/'cluster_summary.csv', index=False)
    # Plot global centers.
    plt.figure(figsize=(8,5))
    colors={'fast-transient':'#2c7fb8','delayed-peak':'#f28e2b','delayed-sustained':'#59a14f'}
    for i in range(3): plt.plot(times, centers[i], lw=2.5, color=colors[names[i]], label=f"{names[i]} (n={(labels==i).sum():,})")
    plt.axvspan(0,150, color='0.92', zorder=-1); plt.xlabel('Time from stimulus onset (ms)'); plt.ylabel('Unit-normalized mean PSTH'); plt.title(f'Triple-N global k-means clusters (0-189 ms), silhouette={sil:.3f}'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/'global_cluster_centers.png', dpi=180); plt.close()
    # Compare G/T image correlation by global cluster, using the established full reliable window.
    relbase = np.flatnonzero(windows[:,1] < 0)
    gtid = np.flatnonzero((windows[:,0] >= 70) & (windows[:,1] <= 189))
    gt_times = windows[gtid].mean(1)
    rows=[]
    for j,u in enumerate(meta.itertuples(index=False)):
        raw=np.asarray(resp[:,:,int(u.unit_global)], np.float32)
        template=raw[gtid].mean(1)
        physical,_=raw_directions(template, gt_times)
        coeff=(raw[gtid]-raw[relbase].mean(0)[None]).T @ physical
        g,t=coeff[:,0],coeff[:,1]
        r=np.corrcoef(g,t)[0,1] if np.std(g)>0 and np.std(t)>0 else np.nan
        rows.append({'unit_global':int(u.unit_global),'cluster_id':labels[j],'cluster_name':names[labels[j]],'native_area':u.native_area,'G_T_image_r':r})
    gt=pd.DataFrame(rows); gt.to_csv(OUT/'unit_G_T_by_global_cluster.csv', index=False)
    areas=['V1','V2','V4','MF','MO','CLC']
    summ=gt[gt.native_area.isin(areas)].groupby(['native_area','cluster_name']).G_T_image_r.agg(['count','median','mean','std']).reset_index()
    summ.to_csv(OUT/'area_cluster_G_T_summary.csv', index=False)
    tests=[]
    for area,q in gt[gt.native_area.isin(areas)].groupby('native_area'):
        gs=[x.G_T_image_r.dropna().to_numpy() for _,x in q.groupby('cluster_name') if x.G_T_image_r.notna().sum()>=5]
        if len(gs)>=2:
            h,p=kruskal(*gs); tests.append({'native_area':area,'test':'Kruskal-Wallis','H':h,'p':p,'n_units':len(q)})
    pd.DataFrame(tests).to_csv(OUT/'area_cluster_G_T_tests.csv', index=False)
    print(f'Global units: {len(labels):,}; silhouette(sample)={sil:.4f}')
    print(pd.Series([names[x] for x in labels]).value_counts().to_string())
    print('\nG/T summary:'); print(summ.to_string(index=False))
    print('\nTests:'); print(pd.DataFrame(tests).to_string(index=False))

if __name__ == '__main__': main()
