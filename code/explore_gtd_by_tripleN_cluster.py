"""Exploratory audit: raw G/T dynamics versus pre-existing Triple-N clusters."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import kruskal,pearsonr
from plot_orthogonal_temporal_modes_2d import raw_directions
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); R=P/'results'; B=P/'proxy_bank_10ms_all_windows'; CL=R/'dynamic_phenotype_clustering_heldout_2026-08-27'; OUT=R/'exploratory_gtd_by_tripleN_cluster_2026-08-29'; OUT.mkdir(parents=True,exist_ok=True)
AREAS=['V1','V2','V4','MF','MO','CLC']
def main():
 meta=pd.read_csv(R/'all_unit_manifest.csv'); cluster=pd.read_csv(CL/'unit_dynamic_phenotypes.csv')[['unit_global','stable_type','heldout_prediction_margin']]; meta=meta.merge(cluster,on='unit_global',how='inner',validate='one_to_one'); meta['analysis_area']=meta.native_area.where(meta.native_area.isin(AREAS),meta.roi.map({'anterior IT':'anterior IT','middle IT':'middle IT','posterior IT':'posterior IT'})); resp=np.load(B/'trin_all_units_10ms_responses.npy',mmap_mode='r'); w=np.load(B/'windows_10ms.npy').astype(int); base=np.flatnonzero(w[:,1]<0); tid=np.flatnonzero((w[:,0]>=70)&(w[:,1]<=189)); times=w[tid].mean(1).astype(float); rows=[]
 for j,u in enumerate(meta.itertuples(index=False)):
  raw=np.asarray(resp[:,:,int(u.unit_global)],np.float32); template=raw[tid].mean(1); physical,_=raw_directions(template,times); rel=(raw[tid]-raw[base].mean(0)[None]).T; coef=rel@physical; g=coef[:,0]; t=coef[:,1]; rows.append({'unit_global':int(u.unit_global),'analysis_area':u.analysis_area,'stable_type':u.stable_type,'G_mean':float(g.mean()),'T_mean':float(t.mean()),'G_sd':float(g.std(ddof=1)),'T_sd':float(t.std(ddof=1)),'G_T_image_r':float(pearsonr(g,t).statistic),'G_T_direction_cos':float(physical[:,0]@physical[:,1]),'repeat_mean_oof_r':float(u.repeat_mean_oof_r),'consistency':float(u.released_independent_consistency)})
 d=pd.DataFrame(rows); d.to_csv(OUT/'unit_G_T_by_cluster.csv',index=False)
 summaries=[]
 for area in AREAS+['anterior IT','middle IT','posterior IT']:
  q=d[d.analysis_area.eq(area)]
  for cl,g in q.groupby('stable_type'):
   summaries.append({'area':area,'cluster':cl,'n_units':len(g),'median_G_T_image_r':g.G_T_image_r.median(),'median_abs_G_T_image_r':g.G_T_image_r.abs().median(),'median_G_sd':g.G_sd.median(),'median_T_sd':g.T_sd.median(),'median_proxy_r':g.repeat_mean_oof_r.median()})
 summary=pd.DataFrame(summaries); summary.to_csv(OUT/'cluster_area_summary.csv',index=False)
 tests=[]
 for area in AREAS+['anterior IT','middle IT','posterior IT']:
  q=d[d.analysis_area.eq(area)].dropna(subset=['stable_type']); groups=[g.G_T_image_r.to_numpy() for _,g in q.groupby('stable_type') if len(g)>=5]; labels=[k for k,g in q.groupby('stable_type') if len(g)>=5];
  if len(groups)>1: h,p=kruskal(*groups); tests.append({'area':area,'metric':'G_T_image_r','clusters': '|'.join(labels),'H':float(h),'p':float(p),'n_units':len(q)})
 tests=pd.DataFrame(tests); tests.to_csv(OUT/'cluster_area_kruskal_tests.csv',index=False)
 print('cluster counts'); print(d.stable_type.value_counts().to_string()); print('\nsummary'); print(summary.to_string(index=False)); print('\ntests'); print(tests.to_string(index=False))
if __name__=='__main__': main()
