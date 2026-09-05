from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
P=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24'); C=P/'results'/'circle5000_axis500_2026-08-29'; R=P/'results'/'six_area_top5_circle_axis_G_2026-08-29'; OUT=R
axes=pd.read_csv(C/'axis_manifest_500.csv'); mm=np.memmap(C/'axis500_circle5000_scores_f16.dat',mode='r',dtype='float16',shape=(500,5000,1000)); top=pd.read_csv(R/'top_axis_per_unit.csv'); mf=top[top.area.eq('MF')].copy(); # one row per MF unit
concepts=['eye','mouth','eyebrow','nose','head','person','ear']
ids={c:int(axes.index[axes.concept.eq(c)][0]) for c in concepts if axes.concept.eq(c).any()}
rows=[]; mats=[]
for u in mf.itertuples(index=False):
 s={c:np.asarray(mm[ids[c],int(u.circle_id)],float) for c in ids};
 for c1 in concepts:
  if c1 not in s: continue
  for c2 in concepts:
   if c2<=c1 or c2 not in s: continue
   rows.append({'unit_global':int(u.unit_global),'circle_id':int(u.circle_id),'concept_1':c1,'concept_2':c2,'pearson_r':float(pearsonr(s[c1],s[c2]).statistic),'spearman_r':float(spearmanr(s[c1],s[c2]).statistic)})
 # unit-specific relation to G from stored top row is not enough; calculate all part-G later from top
 mats.append((int(u.unit_global),int(u.circle_id),s))
pd.DataFrame(rows).to_csv(OUT/'mf_face_part_axis_pairwise_correlations.csv',index=False)
# summary pairs across units
summary=pd.DataFrame(rows).groupby(['concept_1','concept_2']).agg(median_pearson=('pearson_r','median'),min_pearson=('pearson_r','min'),max_pearson=('pearson_r','max')).reset_index(); summary.to_csv(OUT/'mf_face_part_axis_pairwise_summary.csv',index=False)
print('axis ids',ids); print('MF units/circles'); print(mf[['unit_global','proxy_oof_r','circle_id']].to_string(index=False)); print('\nPairwise correlations:'); print(pd.DataFrame(rows).to_string(index=False)); print('\nSummary'); print(summary.to_string(index=False))
