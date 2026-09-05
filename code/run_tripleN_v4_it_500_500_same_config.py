"""Controlled Triple-N 500/500 readout stability: 5 V4 + 5 IT units."""
from pathlib import Path
import json, time, sys
import h5py, numpy as np, torch, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_64d_pilot_disjoint_split_half as base

P=Path(r'D:\Coding\BrainAI'); B=P/'ResNet50_Final_AllUnits_2026-08-24/proxy_bank_64d_high_quality_pilot_113_2026-08-26'; OLD=P/'ResNet50_Final_AllUnits_2026-08-24/proxy_bank_10ms_all_windows'; OUT=B/'controlled_v4_it_500_500'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260904; N=1000; RANK=64; NLAY=17; ROIS=('V4','posterior IT','middle IT','anterior IT'); slug=lambda x:x.lower().replace(' ','_')

def main():
 sel=pd.read_csv(B/'selected_units.csv'); v4=sel[sel.roi.eq('V4')].sort_values('released_independent_consistency',ascending=False).head(5); it=sel[sel.roi.eq('middle IT')].sort_values('released_independent_consistency',ascending=False).head(5); chosen=pd.concat([v4,it],ignore_index=True); chosen.to_csv(OUT/'selected_units.csv',index=False)
 maps=np.load(base.MAPS_PATH,mmap_mode='r'); resp=np.load(OLD/'trin_all_units_10ms_responses.npy',mmap_mode='r'); order=np.random.default_rng(SEED).permutation(N); halves=(np.sort(order[:500]),np.sort(order[500:])); rows=[]
 for roi in ROIS:
  sub=chosen[chosen.roi.eq(roi)];
  if sub.empty: continue
  ids=sub.unit_global.to_numpy(int); source=B/f'{slug(roi)}_proxy_bank_64d.h5'
  with h5py.File(source,'r') as h:
   allids=h['unit_global'][:].astype(int); pos=np.array([int(np.where(allids==x)[0][0]) for x in ids]); order_pos=np.argsort(pos); fields=h['spatial_parameters'][:,pos[order_pos]]; gates=h['layer_gate'][:,pos[order_pos]]; wins=h['windows_ms'][:]
  # restore requested unit order after h5py-compatible sorted read
  inv=np.argsort(order_pos); fields=fields[:,inv]; gates=gates[:,inv]
  target=np.asarray(resp[:,:,ids],np.float32); path=OUT/f'{slug(roi)}.h5'
  with h5py.File(path,'w') as h:
   keep=np.flatnonzero(np.isin(wins[:,0],[70,110,150])); wins=wins[keep]; fields=fields[keep]; gates=gates[keep]
   h['unit_global']=ids; h['windows_ms']=wins; h['train_indices']=np.stack(halves); h['test_indices']=np.stack((halves[1],halves[0])); h['layer_gate']=gates; h.create_dataset('split_normalization_std',(2,NLAY,RANK),'f4'); h.create_dataset('split_channel_weights',(2,len(wins),len(ids),NLAY*RANK),'f4'); h.create_dataset('cross_half_r',(2,len(wins),len(ids)),'f4')
   for sp,tr in enumerate(halves):
    te=halves[1-sp]; rawtr=torch.as_tensor(np.asarray(maps[tr,:,:RANK],np.float32),device='cuda').flatten(3); rawte=torch.as_tensor(np.asarray(maps[te,:,:RANK],np.float32),device='cuda').flatten(3); mean=rawtr.mean((0,3),keepdim=True); std=rawtr.std((0,3),keepdim=True).clamp_min(1e-5); h['split_normalization_std'][sp]=std.squeeze(0).squeeze(-1).cpu().numpy(); ftr=(rawtr-mean)/std; fte=(rawte-mean)/std
    for wi in range(len(wins)):
     fld=torch.as_tensor(base.gaussian_fields(fields[wi]),device='cuda'); xtr=torch.einsum('nldp,up->nuld',ftr,fld); xte=torch.einsum('nldp,up->nuld',fte,fld); scale=torch.as_tensor(np.sqrt(np.clip(gates[wi]*NLAY,1e-5,None)),device='cuda'); xtr=(xtr*scale[None,:,:,None]).flatten(2); xte=(xte*scale[None,:,:,None]).flatten(2); ytr=torch.as_tensor(target[wi,tr],device='cuda'); pred=np.empty((len(te),len(ids)),np.float32)
     target_wi=int(keep[wi])
     for u0 in range(0,len(ids),base.TARGET_BATCH):
      u1=min(len(ids),u0+base.TARGET_BATCH); pr,beta,_,_=base.dual_ridge(xtr[:,u0:u1],torch.as_tensor(target[target_wi,tr,u0:u1],device='cuda'),xte[:,u0:u1]); pred[:,u0:u1]=pr.detach().cpu().numpy(); eff=beta*torch.repeat_interleave(scale[u0:u1],RANK,dim=1); h['split_channel_weights'][sp,wi,u0:u1]=eff.detach().cpu().numpy()
     h['cross_half_r'][sp,wi]=base.corr_columns(pred,target[target_wi,te]);
    del ftr,fte
  print(roi,'done',flush=True)
  with h5py.File(path,'r') as h:
   rr=h['cross_half_r'][:]; W=h['split_channel_weights'][:]; sd=h['split_normalization_std'][:]; G=h['layer_gate'][:]
  # Saved weights already include the frozen depth-gate scale; compare them
  # directly in the common frozen PCA-coordinate space.
  q=W.reshape(2,len(wins),len(ids),-1); q/=np.linalg.norm(q,axis=-1,keepdims=True).clip(1e-12)
  for wi,(a,b) in enumerate(wins):
   for ui,u in enumerate(ids): rows.append({'roi':roi,'unit_global':int(u),'window_start_ms':int(a),'window_end_ms':int(b),'axis_cosine_split0_split1':float(q[0,wi,ui]@q[1,wi,ui]),'r0':float(rr[0,wi,ui]),'r1':float(rr[1,wi,ui])})
 d=pd.DataFrame(rows); print(d.groupby('roi').axis_cosine_split0_split1.agg(['count','min','median','mean','max']).to_string()); print('overall',d.axis_cosine_split0_split1.median()); print('windows',sorted(d.window_start_ms.unique().tolist()));
if __name__=='__main__': main()
