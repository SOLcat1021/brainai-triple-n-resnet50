"""Small scalar-target capacity comparison: PCA8 vs full ResNet channels."""
from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack

ROOT=Path(__file__).resolve().parents[2]
PCA=ROOT/"cache/tvsd/original_fwrf_resnet50/spatial_pca8_14"
FULL=ROOT/"cache/tvsd/original_fwrf_resnet50/spatial_full_14"
DATA=Path(r"G:\BrainAI_Data\TVSD")
OUT=ROOT/"ResNet50_Final_AllUnits_2026-08-24/results/tvsd_scalar_pca8_vs_full_2026-08-30"
LAYERS={"stem":64,"res2":256,"res3":512,"res4":1024,"res5":2048}
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def corr(a,b):
 a=a-a.mean(0,keepdims=True); b=b-b.mean(0,keepdims=True)
 return (a*b).sum(0)/np.sqrt(np.maximum((a*a).sum(0)*(b*b).sum(0),1e-12))

def run_view(xpath, y, fields, channels, view, batch=128, gblock=75):
 n=xpath.shape[0]; u=y.shape[1]; G=len(fields); sx=np.zeros((G,channels),np.float64); sx2=np.zeros_like(sx); cross=np.zeros((G,channels,u),np.float64)
 sy=y.sum(0,dtype=np.float64); sy2=(y*y).sum(0,dtype=np.float64)
 for a in range(0,n,batch):
  b=min(a+batch,n); xb=torch.as_tensor(np.asarray(xpath[a:b],np.float32),device=DEVICE); yb=torch.as_tensor(y[a:b],device=DEVICE)
  for g0 in range(0,G,gblock):
   g1=min(g0+gblock,G); z=torch.einsum('bchw,ghw->bgc',xb,fields[g0:g1]).float()
   sx[g0:g1]+=z.sum(0).cpu().numpy(); sx2[g0:g1]+=(z*z).sum(0).cpu().numpy(); cross[g0:g1]+=torch.einsum('bgc,bu->gcu',z,yb).cpu().numpy()
  if b==n or b%2048==0: print(view,b,'/',n,flush=True)
 meanx=sx/n; meany=sy/n; vx=sx2-sx*sx/n; vy=sy2-sy*sy/n; cov=cross-meanx[:,:,None]*sy[None,None,:]
 score=cov*cov/np.maximum(vx[:,:,None]*vy[None,None,:],1e-12); flat=score.reshape(G*channels,u); pick=np.argmax(flat,0); g=pick//channels; c=pick%channels; num=cov[g,c,np.arange(u)]; den=vx[g,c]; w=num/np.maximum(den,1e-8)
 return g,c,w,meany,view

def predict(xpath, fields, g,c,w,meany, batch=128):
 out=np.empty((xpath.shape[0],len(g)),np.float32)
 for a in range(0,len(xpath),batch):
  b=min(a+batch,len(xpath)); x=torch.as_tensor(np.asarray(xpath[a:b],np.float32),device=DEVICE); z=torch.einsum('bchw,ghw->bgc',x,fields[g]); out[a:b]=(z[:,np.arange(len(g)),c]*torch.as_tensor(w,device=DEVICE)[None]+torch.as_tensor(meany,device=DEVICE)[None]).cpu().numpy()
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True); xs,ys,sg=candidates(); fields=torch.as_tensor(gaussian_mass_stack(xs,ys,sg,14),device=DEVICE)
 rows=[]
 for monkey in ('N','F'):
  with h5py.File(DATA/f'monkey{monkey}/THINGS_normMUA.mat','r') as h:
   ytr=np.asarray(h['train_MUA'],np.float32); yte=np.asarray(h['test_MUA'],np.float32); oracle=np.asarray(h['oracle'],np.float32).reshape(-1)
  sel=np.argsort(oracle)[::-1][:25]; ytr=ytr[:,sel]; yte=yte[:,sel]
  p=np.load(PCA/'train_maps_f16.npy',mmap_mode='r'); pt=np.load(PCA/'test_maps_f16.npy',mmap_mode='r')
  g,c,w,my,v=run_view(p,ytr,fields,40,'PCA8'); pred=predict(pt,fields,g,c,w,my); rs=corr(pred,yte)
  for i,r in enumerate(rs): rows.append({'monkey':monkey,'channel_rank':i,'view':'PCA8','test_r':float(r),'candidate':int(g[i]),'feature_channel':int(c[i])})
  del p,pt
  for layer,ch in LAYERS.items():
   tr=np.load(FULL/f'train_{layer}_f16.npy',mmap_mode='r'); te=np.load(FULL/f'test_{layer}_f16.npy',mmap_mode='r')
   g,c,w,my,v=run_view(tr,ytr,fields,ch,layer); pred=predict(te,fields,g,c,w,my); rs=corr(pred,yte)
   for i,r in enumerate(rs): rows.append({'monkey':monkey,'channel_rank':i,'view':layer,'test_r':float(r),'candidate':int(g[i]),'feature_channel':int(c[i])})
   del tr,te
 df=pd.DataFrame(rows); df.to_csv(OUT/'channel_results.csv',index=False,encoding='utf-8-sig')
 summary=df.groupby('view').test_r.agg(['mean','median','std','count']).reset_index(); summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
 (OUT/'summary.json').write_text(json.dumps({'device':str(DEVICE),'n_channels_per_monkey':25,'target':'official THINGS_normMUA train_MUA/test_MUA scalar','candidate_count':len(xs),'comparison':summary.to_dict('records')},ensure_ascii=False,indent=2),encoding='utf-8')
 print(summary.to_string(index=False))
if __name__=='__main__': main()
