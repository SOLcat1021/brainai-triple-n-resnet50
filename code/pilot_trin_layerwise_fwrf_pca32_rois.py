import io,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch,torchvision,h5py
from PIL import Image
ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'; ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'; META=P/'proxy_bank_10ms_all_windows/unit_metadata.csv'; CAV=P/'tcav_broden500/broden500_native_channel_cav_bank.npz'
ROIS=('V1','V1/V2','V2','V4','MF','MB','MO','CLC','LPP','PF','PITP','AF','AB','AO','AMC'); W=(70,110,150); L=('res5',); BATCH=32; NIMG=1000
def make_fields(dev):
 xs=np.linspace(-10+20/25/2,10-20/25/2,25); ss=np.exp(np.linspace(np.log(.7),np.log(8),8)); xx,yy,sg=np.meshgrid(xs,xs,ss,indexing='ij'); yy=-yy; c=torch.linspace(-10+10/14,10-10/14,14,device=dev); X,Y=torch.meshgrid(c,c,indexing='ij'); return torch.stack([torch.exp(-((X-x)**2+(Y-y)**2)/(2*s*s)) for x,y,s in zip(xx.ravel(),yy.ravel(),sg.ravel())])
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev)
 y=np.load(RESP,mmap_mode='r'); meta=pd.read_csv(META); wi=np.array([(w+20)//10 for w in W]); bank_for={'V1':'v1_proxy_bank.h5','V1/V2':'v2_proxy_bank.h5','V2':'v2_proxy_bank.h5','V4':'v4_proxy_bank.h5','MF':'middle_it_proxy_bank.h5','MB':'middle_it_proxy_bank.h5','MO':'middle_it_proxy_bank.h5','CLC':'middle_it_proxy_bank.h5','LPP':'middle_it_proxy_bank.h5','PF':'posterior_it_proxy_bank.h5','PITP':'posterior_it_proxy_bank.h5','AF':'anterior_it_proxy_bank.h5','AB':'anterior_it_proxy_bank.h5','AO':'anterior_it_proxy_bank.h5','AMC':'anterior_it_proxy_bank.h5'}
 banks={};
 for fn in set(bank_for.values()):
  with h5py.File(P/'proxy_bank_10ms_all_windows'/fn,'r') as h: banks[fn]=(h['unit_global'][:].astype(int),h['oof_r'][:])
 chosen=[]
 for roi in ROIS:
  ids,oo=banks[bank_for[roi]]; idx={int(u):i for i,u in enumerate(ids)}; us=meta.loc[meta.native_area.eq(roi),'unit_global'].astype(int); cand=[u for u in us if u in idx]; vals=sorted([(float(np.min(oo[wi,idx[u]])),float(np.mean(oo[wi,idx[u]])),u) for u in cand],reverse=True); chosen += [x[2] for x in vals[:2]]; print(roi,'units',[(x[2],round(x[0],3),round(x[1],3)) for x in vals[:2]])
 tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(); m=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev); fields=make_fields(dev); center=fields[1125]
 bank=np.load(CAV,allow_pickle=True); concepts=bank['concepts'].astype(str); nodes=bank['nodes'].astype(str); off=bank['offsets'].astype(int)
 for layer in L:
  dim={'res2':256,'res3':512,'res4':1024,'res5':2048}[layer]; center_x=[]; print('extract center/PCA',layer,flush=True)
  with zipfile.ZipFile(ZIP) as z,torch.inference_mode():
   for s in range(0,NIMG,BATCH):
    ims=[tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(s,min(NIMG,s+BATCH))]; x=torch.stack(ims).to(dev); x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x); x=m.layer2(x); x=m.layer3(x) if layer in ('res3','res4','res5') else x; x=m.layer4(x) if layer=='res5' else x; x=torch.nn.functional.interpolate(x.float(),size=(14,14),mode='area'); center_x.append(torch.einsum('bchw,hw->bc',x,center).cpu().numpy())
  cx=np.concatenate(center_x); mu=cx.mean(0); sd=cx.std(0).clip(1e-5); _,_,Vt=np.linalg.svd((cx-mu)/sd,full_matrices=False); V=Vt[:32].T
  feats=[]; print('extract candidate maps',layer,flush=True)
  with zipfile.ZipFile(ZIP) as z,torch.inference_mode():
   for s in range(0,NIMG,BATCH):
    ims=[tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(s,min(NIMG,s+BATCH))]; x=torch.stack(ims).to(dev); x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x); x=m.layer2(x); x=m.layer3(x) if layer in ('res3','res4','res5') else x; x=m.layer4(x) if layer=='res5' else x; x=torch.nn.functional.interpolate(x.float(),size=(14,14),mode='area'); p=torch.einsum('bchw,ghw->bgc',x,fields).cpu().numpy(); feats.append(p)
  X=np.concatenate([(((p-cx.mean(0)[None,None,:])/sd[None,None,:]) @ V).astype(np.float16) for p in feats]).transpose(1,0,2).astype(np.float32)
  ni=int(np.flatnonzero(nodes==layer+'_b1')[0]); cav=bank['cav_full'][:,off[ni]:off[ni+1]].astype(float) @ V
  print('\n###',layer)
  for u in chosen:
   for t,w in enumerate(W):
    yy=np.asarray(y[wi[t],:,u],float); xn=X-X.mean(1,keepdims=True); yc=yy-yy.mean(); cov=np.einsum('gnd,n->gd',xn,yc); den=np.sqrt(np.sum(xn*xn,axis=(1,2))*np.sum(yc*yc)+1e-12); cand=np.argsort(np.sum((cov/den[:,None])**2,axis=1))[::-1][:32]; scores=[]
    for g in cand:
     z=X[g]; zc=z-z.mean(0); yc=yy-yy.mean(); lam=.05*np.trace(zc.T@zc)/32; b=np.linalg.solve(zc.T@zc+lam*np.eye(32),zc.T@yc); pred=zc@b; scores.append((np.corrcoef(pred,yy)[0,1],g,b))
    r,g,b=max(scores,key=lambda a:a[0]); co=cav@b/(np.linalg.norm(cav,axis=1)*np.linalg.norm(b)+1e-12); top=np.argsort(co)[::-1][:5]; print(f'{u} {w}-{w+9} r={r:.3f}: '+' | '.join(f'{concepts[a]}({co[a]:+.3f})' for a in top),flush=True)
if __name__=='__main__': main()
