"""Small OOF prediction comparison: PCA8 versus PCA32.

Same images, windows, folds, RF bank, response targets, and ridge rule; only
the retained response-blind PCA rank is changed. Results are printed only.
"""
import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image

ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'; META=P/'proxy_bank_10ms_all_windows/unit_metadata.csv'
UNITS=(3447,3234,3230,6608,2432); WINDOWS=(70,110,150); LAYERS=('res2','res3','res4','res5'); N=1000; FOLDS=5; SEED=20260905; BATCH=32

def fmap(m,x,l):
 x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x)
 if l=='res2': return x
 x=m.layer2(x)
 if l=='res3': return x
 x=m.layer3(x)
 if l=='res4': return x
 return m.layer4(x)

def fields(device):
 xs=np.linspace(-10+20/25/2,10-20/25/2,25,dtype=np.float32); ss=np.exp(np.linspace(np.log(.7),np.log(8),8)).astype(np.float32); xx,yy,sg=np.meshgrid(xs,xs,ss,indexing='ij'); yy=-yy; c=torch.linspace(-10+10/14,10-10/14,14,device=device); X,Y=torch.meshgrid(c,c,indexing='ij'); return torch.stack([torch.exp(-((X-x)**2+(Y-y)**2)/(2*s*s)) for x,y,s in zip(xx.ravel(),yy.ravel(),sg.ravel())])

def corr(a,b):
 a=a-a.mean(); b=b-b.mean(); return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))

def ridge(x,y):
 xm=x.mean(0); ym=float(y.mean()); z=x-xm; q=y-ym; g=z.T@z; lam=.05*np.trace(g)/g.shape[0]; return np.linalg.solve(g+lam*np.eye(g.shape[0],dtype=np.float32),z.T@q),xm,ym

def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True); model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev); tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(); fs=fields(dev); y=np.load(RESP,mmap_mode='r'); meta=pd.read_csv(META); ui=[int(np.flatnonzero(meta.unit_global.to_numpy()==u)[0]) for u in UNITS]; wi=[(w+20)//10 for w in WINDOWS]; target=np.stack([np.asarray(y[t,:,i],np.float32) for i in ui for t in wi],axis=1); keys=[(u,w) for u in UNITS for w in WINDOWS]; rng=np.random.default_rng(SEED); fold=np.empty(N,int)
 for k,ix in enumerate(np.array_split(rng.permutation(N),FOLDS)): fold[ix]=k
 results=[]
 for rank in (8,32):
  print('RANK',rank,flush=True)
  for layer in LAYERS:
   channels={'res2':256,'res3':512,'res4':1024,'res5':2048}[layer]; center=[]; allmaps=[]
   with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
    for start in range(0,N,BATCH):
     end=min(N,start+BATCH); imgs=torch.stack([tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(start,end)]).to(dev); a=torch.nn.functional.interpolate(fmap(model,imgs,layer).float(),size=(14,14),mode='area'); center.append(torch.einsum('bchw,hw->bc',a,fs[12*25*8+12*8+7]).cpu().numpy()); allmaps.append(a.cpu())
   center=np.concatenate(center); mu=center.mean(0); sd=center.std(0).clip(1e-5); _,_,Vt=np.linalg.svd((center-mu)/sd,full_matrices=False); V=Vt[:rank].T; # reuse pooled maps to avoid another network pass
   p=np.empty((N,fs.shape[0],rank),np.float32)
   for start,a in zip(range(0,N,BATCH),allmaps):
    pooled=torch.einsum('bchw,ghw->bgc',a,fs.cpu()).numpy(); p[start:start+len(pooled)]=((pooled-mu[None,None,:])/sd[None,None,:])@V
   G=fs.shape[0]; sums=np.zeros((FOLDS,G,rank)); sums2=np.zeros_like(sums); cross=np.zeros((FOLDS,G,rank,target.shape[1])); sy=np.zeros((FOLDS,target.shape[1])); sy2=np.zeros_like(sy)
   for f in range(FOLDS):
    ix=np.flatnonzero(fold!=f); q=p[ix]; sums[f]=q.sum(0); sums2[f]=(q*q).sum(0); cross[f]=np.einsum('bgc,bt->gct',q,target[ix]); sy[f]=target[ix].sum(0); sy2[f]=(target[ix]*target[ix]).sum(0)
   pred=np.full_like(target,np.nan)
   for f in range(FOLDS):
    ntr=float(np.sum(fold!=f)); mean=sums[f]/ntr; var=(sums2[f]-sums[f]*sums[f]/ntr).clip(1e-8); cov=cross[f]-mean[:,:,None]*sy[f][None,None,:]; vy=(sy2[f]-sy[f]*sy[f]/ntr).clip(1e-8); score=(cov*cov/(var[:,:,None]*vy[None,None,:])).sum(1); selected=score.argmax(0); uniq=np.unique(selected); pos={int(g):j for j,g in enumerate(uniq)}; X=p[:,uniq,:]
    for t in range(target.shape[1]):
     beta,xm,ym=ridge(X[:,pos[int(selected[t])]][fold!=f],target[fold!=f,t]); pred[fold==f,t]=(X[fold==f,pos[int(selected[t])]]-xm)@beta+ym
   for t,key in enumerate(keys): results.append({'rank':rank,'layer':layer,'unit':key[0],'window_ms':key[1],'oof_r':corr(pred[:,t],target[:,t])})
   print(layer,'done',flush=True)
 q=pd.DataFrame(results); print(q.pivot_table(index=['unit','window_ms'],columns=['rank','layer'],values='oof_r').round(3).to_string()); q8=q[q['rank']==8].set_index(['unit','window_ms','layer']).oof_r; q32=q[q['rank']==32].set_index(['unit','window_ms','layer']).oof_r; d=(q32-q8).dropna(); print('delta PCA32-PCA8: mean',round(float(d.mean()),4),'median',round(float(d.median()),4),'positive',int((d>0).sum()),'/',len(d));
if __name__=='__main__': main()
