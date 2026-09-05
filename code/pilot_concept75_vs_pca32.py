"""OOF pilot: 75 concept scores versus their source PCA32 features."""
import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'; META=P/'proxy_bank_10ms_all_windows/unit_metadata.csv'
Q=P/'tcav_broden1000_pca32/pca32_concept_axis_quality_1000.csv'; BANK=P/'tcav_broden1000_pca32/pca32_concept_axis_bank_1000.npz'; BASE=P/'tcav_broden1000_pca32/pca32_bases.npz'
UNITS=(19783,16431,16010,19254,11947); WINDOWS=(70,110,150); LAYERS=('res2','res3','res4','res5'); N=1000; FOLDS=5; SEED=20260905; BATCH=32
ALPHAS=(.01,.1,1.,10.,100.,1000.)

def fmap(m,x,l):
 x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x)
 if l=='res2': return x
 x=m.layer2(x)
 if l=='res3': return x
 x=m.layer3(x)
 if l=='res4': return x
 return m.layer4(x)
def fields(dev):
 xs=np.linspace(-10+20/25/2,10-20/25/2,25,dtype=np.float32); ss=np.exp(np.linspace(np.log(.7),np.log(8),8)); xx,yy,sg=np.meshgrid(xs,xs,ss,indexing='ij'); yy=-yy; c=torch.linspace(-10+10/14,10-10/14,14,device=dev); X,Y=torch.meshgrid(c,c,indexing='ij'); return torch.stack([torch.exp(-((X-x)**2+(Y-y)**2)/(2*s*s)) for x,y,s in zip(xx.ravel(),yy.ravel(),sg.ravel())])
def corr(a,b):
 a=a-a.mean(); b=b-b.mean(); d=np.sqrt(np.sum(a*a)*np.sum(b*b)); return float(np.sum(a*b)/d) if d>1e-12 else np.nan
def nested_predict(x,y,train,test,seed):
 mu=x[train].mean(0); sd=x[train].std(0).clip(1e-6); z=(x-mu)/sd; inner=KFold(4,shuffle=True,random_state=seed); best=None
 for alpha in ALPHAS:
  ps=np.empty(len(train),np.float32)
  for a,b in inner.split(train):
   tr=train[a]; va=train[b]; model=Ridge(alpha=alpha).fit(z[tr],y[tr]); ps[b]=model.predict(z[va])
  score=corr(ps,y[train])
  if best is None or score>best[0]: best=(score,alpha)
 model=Ridge(alpha=best[1]).fit(z[train],y[train]); return model.predict(z[test]),best[1]
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True); model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev); tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(); fs=fields(dev); meta=pd.read_csv(META); lookup={int(u):i for i,u in enumerate(meta.unit_global)}; resp=np.load(RESP,mmap_mode='r'); wi=[(w+20)//10 for w in WINDOWS]; target=np.stack([np.asarray(resp[t,:,lookup[u]],np.float32) for u in UNITS for t in wi],axis=1); keys=[(u,w) for u in UNITS for w in WINDOWS]; rng=np.random.default_rng(SEED); fold=np.empty(N,int)
 for k,ix in enumerate(np.array_split(rng.permutation(N),FOLDS)): fold[ix]=k
 q=pd.read_csv(Q); count=q.groupby('concept_index').positive_count.first(); candidates=count[count>1000].index; stable=q[q.concept_index.isin(candidates)].groupby('concept_index').repeat_axis_cosine_mean.min(); selected=np.sort(stable[stable>.9].index.to_numpy(int)); bank=np.load(BANK,allow_pickle=True); axes=bank['cav_pca32'][selected].astype(np.float32); bases=np.load(BASE); print('concepts',len(selected),flush=True); rows=[]
 for li,layer in enumerate(LAYERS):
  node=('res2_b3','res3_b4','res4_b6','res5_b3')[li]; mu=bases['mean_'+node].astype(np.float32); sd=bases['scale_'+node].astype(np.float32); V=bases['components_'+node].astype(np.float32).T; chunks=[]
  with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
   for start in range(0,N,BATCH):
    end=min(N,start+BATCH); ims=torch.stack([tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(start,end)]).to(dev); a=torch.nn.functional.interpolate(fmap(model,ims,layer).float(),size=(14,14),mode='area'); pool=torch.einsum('bchw,ghw->bgc',a,fs).cpu().numpy(); chunks.append(((pool-mu[None,None,:])/sd[None,None,:])@V)
  p=np.concatenate(chunks); G=p.shape[1]; # fold-local FWRF selection based on full PCA32, shared by both models
  pred32=np.full_like(target,np.nan); predc=np.full_like(target,np.nan); alpha32=[]; alphac=[]
  for f in range(FOLDS):
   train=np.flatnonzero(fold!=f); test=np.flatnonzero(fold==f); z=p[train]; yy=target[train]; zm=z-z.mean(0); yc=yy-yy.mean(0); cov=np.einsum('ngc,nt->gct',zm,yc); den=np.sum(zm*zm,axis=(0,2))[:,None]*np.sum(yc*yc,axis=0)[None,:]; rf=np.argmax(np.sum(cov*cov,axis=1)/(den+1e-12),axis=0)
   for t in range(target.shape[1]):
    x=p[:,rf[t],:]; cs=x@axes[:,li,:].T; pred32[test,t],a32=nested_predict(x,target[:,t],train,test,SEED+f*100+t); predc[test,t],ac=nested_predict(cs,target[:,t],train,test,SEED+f*100+t); alpha32.append(a32); alphac.append(ac)
  print('\n',layer,'concept-axis rank',np.linalg.matrix_rank(axes[:,li,:]),flush=True)
  for t,(u,w) in enumerate(keys):
   a=corr(pred32[:,t],target[:,t]); b=corr(predc[:,t],target[:,t]); rows.append((layer,u,w,a,b,b-a)); print(f'{u} {w}-{w+9}: PCA32={a:.3f} concept75={b:.3f} delta={b-a:+.3f}',flush=True)
 print('\nSUMMARY',flush=True); a=np.asarray([x[3:] for x in rows]); print('mean PCA32',a[:,0].mean(),'concept75',a[:,1].mean(),'delta',a[:,2].mean(),'wins',np.sum(a[:,2]>0),'/',len(a),flush=True)
 for layer in LAYERS:
  z=np.asarray([x[3:] for x in rows if x[0]==layer]); print(layer,'PCA32',z[:,0].mean(),'concept75',z[:,1].mean(),'delta',z[:,2].mean(),flush=True)
if __name__=='__main__': main()
