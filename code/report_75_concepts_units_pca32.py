"""Aligned PCA32 unit/concept report for the preselected Triple-N units.

Both unit readouts and concept axes use the same layer-specific, response-blind
Broden PCA32 basis. Prints only; no exploratory CSV/cache is written.
"""
import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image

ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'; META=P/'proxy_bank_10ms_all_windows/unit_metadata.csv'
Q=P/'tcav_broden1000_pca32/pca32_concept_axis_quality_1000.csv'; BANK=P/'tcav_broden1000_pca32/pca32_concept_axis_bank_1000.npz'; BASE=P/'tcav_broden1000_pca32/pca32_bases.npz'
UNITS=(19783,18219,16431,16277,16010,17395,19254,18896,11947,5791)
ROIS=('V1','V1/V2','V2','V4','CLC')
W=(70,110,150); LAYERS=('res2','res3','res4','res5'); N=1000; FOLDS=5; SEED=20260905; BATCH=32
def fmap(m,x,l):
 x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x)
 if l=='res2': return x
 x=m.layer2(x)
 if l=='res3': return x
 x=m.layer3(x)
 if l=='res4': return x
 return m.layer4(x)
def make_fields(dev):
 xs=np.linspace(-10+20/25/2,10-20/25/2,25,dtype=np.float32); ss=np.exp(np.linspace(np.log(.7),np.log(8),8)); xx,yy,sg=np.meshgrid(xs,xs,ss,indexing='ij'); yy=-yy; c=torch.linspace(-10+10/14,10-10/14,14,device=dev); X,Y=torch.meshgrid(c,c,indexing='ij'); return torch.stack([torch.exp(-((X-x)**2+(Y-y)**2)/(2*s*s)) for x,y,s in zip(xx.ravel(),yy.ravel(),sg.ravel())])
def corr(a,b):
 a=a-a.mean(); b=b-b.mean(); return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
def ridge(x,y):
 xm=x.mean(0); ym=float(y.mean()); z=x-xm; q=y-ym; g=z.T@z; lam=.05*np.trace(g)/g.shape[0]; return np.linalg.solve(g+lam*np.eye(g.shape[0],dtype=np.float32),z.T@q),xm,ym
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True); model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev); tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(); fields=make_fields(dev); resp=np.load(RESP,mmap_mode='r'); meta=pd.read_csv(META); idx={int(u):i for i,u in enumerate(meta.unit_global)}; units=[u for u in UNITS if u in idx]; ui=[idx[u] for u in units]; wi=[(w+20)//10 for w in W]; target=np.stack([np.asarray(resp[t,:,i],np.float32) for i in ui for t in wi],axis=1); keys=[(u,w) for u in units for w in W]; rng=np.random.default_rng(SEED); fold=np.empty(N,int)
 for k,ix in enumerate(np.array_split(rng.permutation(N),FOLDS)): fold[ix]=k
 q=pd.read_csv(Q); npos=q.groupby('concept_index').positive_count.first(); eligible=npos[npos>1000].index; stab=q[q.concept_index.isin(eligible)].groupby('concept_index').repeat_axis_cosine_mean.min(); selected=stab[stab>.9].index.to_numpy(int); selected=np.sort(selected); print('concepts',len(selected),flush=True)
 bank=np.load(BANK,allow_pickle=True); names=bank['concepts'].astype(str); axes=bank['cav_pca32'].astype(np.float32); bases=np.load(BASE)
 for layer,li in zip(LAYERS,range(4)):
  node=('res2_b3','res3_b4','res4_b6','res5_b3')[li]; mu=bases['mean_'+node].astype(np.float32); sd=bases['scale_'+node].astype(np.float32); V=bases['components_'+node].astype(np.float32).T; print('layer',layer,flush=True); allp=[]
  with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
   for start in range(0,N,BATCH):
    end=min(N,start+BATCH); imgs=torch.stack([tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(start,end)]).to(dev); a=torch.nn.functional.interpolate(fmap(model,imgs,layer).float(),size=(14,14),mode='area'); pooled=torch.einsum('bchw,ghw->bgc',a,fields).cpu().numpy(); allp.append(((pooled-mu[None,None,:])/sd[None,None,:])@V)
  p=np.concatenate(allp); G=p.shape[1]; sums=np.zeros((FOLDS,G,32)); sums2=np.zeros_like(sums); cross=np.zeros((FOLDS,G,32,target.shape[1])); sy=np.zeros((FOLDS,target.shape[1])); sy2=np.zeros_like(sy)
  for f in range(FOLDS):
   ix=np.flatnonzero(fold!=f); z=p[ix]; sums[f]=z.sum(0); sums2[f]=(z*z).sum(0); cross[f]=np.einsum('bgc,bt->gct',z,target[ix]); sy[f]=target[ix].sum(0); sy2[f]=(target[ix]*target[ix]).sum(0)
  pred=np.full_like(target,np.nan); cosine_sum=np.zeros((target.shape[1],len(selected)),np.float32)
  for f in range(FOLDS):
   ixtr=np.flatnonzero(fold!=f); ixte=np.flatnonzero(fold==f); ntr=float(len(ixtr)); mean=sums[f]/ntr; var=(sums2[f]-sums[f]*sums[f]/ntr).clip(1e-8); cov=cross[f]-mean[:,:,None]*sy[f][None,None,:]; vy=(sy2[f]-sy[f]*sy[f]/ntr).clip(1e-8); sel=((cov*cov/(var[:,:,None]*vy[None,None,:])).sum(1)).argmax(0); uniq=np.unique(sel); X=p[:,uniq,:]; pos={int(g):j for j,g in enumerate(uniq)}
   for t in range(target.shape[1]):
    b,xm,ym=ridge(X[ixtr,pos[int(sel[t])]],target[ixtr,t]); pred[ixte,t]=(X[ixte,pos[int(sel[t])]]-xm)@b+ym; ua=np.zeros(32); ua[:]=b; co=axes[selected,li]@ua/(np.linalg.norm(axes[selected,li],axis=1)*np.linalg.norm(ua)+1e-12); cosine_sum[t] += co
  print(f'[{layer}]',flush=True)
  for t,(u,w) in enumerate(keys):
   r=corr(pred[:,t],target[:,t]); co=cosine_sum[t]/FOLDS; top=np.argsort(co)[::-1][:5]; print(f'{u} {w}-{w+9} r={r:.3f}: '+' | '.join(f'{names[selected[a]]}({co[a]:+.3f})' for a in top),flush=True)
if __name__=='__main__': main()
