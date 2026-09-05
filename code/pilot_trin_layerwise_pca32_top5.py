from pathlib import Path
import io, zipfile
import numpy as np, torch, torchvision, h5py
from PIL import Image
from scipy.io import loadmat

ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'
ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'
CAV=P/'tcav_broden500/broden500_native_channel_cav_bank.npz'; WINDOWS=(70,110,150); UNITS=(3447,3234,3230); LAYERS=('res2','res3','res4')
def fields(device,n=14):
 xs=np.linspace(-10+20/25/2,10-20/25/2,25); ss=np.exp(np.linspace(np.log(.7),np.log(8),8)); xx,yy,sg=np.meshgrid(xs,xs,ss,indexing='ij');
 yy=-yy; c=torch.linspace(-10+10/n,10-10/n,n,device=device); X,Y=torch.meshgrid(c,c,indexing='ij'); out=[]
 for x,y,s in zip(xx.ravel(),yy.ravel(),sg.ravel()): out.append(torch.exp(-((X-x)**2+(Y-y)**2)/(2*s*s)))
 return torch.stack(out)
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev)
 y=np.load(RESP,mmap_mode='r'); ids=np.arange(y.shape[2]); ui=[int(np.flatnonzero(ids==u)[0]) for u in UNITS]; wi=[(w+20)//10 for w in WINDOWS]
 Y=np.stack([np.asarray(y[t,:,ui],np.float32) for t in wi],axis=0) # 3 x 1000 x 3
 model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev); tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
 fs=fields(dev); feats={l:[] for l in LAYERS};
 with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
  for st in range(0,1000,32):
   imgs=[]
   for j in range(st,min(st+32,1000)):
    with Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))) as im: imgs.append(tr(im.convert('RGB')))
   x=torch.stack(imgs).to(dev); x=model.relu(model.bn1(model.conv1(x))); x=model.maxpool(x); x=model.layer1(x); maps={'res2':x}; x=model.layer2(x); maps['res3']=x; x=model.layer3(x); maps['res4']=x
   for l in LAYERS:
    a=torch.nn.functional.interpolate(maps[l].float(),size=(14,14),mode='area'); f=fs.to(dev); pooled=torch.einsum('bchw,ghw->bgc',a,f).log1p().sqrt().cpu().numpy(); feats[l].append(pooled)
   print(st+len(imgs),flush=True)
 for l in LAYERS: feats[l]=np.concatenate(feats[l],0) # image x candidate x channel
 bank=np.load(CAV,allow_pickle=True); concepts=bank['concepts'].astype(str); nodes=bank['nodes'].astype(str); off=bank['offsets'].astype(int);
 for l in LAYERS:
  ni=int(np.flatnonzero(nodes==l)[0]); ca,cb=off[ni],off[ni+1]; cav=bank['cav_full'][:,ca:cb].astype(np.float64)
  print('\nLAYER',l)
  for k,u in enumerate(UNITS):
   for tw,w in enumerate(WINDOWS):
    yy=Y[tw,:,k]; Xc=feats[l]; xc=Xc-Xc.mean(0,keepdims=True); yc=yy-yy.mean(); cov=np.einsum('ngc,n->gc',xc,yc); var=np.sum(yc*yc); den=np.sum(xc*xc,axis=(0,2)); score=np.sum(cov*cov,axis=1)/(den*var+1e-8); g=int(np.argmax(score)); Xg=Xc[:,g,:]; mu=Xg.mean(0); sd=Xg.std(0).clip(1e-5); Z=(Xg-mu)/sd; U,S,V=np.linalg.svd(Z,full_matrices=False); V=V[:32].T; z=Z@V; zm=z.mean(0); zz=z-zm; lam=.05*np.trace(zz.T@zz)/32; beta=np.linalg.solve(zz.T@zz+lam*np.eye(32),zz.T@yc); ua=(V@beta)/sd; ca=(cav@V)/sd; co=ca@ua/(np.linalg.norm(ca,axis=1)*np.linalg.norm(ua)+1e-12); top=np.argsort(np.abs(co))[::-1][:5]; pred=(z-zm)@beta+yy.mean(); r=np.corrcoef(pred,yy)[0,1]; print(f'u={u} w={w} rf={g} r={r:.3f}: '+' | '.join(f'{concepts[a]}({co[a]:+.3f})' for a in top))
if __name__=='__main__': main()
