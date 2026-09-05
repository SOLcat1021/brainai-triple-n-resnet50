import io,zipfile
from pathlib import Path
import numpy as np,torch,torchvision,h5py
from PIL import Image
ROOT=Path(r'D:/Coding/BrainAI'); P=ROOT/'ResNet50_Final_AllUnits_2026-08-24'; ZIP=ROOT/'data/TripleN/V1/others/StimuliNNN.zip'; RESP=P/'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'; CAV=P/'tcav_broden500/broden500_native_channel_cav_bank.npz'
U=(13778,2432,2516,53,2,193); W=(70,110,150); L=('res2','res3','res4','res5')
def main():
 d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(); m=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(d); c=torch.linspace(-10+10/14,10-10/14,14,device=d); X,Y=torch.meshgrid(c,c,indexing='ij'); f=torch.exp(-(X*X+Y*Y)/(2*8**2)); feats={x:[] for x in L}
 with zipfile.ZipFile(ZIP) as z,torch.inference_mode():
  for s in range(0,1000,32):
   ims=[tr(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB')) for j in range(s,min(1000,s+32))]; x=torch.stack(ims).to(d); x=m.relu(m.bn1(m.conv1(x))); x=m.maxpool(x); x=m.layer1(x); a={'res2':x}; x=m.layer2(x); a['res3']=x; x=m.layer3(x); a['res4']=x
   x=m.layer4(x); a['res5']=x
   for q in L:
    zq=torch.nn.functional.interpolate(a[q].float(),size=(14,14),mode='area'); feats[q].append(torch.einsum('bchw,hw->bc',zq,f).cpu().numpy())
  print('extracted',flush=True)
 y=np.load(RESP,mmap_mode='r'); ids=np.arange(y.shape[2]); ui=np.array(U); bank=np.load(CAV,allow_pickle=True); con=bank['concepts'].astype(str); nodes=bank['nodes'].astype(str); off=bank['offsets'].astype(int)
 for q in L:
  X=np.concatenate(feats[q]); cavnode=q+'_b1'; ni=int(np.flatnonzero(nodes==cavnode)[0]); cav=bank['cav_full'][:,off[ni]:off[ni+1]].astype(float); print('\n'+q+' (CAV '+cavnode+')')
  for k,u in enumerate(U):
   for t,w in enumerate(W):
    yy=np.asarray(y[(w+20)//10,:,u],float); xm=X.mean(0); sd=X.std(0).clip(1e-5); Z=(X-xm)/sd; _,_,Vt=np.linalg.svd(Z,full_matrices=False); V=Vt[:32].T; z=Z@V; zz=z-z.mean(0); b=np.linalg.solve(zz.T@zz+.05*np.trace(zz.T@zz)/32*np.eye(32),zz.T@(yy-yy.mean())); ua=(V@b)/sd; co=cav@ua/(np.linalg.norm(cav,axis=1)*np.linalg.norm(ua)+1e-12); top=np.argsort(co)[::-1][:5]; print(f'u={u} {w}-{w+9} r={np.corrcoef(zz@b,yy)[0,1]:.3f}: '+' | '.join(f'{con[a]}({co[a]:+.3f})' for a in top))
if __name__=='__main__': main()
