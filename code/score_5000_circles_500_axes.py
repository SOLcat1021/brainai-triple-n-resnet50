from pathlib import Path
import io, zipfile, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

PROJECT=Path(r'D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24')
STIM=Path(r'D:\Coding\BrainAI\data\TripleN\V1\others\StimuliNNN.zip')
CAV=PROJECT/'tcav_broden500'/'broden500_native_channel_cav_bank.npz'
QUAL=PROJECT/'tcav_broden500'/'broden500_cav_quality_by_layer.csv'
OUT=PROJECT/'results'/'circle5000_axis500_2026-08-29'; OUT.mkdir(parents=True,exist_ok=True)
NIMG,NAX,GRID=1000,500,14

def circles():
    centers=np.linspace(0,GRID-1,25); radii=np.linspace(.5,7.,8)
    yy,xx=np.mgrid[0:GRID,0:GRID]; masks=[]; rows=[]
    for cy in centers:
      for cx in centers:
       for ri,r in enumerate(radii):
        m=((xx-cx)**2+(yy-cy)**2<=r*r).astype(np.float32)
        if not m.any(): m[int(round(cy)),int(round(cx))]=1
        m/=m.sum(); masks.append(m); rows.append((len(rows),cx,cy,r,ri))
    masks=np.asarray(masks,np.float32); pd.DataFrame(rows,columns=['circle_id','center_x_grid','center_y_grid','radius_grid','radius_index']).to_csv(OUT/'circle_manifest_5000.csv',index=False); np.save(OUT/'circle_masks_5000_f32.npy',masks); return masks

def main():
    t=time.time(); masks=circles(); bank=np.load(CAV,allow_pickle=True); q=pd.read_csv(QUAL)
    concepts=bank['concepts'].astype(str); nodes=bank['nodes'].astype(str); offs=bank['offsets'].astype(int); axes=bank['cav_full'].astype(np.float32)
    chosen=[]
    for c in concepts:
      z=q[q.concept.eq(c)].sort_values(['heldout_accuracy_mean','repeat_axis_cosine_mean'],ascending=False).iloc[0]; chosen.append(str(z.node))
    by={n:np.flatnonzero(np.asarray(chosen)==n) for n in nodes}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); weights=ResNet50_Weights.IMAGENET1K_V2; model=resnet50(weights=weights).eval().to(device)
    modules={'stem':model.conv1,'res2_b1':model.layer1[0],'res2_b2':model.layer1[1],'res2_b3':model.layer1[2],'res3_b1':model.layer2[0],'res3_b2':model.layer2[1],'res3_b3':model.layer2[2],'res3_b4':model.layer2[3],'res4_b1':model.layer3[0],'res4_b2':model.layer3[1],'res4_b3':model.layer3[2],'res4_b4':model.layer3[3],'res4_b5':model.layer3[4],'res4_b6':model.layer3[5],'res5_b1':model.layer4[0],'res5_b2':model.layer4[1],'res5_b3':model.layer4[2]}
    acts={}; hooks=[]
    for n in set(chosen): hooks.append(modules[n].register_forward_hook(lambda m,i,o,key=n: acts.__setitem__(key,o.detach())))
    out=np.memmap(OUT/'axis500_circle5000_scores_f16.dat',mode='w+',dtype='float16',shape=(NAX,5000,NIMG))
    try:
      with zipfile.ZipFile(STIM) as ar, torch.inference_mode():
       tr=weights.transforms()
       for start in range(0,NIMG,16):
        stop=min(start+16,NIMG); batch=[]
        for ix in range(start,stop):
         with Image.open(io.BytesIO(ar.read(f'{ix+1:04d}.bmp'))) as im: batch.append(tr(im.convert('RGB')))
        acts.clear(); model(torch.stack(batch).to(device))
        for n,idx in by.items():
         if len(idx)==0: continue
         a=acts[n].float(); axis=torch.as_tensor(axes[idx,offs[list(nodes).index(n)]:offs[list(nodes).index(n)+1]],device=device)
         sm=torch.einsum('bchw,ac->bahw',a,axis); sm=F.interpolate(sm.reshape(len(idx)*(stop-start),1,sm.shape[-2],sm.shape[-1]),size=(GRID,GRID),mode='bilinear',align_corners=False).reshape(stop-start,len(idx),GRID,GRID)
         val=torch.einsum('bahw,chw->bac',sm,torch.as_tensor(masks,device=device)).permute(1,2,0).cpu().numpy(); out[idx,:,start:stop]=val.astype(np.float16)
        if stop%100==0 or stop==NIMG: print(f'{stop}/{NIMG}',flush=True)
    finally:
      for h in hooks: h.remove()
    out.flush(); pd.DataFrame({'axis_id':np.arange(NAX),'concept':concepts,'layer':chosen}).to_csv(OUT/'axis_manifest_500.csv',index=False); (OUT/'audit.txt').write_text(f'axes={NAX}; circles=5000; images={NIMG}; dtype=float16; seconds={time.time()-t:.1f}; layers=quality-best per concept; spatial=14x14 native CAV map; scoring=normalized circle mask dot product')
    print(f'Saved {out.filename}; size={out.nbytes/1024**3:.2f} GB; elapsed={time.time()-t:.1f}s')
if __name__=='__main__': main()
