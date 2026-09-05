"""Rank 500 Broden concepts by MLP TCAV directional derivative."""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch

ROOT=Path(r"D:\Coding\BrainAI"); PROJECT=ROOT/"ResNet50_Final_AllUnits_2026-08-24"; CODE=PROJECT/"code"
sys.path.insert(0,str(CODE))
import run_trin_closed_loop_validation as folds
from pilot_tvsd_continuous_gaussian import continuous_gaussian

MAPS=ROOT/"cache/trin/original_fwrf_resnet50/block_pca8_14_tvsd_frozen/maps_f16.npy"
RESP=ROOT/"cache/trin/population_time10_area_starts_all_unittypes/middle_it_responses.npy"
META=ROOT/"cache/trin/population_time10_area_starts_all_unittypes/middle_it_metadata.npz"
BANK=PROJECT/"proxy_bank_10ms_all_windows/middle_it_proxy_bank.h5"
CAV=PROJECT/"tcav_broden500/broden500_native_channel_cav_bank.npz"
BASIS=ROOT/"cache/tvsd/original_fwrf_resnet50/block_pca8_14/pca_basis.npz"
UNIT, WINDOW, LAYER, EPOCHS, SEED = 3447, (150,159), "res4_b3", 120, 20260910

def train(x,y,seed):
    torch.manual_seed(seed); xm=x.mean(0); xs=x.std(0).clip(1e-5); xn=(x-xm)/xs
    ym,ys=float(y.mean()),float(y.std() or 1.0); yn=(y-ym)/ys
    m=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,1)).cuda()
    o=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-3); xt=torch.as_tensor(xn,device='cuda'); yt=torch.as_tensor(yn[:,None],device='cuda')
    for _ in range(EPOCHS):
        loss=((m(xt)-yt)**2).mean(); o.zero_grad(); loss.backward(); o.step()
    return m,xm,xs

def main():
    maps=np.load(MAPS,mmap_mode='r'); responses=np.load(RESP,mmap_mode='r'); windows=np.load(META,allow_pickle=True)['windows']; wi=int(np.flatnonzero(np.all(windows==WINDOW,axis=1))[0])
    with h5py.File(BANK,'r') as h:
        ui=int(np.flatnonzero(h['unit_global'][:]==UNIT)[0]); widx=int(np.flatnonzero(np.all(h['windows_ms'][:]==WINDOW,axis=1))[0]); cx,cy,sigma=h['spatial_parameters'][widx,ui]; gate=h['layer_gate'][widx,ui].astype(np.float32); nm=h['normalization_mean'][:].astype(np.float32); ns=h['normalization_std'][:].astype(np.float32)
    y=responses[wi,:,ui].astype(np.float32); mn=(maps.astype(np.float32)-nm[None,:,None,None])/ns[None,:,None,None]
    field=continuous_gaussian(torch.as_tensor([cx],device='cuda'),torch.as_tensor([cy],device='cuda'),torch.as_tensor([sigma],device='cuda'),14).reshape(14,14).cpu().numpy()
    X=np.einsum('ncpq,pq->nc',mn,field).astype(np.float32); scale=np.repeat(np.sqrt(np.clip(gate,1e-5,None)*17),8).astype(np.float32); X*=scale[None]
    # Native CAV at the proxy's maximum-gate layer -> the corresponding PCA8 block.
    cav=np.load(CAV,allow_pickle=True); concepts=cav['concepts'].astype(str); nodes=cav['nodes'].astype(str); offs=cav['offsets'].astype(int); li=int(np.flatnonzero(nodes==LAYER)[0]); a,b=int(offs[li]),int(offs[li+1]); native=cav['cav_full'][:,a:b].astype(np.float32)
    basis=np.load(BASIS,allow_pickle=True)[LAYER+'_basis'].astype(np.float32); axes=native@basis
    axes/=np.linalg.norm(axes,axis=1,keepdims=True).clip(1e-12); vec=np.zeros((len(axes),X.shape[1]),np.float32); vec[:,li*8:(li+1)*8]=axes*scale[li]
    signed=[]; absolute=[]; positive=[]; linear=[]
    for fi,(_,test) in enumerate(folds.fixed_folds(len(X))):
        tr=np.setdiff1d(np.arange(len(X)),test,assume_unique=True); m,xm,xs=train(X[tr],y[tr],SEED+fi)
        xc=X[tr]-X[tr].mean(0); lam=.05*np.trace(xc.T@xc)/X.shape[1]
        lb=np.linalg.solve(xc.T@xc+lam*np.eye(X.shape[1]), xc.T@(y[tr]-y[tr].mean()))
        linear.append(lb@vec.T)
        z=torch.as_tensor((X[test]-xm)/xs,device='cuda').requires_grad_(); out=m(z).sum(); g=torch.autograd.grad(out,z)[0].detach().cpu().numpy()/xs[None]
        d=g@vec.T; signed.append(d.mean(0)); absolute.append(np.abs(d).mean(0)); positive.append((d>0).mean(0)); print(f'fold {fi+1}/5 complete',flush=True)
    signed=np.stack(signed); absolute=np.stack(absolute); positive=np.stack(positive)
    table=pd.DataFrame({'concept':concepts,'mean_signed_directional_derivative':signed.mean(0),'mean_abs_directional_derivative':absolute.mean(0),'positive_fraction':positive.mean(0),'fold_signed_sd':signed.std(0,ddof=1)})
    print('\nTOP 5 POSITIVE MEAN SIGNED DERIVATIVE'); print(table.sort_values('mean_signed_directional_derivative',ascending=False).head(5).to_string(index=False))
    print('\nTOP 5 ABSOLUTE DERIVATIVE'); print(table.sort_values('mean_abs_directional_derivative',ascending=False).head(5).to_string(index=False))
    print('\nTOP 5 POSITIVE-FRACTION'); print(table.sort_values('positive_fraction',ascending=False).head(5).to_string(index=False))
    ltab=pd.DataFrame({'concept':concepts,'linear_directional_derivative':np.stack(linear).mean(0)})
    print('\nTOP 5 LINEAR DIRECTIONAL DERIVATIVE'); print(ltab.sort_values('linear_directional_derivative',ascending=False).head(5).to_string(index=False))

if __name__=='__main__': main()
