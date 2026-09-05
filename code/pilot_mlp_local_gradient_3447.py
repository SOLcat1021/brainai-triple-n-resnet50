"""Compare linear unit axis with image-dependent MLP local gradients."""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
import run_trin_closed_loop_validation as folds
from pilot_tvsd_continuous_gaussian import continuous_gaussian

MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"
RESP = ROOT / "cache" / "trin" / "population_time10_area_starts_all_unittypes" / "middle_it_responses.npy"
META = ROOT / "cache" / "trin" / "population_time10_area_starts_all_unittypes" / "middle_it_metadata.npz"
BANK = PROJECT / "proxy_bank_10ms_all_windows" / "middle_it_proxy_bank.h5"
UNIT, WINDOW = 3447, (150, 159)
RIDGE, SEED, EPOCHS = 0.05, 20260909, 120

def cos_rows(a, b):
    den = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a*b, axis=1), den, out=np.full(a.shape[0], np.nan), where=den>1e-12)

def q(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return f"median={np.median(x):.3f}, P5={np.percentile(x,5):.3f}, P95={np.percentile(x,95):.3f}"

def ridge(x, y):
    xm, ym = x.mean(0), y.mean(); xc=x-xm
    lam=RIDGE*np.trace(xc.T@xc)/x.shape[1]
    b=np.linalg.solve(xc.T@xc+lam*np.eye(x.shape[1]), xc.T@(y-ym))
    return b, xm

def train_mlp(x, y, seed):
    torch.manual_seed(seed)
    xm, xs=x.mean(0), x.std(0).clip(1e-5); xn=(x-xm)/xs
    ym, ys=float(y.mean()), float(y.std() or 1.0); yn=(y-ym)/ys
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,1)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-3)
    xt=torch.as_tensor(xn,device='cuda'); yt=torch.as_tensor(yn[:,None],device='cuda')
    for _ in range(EPOCHS):
        loss=((model(xt)-yt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model, xm, xs, ym, ys

def main():
    maps=np.load(MAPS,mmap_mode='r'); responses=np.load(RESP,mmap_mode='r')
    windows=np.load(META,allow_pickle=True)['windows']; wi=int(np.flatnonzero(np.all(windows==WINDOW,axis=1))[0])
    with h5py.File(BANK,'r') as h:
        ui=int(np.flatnonzero(h['unit_global'][:]==UNIT)[0]); widx=int(np.flatnonzero(np.all(h['windows_ms'][:]==WINDOW,axis=1))[0])
        cx,cy,sigma=h['spatial_parameters'][widx,ui]; gate=h['layer_gate'][widx,ui].astype(np.float32)
        norm_mean=h['normalization_mean'][:].astype(np.float32); norm_std=h['normalization_std'][:].astype(np.float32)
    y=responses[wi,:,ui].astype(np.float32)
    maps_n=(maps.astype(np.float32)-norm_mean[None,:,None,None])/norm_std[None,:,None,None]
    field=continuous_gaussian(torch.as_tensor([cx],device='cuda'),torch.as_tensor([cy],device='cuda'),torch.as_tensor([sigma],device='cuda'),14).reshape(14,14).cpu().numpy()
    X=np.einsum('ncpq,pq->nc',maps_n,field).astype(np.float32)
    X*=np.repeat(np.sqrt(np.clip(gate,1e-5,None)*17),8)[None]
    print(f"device=cuda; unit={UNIT}; window={WINDOW}; X={X.shape}; one-hidden-layer ReLU MLP")
    grad_axis_cos=[]; grad_pair_cos=[]; grad_norms=[]; linear_cos=[]
    for fi,(_,test) in enumerate(folds.fixed_folds(len(X))):
        train=np.setdiff1d(np.arange(len(X)),test,assume_unique=True)
        b_raw,_=ridge(X[train],y[train])
        xm=X[train].mean(0); xs=X[train].std(0).clip(1e-5)
        b_std=b_raw*xs
        model,mm,ss,ym,ys=train_mlp(X[train],y[train],SEED+fi)
        z=torch.as_tensor((X[test]-mm)/ss,device='cuda').requires_grad_()
        pred=model(z).sum(); g=torch.autograd.grad(pred,z)[0].detach().cpu().numpy()
        # Convert MLP gradients from standardized coordinates back to X coordinates.
        g_raw=g/ss[None]
        axis=np.broadcast_to(b_raw[None],g_raw.shape)
        c=cos_rows(g_raw,axis); grad_axis_cos.extend(c.tolist()); grad_norms.extend(np.linalg.norm(g_raw,axis=1).tolist())
        linear_cos.extend(np.ones(len(test)).tolist())
        for i in range(len(g_raw)):
            for j in range(i): grad_pair_cos.append(float(np.dot(g_raw[i],g_raw[j])/(np.linalg.norm(g_raw[i])*np.linalg.norm(g_raw[j])+1e-12)))
        print(f"fold {fi+1}/5: MLP gradient-vs-linear-axis {q(c)}; gradient norm {q(np.linalg.norm(g_raw,axis=1))}",flush=True)
    print("\nLINEAR: gradient-vs-linear-axis",q(linear_cos),"(theoretical exact cosine=1)")
    print("MLP: local gradient-vs-linear unit axis",q(grad_axis_cos))
    print("MLP: pairwise local-gradient cosine",q(grad_pair_cos))
    print("MLP: local gradient norm",q(grad_norms))

if __name__=='__main__': main()
