"""Small exploratory comparison of shared spaces for time-specific unit axes."""
from pathlib import Path
from itertools import combinations
import sys
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as base

ROOT = Path(r"D:\Coding\BrainAI")
FEATURES = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_full_14"
OUT = ROOT / "ResNet50_Final_AllUnits_2026-08-24" / "results" / "pilot_shared_space_axis_stability"
SEED = 20260904
N_REP = 5
SAMPLE = 1000
WINDOWS = np.array([0, 1, 3, 5])  # 50, 70, 110, 150 ms in the source pilot

def corr(a, b):
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
    return np.sum(a*b, 0) / np.sqrt(np.sum(a*a, 0)*np.sum(b*b, 0)).clip(1e-12)

def cosine(a, b):
    return np.sum(a*b, -1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).clip(1e-12)

def fit_axis(x, y, lam=0.05):
    xm, ym = x.mean(0), y.mean(0); xc, yc = x-xm, y-ym
    g = xc.T @ xc; d = g.shape[0]
    return np.linalg.solve(g + np.eye(d, dtype=np.float32)*(lam*np.trace(g)/d), xc.T @ yc)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y, _, chosen, _, _ = base.load_targets()
    y = y[:, WINDOWS, :5].astype(np.float32)
    maps = np.load(FEATURES / 'train_res4_f16.npy', mmap_mode='r')
    xs, ys, sig = base.candidates(); j = int(np.argmin((xs**2+ys**2)+(sig-8)**2))
    field = base.gaussian_mass_stack(xs[j:j+1], ys[j:j+1], sig[j:j+1], 14)[0]
    x = np.empty((len(maps), maps.shape[1]), np.float32)
    for a in range(0, len(maps), 512):
        b=min(a+512,len(maps)); x[a:b] = np.einsum('nchw,hw->nc', np.asarray(maps[a:b],np.float32), field)
    rng=np.random.default_rng(SEED); rows=[]; stored={}
    for rep in range(N_REP):
        idx=rng.choice(len(x), SAMPLE, replace=False)
        tr=np.sort(idx); te=np.setdiff1d(np.arange(len(x)), tr)[:1000]
        # Shared transforms are learned inside each fold, then all time windows fit independently.
        spaces={'raw1024':(x[tr],x[te],np.eye(x.shape[1],dtype=np.float32))}
        mu=x[tr].mean(0); _,_,v=np.linalg.svd(x[tr]-mu, full_matrices=False); v=v[:64].T
        spaces['pca64']=((x[tr]-mu)@v,(x[te]-mu)@v,v)
        # Shared supervised subspace from train-only cross-covariance SVD.
        # Each time window is still fitted independently after this transform.
        xc=x[tr]-x[tr].mean(0); yc=y[tr].reshape(len(tr),-1)-y[tr].reshape(len(tr),-1).mean(0)
        uu,_,_=np.linalg.svd(xc.T @ yc, full_matrices=False); q=uu[:,:16]
        spaces['cov16']=(xc@q,(x[te]-x[tr].mean(0))@q,q)
        for name,(xt,xv,proj) in spaces.items():
            axes=fit_axis(xt,y[tr].reshape(len(tr),-1)).T.reshape(len(WINDOWS),5,-1)
            # Convert every fold's coefficient vectors back to the common
            # original res4 coordinate system before cross-fold comparison.
            stored.setdefault(name, []).append(np.einsum('wud,fd->wuf', axes, proj))
            pred=np.stack([xv@axes[w].T + y[tr,w,:].mean() for w in range(len(WINDOWS))])
            for w in range(len(WINDOWS)):
                for u in range(5): rows.append({'rep':rep,'space':name,'window':int(WINDOWS[w]),'unit':int(chosen[u]),'test_r':float(corr(pred[w,:,u],y[te,w,u]))})
            for a,b in combinations(range(len(WINDOWS)),2):
                for u in range(5): rows.append({'rep':rep,'space':name,'window':f'{WINDOWS[a]}-{WINDOWS[b]}','unit':int(chosen[u]),'axis_cosine':float(cosine(axes[a][u:u+1],axes[b][u:u+1])[0])})
    import pandas as pd
    d=pd.DataFrame(rows)
    cross=[]
    for name, reps in stored.items():
        for w in range(len(WINDOWS)):
            for u in range(5):
                vals=[cosine(reps[a][w,u:u+1], reps[b][w,u:u+1])[0] for a,b in combinations(range(N_REP),2)]
                cross.append({'space':name,'window':int(WINDOWS[w]),'unit':int(chosen[u]),'cross_rep_axis_cosine':float(np.median(vals))})
    c=pd.DataFrame(cross); c.to_csv(OUT/'cross_fold_axis_stability.csv',index=False)
    d.to_csv(OUT/'detail.csv',index=False)
    print(d.groupby('space').agg(test_r=('test_r','median'), within_rep_window_axis_cosine=('axis_cosine','median')).to_string())
    print('\nCross-fold same-window axis cosine:')
    print(c.groupby('space').cross_rep_axis_cosine.median().to_string())
    print('saved',OUT/'detail.csv')
if __name__=='__main__': main()
