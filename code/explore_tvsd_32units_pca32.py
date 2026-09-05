from pathlib import Path
import sys
import numpy as np
import torch
from scipy.io import loadmat

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from extract_trin_resnet50_modelspace import gaussian_mass_stack

ROOT = Path(r"D:\Coding\BrainAI")
PCA8 = ROOT / "cache/tvsd/original_fwrf_resnet50/spatial_pca8_14/train_maps_f16.npy"
FULL = ROOT / "cache/tvsd/original_fwrf_resnet50/spatial_full_14/train_res4_f16.npy"
RAW = ROOT / "cache/tvsd/vit_features_targets/monkeyN_raw_windows_f16.npy"
MAT = Path(r"G:\BrainAI_Data\TVSD\monkeyN\THINGS_MUA_trials.mat")
MAP = Path(r"G:\BrainAI_Data\TVSD_repo_metadata\monkeyN\_logs\1024chns_mapping_20220105.mat")
N = 22248
WINDOWS = (70, 110, 150)
N_FOLDS = 5
SEED = 20260904

def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    d = np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0), d, out=np.full(d.shape, np.nan), where=d > 1e-8)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def project(path, idx, field):
    m = np.load(path, mmap_mode='r')
    out = np.empty((len(idx), m.shape[1]), np.float32)
    for a in range(0, len(idx), 64):
        b = min(a + 64, len(idx))
        z = torch.as_tensor(np.asarray(m[idx[a:b]], np.float32), device=DEVICE).flatten(2)
        out[a:b] = torch.einsum('ncp,p->nc', z, torch.as_tensor(field, device=DEVICE)).cpu().numpy()
    return out

def main():
    with __import__('h5py').File(MAT, 'r') as h:
        mat = np.asarray(h['ALLMAT'])
    raw = np.load(RAW, mmap_mode='r')
    train_pos = np.flatnonzero(mat[1] > 0)
    train_ids = mat[1, train_pos].astype(int) - 1
    order = np.argsort(train_ids)
    raw_train = np.asarray(raw[train_pos[order]], np.float32)
    mapping = np.asarray(loadmat(MAP, squeeze_me=True)['mapping']).reshape(-1).astype(int) - 1
    base = np.asarray(['V1']*512 + ['V4']*256 + ['IT']*256)
    roi = base[mapping]
    # Official TVSD order has no native posterior/middle/anterior IT labels.
    it = np.flatnonzero(roi == 'IT')
    labels = np.array(['V1']*1024, dtype=object)
    labels[roi == 'V4'] = 'V4'; labels[roi == 'IT'] = 'IT'
    labels[it[:len(it)//3]] = 'posterior IT'
    labels[it[len(it)//3:2*len(it)//3]] = 'middle IT'
    labels[it[2*len(it)//3:]] = 'anterior IT'
    # 20-ms source windows start at 30,50,...,170; requested 70/110/150 are indices 2/4/6.
    wi = [2, 4, 6]
    groups = [('V1', 10), ('V4', 10), ('posterior IT', 4), ('middle IT', 4), ('anterior IT', 4)]
    selected = []
    for name, k in groups:
        ch = np.flatnonzero(labels == name)
        score = raw_train[:, ch][:, :, wi].var(axis=(0, 2))
        selected.extend(ch[np.argsort(score)[::-1][:k]])
    selected = np.asarray(selected, int)
    print('selected channels:', [(int(c), str(labels[c])) for c in selected])
    y = raw_train[:, selected][:, :, wi]  # image x unit x time
    y = np.transpose(y, (0, 2, 1)).astype(np.float32)
    maps8 = np.load(PCA8, mmap_mode='r'); full = np.load(FULL, mmap_mode='r')
    fields_xy = np.linspace(-10+20/25/2, 10-20/25/2, 25).astype(np.float32)
    sig = np.exp(np.linspace(np.log(.7), np.log(8), 8)).astype(np.float32)
    xx, yy, ss = np.meshgrid(fields_xy, fields_xy, sig, indexing='ij')
    fields = gaussian_mass_stack(xx.ravel(), yy.ravel(), ss.ravel(), 14).reshape(-1, 196).astype(np.float32)
    rng = np.random.default_rng(SEED); sample = np.sort(rng.choice(N, 5000, replace=False))
    foldid = np.arange(5000) % N_FOLDS
    axes = np.zeros((N_FOLDS, 3, len(selected), 1024), np.float32)
    test_r = np.zeros((N_FOLDS, 3, len(selected)), np.float32)
    rf = np.zeros((N_FOLDS, 3, len(selected)), np.int32)
    for fi in range(N_FOLDS):
        tr = sample[foldid != fi]; te = sample[foldid == fi]
        f = np.asarray(maps8[tr], np.float32).reshape(len(tr), 40, -1)
        yt = y[tr]
        # Candidate selection is fold-local and uses PCA8 maps, as in the existing fwRF code.
        fm = f - f.mean(0); ym = yt - yt.mean(0)
        cov = np.einsum('ncp,ntu->cptu', fm, ym)
        ff = np.einsum('ncp,ncq->cpq', fm, fm)
        vy = np.sum(ym*ym, 0).clip(1e-8)
        vx = np.einsum('gp,cpq,gq->gc', fields, ff, fields).clip(1e-8)
        num = np.einsum('gp,cptu->gctu', fields, cov)
        energy = (num*num/(vx[:, :, None, None]*vy[None, None, :, :])).sum(1)
        # 40 PCA channels are not neural channels; candidate is shared per selected unit/time.
        sel = energy.argmax(0)  # time x unit
        for ti in range(3):
            for ui in range(len(selected)):
                g = int(sel[ti, ui]); rf[fi, ti, ui] = g
                z = project(FULL, tr, fields[g])
                mu, sd = z.mean(0), z.std(0).clip(1e-5)
                zz = (z-mu)/sd
                _, _, vh = torch.linalg.svd(torch.as_tensor(zz, device=DEVICE), full_matrices=False)
                v = vh[:16].T.cpu().numpy()
                z32 = zz @ v; xm, ym0 = z32.mean(0), yt[:, ti, ui].mean()
                xc = z32-xm; yc = yt[:, ti, ui]-ym0
                lam = .05*np.trace(xc.T@xc)/16
                beta = np.linalg.solve(xc.T@xc + np.eye(16)*lam, xc.T@yc)
                axes[fi,ti,ui] = (v@beta)/sd
                zte = project(FULL, te, fields[g])
                pred = ((zte-mu)/sd@v-xm)@beta + ym0
                test_r[fi,ti,ui] = corr(pred[:,None], y[te,ti,ui,None])[0]
        print('fold', fi+1, 'done', flush=True)
    print('\nSummary (median across units):')
    for ti, w in enumerate(WINDOWS):
        vals=[]
        for gi,(name,k) in enumerate(groups):
            sl=slice(sum(x[1] for x in groups[:gi]),sum(x[1] for x in groups[:gi+1]))
            pair=[]
            for u in range(sl.start,sl.stop):
                q=axes[:,ti,u]; q=q/np.linalg.norm(q,axis=1,keepdims=True)
                pair.extend([float(q[a]@q[b]) for a in range(5) for b in range(a+1,5)])
            vals.append((name, np.nanmedian(pair), np.nanmedian(test_r[:,ti,sl]), np.median([len(set(rf[:,ti,u]))/5 for u in range(sl.start,sl.stop)])))
        print(w, vals)
    print('\nPer-unit axis stability/test-r:')
    for gi,(name,k) in enumerate(groups):
        sl=slice(sum(x[1] for x in groups[:gi]),sum(x[1] for x in groups[:gi+1]))
        for u in range(sl.start,sl.stop):
            cos=[]
            for ti in range(3):
                q=axes[:,ti,u]; q=q/np.linalg.norm(q,axis=1,keepdims=True); cos += [q[a]@q[b] for a in range(5) for b in range(a+1,5)]
            print(name, int(selected[u]), 'cos', round(float(np.median(cos)),3), 'r', np.round(np.median(test_r[:,:,u],axis=0),3).tolist(), 'rfuniq', [round(float(len(set(rf[:,ti,u]))/5),2) for ti in range(3)])

if __name__ == '__main__': main()
