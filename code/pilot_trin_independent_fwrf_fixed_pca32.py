"""Exploratory Triple-N layerwise fwRF/PCA32/CAV pilot.

The order follows the fwRF method: response-blind center-field PCA32,
training-fold covariance-based Gaussian-field selection, then fixed-field
ridge channel weights. All reported r values are five-fold out-of-fold.
No exploratory output is written to disk.
"""
import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torchvision, h5py
from PIL import Image

ROOT = Path(r'D:/Coding/BrainAI')
P = ROOT / 'ResNet50_Final_AllUnits_2026-08-24'
ZIP = ROOT / 'data/TripleN/V1/others/StimuliNNN.zip'
RESP = P / 'proxy_bank_10ms_all_windows/trin_all_units_10ms_responses.npy'
META = P / 'proxy_bank_10ms_all_windows/unit_metadata.csv'
CAV = P / 'tcav_broden500/broden500_native_channel_cav_bank.npz'
ROIS = ('V1','V1/V2','V2','V4','MF','MB','MO','CLC','LPP','PF','PITP','AF','AB','AO','AMC')
W = (70, 110, 150); LAYERS = ('res2', 'res3', 'res4', 'res5')
BATCH = 16; N = 1000; FOLDS = 5; SEED = 20260904
# Optional matched-unit control; None uses the per-ROI quality selection above.
FORCED_UNITS = None
# CAV node used for cosine comparison. The default preserves the historical
# pilot; "last" matches the stage output returned by fmap().
CAV_BLOCK = 'b1'


def fmap(m, x, layer):
    # torchvision ResNet naming: layer1=res2, layer2=res3, layer3=res4, layer4=res5.
    x = m.relu(m.bn1(m.conv1(x))); x = m.maxpool(x); x = m.layer1(x)
    if layer == 'res2': return x
    x = m.layer2(x)
    if layer == 'res3': return x
    x = m.layer3(x)
    if layer == 'res4': return x
    return m.layer4(x)


def make_fields(device):
    xs = np.linspace(-10 + 20/25/2, 10 - 20/25/2, 25, dtype=np.float32)
    sigmas = np.exp(np.linspace(np.log(.7), np.log(8), 8)).astype(np.float32)
    xx, yy, ss = np.meshgrid(xs, xs, sigmas, indexing='ij'); yy = -yy
    c = torch.linspace(-10 + 10/14, 10 - 10/14, 14, device=device)
    X, Y = torch.meshgrid(c, c, indexing='ij')
    fields = torch.stack([torch.exp(-((X-x)**2 + (Y-y)**2)/(2*s*s))
                          for x, y, s in zip(xx.ravel(), yy.ravel(), ss.ravel())])
    return fields, xs, sigmas


def read_batch(z, transform, start, end):
    return torch.stack([transform(Image.open(io.BytesIO(z.read(f'{j+1:04d}.bmp'))).convert('RGB'))
                        for j in range(start, end)])


def project_batch(a, fields, mu, sd, V):
    """Spatially pool candidate fields and project into the fixed PCA32 basis."""
    out = np.empty((a.shape[0], fields.shape[0], 32), np.float32)
    for g0 in range(0, fields.shape[0], 64):
        g1 = min(fields.shape[0], g0 + 64)
        pooled = torch.einsum('bchw,ghw->bgc', a, fields[g0:g1]).cpu().numpy()
        out[:, g0:g1] = ((pooled - mu[None, None, :]) / sd[None, None, :]) @ V
    return out


def ridge_fit(x, y):
    xm = x.mean(0); ym = float(y.mean()); xc = x - xm; yc = y - ym
    gram = xc.T @ xc; lam = .05 * np.trace(gram) / gram.shape[0]
    beta = np.linalg.solve(gram + np.eye(gram.shape[0], dtype=np.float32) * lam, xc.T @ yc)
    return beta, xm, ym


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); den = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return float(np.sum(a*b) / den) if den > 1e-12 else np.nan


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True; print('device', device, flush=True)
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    fields, xs, sigmas = make_fields(device)
    response = np.load(RESP, mmap_mode='r'); metadata = pd.read_csv(META)
    wi = np.asarray([(w + 20) // 10 for w in W], int)
    bankfiles = {'V1':'v1_proxy_bank.h5','V1/V2':'v2_proxy_bank.h5','V2':'v2_proxy_bank.h5',
                 'V4':'v4_proxy_bank.h5','MF':'middle_it_proxy_bank.h5','MB':'middle_it_proxy_bank.h5',
                 'MO':'middle_it_proxy_bank.h5','CLC':'middle_it_proxy_bank.h5','LPP':'middle_it_proxy_bank.h5',
                 'PF':'posterior_it_proxy_bank.h5','PITP':'posterior_it_proxy_bank.h5',
                 'AF':'anterior_it_proxy_bank.h5','AB':'anterior_it_proxy_bank.h5',
                 'AO':'anterior_it_proxy_bank.h5','AMC':'anterior_it_proxy_bank.h5'}
    banks = {}
    for filename in set(bankfiles.values()):
        with h5py.File(P / 'proxy_bank_10ms_all_windows' / filename, 'r') as h:
            banks[filename] = (h['unit_global'][:].astype(int), h['oof_r'][:])
    chosen = []
    if FORCED_UNITS is not None:
        chosen = [int(u) for u in FORCED_UNITS]
        print('forced units', chosen, flush=True)
    else:
        for roi in ROIS:
            ids, oof = banks[bankfiles[roi]]; lookup = {int(u): i for i, u in enumerate(ids)}
            units = metadata.loc[metadata.native_area.eq(roi), 'unit_global'].astype(int)
            ranked = sorted([(float(np.min(oof[wi, lookup[u]])), float(np.mean(oof[wi, lookup[u]])), u)
                             for u in units if u in lookup], reverse=True)
            chosen.extend([x[2] for x in ranked[:2]])
            print(roi, [(x[2], round(x[0], 3), round(x[1], 3)) for x in ranked[:2]], flush=True)
    target = np.stack([np.asarray(response[t, :, u], np.float32) for u in chosen for t in wi], axis=1)
    target_keys = [(u, w) for u in chosen for w in W]; n_targets = target.shape[1]
    rng = np.random.default_rng(SEED); fold_id = np.empty(N, int)
    for k, ix in enumerate(np.array_split(rng.permutation(N), FOLDS)): fold_id[ix] = k
    cav = np.load(CAV, allow_pickle=True); concepts = cav['concepts'].astype(str)
    nodes = cav['nodes'].astype(str); offsets = cav['offsets'].astype(int)

    for layer in LAYERS:
        channels = {'res2':256, 'res3':512, 'res4':1024, 'res5':2048}[layer]
        print(f'PCA {layer} ({channels} native channels)', flush=True)
        # Center x=y=0, sigma=8: response-blind PCA common to all folds and units.
        center_field = fields[12 * 25 * 8 + 12 * 8 + 7]; center = []
        with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
            for start in range(0, N, BATCH):
                end = min(N, start + BATCH)
                a = torch.nn.functional.interpolate(fmap(model, read_batch(z, transform, start, end).to(device), layer).float(), size=(14,14), mode='area')
                center.append(torch.einsum('bchw,hw->bc', a, center_field).cpu().numpy())
        center = np.concatenate(center); mu = center.mean(0); sd = center.std(0).clip(1e-5)
        _, _, Vt = np.linalg.svd((center - mu) / sd, full_matrices=False); V = Vt[:32].T

        # Fold-local sufficient statistics for independent fwRF field selection.
        G = fields.shape[0]; sums = np.zeros((FOLDS,G,32), np.float64)
        sums2 = np.zeros_like(sums); cross = np.zeros((FOLDS,G,32,n_targets), np.float64)
        sum_y = np.zeros((FOLDS,n_targets), np.float64); sum_y2 = np.zeros_like(sum_y)
        with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
            for start in range(0, N, BATCH):
                end = min(N, start + BATCH); ids = np.arange(start, end)
                a = torch.nn.functional.interpolate(fmap(model, read_batch(z, transform, start, end).to(device), layer).float(), size=(14,14), mode='area')
                p = project_batch(a, fields, mu, sd, V)
                for fold in range(FOLDS):
                    take = ids[fold_id[ids] != fold]
                    if len(take) == 0: continue
                    q = p[take - start]; yy = target[take]
                    sums[fold] += q.sum(0); sums2[fold] += (q*q).sum(0)
                    cross[fold] += np.einsum('bgc,bt->gct', q, yy)
                    sum_y[fold] += yy.sum(0); sum_y2[fold] += (yy*yy).sum(0)

        pred_oof = np.full((N, n_targets), np.nan, np.float32)
        cosine_folds = np.full((FOLDS, n_targets, len(concepts)), np.nan, np.float32)
        cosine_native_folds = np.full((FOLDS, n_targets, len(concepts)), np.nan, np.float32)
        rf_ids = np.empty((FOLDS, n_targets), np.int32)
        cav_node = (layer + '_b1') if CAV_BLOCK == 'b1' else {
            'res2': 'res2_b3', 'res3': 'res3_b4', 'res4': 'res4_b6', 'res5': 'res5_b3'
        }[layer]
        cv_start = int(np.flatnonzero(nodes == cav_node)[0])
        cav_native = cav['cav_full'][:, offsets[cv_start]:offsets[cv_start+1]].astype(np.float64)
        cav_pca = (cav_native / sd[None,:]) @ V
        for fold in range(FOLDS):
            train = np.flatnonzero(fold_id != fold); test = np.flatnonzero(fold_id == fold); ntr = float(len(train))
            mean = sums[fold] / ntr; var = (sums2[fold] - sums[fold]*sums[fold]/ntr).clip(1e-8)
            cov = cross[fold] - mean[:, :, None] * sum_y[fold][None, None, :]
            vy = (sum_y2[fold] - sum_y[fold]*sum_y[fold]/ntr).clip(1e-8)
            score = (cov*cov / (var[:,:,None] * vy[None,None,:])).sum(1)
            selected = score.argmax(0); rf_ids[fold] = selected; uniq = np.unique(selected)
            xsel = np.empty((N, len(uniq), 32), np.float32); pos = {int(g): j for j, g in enumerate(uniq)}
            with zipfile.ZipFile(ZIP) as z, torch.inference_mode():
                chosen_fields = fields[torch.as_tensor(uniq, device=device)]
                for start in range(0, N, BATCH):
                    end = min(N, start + BATCH)
                    a = torch.nn.functional.interpolate(fmap(model, read_batch(z, transform, start, end).to(device), layer).float(), size=(14,14), mode='area')
                    xsel[start:end] = project_batch(a, chosen_fields, mu, sd, V)
            for t in range(n_targets):
                x = xsel[:, pos[int(selected[t])]]; beta, xm, ym = ridge_fit(x[train], target[train, t])
                pred_oof[test, t] = (x[test] - xm) @ beta + ym
                den = np.linalg.norm(cav_pca, axis=1) * np.linalg.norm(beta)
                cosine_folds[fold, t] = np.divide(cav_pca @ beta, den, out=np.zeros(len(concepts)), where=den > 1e-12)
                native_axis = (V @ beta) / sd
                native_den = np.linalg.norm(cav_native, axis=1) * np.linalg.norm(native_axis)
                cosine_native_folds[fold, t] = np.divide(cav_native @ native_axis, native_den,
                                                          out=np.zeros(len(concepts)), where=native_den > 1e-12)
            print(f'{layer}: fold {fold+1}/{FOLDS} complete', flush=True)

        print(f'\n### {layer} (five-fold OOF r; Top-5 mean signed cosine; CAV={cav_node})', flush=True)
        for t, (unit, window) in enumerate(target_keys):
            r = corr(pred_oof[:,t], target[:,t]); co = np.nanmean(cosine_folds[:,t], axis=0)
            top = np.argsort(co)[::-1][:5]; rf = rf_ids[:,t]
            native_co = np.nanmean(cosine_native_folds[:,t], axis=0)
            native_top = np.argsort(native_co)[::-1][:5]
            unique_frac = len(np.unique(rf)) / FOLDS
            params = [(float(xs[g//(25*8)]), float(-xs[(g//8)%25]), float(sigmas[g%8])) for g in rf]
            print(f'{unit} {window}-{window+9} ms OOF_r={r:.3f} RF_unique={unique_frac:.2f}: '
                  + ' | '.join(f'{concepts[a]}({co[a]:+.3f})' for a in top), flush=True)
            print('  native_cos_top5: ' + ' | '.join(f'{concepts[a]}({native_co[a]:+.3f})' for a in native_top), flush=True)


if __name__ == '__main__': main()
