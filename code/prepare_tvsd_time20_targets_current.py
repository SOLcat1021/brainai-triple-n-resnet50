"""Rebuild the audited TVSD 20-ms targets from raw trials."""
from __future__ import annotations
import json, sys, zlib
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"G:\BrainAI_Data\TVSD")
META = Path(r"G:\BrainAI_Data\TVSD_repo_metadata")
OUT = ROOT / "cache" / "tvsd" / "vit_features_targets"
WINDOWS = np.arange(30, 190, 20)
BLOCK = 512

def roi_vector(monkey):
    base = np.asarray(["V1"]*512 + (["V4"]*256+["IT"]*256 if monkey == "N" else ["IT"]*320+["V4"]*192))
    mapping = np.asarray(loadmat(META / f"monkey{monkey}" / "_logs" / "1024chns_mapping_20220105.mat", squeeze_me=True)["mapping"], int) - 1
    return base[mapping], mapping

def read_chunk(dataset, start, stop):
    return np.asarray(dataset[:, start:stop, :], dtype=np.float32)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for monkey in ("N", "F"):
        out = OUT / f"monkey{monkey}_time20_mua_pilot_f16.npy"
        test_out = OUT / f"monkey{monkey}_test_time20_mua_pilot_f16.npy"
        if out.exists() and test_out.exists(): continue
        source = DATA / f"monkey{monkey}" / "THINGS_MUA_trials.mat"
        roi, mapping = roi_vector(monkey)
        with h5py.File(source, "r") as h:
            mua, mat, tb = h["ALLMUA"], np.asarray(h["ALLMAT"]), np.asarray(h["tb"]).reshape(-1)
            train_p = np.flatnonzero(mat[1] > 0); test_p = np.flatnonzero(mat[2] > 0)
            train_id = mat[1, train_p].astype(int)-1; test_id = mat[2, test_p].astype(int)-1
            day = mat[-1].astype(int); nday = int(day.max())
            raw_path = OUT / f"monkey{monkey}_raw_windows_f16.npy"
            raw_cache = np.lib.format.open_memmap(raw_path, mode="r+" if raw_path.exists() else "w+", dtype=np.float16, shape=(mat.shape[1],1024,8))
            sums=np.zeros((nday+1,1024)); sq=np.zeros_like(sums); cnt=np.zeros_like(sums)
            time=np.stack([(tb>s)&(tb<=s+20) for s in WINDOWS])
            need_scan = not (raw_path.exists() and raw_path.stat().st_size >= mat.shape[1]*1024*8*2)
            for a in range(0,mat.shape[1],BLOCK):
                b=min(a+BLOCK,mat.shape[1])
                if need_scan:
                    raw=read_chunk(mua,a,b)
                    raw_cache[a:b]=np.stack([np.nanmean(raw[t],axis=0) for t in time],axis=-1).astype(np.float16)
                else:
                    raw = None
                q=np.flatnonzero(mat[2,a:b]>0)
                for d in np.unique(day[a:b][q]):
                    z = (read_chunk(mua, a, b) if raw is None else raw)[:, q[day[a:b][q]==d], :]
                    sums[d] += np.nansum(z, axis=(0, 1))
                    sq[d] += np.nansum(z * z, axis=(0, 1))
                    cnt[d] += np.isfinite(z).sum(axis=(0, 1))
                if b%512==0 or b==mat.shape[1]: print(monkey,'raw',b,'/',mat.shape[1],flush=True)
            mean=sums/np.maximum(cnt,1); var=(sq-sums*sums/np.maximum(cnt,1))/np.maximum(cnt-1,1); scale=np.sqrt(np.maximum(var,1e-12))
            te=np.zeros((100,1024,8),np.float32); tc=np.zeros(100,int)
            for a in range(0,len(test_p),BLOCK):
                b=min(a+BLOCK,len(test_p)); p=test_p[a:b]; v=np.asarray(raw_cache[p],np.float32)
                for d in np.unique(day[p]):
                    q=np.flatnonzero(day[p]==d); v[q]=(v[q]-mean[d,:,None])/scale[d,:,None]
                for i,k in enumerate(test_id[a:b]): te[k]+=v[i]; tc[k]+=1
            te/=np.maximum(tc[:,None,None],1); np.save(test_out,te[:,mapping].astype(np.float16))
            tr=np.lib.format.open_memmap(out,mode='w+',dtype=np.float16,shape=(22248,1024,8))
            for a in range(0,len(train_p),BLOCK):
                b=min(a+BLOCK,len(train_p)); p=train_p[a:b]; v=np.asarray(raw_cache[p],np.float32)
                for d in np.unique(day[p]):
                    q=np.flatnonzero(day[p]==d); v[q]=(v[q]-mean[d,:,None])/scale[d,:,None]
                tr[train_id[a:b]]=v[:,mapping].astype(np.float16)
                if b%512==0 or b==len(train_p): tr.flush(); print(monkey,'train',b,'/',len(train_p),flush=True)
    (OUT/'target_summary.json').write_text(json.dumps({'windows_ms':[(int(s),int(s+19)) for s in WINDOWS],'source':'THINGS_MUA_trials official per-day normalization'},indent=2),encoding='utf-8')

if __name__=='__main__': main()
