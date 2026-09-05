"""Train 1000 Broden concept axes per ResNet stage in fixed PCA32 spaces.

The PCA bases are fit only from response-blind Broden ResNet GAP features.
Concept positives/negatives are transformed by the same stage-specific basis,
then a linear TCAV classifier is trained in that 32-D space.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:/Coding/BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
SRC = PROJECT / "tcav_broden500"
OUT = PROJECT / "tcav_broden1000_pca32"
NODES = ("res2_b3", "res3_b4", "res4_b6", "res5_b3")
SLICES = {"res2_b3": (512, 768), "res3_b4": (2560, 3072),
          "res4_b6": (7680, 8704), "res5_b3": (13120, 15168)}
FEATURES = SRC / "broden_resnet50_native_gap_f16.npy"
REPEATS = 5

def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

def classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3,
                         random_state=seed)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    concepts = pd.read_csv(SRC / "broden500_concepts.csv")
    # Reconstruct the full Broden label list and the same deterministic policy.
    labels = pd.read_csv(ROOT / "data" / "Broden" / "label.csv")
    labels["primary_category"] = labels.category.astype(str).str.split("(").str[0]
    labels = labels[labels.primary_category.isin(("color", "object", "part", "material", "scene", "texture"))]
    # The 1000-axis request requires relaxing the original 500-axis frequency
    # gate; keep the deterministic frequency ranking and expose counts below.
    labels = labels[labels.name.notna()].copy()
    labels = labels.sort_values(["frequency", "number"], ascending=[False, True])
    # Keep the established 500 concepts, then add 500 concepts by an explicit
    # seeded random draw from the remaining Broden labels (no frequency winner
    # selection for the exploratory expansion).
    established = set(pd.read_csv(SRC / "broden500_concepts.csv").number.astype(int))
    base = labels[labels.number.isin(established)].copy()
    if len(base) != 500:
        raise RuntimeError(f"Expected 500 established concepts, found {len(base)}")
    remaining = labels[~labels.number.isin(established)]
    extra = remaining.sample(n=500, random_state=args.seed)
    labels = pd.concat([base, extra], ignore_index=True)
    if len(labels) != 1000:
        raise RuntimeError(f"Only {len(labels)} concepts meet frequency threshold")
    labels.insert(0, "concept_index", np.arange(1000, dtype=np.int32))
    labels = labels[["concept_index", "number", "name", "category", "primary_category", "frequency", "coverage", "syns"]]
    # Presence is reusable because the original manifest scans all labels; build
    # the 1000-label CSR once by reading the Broden annotation table.
    presence_path = OUT / "presence_1000.npz"
    if not presence_path.exists() or args.force:
        old = np.load(SRC / "broden500_presence_csr.npz")
        old_concepts = pd.read_csv(SRC / "broden500_concepts.csv")
        old_map = {int(n): i for i, n in enumerate(old_concepts.number)}
        # For labels beyond 500, scan Broden index cells; this is response-blind.
        from PIL import Image
        from collections import defaultdict
        broden = ROOT / "data" / "Broden"
        images = pd.read_csv(SRC / "broden_image_manifest.csv")
        wanted = set(labels.number.astype(int)); by_num = defaultdict(list)
        def nums(cell):
            if pd.isna(cell) or not str(cell).strip(): return set()
            vals = [x.strip() for x in str(cell).split(";") if x.strip()]
            try: return {int(float(x)) for x in vals}
            except ValueError:
                out = set()
                for p in vals:
                    with Image.open(broden / "images" / p) as im:
                        a = np.asarray(im.convert("RGB"), dtype=np.uint16)
                    out.update(int(x) for x in np.unique(a[...,0] + 256*a[...,1]) if int(x))
                return out
        if old_map and all(int(n) in old_map for n in labels.number.head(500)):
            for row in labels.itertuples(index=False):
                if int(row.number) in old_map:
                    i = old_map[int(row.number)]; by_num[int(row.number)] = old["indices"][old["indptr"][i]:old["indptr"][i+1]].tolist()
        raw = pd.read_csv(broden / "index.csv")
        for i, row in raw.iterrows():
            for col in ("color", "object", "part", "material", "scene", "texture"):
                by_num.update({})
                for n in nums(row.get(col, "")) & wanted:
                    by_num[n].append(int(i))
        indptr=[0]; indices=[]
        for n in labels.number:
            vals=sorted(set(by_num[int(n)])); indices.extend(vals); indptr.append(len(indices))
        labels["observed_image_count"]=[indptr[i+1]-indptr[i] for i in range(1000)]
        np.savez_compressed(presence_path, indptr=np.asarray(indptr,np.int64), indices=np.asarray(indices,np.int32), n_images=np.asarray([len(images)],np.int64))
    presence=np.load(presence_path); features=np.load(FEATURES,mmap_mode="r")
    if features.shape[0] != int(presence["n_images"][0]): raise RuntimeError("feature/manifest mismatch")
    bases={}
    basis_path=OUT/"pca32_bases.npz"
    if basis_path.exists() and not args.force:
        b=np.load(basis_path); bases={n:(b[f"mean_{n}"],b[f"scale_{n}"],b[f"components_{n}"],b[f"explained_{n}"]) for n in NODES}
    else:
        for node in NODES:
            a,bx=SLICES[node]; X=np.asarray(features[:,a:bx],np.float32); mu=X.mean(0); sd=X.std(0).clip(1e-6); Z=(X-mu)/sd
            pca=PCA(n_components=32, svd_solver="randomized", random_state=args.seed).fit(Z)
            bases[node]=(mu,sd,pca.components_.astype(np.float32),pca.explained_variance_ratio_.astype(np.float32))
            print(f"PCA {node}: variance32={pca.explained_variance_ratio_.sum():.4f}",flush=True)
        np.savez_compressed(basis_path, **{f"{k}_{n}":v for n,(mu,sd,v,e) in bases.items() for k,v in (("mean",mu),("scale",sd),("components",v),("explained",e))})
    shards=OUT/"axis_shards"; shards.mkdir(exist_ok=True)
    indptr,indices=presence["indptr"],presence["indices"]; all_idx=np.arange(features.shape[0],dtype=np.int32)
    for ci,row in labels.iloc[max(0,args.start):min(1000,args.stop)].iterrows():
        out=shards/f"{ci:04d}_{safe(str(row['name']))}.npz"
        if out.exists() and not args.force: continue
        pos=np.asarray(indices[indptr[ci]:indptr[ci+1]],np.int32); mask=np.zeros(len(all_idx),bool); mask[pos]=1; neg=all_idx[~mask]; n=min(len(pos),len(neg)); rng=np.random.default_rng(args.seed+ci*100003)
        full=np.zeros((4,32),np.float32); rep=np.zeros((args.repeats,4,32),np.float16); acc=np.zeros((args.repeats,4),np.float32)
        for li,node in enumerate(NODES):
            mu,sd,V,_=bases[node]; a,bx=SLICES[node]; X=(np.asarray(features[:,a:bx],np.float32)-mu)/sd@V.T
            fp=rng.choice(pos,n,False); fn=rng.choice(neg,n,False); ix=np.r_[fp,fn]; yy=np.r_[np.ones(n,np.int8),np.zeros(n,np.int8)]
            full[li]=classifier(args.seed+ci*100003+li).fit(X[ix],yy).coef_[0]; full[li]/=max(np.linalg.norm(full[li]),1e-12)
            for r in range(args.repeats):
                rp=rng.choice(pos,n,False); rn=rng.choice(neg,n,False); sx=np.r_[rp,rn]; sy=np.r_[np.ones(n,np.int8),np.zeros(n,np.int8)]
                tr,te,ty,vy=train_test_split(sx,sy,test_size=.33,stratify=sy,random_state=args.seed+ci*100003+r)
                q=classifier(args.seed+ci*100003+r*101+li).fit(X[tr],ty); rep[r,li]=q.coef_[0]/max(np.linalg.norm(q.coef_[0]),1e-12); acc[r,li]=accuracy_score(vy,q.predict(X[te]))
        np.savez_compressed(out, concept_index=np.asarray([ci]), concept_name=np.asarray([str(row.name)]), cav_pca32=full, cav_repeated=rep, heldout_accuracy=acc, positive_count=np.asarray([len(pos)]), balanced_count=np.asarray([n]))
        if ci%25==0: print(f"axes {ci+1}/1000",flush=True)
    rows=[]; bank=np.zeros((1000,4,32),np.float32); repeats=np.zeros((1000,args.repeats,4,32),np.float16)
    for ci,row in labels.iterrows():
        sh=np.load(next((shards/f"{ci:04d}_{safe(str(row['name']))}.npz" for _ in [0])))
        bank[ci]=sh["cav_pca32"]; repeats[ci]=sh["cav_repeated"]
        for li,node in enumerate(NODES):
            v=sh["cav_repeated"][:,li].astype(np.float32); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12); c=v@v.T; upper=c[np.triu_indices(args.repeats,1)]
            rows.append({"concept_index":ci,"concept":row.name,"node":node,"pca_dim":32,"positive_count":int(indptr[ci+1]-indptr[ci]),"balanced_count":int(sh["balanced_count"][0]),"heldout_accuracy_mean":float(sh["heldout_accuracy"][:,li].mean()),"repeat_axis_cosine_mean":float(upper.mean())})
    np.savez_compressed(OUT/"pca32_concept_axis_bank_1000.npz", concepts=labels.name.astype(str).to_numpy(), nodes=np.asarray(NODES), cav_pca32=bank, cav_repeated=repeats)
    pd.DataFrame(rows).to_csv(OUT/"pca32_concept_axis_quality_1000.csv",index=False)
    labels.to_csv(OUT/"concepts_1000.csv",index=False)
    (OUT/"METHOD_AUDIT.json").write_text(json.dumps({"concept_count":1000,"nodes":NODES,"pca_dim":32,"response_blind":True,"source_features":str(FEATURES),"classifier":"SGDClassifier(alpha=.01)","repeats":args.repeats,"native_to_pca":"z=V32 @ ((x-mu)/sigma)"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"DONE: {OUT}",flush=True)

if __name__ == "__main__": main()
