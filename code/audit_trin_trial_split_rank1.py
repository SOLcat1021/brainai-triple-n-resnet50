"""Strict trial-split test of temporal rank in the released TriN responses.

This analysis starts from ``raster_matrix_img`` rather than the condition-mean
``response_matrix_img`` cache.  Raw spikes are summed in the project's frozen
official non-overlapping 20-ms windows. Each image contributes an equal number of
trials to two independent halves (two or three per half, fixed within session).
Image folds, not trial halves, define the outer cross-validation for the
low-rank model.

No visual model, PCA feature, CAV, concept label, or proxy prediction is used.
The script is intentionally report-only and writes no result files.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import beta, pearsonr


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
RAW = Path(r"D:\Coding\BrainAI\data\TripleN\V1\Raw\H5FILES")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OLD_COHORT = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "selected_units.csv"

ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_COUNTS = {"V1": 12, "V2": 12, "V4": 12, "posterior IT": 12,
              "middle IT": 40, "anterior IT": 25}
SEED = 20260905
N_SPLITS = 30
N_IMAGE_FOLDS = 5
REL_FULL_THRESHOLD = 0.40
TIME_ZERO_INDEX = 49
WINDOWS = np.asarray([
    (-20, -1), (30, 49), (50, 69), (70, 89), (90, 109),
    (110, 129), (130, 149), (150, 169), (170, 189),
], dtype=int)


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlations without constructing large matrices."""
    a = np.asarray(a, float) - np.asarray(a, float).mean(axis=0, keepdims=True)
    b = np.asarray(b, float) - np.asarray(b, float).mean(axis=0, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    return np.divide(np.sum(a * b, axis=0), den, out=np.full(a.shape[1], np.nan), where=den > 0)


def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float) - np.asarray(a, float).mean(axis=0, keepdims=True)
    b = np.asarray(b, float) - np.asarray(b, float).mean(axis=0, keepdims=True)
    an = np.sqrt(np.sum(a * a, axis=0)); bn = np.sqrt(np.sum(b * b, axis=0))
    den = an[:, None] * bn[None, :]
    return np.divide(a.T @ b, den, out=np.full(den.shape, np.nan), where=den > 0)


def fisher_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, float)
    clipped = np.clip(values, -0.999999, 0.999999)
    z = np.arctanh(clipped)
    count = np.sum(np.isfinite(z), axis=axis)
    mean = np.divide(np.nansum(z, axis=axis), count,
                     out=np.full(np.asarray(count).shape, np.nan, float), where=count > 0)
    return np.tanh(mean)


def spearman_brown(r: np.ndarray) -> np.ndarray:
    """Reliability after averaging the two equal independent trial halves."""
    r = np.asarray(r, float)
    return np.divide(2 * r, 1 + r, out=np.full_like(r, np.nan), where=np.abs(1 + r) > 1e-12)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    q = p[finite]
    order = np.argsort(q)
    ranked = q[order] * len(q) / np.arange(1, len(q) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    out[finite] = restored
    return out


def session_paths(session: int, source_h5: str) -> tuple[Path, Path]:
    h5_path = Path(source_h5)
    if not h5_path.exists():
        found = sorted(RAW.glob(f"ses{session:02d}_*.h5"))
        if len(found) != 1:
            raise FileNotFoundError(f"session {session}: expected one H5, found {found}")
        h5_path = found[0]
    info = h5_path.with_name(h5_path.stem + "_info.mat")
    if not info.exists():
        raise FileNotFoundError(info)
    return h5_path, info


def load_session_counts(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int], int]:
    """Return image trial table and 20-ms spike counts: trial x time x selected unit."""
    session = int(group.session.iloc[0])
    h5_path, info_path = session_paths(session, str(group.source_h5.iloc[0]))
    image_id = np.asarray(loadmat(info_path, squeeze_me=True)["img_idx"], int).ravel()
    unit_index = group.unit_index_zero_based.to_numpy(int)
    if not np.all(np.diff(unit_index) >= 0):
        raise ValueError("unit indices must be sorted before HDF5 fancy indexing")
    with h5py.File(h5_path, "r") as handle:
        source = handle["raster_matrix_img"]
        if source.shape[1] != len(image_id):
            raise RuntimeError(f"session {session}: img_idx/raster trial count mismatch")
        slab = np.asarray(source[TIME_ZERO_INDEX - 20:TIME_ZERO_INDEX + 190, :, unit_index], np.float32)
    counts = np.stack([
        slab[start + 20:end + 21].sum(axis=0) for start, end in WINDOWS
    ], axis=1)
    repeat_counts = np.bincount(image_id, minlength=1001)[1:1001]
    table = np.full((1000, int(repeat_counts.max())), -1, int)
    for image in range(1, 1001):
        trials = np.flatnonzero(image_id == image)
        if len(trials) == 0:
            raise RuntimeError(f"session {session}, image {image}: no trials")
        table[image - 1, :len(trials)] = trials
    eligible = []
    for candidate in (3, 2):
        fraction = float(np.mean(repeat_counts >= 2 * candidate))
        if fraction >= 0.95:
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError(f"session {session}: fewer than 95% of images support 2-vs-2 trials")
    half_size = max(eligible)
    keep = repeat_counts >= 2 * half_size
    return (table[keep], counts, half_size,
            (int(repeat_counts.min()), int(repeat_counts.max())), int(keep.sum()))


def split_trials(table: np.ndarray, counts: np.ndarray, half_size: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_images = len(table)
    a_idx = np.empty((n_images, half_size), int); b_idx = np.empty((n_images, half_size), int)
    for image in range(n_images):
        available = table[image, table[image] >= 0]
        chosen = rng.permutation(available)[:2 * half_size]
        a_idx[image], b_idx[image] = chosen[:half_size], chosen[half_size:]
    a = counts[a_idx].mean(axis=1)
    b = counts[b_idx].mean(axis=1)
    return a, b


def conditional_rank2_threshold(n_time: int) -> tuple[float, float]:
    """Null after conditioning on a perfectly shared first temporal mode.

    Two random second modes live in the (T-1)-dimensional orthogonal complement.
    Their squared cosine follows Beta(1/2, (T-2)/2).  The corresponding rank-2
    subspace score is the mean of the shared first-mode cosine squared (one)
    and the random second-mode cosine squared.
    """
    dimension = n_time - 1
    cosine_sq_95 = float(beta.ppf(0.95, 0.5, (dimension - 1) / 2))
    return float(np.sqrt(cosine_sq_95)), float(0.5 * (1.0 + cosine_sq_95))


def temporal_basis_stability(full: np.ndarray) -> tuple[float, float, float]:
    """Disjoint-image stability of the first mode, second mode, and rank-2 subspace."""
    order = np.random.default_rng(SEED + 991).permutation(len(full))
    bases = []
    for idx in (order[:500], order[500:]):
        z = full[idx] - full[idx].mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        bases.append(vt)
    a, b = bases
    mode1 = float(abs(a[0] @ b[0]))
    mode2 = float(abs(a[1] @ b[1])) if a.shape[0] >= 2 else np.nan
    singular = np.linalg.svd(a[:2] @ b[:2].T, compute_uv=False)
    subspace2 = float(np.mean(singular ** 2))
    return mode1, mode2, subspace2


def rank_cv(full: np.ndarray, split_pairs: list[tuple[np.ndarray, np.ndarray]],
            max_rank: int = 5) -> dict[str, np.ndarray | float]:
    """Outer image-CV fraction of independent-repeat covariance captured by rank k."""
    n_images = len(full)
    image_folds = np.array_split(np.random.default_rng(SEED + 313).permutation(n_images), N_IMAGE_FOLDS)
    fractions = []
    coefficient_r = []
    coefficient_p = []
    component_products = []
    crossselected_shares = []
    concentration_fractions = np.asarray([0.01, 0.05, 0.10, 0.20])
    for split_i, (a, b) in enumerate(split_pairs):
        numer = np.zeros(max_rank, float); denominator = 0.0
        ca_all = [[] for _ in range(max_rank)]
        cb_all = [[] for _ in range(max_rank)]
        products_by_image = np.full((n_images, max_rank), np.nan, float)
        coefficient_a_by_image = np.full((n_images, max_rank), np.nan, float)
        coefficient_b_by_image = np.full((n_images, max_rank), np.nan, float)
        fold_by_image = np.full(n_images, -1, int)
        for fold_i, test in enumerate(image_folds):
            train = np.setdiff1d(np.arange(n_images), test, assume_unique=False)
            z = full[train] - full[train].mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            aa = a[test] - a[test].mean(axis=0, keepdims=True)
            bb = b[test] - b[test].mean(axis=0, keepdims=True)
            denominator += float(np.sum(aa * bb))
            for k in range(max_rank):
                va = aa @ vt[k]
                vb = bb @ vt[k]
                numer[k:] += float(np.sum(va * vb))
                products_by_image[test, k] = va * vb
                coefficient_a_by_image[test, k] = va
                coefficient_b_by_image[test, k] = vb
                fold_by_image[test] = fold_i
                # Within-fold standardization prevents arbitrary fold scale from
                # inflating the pooled coefficient correlation.
                va = (va - va.mean()) / max(float(va.std()), 1e-12)
                vb = (vb - vb.mean()) / max(float(vb.std()), 1e-12)
                ca_all[k].append(va); cb_all[k].append(vb)
        fractions.append(numer / denominator if denominator > 0 else np.full(max_rank, np.nan))
        rr, pp = [], []
        for k in range(max_rank):
            r, p = pearsonr(np.concatenate(ca_all[k]), np.concatenate(cb_all[k]))
            rr.append(float(r)); pp.append(float(p))
        coefficient_r.append(rr); coefficient_p.append(pp)
        component_products.append(products_by_image)
        shares = np.full((max_rank, len(concentration_fractions)), np.nan, float)
        for k in range(max_rank):
            product_k = products_by_image[:, k]
            total_k = float(np.nansum(product_k))
            if total_k <= 0:
                continue
            for qi, fraction in enumerate(concentration_fractions):
                numerator_a = 0.0; numerator_b = 0.0
                for fold_i in range(N_IMAGE_FOLDS):
                    idx = np.flatnonzero(fold_by_image == fold_i)
                    count = max(1, int(np.ceil(len(idx) * fraction)))
                    choose_a = idx[np.argsort(np.abs(coefficient_a_by_image[idx, k]))[-count:]]
                    choose_b = idx[np.argsort(np.abs(coefficient_b_by_image[idx, k]))[-count:]]
                    numerator_a += float(np.nansum(product_k[choose_a]))
                    numerator_b += float(np.nansum(product_k[choose_b]))
                shares[k, qi] = 0.5 * (numerator_a + numerator_b) / total_k
        crossselected_shares.append(shares)
    return {
        "fractions": np.asarray(fractions),
        "coefficient_r": np.asarray(coefficient_r),
        "coefficient_p_primary": np.asarray(coefficient_p)[0],
        "component_products": np.asarray(component_products),
        "crossselected_shares": np.asarray(crossselected_shares),
        "concentration_fractions": concentration_fractions,
    }


def contribution_concentration(products: np.ndarray, prefix: str) -> dict[str, float]:
    """Concentration of one mode's repeatable covariance across held-out images."""
    per_image = np.nanmean(np.asarray(products, float), axis=0)
    positive = np.maximum(per_image, 0)
    total = float(positive.sum())
    if total <= 0:
        return {f"{prefix}_{suffix}": np.nan for suffix in [
            "top1_share", "top5_share", "top10_share", "top20_share",
            "effective_image_fraction", "images_for_half_mass_fraction",
            "positive_image_fraction", "consistently_positive_fraction",
        ]}
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered)
    n = len(positive)
    def top_share(fraction: float) -> float:
        count = max(1, int(np.ceil(n * fraction)))
        return float(ordered[:count].sum() / total)
    effective_n = total ** 2 / max(float(np.sum(positive ** 2)), 1e-12)
    half_n = int(np.searchsorted(cumulative, 0.5 * total) + 1)
    sign_support = np.mean(np.asarray(products, float) > 0, axis=0)
    return {
        f"{prefix}_top1_share": top_share(0.01),
        f"{prefix}_top5_share": top_share(0.05),
        f"{prefix}_top10_share": top_share(0.10),
        f"{prefix}_top20_share": top_share(0.20),
        f"{prefix}_effective_image_fraction": float(effective_n / n),
        f"{prefix}_images_for_half_mass_fraction": float(half_n / n),
        f"{prefix}_positive_image_fraction": float(np.mean(per_image > 0)),
        f"{prefix}_consistently_positive_fraction": float(np.mean(sign_support >= 0.75)),
    }


def analyze_unit(full: np.ndarray, split_pairs: list[tuple[np.ndarray, np.ndarray]],
                 roi: str) -> tuple[dict, list[dict]]:
    rel_repeats = np.stack([corr_columns(a, b) for a, b in split_pairs])
    rel_half = fisher_mean(rel_repeats)
    rel_full = spearman_brown(rel_half)
    physiological = WINDOWS[:, 0] >= 0
    reliable = physiological & (rel_full >= REL_FULL_THRESHOLD)

    window_rows = [{
        "window_start_ms": int(WINDOWS[t, 0]),
        "rel_half": float(rel_half[t]),
        "rel_full_sb": float(rel_full[t]),
        "in_physiological_interval": bool(physiological[t]),
        "used_for_rank": bool(reliable[t]),
    } for t in range(len(WINDOWS))]

    row = {
        "n_reliable_windows": int(reliable.sum()),
        "median_rel_half_reliable": float(np.nanmedian(rel_half[reliable])) if reliable.any() else np.nan,
        "median_rel_full_reliable": float(np.nanmedian(rel_full[reliable])) if reliable.any() else np.nan,
    }
    if reliable.sum() < 3:
        row.update({k: np.nan for k in [
            "raw_cross_time", "corrected_cross_time", "independent_corrected_cross_time",
            "rank1_fraction", "rank2_fraction", "rank3_fraction", "rank2_increment",
            "rank3_increment", "mode1_stability", "mode2_stability", "rank2_subspace_stability",
            "mode2_coefficient_r", "mode2_p_primary", "rank2_null95",
            "mode2_top1_share", "mode2_top5_share", "mode2_top10_share",
            "mode2_top20_share", "mode2_effective_image_fraction",
            "mode2_images_for_half_mass_fraction", "mode2_positive_image_fraction",
            "mode2_consistently_positive_fraction",
            "rank1_top1_share", "rank1_top5_share", "rank1_top10_share",
            "rank1_top20_share", "rank1_effective_image_fraction",
            "rank1_images_for_half_mass_fraction", "rank1_positive_image_fraction",
            "rank1_consistently_positive_fraction",
            "rank1_cvtop1_share", "rank1_cvtop5_share", "rank1_cvtop10_share", "rank1_cvtop20_share",
            "mode2_cvtop1_share", "mode2_cvtop5_share", "mode2_cvtop10_share", "mode2_cvtop20_share",
        ]})
        return row, window_rows

    use = np.flatnonzero(reliable)
    full_use = full[:, use]
    # The raw full-mean matrix and its Spearman-Brown reliability use exactly
    # the same 2k trials.  Averaging Fisher z over randomized partitions also
    # rotates which trial is omitted when an image has an odd repeat count.
    raw = fisher_mean(np.stack([
        corr_matrix(0.5 * (a[:, use] + b[:, use]), 0.5 * (a[:, use] + b[:, use]))
        for a, b in split_pairs
    ]))
    corrected = raw / np.sqrt(rel_full[use, None] * rel_full[None, use])
    off = np.triu_indices(len(use), 1)

    independent_matrices = []
    for a, b in split_pairs:
        aa, bb = a[:, use], b[:, use]
        cross = 0.5 * (corr_matrix(aa, bb) + corr_matrix(bb, aa))
        rhalf = corr_columns(aa, bb)
        product = rhalf[:, None] * rhalf[None, :]
        denom = np.sqrt(np.maximum(product, 0))
        independent_matrices.append(np.divide(
            cross, denom, out=np.full_like(cross, np.nan), where=product > 0))
    independent_corrected = np.nanmedian(np.stack(independent_matrices), axis=0)

    max_rank = min(5, len(use))
    cv = rank_cv(full_use, [(a[:, use], b[:, use]) for a, b in split_pairs], max_rank=max_rank)
    concentration = {
        **contribution_concentration(cv["component_products"][:, :, 0], "rank1"),
        **contribution_concentration(cv["component_products"][:, :, 1], "mode2"),
    }
    cv_shares = np.nanmedian(cv["crossselected_shares"], axis=0)
    for mode, prefix in ((0, "rank1"), (1, "mode2")):
        for qi, label in enumerate((1, 5, 10, 20)):
            concentration[f"{prefix}_cvtop{label}_share"] = float(cv_shares[mode, qi])
    fractions = np.nanmedian(cv["fractions"], axis=0)
    mode1, mode2, subspace2 = temporal_basis_stability(full_use)
    mode2_null95, null95 = conditional_rank2_threshold(len(use))
    row.update({
        "raw_cross_time": float(np.nanmedian(raw[off])),
        "corrected_cross_time": float(np.nanmedian(corrected[off])),
        "independent_corrected_cross_time": float(np.nanmedian(independent_corrected[off])),
        "corrected_gt1_fraction": float(np.mean(np.abs(corrected[off]) > 1)),
        "rank1_fraction": float(fractions[0]),
        "rank2_fraction": float(fractions[1]),
        "rank3_fraction": float(fractions[2]) if max_rank >= 3 else np.nan,
        "rank2_increment": float(fractions[1] - fractions[0]),
        "rank3_increment": float(fractions[2] - fractions[1]) if max_rank >= 3 else np.nan,
        "mode1_stability": mode1,
        "mode2_stability": mode2,
        "mode2_null95": mode2_null95,
        "rank2_subspace_stability": subspace2,
        "mode2_coefficient_r": float(np.nanmedian(cv["coefficient_r"][:, 1])),
        "mode2_p_primary": float(cv["coefficient_p_primary"][1]),
        "rank2_null95": null95,
        **concentration,
    })
    return row, window_rows


def fmt_table(frame: pd.DataFrame, digits: int = 3) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include=[np.number]):
        if not pd.api.types.is_integer_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{x:.{digits}f}" if np.isfinite(x) else "NA")
    return shown.to_string(index=False)


def main() -> None:
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    old_cohort = pd.read_csv(OLD_COHORT)
    selected = []
    for roi in ROIS:
        q = meta[
            meta.roi.eq(roi) & meta.released_independent_consistency.ge(0.4)
        ].sort_values(["released_independent_consistency", "unit_global"], ascending=[False, True])
        selected.append(q.head(ROI_COUNTS[roi]))
    meta = pd.concat(selected, ignore_index=True)
    cohort_overlap = int(meta.unit_global.isin(old_cohort.unit_global).sum())
    meta = meta[meta.roi.isin(ROIS)].sort_values(["session", "unit_index_zero_based"]).reset_index(drop=True)
    rows, window_rows, repeat_designs = [], [], []

    for session, group in meta.groupby("session", sort=True):
        group = group.sort_values("unit_index_zero_based").reset_index(drop=True)
        table, counts, half_size, repeat_range, n_images = load_session_counts(group)
        repeat_designs.append((int(session), half_size, *repeat_range, n_images))
        split_pairs_all = []
        for split in range(N_SPLITS):
            split_pairs_all.append(split_trials(
                table, counts, half_size,
                np.random.default_rng(SEED + int(session) * 1009 + split)))
        full = np.stack([counts[table[i, table[i] >= 0]].mean(axis=0) for i in range(n_images)])

        for local, unit in group.iterrows():
            split_pairs = [(a[:, :, local], b[:, :, local]) for a, b in split_pairs_all]
            result, win = analyze_unit(full[:, :, local], split_pairs, str(unit.roi))
            base = {
                "unit_global": int(unit.unit_global), "roi": str(unit.roi),
                "session": int(session), "released_consistency": float(unit.released_independent_consistency),
            }
            rows.append({**base, **result})
            window_rows.extend([{**base, **x} for x in win])
        print(f"session {int(session):02d}: {len(group)} units, {half_size} vs {half_size} trials, "
              f"{n_images} images", flush=True)

    units = pd.DataFrame(rows)
    units["mode2_p_bh"] = bh_adjust(units.mode2_p_primary.to_numpy(float))
    units["reliable_nonrank1"] = (
        units.mode2_p_bh.lt(0.05)
        & units.rank2_increment.gt(0)
        & units.mode2_stability.gt(units.mode2_null95)
        & units.rank2_subspace_stability.gt(units.rank2_null95)
    )
    valid = units.n_reliable_windows.ge(3)
    units["reliable_nonrank1_for_valid"] = units.reliable_nonrank1.where(valid)
    windows = pd.DataFrame(window_rows)

    rel_summary = windows.groupby(["roi", "window_start_ms"], as_index=False).agg(
        n_units=("unit_global", "size"),
        median_rel_half=("rel_half", "median"),
        median_rel_full_sb=("rel_full_sb", "median"),
        fraction_full_rel_ge_040=("rel_full_sb", lambda x: float(np.mean(x >= 0.4))),
    )
    roi = units.groupby("roi", as_index=False).agg(
        n_units=("unit_global", "size"),
        n_rank_units=("n_reliable_windows", lambda x: int(np.sum(x >= 3))),
        median_n_reliable_windows=("n_reliable_windows", "median"),
        median_rel_full=("median_rel_full_reliable", "median"),
        median_raw_cross_time=("raw_cross_time", "median"),
        median_corrected_cross_time=("corrected_cross_time", "median"),
        median_independent_corrected=("independent_corrected_cross_time", "median"),
        median_rank1=("rank1_fraction", "median"),
        median_rank2=("rank2_fraction", "median"),
        median_rank2_increment=("rank2_increment", "median"),
        median_rank3_increment=("rank3_increment", "median"),
        median_mode1_stability=("mode1_stability", "median"),
        median_mode2_stability=("mode2_stability", "median"),
        median_rank2_subspace_stability=("rank2_subspace_stability", "median"),
        reliable_nonrank1_fraction=("reliable_nonrank1_for_valid", "mean"),
        median_mode2_top1_share=("mode2_top1_share", "median"),
        median_mode2_top5_share=("mode2_top5_share", "median"),
        median_mode2_top10_share=("mode2_top10_share", "median"),
        median_mode2_top20_share=("mode2_top20_share", "median"),
        median_mode2_effective_image_fraction=("mode2_effective_image_fraction", "median"),
        median_mode2_half_mass_image_fraction=("mode2_images_for_half_mass_fraction", "median"),
        median_mode2_positive_image_fraction=("mode2_positive_image_fraction", "median"),
        median_mode2_consistently_positive_fraction=("mode2_consistently_positive_fraction", "median"),
        median_rank1_top10_share=("rank1_top10_share", "median"),
        median_rank1_effective_image_fraction=("rank1_effective_image_fraction", "median"),
        median_rank1_half_mass_image_fraction=("rank1_images_for_half_mass_fraction", "median"),
        median_rank1_cvtop1_share=("rank1_cvtop1_share", "median"),
        median_rank1_cvtop5_share=("rank1_cvtop5_share", "median"),
        median_rank1_cvtop10_share=("rank1_cvtop10_share", "median"),
        median_rank1_cvtop20_share=("rank1_cvtop20_share", "median"),
        median_mode2_cvtop1_share=("mode2_cvtop1_share", "median"),
        median_mode2_cvtop5_share=("mode2_cvtop5_share", "median"),
        median_mode2_cvtop10_share=("mode2_cvtop10_share", "median"),
        median_mode2_cvtop20_share=("mode2_cvtop20_share", "median"),
    )
    roi["_order"] = roi.roi.map({r: i for i, r in enumerate(ROIS)})
    roi = roi.sort_values("_order").drop(columns="_order")

    print("\nAUDIT", flush=True)
    print({
        "cohort_units": len(units),
        "cohort_selection": "top released_independent_consistency within ROI; no model metric",
        "roi_counts": ROI_COUNTS, "overlap_with_old_proxy_selected_113": cohort_overlap,
        "trial_halves": "equal independent halves; session-specific 2 vs 2 or 3 vs 3",
        "repeat_design_counts": pd.DataFrame(
            repeat_designs, columns=["session", "half_size", "min_repeats", "max_repeats", "n_images"]
        ).groupby(["half_size", "min_repeats", "max_repeats", "n_images"]).size().to_dict(),
        "trial_split_repeats": N_SPLITS, "outer_image_folds": N_IMAGE_FOLDS,
        "reliability_for_correction": "split-half for independent cross-half; Spearman-Brown full-mean for full response",
        "rank_target": "cross-validated independent-repeat covariance after removing per-window image mean",
        "raw_source": "unsmoothed 1-ms raster_matrix_img summed into the frozen official non-overlapping 20-ms windows",
        "trial_image_mapping_check": "len(img_idx) equals raster trial dimension; every retained image has 2k trials",
        "files_written": False,
    }, flush=True)
    print("\nRELIABILITY_BY_ROI_WINDOW", flush=True)
    print(fmt_table(rel_summary), flush=True)
    print("\nROI_SUMMARY", flush=True)
    print(fmt_table(roi), flush=True)
    print("\nMOST_RELIABLE_NONRANK1_UNITS", flush=True)
    cols = ["unit_global", "roi", "n_reliable_windows", "rank1_fraction", "rank2_increment",
            "rank3_increment", "mode2_coefficient_r", "mode2_p_bh", "rank2_subspace_stability",
            "rank2_null95", "mode2_stability", "mode2_null95", "reliable_nonrank1"]
    strongest = units.sort_values(["reliable_nonrank1", "rank2_increment"], ascending=False).head(20)
    print(fmt_table(strongest[cols]), flush=True)
    print("\nMODE2_IMAGE_CONCENTRATION", flush=True)
    concentration_cols = [
        "roi", "n_rank_units", "median_rank1_top10_share", "median_mode2_top1_share", "median_mode2_top5_share",
        "median_mode2_top10_share", "median_mode2_top20_share",
        "median_rank1_effective_image_fraction", "median_mode2_effective_image_fraction",
        "median_rank1_half_mass_image_fraction", "median_mode2_half_mass_image_fraction",
        "median_mode2_positive_image_fraction", "median_mode2_consistently_positive_fraction",
    ]
    print(fmt_table(roi[concentration_cols]), flush=True)
    print("\nCROSS_TRIAL_SELECTED_CONCENTRATION", flush=True)
    cv_concentration_cols = [
        "roi", "n_rank_units", "median_rank1_cvtop1_share", "median_rank1_cvtop5_share",
        "median_rank1_cvtop10_share", "median_rank1_cvtop20_share",
        "median_mode2_cvtop1_share", "median_mode2_cvtop5_share",
        "median_mode2_cvtop10_share", "median_mode2_cvtop20_share",
    ]
    print(fmt_table(roi[cv_concentration_cols]), flush=True)
    print("\nGLOBAL", flush=True)
    print({
        "units_with_3plus_reliable_windows": int(valid.sum()),
        "median_rank1_repeatable_fraction": float(units.loc[valid, "rank1_fraction"].median()),
        "median_rank2_increment": float(units.loc[valid, "rank2_increment"].median()),
        "median_rank3_increment": float(units.loc[valid, "rank3_increment"].median()),
        "reliable_nonrank1_units": int(units.loc[valid, "reliable_nonrank1"].sum()),
        "reliable_nonrank1_fraction": float(units.loc[valid, "reliable_nonrank1"].mean()),
    }, flush=True)


if __name__ == "__main__":
    main()
