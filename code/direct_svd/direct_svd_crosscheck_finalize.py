#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import numpy as np

BASE = Path.home() / "castillo_lapack_baseline"
ROOT = BASE / "results" / "direct_svd_crosscheck"
TASKS = ROOT / "tasks"
FINAL = ROOT / "final"
METHODS = ("R0_C0", "R0_C1", "R0_C2", "R1_C1", "R2_C2")
LAPACK = "LAPACK_xGETRF_xGETRI"
ALL = METHODS + (LAPACK,)
DTYPES = ("float32", "float64")
EXPECTED_TASKS = 832
EXPECTED_MATRICES = 2496
EXPECTED_ROWS = EXPECTED_MATRICES * 2 * 6
BOOT_REPS = 2000
BOOT_SEED = 20260908


def B(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def F(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def Q(v, p):
    a = np.asarray([x for x in v if math.isfinite(x)], dtype=float)
    return float(np.quantile(a, p)) if a.size else math.nan


def cluster_boot_mean(blockvals, seed):
    bids = np.array(sorted(blockvals), dtype=int)
    if not bids.size:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        draw = rng.choice(bids, size=bids.size, replace=True)
        vals = []
        for b in draw:
            vals.extend(blockvals[int(b)])
        out[i] = np.mean(np.asarray(vals, dtype=float))
    return (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


def cluster_boot_median(blockvals, seed):
    bids = np.array(sorted(blockvals), dtype=int)
    if not bids.size:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        draw = rng.choice(bids, size=bids.size, replace=True)
        vals = []
        for b in draw:
            vals.extend(blockvals[int(b)])
        out[i] = np.median(np.asarray(vals, dtype=float))
    return (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


def main():
    files = sorted(TASKS.glob("direct_svd_task_????.csv"))
    ids = [int(p.stem.rsplit("_", 1)[1]) for p in files]
    if len(files) != EXPECTED_TASKS or ids != list(range(EXPECTED_TASKS)):
        raise RuntimeError(f"task coverage failure: {len(files)} files")

    rows = []
    for p in files:
        with p.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")

    mp = defaultdict(dict)
    for r in rows:
        k = (int(r["block_index"]), int(r["selection_rank"]), r["dtype_name"])
        if r["method"] in mp[k]:
            raise RuntimeError(f"duplicate {k} {r['method']}")
        mp[k][r["method"]] = r

    if len(mp) != EXPECTED_MATRICES * 2:
        raise RuntimeError(f"bad matrix/precision groups {len(mp)}")

    # Strict six-method common-success population, augmented only by the
    # requirement that both diagnostics be finite and nonnegative.  Zero
    # residuals are valid and are NOT removed from the common population.
    common = {}
    for k, rr in mp.items():
        if set(rr) != set(ALL):
            raise RuntimeError(f"method coverage failure {k}")
        ok = True
        for m in ALL:
            est = F(rr[m]["rR2_nu6_over_u"])
            ref = F(rr[m]["rR2_svd_over_u"])
            ok = (
                ok
                and B(rr[m]["success"])
                and math.isfinite(est)
                and est >= 0.0
                and math.isfinite(ref)
                and ref >= 0.0
            )
        common[k] = ok

    FINAL.mkdir(parents=True, exist_ok=True)

    # Balanced-subsample pooled cross-check.  These are NOT a literal
    # recomputation of the full-campaign Table-4 blockwise statistic.
    fields = [
        "dtype_name", "method", "strict_common6_N", "strict_common6_blocks",
        "nu6_q50_rR2_over_u", "nu6_q95_rR2_over_u",
        "svd_q50_rR2_over_u", "svd_q95_rR2_over_u",
        "median_nu6_over_svd_positive_reference",
        "nu6_over_svd_positive_reference_N",
    ]
    table = []
    for dtype in DTYPES:
        keys = [k for k in mp if k[2] == dtype and common[k]]
        for m in ALL:
            est = [F(mp[k][m]["rR2_nu6_over_u"]) for k in keys]
            ref = [F(mp[k][m]["rR2_svd_over_u"]) for k in keys]
            rat = [a / b for a, b in zip(est, ref) if b > 0.0]
            table.append({
                "dtype_name": dtype,
                "method": m,
                "strict_common6_N": len(keys),
                "strict_common6_blocks": len({k[0] for k in keys}),
                "nu6_q50_rR2_over_u": Q(est, 0.5),
                "nu6_q95_rR2_over_u": Q(est, 0.95),
                "svd_q50_rR2_over_u": Q(ref, 0.5),
                "svd_q95_rR2_over_u": Q(ref, 0.95),
                "median_nu6_over_svd_positive_reference": Q(rat, 0.5),
                "nu6_over_svd_positive_reference_N": len(rat),
            })

    with (FINAL / "direct_svd_balanced_subsample_crosscheck.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    # Estimator accuracy, method by method, on every successful record for
    # which the SVD-reference residual is positive (the ratio is undefined
    # when the reference residual is exactly zero).
    fields2 = [
        "dtype_name", "method", "N", "min", "q05", "median", "q95", "q99",
        "max", "fraction_within_1pct", "fraction_within_5pct",
        "fraction_within_10pct",
    ]
    acc = []
    for dtype in DTYPES:
        for m in ALL:
            vals = []
            for r in rows:
                if r["dtype_name"] == dtype and r["method"] == m and B(r["success"]):
                    x = F(r["rR2_nu6_over_svd"])
                    if math.isfinite(x) and x > 0:
                        vals.append(x)
            a = np.asarray(vals, dtype=float)
            acc.append({
                "dtype_name": dtype,
                "method": m,
                "N": len(vals),
                "min": float(np.min(a)) if a.size else math.nan,
                "q05": Q(vals, 0.05),
                "median": Q(vals, 0.5),
                "q95": Q(vals, 0.95),
                "q99": Q(vals, 0.99),
                "max": float(np.max(a)) if a.size else math.nan,
                "fraction_within_1pct": float(np.mean(np.abs(a - 1) <= 0.01)) if a.size else math.nan,
                "fraction_within_5pct": float(np.mean(np.abs(a - 1) <= 0.05)) if a.size else math.nan,
                "fraction_within_10pct": float(np.mean(np.abs(a - 1) <= 0.10)) if a.size else math.nan,
            })

    with (FINAL / "direct_svd_estimator_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(acc)

    # Paired direct-SVD Castillo/LAPACK comparison.  The win fraction is
    # defined even when LAPACK's direct-SVD residual is zero.  Ratio summaries
    # are computed only where the LAPACK denominator is positive.
    fields3 = [
        "dtype_name", "castillo_method", "strict_common6_N",
        "strict_common6_blocks", "fraction_castillo_lt_lapack_svd",
        "fraction_castillo_eq_lapack_svd", "fraction_ci025", "fraction_ci975",
        "ratio_positive_lapack_N", "ratio_positive_lapack_blocks",
        "median_castillo_over_lapack_svd", "median_ci025", "median_ci975",
        "q05_ratio", "q95_ratio",
    ]
    pairs = []
    for dtype in DTYPES:
        keys = [k for k in mp if k[2] == dtype and common[k]]
        for im, m in enumerate(METHODS):
            win_vals = []
            win_by_block = defaultdict(list)
            ratio_vals = []
            ratio_by_block = defaultdict(list)
            equal_count = 0

            for k in keys:
                c = F(mp[k][m]["rR2_svd_over_u"])
                l = F(mp[k][LAPACK]["rR2_svd_over_u"])
                win = 1.0 if c < l else 0.0
                win_vals.append(win)
                win_by_block[k[0]].append(win)
                if c == l:
                    equal_count += 1
                if l > 0.0:
                    rr = c / l
                    ratio_vals.append(rr)
                    ratio_by_block[k[0]].append(rr)

            flo, fhi = cluster_boot_mean(
                win_by_block,
                BOOT_SEED + im + (100 if dtype == "float64" else 0),
            )
            mlo, mhi = cluster_boot_median(
                ratio_by_block,
                BOOT_SEED + 1000 + im + (100 if dtype == "float64" else 0),
            )

            pairs.append({
                "dtype_name": dtype,
                "castillo_method": m,
                "strict_common6_N": len(keys),
                "strict_common6_blocks": len({k[0] for k in keys}),
                "fraction_castillo_lt_lapack_svd": float(np.mean(win_vals)) if win_vals else math.nan,
                "fraction_castillo_eq_lapack_svd": equal_count / len(keys) if keys else math.nan,
                "fraction_ci025": flo,
                "fraction_ci975": fhi,
                "ratio_positive_lapack_N": len(ratio_vals),
                "ratio_positive_lapack_blocks": len(ratio_by_block),
                "median_castillo_over_lapack_svd": Q(ratio_vals, 0.5),
                "median_ci025": mlo,
                "median_ci975": mhi,
                "q05_ratio": Q(ratio_vals, 0.05),
                "q95_ratio": Q(ratio_vals, 0.95),
            })

    with (FINAL / "direct_svd_paired_vs_lapack.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields3)
        w.writeheader()
        w.writerows(pairs)

    cc = {
        d: sum(1 for k in common if k[2] == d and common[k])
        for d in DTYPES
    }
    ccb = {
        d: len({k[0] for k in common if k[2] == d and common[k]})
        for d in DTYPES
    }

    summary = {
        "design": {
            "balanced_blocks": 832,
            "selected_ranks": [0, 1000, 1999],
            "matrices": 2496,
            "methods": list(ALL),
            "precisions": list(DTYPES),
            "reference": "direct binary64 SVD of working-precision A, inverse, and binary64-formed residual",
            "production_estimator": "six deterministic power iterations",
            "aggregation_note": "balanced-subsample pooled quantiles; not a literal recomputation of the full-campaign blockwise Table 4 statistic",
            "block_bootstrap_repetitions": BOOT_REPS,
            "block_bootstrap_seed": BOOT_SEED,
            "zero_residual_policy": "zero residuals retained in common-success quantiles and win fractions; estimator/reference and Castillo/LAPACK ratios require positive denominator",
        },
        "coverage": {
            "task_files": len(files),
            "records": len(rows),
            "matrix_precision_groups": len(mp),
            "strict_common6_N": cc,
            "strict_common6_blocks": ccb,
        },
        "gate": True,
    }

    (FINAL / "direct_svd_crosscheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("===== BALANCED SUBSAMPLE =====")
    for r in table:
        print(r)
    print("===== PAIRED VS LAPACK =====")
    for r in pairs:
        print(r)
    print("DIRECT_SVD_CROSSCHECK_FINALIZE_OK")


if __name__ == "__main__":
    main()
