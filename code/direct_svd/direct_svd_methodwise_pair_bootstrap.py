#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv
import math
import sys
import numpy as np

import os
RELEASE_ROOT = Path(os.environ.get("CASTILLO_REPRO_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
ROOT = Path(os.environ.get("CASTILLO_DIRECT_SVD_WORKDIR", str(RELEASE_ROOT / "work" / "direct_svd_crosscheck"))).resolve()
TASKS = ROOT / "tasks"
FINAL = ROOT / "final"

METHODS = ("R0_C0", "R0_C1", "R0_C2", "R1_C1", "R2_C2")
LAPACK = "LAPACK_xGETRF_xGETRI"
ALL = METHODS + (LAPACK,)
DTYPES = ("float32", "float64")

EXPECTED_TASKS = 832
EXPECTED_MATRICES = 2496
EXPECTED_ROWS = EXPECTED_MATRICES * 2 * 6
EXPECTED_GROUPS = EXPECTED_MATRICES * 2

# Deliberately identical to the original direct-SVD finalizer.
BOOT_REPS = 2000
BOOT_SEED = 20260908

EXPECTED_METHODWISE_N32 = {
    "R0_C0": 2486,
    "R0_C1": 2496,
    "R0_C2": 2493,
    "R1_C1": 2490,
    "R2_C2": 2484,
}


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
    if bids.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        draw = rng.choice(bids, size=bids.size, replace=True)
        vals = []
        for b in draw:
            vals.extend(blockvals[int(b)])
        out[i] = np.mean(np.asarray(vals, dtype=float))
    return (
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    )


def cluster_boot_median(blockvals, seed):
    bids = np.array(sorted(blockvals), dtype=int)
    if bids.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        draw = rng.choice(bids, size=bids.size, replace=True)
        vals = []
        for b in draw:
            vals.extend(blockvals[int(b)])
        out[i] = np.median(np.asarray(vals, dtype=float))
    return (
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    )


def main():
    files = sorted(TASKS.glob("direct_svd_task_????.csv"))
    ids = [int(p.stem.rsplit("_", 1)[1]) for p in files]

    if len(files) != EXPECTED_TASKS or ids != list(range(EXPECTED_TASKS)):
        raise RuntimeError(
            f"task coverage failure: {len(files)} files; "
            f"expected task ids 0..{EXPECTED_TASKS-1}"
        )

    rows = []
    for p in files:
        with p.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")

    mp = defaultdict(dict)
    for r in rows:
        key = (
            int(r["block_index"]),
            int(r["selection_rank"]),
            r["dtype_name"],
        )
        method = r["method"]
        if method in mp[key]:
            raise RuntimeError(f"duplicate record: {key} {method}")
        mp[key][method] = r

    if len(mp) != EXPECTED_GROUPS:
        raise RuntimeError(
            f"expected {EXPECTED_GROUPS} matrix/precision groups, got {len(mp)}"
        )

    for key, rr in mp.items():
        if set(rr) != set(ALL):
            raise RuntimeError(f"method coverage failure for {key}")

    # Strict-common6 mask retained only as a reference to quantify the
    # correction. It is NOT used for the definitive methodwise inference.
    strict6 = {}
    for key, rr in mp.items():
        ok = True
        for method in ALL:
            est = F(rr[method]["rR2_nu6_over_u"])
            ref = F(rr[method]["rR2_svd_over_u"])
            ok = (
                ok
                and B(rr[method]["success"])
                and math.isfinite(est)
                and est >= 0.0
                and math.isfinite(ref)
                and ref >= 0.0
            )
        strict6[key] = ok

    output_rows = []

    for dtype in DTYPES:
        for im, method in enumerate(METHODS):
            keys = []

            for key, rr in mp.items():
                if key[2] != dtype:
                    continue

                c = rr[method]
                l = rr[LAPACK]

                cr = F(c["rR2_svd_over_u"])
                lr = F(l["rR2_svd_over_u"])

                pair_ok = (
                    B(c["success"])
                    and B(l["success"])
                    and math.isfinite(cr)
                    and cr >= 0.0
                    and math.isfinite(lr)
                    and lr >= 0.0
                )

                if pair_ok:
                    keys.append(key)

            win_vals = []
            win_by_block = defaultdict(list)
            ratio_vals = []
            ratio_by_block = defaultdict(list)
            equal_count = 0

            for key in keys:
                c = F(mp[key][method]["rR2_svd_over_u"])
                l = F(mp[key][LAPACK]["rR2_svd_over_u"])

                win = 1.0 if c < l else 0.0
                win_vals.append(win)
                win_by_block[key[0]].append(win)

                if c == l:
                    equal_count += 1

                # Ratio is undefined only if LAPACK's residual is exactly zero.
                if l > 0.0:
                    ratio = c / l
                    ratio_vals.append(ratio)
                    ratio_by_block[key[0]].append(ratio)

            frac_lo, frac_hi = cluster_boot_mean(
                win_by_block,
                BOOT_SEED + im + (100 if dtype == "float64" else 0),
            )
            med_lo, med_hi = cluster_boot_median(
                ratio_by_block,
                BOOT_SEED + 1000 + im + (100 if dtype == "float64" else 0),
            )

            strict_keys = [key for key in keys if strict6[key]]

            output_rows.append({
                "dtype_name": dtype,
                "castillo_method": method,
                "methodwise_N": len(keys),
                "methodwise_blocks": len({key[0] for key in keys}),
                "strict_common6_N": len(strict_keys),
                "added_vs_strict6_N": len(keys) - len(strict_keys),
                "fraction_castillo_lt_lapack_svd":
                    float(np.mean(win_vals)) if win_vals else math.nan,
                "fraction_castillo_eq_lapack_svd":
                    equal_count / len(keys) if keys else math.nan,
                "fraction_ci025": frac_lo,
                "fraction_ci975": frac_hi,
                "ratio_positive_lapack_N": len(ratio_vals),
                "ratio_positive_lapack_blocks": len(ratio_by_block),
                "median_castillo_over_lapack_svd":
                    Q(ratio_vals, 0.50),
                "median_ci025": med_lo,
                "median_ci975": med_hi,
                "q05_ratio": Q(ratio_vals, 0.05),
                "q95_ratio": Q(ratio_vals, 0.95),
                "bootstrap_unit": "block_index",
                "bootstrap_repetitions": BOOT_REPS,
                "bootstrap_seed_fraction":
                    BOOT_SEED + im + (100 if dtype == "float64" else 0),
                "bootstrap_seed_median":
                    BOOT_SEED + 1000 + im + (100 if dtype == "float64" else 0),
            })

    # Gates: exact sample sizes already independently audited in the previous
    # methodwise-pair calculation, plus full float64 coverage.
    gate = True
    for r in output_rows:
        if r["dtype_name"] == "float32":
            gate = (
                gate
                and r["methodwise_N"]
                == EXPECTED_METHODWISE_N32[r["castillo_method"]]
            )
        else:
            gate = gate and r["methodwise_N"] == 2496

        gate = (
            gate
            and r["ratio_positive_lapack_N"] == r["methodwise_N"]
            and math.isfinite(r["fraction_ci025"])
            and math.isfinite(r["fraction_ci975"])
            and math.isfinite(r["median_ci025"])
            and math.isfinite(r["median_ci975"])
        )

    FINAL.mkdir(parents=True, exist_ok=True)
    out = FINAL / "direct_svd_methodwise_pair_bootstrap.csv"

    fields = list(output_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output_rows)

    print("============================================================")
    print("DIRECT-SVD METHODWISE PAIRED BLOCK BOOTSTRAP")
    print("============================================================")
    print("TASK_FILES =", len(files))
    print("TOTAL_ROWS =", len(rows))
    print("MATRIX_PRECISION_GROUPS =", len(mp))
    print("BOOT_REPS =", BOOT_REPS)
    print("BOOT_SEED =", BOOT_SEED)

    for r in output_rows:
        print()
        print(
            r["dtype_name"],
            r["castillo_method"],
            "N=", r["methodwise_N"],
            "blocks=", r["methodwise_blocks"],
        )
        print(
            "  win_fraction =",
            r["fraction_castillo_lt_lapack_svd"],
            "CI95=[",
            r["fraction_ci025"],
            ",",
            r["fraction_ci975"],
            "]",
        )
        print(
            "  median_ratio =",
            r["median_castillo_over_lapack_svd"],
            "CI95=[",
            r["median_ci025"],
            ",",
            r["median_ci975"],
            "]",
        )

    print()
    print("OUTPUT =", out)
    print("METHODWISE_BOOTSTRAP_GATE =", gate)

    if gate:
        print("DIRECT_SVD_METHODWISE_BOOTSTRAP_OK")
    else:
        print("DIRECT_SVD_METHODWISE_BOOTSTRAP_REQUIRES_REVIEW")


if __name__ == "__main__":
    main()
