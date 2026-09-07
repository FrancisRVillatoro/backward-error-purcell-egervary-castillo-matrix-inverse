#!/usr/bin/env python3
"""Stratified validation of the frozen six-iteration spectral-norm estimator.

This is an independent audit utility. It does not modify the scientific code or
campaign outputs. Each Slurm array task reconstructs one deterministic balanced
matrix and evaluates the five Castillo variants plus LAPACK in both precisions.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

HOME = Path.home()
BASE = HOME / "castillo_lapack_baseline"
SOURCE = BASE / "source"
SELDIR = BASE / "exact_balanced_selection" / "selection"
OUTDIR = BASE / "results" / "spectral_estimator_validation" / "tasks"

TARGET_N = (8, 64, 256, 512)
METHODS = ("R0_C0", "R0_C1", "R0_C2", "R1_C1", "R2_C2")
DTYPES = (("float32", np.float32), ("float64", np.float64))
SPECTRAL_ITERATIONS = 6

sys.path.insert(0, str((SOURCE / "code").resolve()))
from families import build_matrix, stable_seed  # noqa: E402
from castillo import castillo_inverse, spectral_norm_estimate, unit_roundoff  # noqa: E402


def load_manifest(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            item["block_index"] = int(row["block_index"])
            item["n"] = int(row["n"])
            item["stochastic"] = str(row["stochastic"]).strip().lower() == "true"
            item["parameters"] = json.loads(row["parameters_json"])
            rows.append(item)
    return rows


def rank_zero_replicas(selection_dir: Path):
    result = {}
    paths = sorted(selection_dir.glob("task_????.npz"))
    if not paths:
        raise RuntimeError(f"No balanced-selection files in {selection_dir}")
    for task_id, path in enumerate(paths):
        with np.load(str(path), allow_pickle=False) as z:
            blocks = np.asarray(z["block"], dtype=np.int64)
            replicas = np.asarray(z["replica"], dtype=np.int64)
            ranks = np.asarray(z["rank"], dtype=np.int64)
        mask = ranks == 0
        for block, replica in zip(blocks[mask], replicas[mask]):
            b = int(block); r = int(replica)
            if b in result and result[b][1] != r:
                raise RuntimeError(f"Duplicate rank-zero selection for block {b}")
            result[b] = (task_id, r)
    return result


def validation_cases():
    manifest = load_manifest(SOURCE / "reports" / "canonical" / "canonical_manifest.csv")
    rank0 = rank_zero_replicas(SELDIR)
    grouped = defaultdict(list)
    for row in manifest:
        if row["stochastic"] and row["n"] in TARGET_N and row["block_index"] in rank0:
            grouped[(row["family"], row["n"])].append(row)

    cases = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda r: (r["cell_id"], r["block_index"]))
        chosen = [rows[0]]
        if len(rows) > 1 and rows[-1]["block_index"] != rows[0]["block_index"]:
            chosen.append(rows[-1])
        for row in chosen:
            task_id, replica = rank0[row["block_index"]]
            cases.append((row, task_id, replica))
    return cases


def norm2_exact(a):
    a64 = np.asarray(a, dtype=np.float64)
    return float(np.linalg.norm(a64, ord=2))


def ratio(est, exact):
    if exact == 0.0:
        if est == 0.0:
            return math.nan, "both_zero"
        return math.inf, "exact_zero_est_nonzero"
    return est / exact, "positive_exact"


def lapack_inverse(a):
    af = np.array(a, order="F", copy=True)
    getrf, getri, getri_lwork = get_lapack_funcs(("getrf", "getri", "getri_lwork"), (af,))
    lu, piv, info_rf = getrf(af, overwrite_a=True)
    if int(info_rf) != 0:
        return None, f"getrf_info_{int(info_rf)}"
    lwork, info_lw = getri_lwork(af.shape[0])
    if int(info_lw) != 0:
        return None, f"getri_lwork_info_{int(info_lw)}"
    lw = max(1, int(float(np.real(lwork))))
    x, info_ri = getri(lu, piv, lwork=lw, overwrite_lu=True)
    if int(info_ri) != 0:
        return None, f"getri_info_{int(info_ri)}"
    if not np.all(np.isfinite(x)):
        return None, "nonfinite_inverse"
    return x, "completed"


def evaluate(a_work, x_work, a_for_residual):
    a64 = np.asarray(a_for_residual, dtype=np.float64)
    x64 = np.asarray(x_work, dtype=np.float64)
    eye = np.eye(a64.shape[0], dtype=np.float64)
    residual = eye - a64 @ x64

    a_est = spectral_norm_estimate(a_work, SPECTRAL_ITERATIONS)
    x_est = spectral_norm_estimate(x_work, SPECTRAL_ITERATIONS)
    r_est_norm = spectral_norm_estimate(residual, SPECTRAL_ITERATIONS)

    a_exact = norm2_exact(a64)
    x_exact = norm2_exact(x64)
    r_exact_norm = norm2_exact(residual)

    a_ratio, a_state = ratio(a_est, a_exact)
    x_ratio, x_state = ratio(x_est, x_exact)
    r_ratio, r_state = ratio(r_est_norm, r_exact_norm)

    denom_est = a_est * x_est
    denom_exact = a_exact * x_exact
    residual_est = r_est_norm / denom_est if denom_est > 0 else math.nan
    residual_exact = r_exact_norm / denom_exact if denom_exact > 0 else math.nan
    residual_ratio, residual_state = ratio(residual_est, residual_exact)

    return {
        "A_nu6": a_est, "A_norm2": a_exact, "A_nu6_over_norm2": a_ratio, "A_state": a_state,
        "X_nu6": x_est, "X_norm2": x_exact, "X_nu6_over_norm2": x_ratio, "X_state": x_state,
        "R_nu6": r_est_norm, "R_norm2": r_exact_norm, "R_nu6_over_norm2": r_ratio, "R_state": r_state,
        "residual_nu6": residual_est, "residual_norm2": residual_exact,
        "residual_nu6_over_norm2": residual_ratio, "residual_state": residual_state,
    }


def main():
    cases = validation_cases()
    print("validation_cases =", len(cases))
    print("target_n =", TARGET_N)
    print("expected_array_indices = 0..{}".format(len(cases)-1))

    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CASE_INDEX", "0")))
    if idx < 0 or idx >= len(cases):
        print("INDEX_OUTSIDE_ACTUAL_CASES", idx)
        return

    canonical = json.loads((SOURCE / "config" / "canonical.json").read_text(encoding="utf-8"))
    execution = json.loads((SOURCE / "config" / "execution.json").read_text(encoding="utf-8"))
    if int(execution["spectral_norm_iterations"]) != SPECTRAL_ITERATIONS:
        raise RuntimeError("Frozen campaign does not use six spectral iterations")

    row, selection_task, replica = cases[idx]
    seed = stable_seed(canonical["seed_namespace"], row["cell_id"], row["n"], replica)
    aref = build_matrix(family=row["family"], n=row["n"], parameters=row["parameters"], seed=seed)

    common = {
        "case_index": idx,
        "family": row["family"],
        "cell_id": row["cell_id"],
        "block_index": row["block_index"],
        "selection_task": selection_task,
        "n": row["n"],
        "replica": replica,
        "seed": seed,
    }
    rows_out = []
    for dtype_name, dtype in DTYPES:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            a_work = np.asarray(aref, dtype=dtype)
        if not np.all(np.isfinite(a_work)):
            for method in METHODS + ("LAPACK_xGETRF_xGETRI",):
                rows_out.append({**common, "dtype_name": dtype_name, "method": method,
                                 "success": False, "status": "nonrepresentable_input"})
            continue

        for method in METHODS:
            t0 = time.perf_counter()
            result = castillo_inverse(aref, method, dtype_name,
                                      spectral_iterations=SPECTRAL_ITERATIONS, audit=False)
            out = {**common, "dtype_name": dtype_name, "method": method,
                   "success": bool(result.success), "status": result.failure_class,
                   "elapsed_seconds": time.perf_counter()-t0}
            if result.success and np.all(np.isfinite(result.inverse)):
                apermuted = a_work[np.asarray(result.row_order, dtype=int), :]
                out.update(evaluate(a_work, result.inverse, apermuted))
            rows_out.append(out)

        t0 = time.perf_counter()
        x, status = lapack_inverse(a_work)
        out = {**common, "dtype_name": dtype_name, "method": "LAPACK_xGETRF_xGETRI",
               "success": x is not None, "status": status,
               "elapsed_seconds": time.perf_counter()-t0}
        if x is not None:
            out.update(evaluate(a_work, x, a_work))
        rows_out.append(out)

    fields = [
        "case_index","family","cell_id","block_index","selection_task","n","replica","seed",
        "dtype_name","method","success","status","elapsed_seconds",
        "A_nu6","A_norm2","A_nu6_over_norm2","A_state",
        "X_nu6","X_norm2","X_nu6_over_norm2","X_state",
        "R_nu6","R_norm2","R_nu6_over_norm2","R_state",
        "residual_nu6","residual_norm2","residual_nu6_over_norm2","residual_state",
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"validation_task_{idx:04d}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows_out)
    print("output =", path)
    print("records =", len(rows_out))
    print("SPECTRAL_ESTIMATOR_VALIDATION_TASK_OK")

if __name__ == "__main__":
    main()
