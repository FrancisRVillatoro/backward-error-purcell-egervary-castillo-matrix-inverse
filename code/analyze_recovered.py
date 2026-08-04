#!/usr/bin/env python3
"""Streaming analysis of the recovered Castillo campaign.

Subcommands
-----------
select
    Build an exact deterministic 2000-matrix sample for every stochastic
    parameter cell and every recovered dimension n <= 512.

shard
    Read one recovered inverse/solution shard and write one compact NPZ
    partial containing: exact core data, balanced-sample data, all-data
    failure counts, deterministic records, and task-level censoring summaries.

finalize
    Merge all shard partials, compute exact quantiles, model fits, failure
    rates, refinement effects, censoring diagnostics, bootstrap inputs, and
    preliminary figures/reports.

bootstrap
    Bootstrap one (cell, method, precision) combination with cluster
    resampling at matrix level, stratified by dimension.

report
    Merge bootstrap outputs and write the final technical report and
    multipanel PNG figure.

The script is compatible with Python 3.8 and uses only the standard library,
NumPy, and Matplotlib (Matplotlib is only needed by the report command).
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


FAILURE_NAMES = (
    "total",
    "success",
    "input_not_representable",
    "no_pivot",
    "nonfinite_B",
    "nonfinite_tableau",
    "other_failure",
)

INV_METRIC_NAMES = (
    "rRu",
    "gamma_over_n",
    "eta_over_u",
    "max_tableau_norm_inf",
    "max_B_norm_inf",
    "max_inverse_tableau_norm_inf",
    "max_multiplier",
    "elapsed_seconds",
)

SOL_METRIC_NAMES = (
    "forward_error_2",
    "normwise_backward_error_inf",
    "componentwise_backward_error",
    "relative_residual_inf",
)

QUANTILE_NAMES = (
    (0.50, "q50"),
    (0.90, "q90"),
    (0.95, "q95"),
    (0.99, "q99"),
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def load_config(path):
    path = Path(path)
    raw = load_json(path)
    base_name = raw.get("base_config")
    if base_name is None:
        return raw
    base = load_config(path.parent / str(base_name))
    for key, value in raw.items():
        if key != "base_config":
            base[key] = value
    return base


def read_manifest(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            item["block_index"] = int(row["block_index"])
            item["n"] = int(row["n"])
            item["case_start"] = int(row["case_start"])
            item["replicas"] = int(row["replicas"])
            item["stochastic"] = str(row["stochastic"]).strip().lower() == "true"
            item["parameters"] = json.loads(row["parameters_json"])
            rows.append(item)
    rows.sort(key=lambda x: x["block_index"])
    return rows


def hash64(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def as_bool(value):
    return str(value).strip().lower() == "true"


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result


def quantile_values(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    result = {"finite_count": int(x.size)}
    if x.size == 0:
        for _, name in QUANTILE_NAMES:
            result[name] = math.nan
        return result
    q = np.quantile(x, [item[0] for item in QUANTILE_NAMES])
    for index, (_, name) in enumerate(QUANTILE_NAMES):
        result[name] = float(q[index])
    return result


def combo_maps(config):
    methods = [str(value) for value in config["methods"]]
    dtypes = [str(value) for value in config["dtypes"]]
    rhs_types = [str(value) for value in config["solution"]["rhs_types"]]
    refinements = [int(value) for value in config["solution"]["refinement_steps"]]

    inv_combo = {}
    inv_decode = []
    for dtype_index, dtype_name in enumerate(dtypes):
        for method_index, method in enumerate(methods):
            index = len(inv_decode)
            inv_combo[(method, dtype_name)] = index
            inv_decode.append(
                {
                    "combo": index,
                    "method": method,
                    "dtype_name": dtype_name,
                    "method_index": method_index,
                    "dtype_index": dtype_index,
                }
            )

    sol_combo = {}
    sol_decode = []
    for dtype_name in dtypes:
        for method in methods:
            for rhs_type in rhs_types:
                for refinement in refinements:
                    index = len(sol_decode)
                    sol_combo[(method, dtype_name, rhs_type, refinement)] = index
                    sol_decode.append(
                        {
                            "combo": index,
                            "method": method,
                            "dtype_name": dtype_name,
                            "rhs_type": rhs_type,
                            "refinement_steps": refinement,
                        }
                    )

    return {
        "methods": methods,
        "dtypes": dtypes,
        "rhs_types": rhs_types,
        "refinements": refinements,
        "inv_combo": inv_combo,
        "inv_decode": inv_decode,
        "sol_combo": sol_combo,
        "sol_decode": sol_decode,
        "num_inv_combo": len(inv_decode),
        "num_sol_combo": len(sol_decode),
    }


def failure_index(row):
    if as_bool(row.get("success", "False")):
        return 1
    failure_class = str(row.get("failure_class", ""))
    reason = str(row.get("failure_reason", ""))
    text = failure_class + "|" + reason
    if "input_not_representable" in text or "input_nonfinite" in text:
        return 2
    if "no_pivot" in text or "no_finite_nonzero_pivot" in text:
        return 3
    if "nonfinite_B" in text or "nonfinite_elementary_transformation" in text:
        return 4
    if "nonfinite_tableau" in text:
        return 5
    return 6


def inverse_metrics(row, n):
    u = as_float(row.get("u"))
    success = as_bool(row.get("success", "False"))
    values = np.full(len(INV_METRIC_NAMES), np.nan, dtype=np.float64)
    if success:
        residual = as_float(row.get("right_inverse_scaled_residual_inf"))
        gamma = as_float(row.get("gamma_inf"))
        eta = as_float(row.get("inverse_backward_error_inf"))
        if math.isfinite(residual) and math.isfinite(u) and u > 0.0:
            values[0] = residual / u
        if math.isfinite(gamma) and n > 0:
            values[1] = gamma / float(n)
        if (
            as_bool(row.get("eta_inv_reliable", "False"))
            and math.isfinite(eta)
            and math.isfinite(u)
            and u > 0.0
        ):
            values[2] = eta / u
        values[3] = as_float(row.get("max_tableau_norm_inf"))
        values[4] = as_float(row.get("max_B_norm_inf"))
        if as_bool(row.get("inverse_tableau_norm_reliable", "False")):
            values[5] = as_float(row.get("max_inverse_tableau_norm_inf"))
        values[6] = as_float(row.get("max_multiplier"))
    values[7] = as_float(row.get("elapsed_seconds"))
    return values


def solution_metrics(row):
    values = np.full(len(SOL_METRIC_NAMES), np.nan, dtype=np.float64)
    if as_bool(row.get("success", "False")):
        for index, name in enumerate(SOL_METRIC_NAMES):
            values[index] = as_float(row.get(name))
    return values


def command_select(args):
    manifest = read_manifest(args.manifest)
    status_paths = sorted(Path(args.status_dir).glob("task_*.json"))
    statuses = {}
    for path in status_paths:
        status = load_json(path)
        if status.get("status") != "RECOVERY_OK":
            raise RuntimeError("Non-OK recovery status: {}".format(path))
        statuses[int(status["task_id"])] = status

    missing = [task for task in range(args.num_tasks) if task not in statuses]
    if missing:
        raise RuntimeError("Missing recovered task statuses: {}".format(missing[:20]))

    outdir = Path(args.outdir)
    selection_dir = outdir / "selection"
    if outdir.exists():
        shutil.rmtree(str(outdir))
    selection_dir.mkdir(parents=True, exist_ok=True)

    per_task = [[] for _ in range(args.num_tasks)]
    selected_blocks = []
    cell_dimensions = defaultdict(list)

    for row in manifest:
        if not row["stochastic"] or row["n"] > args.max_n:
            continue

        block = row["block_index"]
        case_start = row["case_start"]
        replicas = row["replicas"]
        candidates = []
        recovered_total = 0

        for task_id in range(args.num_tasks):
            count = int(statuses[task_id].get("block_counts", {}).get(str(block), 0))
            if count <= 0:
                continue
            first = (task_id - case_start) % args.num_tasks
            assigned_count = 0
            replica = first
            while replica < replicas:
                assigned_count += 1
                replica += args.num_tasks
            if count > assigned_count:
                raise RuntimeError(
                    "Block {} task {} recovered {} > assigned {}".format(
                        block, task_id, count, assigned_count
                    )
                )
            for local_index in range(count):
                replica = first + local_index * args.num_tasks
                score = hash64(
                    "castillo-balanced|{}|{}|{}".format(
                        row["cell_id"], row["n"], replica
                    )
                )
                candidates.append((score, task_id, replica))
            recovered_total += count

        if recovered_total != len(candidates):
            raise RuntimeError("Internal candidate count mismatch for block {}".format(block))
        if recovered_total < args.sample_size:
            raise RuntimeError(
                "Block {} has only {} recovered matrices; need {}".format(
                    block, recovered_total, args.sample_size
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = candidates[: args.sample_size]
        for rank, (_, task_id, replica) in enumerate(chosen):
            per_task[task_id].append((block, replica, rank))

        selected_blocks.append(
            {
                "block_index": block,
                "cell_id": row["cell_id"],
                "family": row["family"],
                "n": row["n"],
                "parameters_json": row["parameters_json"],
                "recovered_count": recovered_total,
                "selected_count": args.sample_size,
            }
        )
        cell_dimensions[row["cell_id"]].append(row["n"])

    expected_blocks = 64 * 13
    if len(selected_blocks) != expected_blocks:
        raise RuntimeError(
            "Expected {} selected stochastic blocks, found {}".format(
                expected_blocks, len(selected_blocks)
            )
        )

    total_selected = 0
    for task_id, items in enumerate(per_task):
        items.sort()
        if items:
            array = np.asarray(items, dtype=np.int32)
            blocks = array[:, 0]
            replicas = array[:, 1]
            ranks = array[:, 2]
        else:
            blocks = np.empty(0, dtype=np.int32)
            replicas = np.empty(0, dtype=np.int32)
            ranks = np.empty(0, dtype=np.int32)
        total_selected += int(blocks.size)
        np.savez_compressed(
            str(selection_dir / "task_{:04d}.npz".format(task_id)),
            block=blocks,
            replica=replicas,
            rank=ranks,
        )

    expected_selected = expected_blocks * args.sample_size
    if total_selected != expected_selected:
        raise RuntimeError(
            "Selected matrix total {} != {}".format(total_selected, expected_selected)
        )

    for cell_id, dimensions in cell_dimensions.items():
        if sorted(dimensions) != sorted(set(dimensions)) or len(dimensions) != 13:
            raise RuntimeError("Cell {} does not have 13 selected dimensions".format(cell_id))

    summary = {
        "status": "ANALYSIS_SELECTION_OK",
        "num_tasks": args.num_tasks,
        "sample_size": args.sample_size,
        "max_n": args.max_n,
        "selected_block_count": len(selected_blocks),
        "selected_matrix_count": total_selected,
        "stochastic_cell_count": len(cell_dimensions),
        "selected_blocks": sorted(selected_blocks, key=lambda x: x["block_index"]),
        "selection_dir": str(selection_dir),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ANALYSIS_SELECTION_OK")


def command_shard(args):
    config = load_config(args.config)
    maps = combo_maps(config)
    manifest = read_manifest(args.manifest)
    manifest_by_block = {row["block_index"]: row for row in manifest}

    selection_path = Path(args.selection_dir) / "task_{:04d}.npz".format(args.task_id)
    selection_data = np.load(str(selection_path), allow_pickle=False)
    selected_lookup = {}
    for block, replica, rank in zip(
        selection_data["block"], selection_data["replica"], selection_data["rank"]
    ):
        key = (int(block) << 32) | int(replica)
        selected_lookup[key] = int(rank)

    recovered_root = Path(args.recovered_root)
    inverse_path = recovered_root / "inverse" / "inverse_{:04d}.csv.gz".format(args.task_id)
    solution_path = recovered_root / "solution" / "solution_{:04d}.csv.gz".format(args.task_id)
    recovery_status = load_json(
        recovered_root / "status" / "task_{:04d}.json".format(args.task_id)
    )

    all_counts = {}
    core_gid = []
    core_rRu = []
    core_gamma = []
    sel_inv_lid = []
    sel_inv_success = []
    sel_inv_metric = []
    det_gid = []
    det_success = []
    det_metric = []
    selected_mid = {}

    censor = np.zeros((maps["num_inv_combo"], 6), dtype=np.float64)
    # columns: total, success, sum(log10 rRu), sum(log10 gamma/n),
    #          sum(log10 max tableau), sum(elapsed)

    start = time.perf_counter()
    inverse_rows = 0

    with gzip.open(str(inverse_path), "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            inverse_rows += 1
            block = int(row["block_index"])
            replica = int(row["replica"])
            meta = manifest_by_block[block]
            combo = maps["inv_combo"][(row["method"], row["dtype_name"])]
            gid = block * maps["num_inv_combo"] + combo
            counts = all_counts.get(gid)
            if counts is None:
                counts = np.zeros(len(FAILURE_NAMES), dtype=np.int64)
                all_counts[gid] = counts
            counts[0] += 1
            category = failure_index(row)
            counts[category] += 1

            success = as_bool(row.get("success", "False"))
            metrics = inverse_metrics(row, meta["n"])

            if meta["stochastic"] and meta["n"] <= args.core_max_n:
                core_gid.append(gid)
                core_rRu.append(metrics[0])
                core_gamma.append(metrics[1])

            if not meta["stochastic"]:
                det_gid.append(gid)
                det_success.append(1 if success else 0)
                det_metric.append(metrics)

            packed = (block << 32) | replica
            rank = selected_lookup.get(packed)
            if rank is not None:
                lid = ((block * args.sample_size + rank) * maps["num_inv_combo"]) + combo
                sel_inv_lid.append(lid)
                sel_inv_success.append(1 if success else 0)
                sel_inv_metric.append(metrics)
                selected_mid[row["matrix_id"]] = (block, rank)

            if meta["stochastic"] and meta["n"] == args.censor_n:
                censor[combo, 0] += 1.0
                if success:
                    censor[combo, 1] += 1.0
                    if math.isfinite(metrics[0]) and metrics[0] > 0.0:
                        censor[combo, 2] += math.log10(metrics[0])
                    if math.isfinite(metrics[1]) and metrics[1] > 0.0:
                        censor[combo, 3] += math.log10(metrics[1])
                    if math.isfinite(metrics[3]) and metrics[3] > 0.0:
                        censor[combo, 4] += math.log10(metrics[3])
                if math.isfinite(metrics[7]):
                    censor[combo, 5] += metrics[7]

    expected_selected_matrices = len(selected_lookup)
    expected_selected_inverse = expected_selected_matrices * maps["num_inv_combo"]
    if len(sel_inv_lid) != expected_selected_inverse:
        raise RuntimeError(
            "Task {} selected inverse rows {} != {}".format(
                args.task_id, len(sel_inv_lid), expected_selected_inverse
            )
        )
    if len(selected_mid) != expected_selected_matrices:
        raise RuntimeError(
            "Task {} selected matrix IDs {} != {}".format(
                args.task_id, len(selected_mid), expected_selected_matrices
            )
        )

    sel_sol_lid = []
    sel_sol_success = []
    sel_sol_metric = []
    solution_rows = 0
    selected_solution_rows = 0

    with gzip.open(str(solution_path), "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            solution_rows += 1
            location = selected_mid.get(row["matrix_id"])
            if location is None:
                continue
            block, rank = location
            combo = maps["sol_combo"][(
                row["method"],
                row["dtype_name"],
                row["rhs_type"],
                int(row["refinement_steps"]),
            )]
            lid = ((block * args.sample_size + rank) * maps["num_sol_combo"]) + combo
            sel_sol_lid.append(lid)
            sel_sol_success.append(1 if as_bool(row.get("success", "False")) else 0)
            sel_sol_metric.append(solution_metrics(row))
            selected_solution_rows += 1

    expected_selected_solution = expected_selected_matrices * maps["num_sol_combo"]
    if selected_solution_rows != expected_selected_solution:
        raise RuntimeError(
            "Task {} selected solution rows {} != {}".format(
                args.task_id, selected_solution_rows, expected_selected_solution
            )
        )

    all_gid = np.asarray(sorted(all_counts), dtype=np.int32)
    all_count_matrix = np.asarray([all_counts[int(gid)] for gid in all_gid], dtype=np.int64)

    def metric_array(items, width):
        if not items:
            return np.empty((0, width), dtype=np.float64)
        return np.asarray(items, dtype=np.float64)

    meta = {
        "status": "ANALYSIS_SHARD_OK",
        "task_id": args.task_id,
        "last_case_index": recovery_status.get("last_case_index"),
        "recovered_matrices": recovery_status.get("recovered_matrices"),
        "inverse_rows_read": inverse_rows,
        "solution_rows_read": solution_rows,
        "selected_matrices": expected_selected_matrices,
        "selected_inverse_rows": len(sel_inv_lid),
        "selected_solution_rows": len(sel_sol_lid),
        "elapsed_seconds": time.perf_counter() - start,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "task_{:04d}.npz".format(args.task_id)
    tmp_path = outdir / "task_{:04d}.npz.tmp".format(args.task_id)
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            all_gid=all_gid,
            all_counts=all_count_matrix,
            core_gid=np.asarray(core_gid, dtype=np.int32),
            core_rRu=np.asarray(core_rRu, dtype=np.float64),
            core_gamma=np.asarray(core_gamma, dtype=np.float64),
            sel_inv_lid=np.asarray(sel_inv_lid, dtype=np.int64),
            sel_inv_success=np.asarray(sel_inv_success, dtype=np.uint8),
            sel_inv_metric=metric_array(sel_inv_metric, len(INV_METRIC_NAMES)),
            sel_sol_lid=np.asarray(sel_sol_lid, dtype=np.int64),
            sel_sol_success=np.asarray(sel_sol_success, dtype=np.uint8),
            sel_sol_metric=metric_array(sel_sol_metric, len(SOL_METRIC_NAMES)),
            det_gid=np.asarray(det_gid, dtype=np.int32),
            det_success=np.asarray(det_success, dtype=np.uint8),
            det_metric=metric_array(det_metric, len(INV_METRIC_NAMES)),
            censor=censor,
            meta=np.asarray(json.dumps(meta, sort_keys=True)),
        )
    os.replace(str(tmp_path), str(output_path))
    print(json.dumps(meta, indent=2, sort_keys=True))
    print("ANALYSIS_SHARD_OK={:04d}".format(args.task_id))


def ols_fit(x, y, design):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < design + 1:
        return None
    if design == 2:
        matrix = np.column_stack([np.ones(x.size), x])
    elif design == 3:
        matrix = np.column_stack([np.ones(x.size), x, x * x])
    else:
        raise ValueError("Unsupported design")
    coef, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = matrix @ coef
    residual = y - fitted
    sse = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    sst = float(np.sum(centered * centered))
    r2 = 1.0 - sse / sst if sst > 0.0 else math.nan
    k = matrix.shape[1]
    aic = x.size * math.log(max(sse / x.size, np.finfo(np.float64).tiny)) + 2.0 * k
    return {
        "coef": [float(value) for value in coef],
        "sse": sse,
        "r2": r2,
        "aic": aic,
        "n_points": int(x.size),
    }


def fit_curve(n_values, q_values):
    n = np.asarray(n_values, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    valid = np.isfinite(q) & (q > 0.0)
    n = n[valid]
    q = q[valid]
    if n.size < 3:
        return None
    logn = np.log(n)
    logq = np.log(q)
    power = ols_fit(logn, logq, 2)
    quadratic = ols_fit(logn, logq, 3)
    exponential = ols_fit(n, logq, 2)
    if power is None:
        return None

    slopes = []
    for omit in range(n.size):
        mask = np.ones(n.size, dtype=bool)
        mask[omit] = False
        trial = ols_fit(logn[mask], logq[mask], 2)
        if trial is not None:
            slopes.append(trial["coef"][1])
    slope = power["coef"][1]
    lono_max = max([abs(value - slope) for value in slopes] or [math.nan])

    local = {}
    lookup = {int(value): index for index, value in enumerate(n.astype(int))}
    for left, right, name in ((45, 64, "s45_64"), (64, 128, "s64_128"), (128, 181, "s128_181")):
        if left in lookup and right in lookup:
            i = lookup[left]
            j = lookup[right]
            local[name] = float((logq[j] - logq[i]) / (logn[j] - logn[i]))
        else:
            local[name] = math.nan
    neighbors = [local["s45_64"], local["s128_181"]]
    neighbors = [value for value in neighbors if math.isfinite(value)]
    anomaly = (
        local["s64_128"] - float(np.mean(neighbors))
        if math.isfinite(local["s64_128"]) and neighbors
        else math.nan
    )

    return {
        "power_intercept": power["coef"][0],
        "power_slope": power["coef"][1],
        "power_r2": power["r2"],
        "power_aic": power["aic"],
        "quadratic_intercept": quadratic["coef"][0],
        "quadratic_linear": quadratic["coef"][1],
        "quadratic_curvature": quadratic["coef"][2],
        "quadratic_r2": quadratic["r2"],
        "quadratic_aic": quadratic["aic"],
        "exponential_intercept": exponential["coef"][0],
        "exponential_rate": exponential["coef"][1],
        "exponential_r2": exponential["r2"],
        "exponential_aic": exponential["aic"],
        "lono_max_abs_slope_change": lono_max,
        "s45_64": local["s45_64"],
        "s64_128": local["s64_128"],
        "s128_181": local["s128_181"],
        "anomaly_64_128": anomaly,
        "n_points": power["n_points"],
    }


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(tmp), str(path))


def standardized_difference(low, high):
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if low.size < 2 or high.size < 2:
        return math.nan
    variance = ((low.size - 1) * np.var(low, ddof=1) + (high.size - 1) * np.var(high, ddof=1))
    variance /= float(low.size + high.size - 2)
    if variance <= 0.0:
        return 0.0
    return float((np.mean(high) - np.mean(low)) / math.sqrt(variance))


def command_finalize(args):
    config = load_config(args.config)
    analysis_config = load_json(args.analysis_config)
    maps = combo_maps(config)
    manifest = read_manifest(args.manifest)
    manifest_by_block = {row["block_index"]: row for row in manifest}
    selection_summary = load_json(args.selection_summary)
    sample_size = int(selection_summary["sample_size"])
    selected_blocks = [int(item["block_index"]) for item in selection_summary["selected_blocks"]]
    selected_blocks.sort()
    block_to_selected = np.full(len(manifest), -1, dtype=np.int32)
    for index, block in enumerate(selected_blocks):
        block_to_selected[block] = index

    partial_paths = sorted(Path(args.partial_dir).glob("task_*.npz"))
    if len(partial_paths) != args.num_tasks:
        raise RuntimeError(
            "Expected {} analysis partials, found {}".format(args.num_tasks, len(partial_paths))
        )

    num_inv = maps["num_inv_combo"]
    num_sol = maps["num_sol_combo"]
    max_gid = len(manifest) * num_inv
    core_counts = np.zeros(max_gid, dtype=np.int64)
    all_counts = np.zeros((max_gid, len(FAILURE_NAMES)), dtype=np.int64)
    det_gid_parts = []
    det_success_parts = []
    det_metric_parts = []
    task_censor = []
    task_progress = []
    partial_meta = []

    for path in partial_paths:
        with np.load(str(path), allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            partial_meta.append(meta)
            gids = data["core_gid"]
            if gids.size:
                core_counts += np.bincount(gids, minlength=max_gid)
            np.add.at(all_counts, data["all_gid"], data["all_counts"])
            if data["det_gid"].size:
                det_gid_parts.append(data["det_gid"].copy())
                det_success_parts.append(data["det_success"].copy())
                det_metric_parts.append(data["det_metric"].copy())
            task_censor.append(data["censor"].copy())
            task_progress.append(float(meta.get("last_case_index", math.nan)))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(args.tmpdir)
    if tmpdir.exists():
        shutil.rmtree(str(tmpdir))
    tmpdir.mkdir(parents=True, exist_ok=True)

    core_offsets = np.zeros(max_gid + 1, dtype=np.int64)
    core_offsets[1:] = np.cumsum(core_counts)
    core_total = int(core_offsets[-1])
    core_rRu_path = tmpdir / "core_rRu.dat"
    core_gamma_path = tmpdir / "core_gamma.dat"
    core_rRu = np.memmap(str(core_rRu_path), dtype=np.float64, mode="w+", shape=(core_total,))
    core_gamma = np.memmap(str(core_gamma_path), dtype=np.float64, mode="w+", shape=(core_total,))
    core_cursor = core_offsets[:-1].copy()

    bcount = len(selected_blocks)
    inv_shape = (bcount, sample_size, num_inv, len(INV_METRIC_NAMES))
    inv_success_shape = (bcount, sample_size, num_inv)
    sol_shape = (bcount, sample_size, num_sol, len(SOL_METRIC_NAMES))
    sol_success_shape = (bcount, sample_size, num_sol)

    sel_inv_path = tmpdir / "selected_inverse.dat"
    sel_inv_success_path = tmpdir / "selected_inverse_success.dat"
    sel_sol_path = tmpdir / "selected_solution.dat"
    sel_sol_success_path = tmpdir / "selected_solution_success.dat"

    sel_inv = np.memmap(str(sel_inv_path), dtype=np.float64, mode="w+", shape=inv_shape)
    sel_inv[:] = np.nan
    sel_inv_success = np.memmap(
        str(sel_inv_success_path), dtype=np.uint8, mode="w+", shape=inv_success_shape
    )
    sel_inv_success[:] = 255
    sel_sol = np.memmap(str(sel_sol_path), dtype=np.float64, mode="w+", shape=sol_shape)
    sel_sol[:] = np.nan
    sel_sol_success = np.memmap(
        str(sel_sol_success_path), dtype=np.uint8, mode="w+", shape=sol_success_shape
    )
    sel_sol_success[:] = 255

    for path in partial_paths:
        with np.load(str(path), allow_pickle=False) as data:
            gids = data["core_gid"]
            if gids.size:
                order = np.argsort(gids, kind="stable")
                sorted_gid = gids[order]
                sorted_r = data["core_rRu"][order]
                sorted_g = data["core_gamma"][order]
                unique, starts, counts = np.unique(sorted_gid, return_index=True, return_counts=True)
                for gid, start, count in zip(unique, starts, counts):
                    gid = int(gid)
                    begin = int(core_cursor[gid])
                    end = begin + int(count)
                    core_rRu[begin:end] = sorted_r[start : start + count]
                    core_gamma[begin:end] = sorted_g[start : start + count]
                    core_cursor[gid] = end

            lids = data["sel_inv_lid"]
            if lids.size:
                combo = lids % num_inv
                work = lids // num_inv
                rank = work % sample_size
                block = work // sample_size
                bindex = block_to_selected[block]
                if np.any(bindex < 0):
                    raise RuntimeError("Selected inverse row references a non-selected block")
                sel_inv[bindex, rank, combo, :] = data["sel_inv_metric"]
                sel_inv_success[bindex, rank, combo] = data["sel_inv_success"]

            lids = data["sel_sol_lid"]
            if lids.size:
                combo = lids % num_sol
                work = lids // num_sol
                rank = work % sample_size
                block = work // sample_size
                bindex = block_to_selected[block]
                if np.any(bindex < 0):
                    raise RuntimeError("Selected solution row references a non-selected block")
                sel_sol[bindex, rank, combo, :] = data["sel_sol_metric"]
                sel_sol_success[bindex, rank, combo] = data["sel_sol_success"]

    if not np.array_equal(core_cursor, core_offsets[1:]):
        bad = np.flatnonzero(core_cursor != core_offsets[1:])
        raise RuntimeError("Core fill mismatch in groups {}".format(bad[:20].tolist()))
    missing_inv = int(np.count_nonzero(sel_inv_success == 255))
    missing_sol = int(np.count_nonzero(sel_sol_success == 255))
    if missing_inv or missing_sol:
        raise RuntimeError(
            "Missing selected entries: inverse={} solution={}".format(missing_inv, missing_sol)
        )

    core_rows = []
    for gid in np.flatnonzero(core_counts):
        block = int(gid // num_inv)
        combo = int(gid % num_inv)
        meta = manifest_by_block[block]
        if not meta["stochastic"] or meta["n"] > int(analysis_config["complete_core_max_n"]):
            continue
        begin = int(core_offsets[gid])
        end = int(core_offsets[gid + 1])
        rstats = quantile_values(core_rRu[begin:end])
        gstats = quantile_values(core_gamma[begin:end])
        counts = all_counts[gid]
        row = {
            "block_index": block,
            "cell_id": meta["cell_id"],
            "family": meta["family"],
            "n": meta["n"],
            "parameters_json": meta["parameters_json"],
            "method": maps["inv_decode"][combo]["method"],
            "dtype_name": maps["inv_decode"][combo]["dtype_name"],
            "total": int(counts[0]),
            "success": int(counts[1]),
            "failure_rate": 1.0 - float(counts[1]) / float(counts[0]) if counts[0] else math.nan,
        }
        for name in ("q50", "q90", "q95", "q99"):
            row["rRu_" + name] = rstats[name]
            row["gamma_over_n_" + name] = gstats[name]
        core_rows.append(row)

    balanced_inverse_rows = []
    for bindex, block in enumerate(selected_blocks):
        meta = manifest_by_block[block]
        for combo in range(num_inv):
            success = sel_inv_success[bindex, :, combo] == 1
            row = {
                "block_index": block,
                "cell_id": meta["cell_id"],
                "family": meta["family"],
                "n": meta["n"],
                "parameters_json": meta["parameters_json"],
                "method": maps["inv_decode"][combo]["method"],
                "dtype_name": maps["inv_decode"][combo]["dtype_name"],
                "total": sample_size,
                "success": int(np.count_nonzero(success)),
                "failure_rate": 1.0 - float(np.count_nonzero(success)) / float(sample_size),
            }
            for metric_index, metric_name in enumerate(INV_METRIC_NAMES):
                stats = quantile_values(sel_inv[bindex, :, combo, metric_index])
                row[metric_name + "_finite_count"] = stats["finite_count"]
                for name in ("q50", "q90", "q95", "q99"):
                    row[metric_name + "_" + name] = stats[name]
            balanced_inverse_rows.append(row)

    balanced_solution_rows = []
    for bindex, block in enumerate(selected_blocks):
        meta = manifest_by_block[block]
        for combo in range(num_sol):
            success = sel_sol_success[bindex, :, combo] == 1
            decode = maps["sol_decode"][combo]
            row = {
                "block_index": block,
                "cell_id": meta["cell_id"],
                "family": meta["family"],
                "n": meta["n"],
                "parameters_json": meta["parameters_json"],
                "method": decode["method"],
                "dtype_name": decode["dtype_name"],
                "rhs_type": decode["rhs_type"],
                "refinement_steps": decode["refinement_steps"],
                "total": sample_size,
                "success": int(np.count_nonzero(success)),
                "failure_rate": 1.0 - float(np.count_nonzero(success)) / float(sample_size),
            }
            for metric_index, metric_name in enumerate(SOL_METRIC_NAMES):
                stats = quantile_values(sel_sol[bindex, :, combo, metric_index])
                row[metric_name + "_finite_count"] = stats["finite_count"]
                for name in ("q50", "q90", "q95", "q99"):
                    row[metric_name + "_" + name] = stats[name]
            balanced_solution_rows.append(row)

    failure_rows = []
    for gid in np.flatnonzero(all_counts[:, 0]):
        block = int(gid // num_inv)
        combo = int(gid % num_inv)
        meta = manifest_by_block[block]
        counts = all_counts[gid]
        row = {
            "block_index": block,
            "cell_id": meta["cell_id"],
            "family": meta["family"],
            "stochastic": meta["stochastic"],
            "n": meta["n"],
            "parameters_json": meta["parameters_json"],
            "method": maps["inv_decode"][combo]["method"],
            "dtype_name": maps["inv_decode"][combo]["dtype_name"],
        }
        for index, name in enumerate(FAILURE_NAMES):
            row[name] = int(counts[index])
        row["failure_rate"] = 1.0 - float(counts[1]) / float(counts[0])
        failure_rows.append(row)

    if det_gid_parts:
        det_gid = np.concatenate(det_gid_parts)
        det_success = np.concatenate(det_success_parts)
        det_metric = np.concatenate(det_metric_parts, axis=0)
    else:
        det_gid = np.empty(0, dtype=np.int32)
        det_success = np.empty(0, dtype=np.uint8)
        det_metric = np.empty((0, len(INV_METRIC_NAMES)), dtype=np.float64)

    deterministic_rows = []
    for index in range(det_gid.size):
        gid = int(det_gid[index])
        block = gid // num_inv
        combo = gid % num_inv
        meta = manifest_by_block[block]
        row = {
            "block_index": block,
            "cell_id": meta["cell_id"],
            "family": meta["family"],
            "n": meta["n"],
            "parameters_json": meta["parameters_json"],
            "method": maps["inv_decode"][combo]["method"],
            "dtype_name": maps["inv_decode"][combo]["dtype_name"],
            "success": int(det_success[index]),
        }
        for metric_index, metric_name in enumerate(INV_METRIC_NAMES):
            row[metric_name] = float(det_metric[index, metric_index])
        deterministic_rows.append(row)

    refinement_rows = []
    sol_lookup = maps["sol_combo"]
    for bindex, block in enumerate(selected_blocks):
        meta = manifest_by_block[block]
        for dtype_name in maps["dtypes"]:
            for method in maps["methods"]:
                for rhs_type in maps["rhs_types"]:
                    if 0 not in maps["refinements"] or 2 not in maps["refinements"]:
                        continue
                    c0 = sol_lookup[(method, dtype_name, rhs_type, 0)]
                    c2 = sol_lookup[(method, dtype_name, rhs_type, 2)]
                    f0 = sel_sol[bindex, :, c0, 0]
                    f2 = sel_sol[bindex, :, c2, 0]
                    valid = np.isfinite(f0) & np.isfinite(f2) & (f0 > 0.0)
                    ratio = np.full(sample_size, np.nan, dtype=np.float64)
                    ratio[valid] = f2[valid] / f0[valid]
                    stats = quantile_values(ratio)
                    refinement_rows.append(
                        {
                            "block_index": block,
                            "cell_id": meta["cell_id"],
                            "family": meta["family"],
                            "n": meta["n"],
                            "method": method,
                            "dtype_name": dtype_name,
                            "rhs_type": rhs_type,
                            "paired_count": stats["finite_count"],
                            "forward_error_ratio_step2_over_step0_q50": stats["q50"],
                            "forward_error_ratio_step2_over_step0_q90": stats["q90"],
                            "forward_error_ratio_step2_over_step0_q95": stats["q95"],
                            "success_step0": int(np.count_nonzero(sel_sol_success[bindex, :, c0] == 1)),
                            "success_step2": int(np.count_nonzero(sel_sol_success[bindex, :, c2] == 1)),
                        }
                    )

    # Exact core versus balanced-sample sensitivity.
    core_lookup = {}
    for row in core_rows:
        key = (row["block_index"], row["method"], row["dtype_name"])
        core_lookup[key] = row
    sensitivity_rows = []
    for row in balanced_inverse_rows:
        if int(row["n"]) > int(analysis_config["complete_core_max_n"]):
            continue
        key = (row["block_index"], row["method"], row["dtype_name"])
        full = core_lookup[key]
        item = {
            "block_index": row["block_index"],
            "cell_id": row["cell_id"],
            "family": row["family"],
            "n": row["n"],
            "method": row["method"],
            "dtype_name": row["dtype_name"],
        }
        for name in ("q50", "q90", "q95", "q99"):
            a = float(full["rRu_" + name])
            b = float(row["rRu_" + name])
            item["rRu_{}_relative_difference".format(name)] = (
                (b - a) / a if math.isfinite(a) and a != 0.0 and math.isfinite(b) else math.nan
            )
        sensitivity_rows.append(item)

    # Fits by parameter cell.
    balanced_by_key = defaultdict(list)
    for row in balanced_inverse_rows:
        for _, qname in QUANTILE_NAMES:
            balanced_by_key[(row["cell_id"], row["family"], row["method"], row["dtype_name"], qname)].append(
                (int(row["n"]), float(row["rRu_" + qname]))
            )

    fit_rows = []
    for key, points in sorted(balanced_by_key.items()):
        points.sort()
        fit = fit_curve([item[0] for item in points], [item[1] for item in points])
        if fit is None:
            continue
        cell_id, family, method, dtype_name, qname = key
        row = {
            "cell_id": cell_id,
            "family": family,
            "method": method,
            "dtype_name": dtype_name,
            "quantile": qname,
            "n_min": min(item[0] for item in points),
            "n_max": max(item[0] for item in points),
        }
        row.update(fit)
        fit_rows.append(row)

    # Equal-cell-weight family curves and fits.
    family_points = defaultdict(lambda: defaultdict(list))
    for row in balanced_inverse_rows:
        for _, qname in QUANTILE_NAMES:
            family_points[(row["family"], row["method"], row["dtype_name"], qname)][int(row["n"])].append(
                float(row["rRu_" + qname])
            )
    family_fit_rows = []
    for key, by_n in sorted(family_points.items()):
        n_values = sorted(by_n)
        values = [float(np.nanmedian(np.asarray(by_n[n], dtype=np.float64))) for n in n_values]
        fit = fit_curve(n_values, values)
        if fit is None:
            continue
        family, method, dtype_name, qname = key
        row = {
            "family": family,
            "method": method,
            "dtype_name": dtype_name,
            "quantile": qname,
            "n_min": min(n_values),
            "n_max": max(n_values),
            "cell_count": max(len(by_n[n]) for n in n_values),
        }
        row.update(fit)
        family_fit_rows.append(row)

    # Precision consistency from matched cell fits.
    fit_index = {
        (row["cell_id"], row["method"], row["quantile"], row["dtype_name"]): row
        for row in fit_rows
    }
    precision_rows = []
    for row in fit_rows:
        if row["dtype_name"] != "float32":
            continue
        other = fit_index.get((row["cell_id"], row["method"], row["quantile"], "float64"))
        if other is None:
            continue
        precision_rows.append(
            {
                "cell_id": row["cell_id"],
                "family": row["family"],
                "method": row["method"],
                "quantile": row["quantile"],
                "s_float32": row["power_slope"],
                "s_float64": other["power_slope"],
                "s_float32_minus_float64": row["power_slope"] - other["power_slope"],
                "r2_float32": row["power_r2"],
                "r2_float64": other["power_r2"],
            }
        )

    # Censoring audit: compare slowest and fastest progress quartiles at n=181.
    censor_array = np.asarray(task_censor, dtype=np.float64)
    progress = np.asarray(task_progress, dtype=np.float64)
    finite_progress = np.isfinite(progress)
    q25, q75 = np.quantile(progress[finite_progress], [0.25, 0.75])
    low_mask = progress <= q25
    high_mask = progress >= q75
    censor_rows = []
    for combo in range(num_inv):
        total = censor_array[:, combo, 0]
        success = censor_array[:, combo, 1]
        failure_rate = np.where(total > 0.0, 1.0 - success / total, np.nan)
        mean_log_r = np.where(success > 0.0, censor_array[:, combo, 2] / success, np.nan)
        mean_log_g = np.where(success > 0.0, censor_array[:, combo, 3] / success, np.nan)
        mean_log_v = np.where(success > 0.0, censor_array[:, combo, 4] / success, np.nan)
        mean_elapsed = np.where(total > 0.0, censor_array[:, combo, 5] / total, np.nan)
        decode = maps["inv_decode"][combo]
        for metric_name, values in (
            ("failure_rate", failure_rate),
            ("mean_log10_rRu", mean_log_r),
            ("mean_log10_gamma_over_n", mean_log_g),
            ("mean_log10_max_tableau", mean_log_v),
            ("mean_elapsed_seconds", mean_elapsed),
        ):
            low = values[low_mask]
            high = values[high_mask]
            censor_rows.append(
                {
                    "method": decode["method"],
                    "dtype_name": decode["dtype_name"],
                    "metric": metric_name,
                    "low_progress_tasks": int(np.count_nonzero(np.isfinite(low))),
                    "high_progress_tasks": int(np.count_nonzero(np.isfinite(high))),
                    "low_mean": float(np.nanmean(low)),
                    "high_mean": float(np.nanmean(high)),
                    "high_minus_low": float(np.nanmean(high) - np.nanmean(low)),
                    "standardized_difference": standardized_difference(low, high),
                    "progress_q25": float(q25),
                    "progress_q75": float(q75),
                }
            )

    core_fields = list(core_rows[0].keys())
    inv_fields = list(balanced_inverse_rows[0].keys())
    sol_fields = list(balanced_solution_rows[0].keys())
    failure_fields = list(failure_rows[0].keys())
    det_fields = list(deterministic_rows[0].keys()) if deterministic_rows else []
    refinement_fields = list(refinement_rows[0].keys())
    sensitivity_fields = list(sensitivity_rows[0].keys())
    fit_fields = list(fit_rows[0].keys())
    family_fit_fields = list(family_fit_rows[0].keys())
    precision_fields = list(precision_rows[0].keys())
    censor_fields = list(censor_rows[0].keys())

    write_csv(report_dir / "inverse_core_exact_quantiles.csv", core_fields, core_rows)
    write_csv(report_dir / "inverse_balanced_quantiles.csv", inv_fields, balanced_inverse_rows)
    write_csv(report_dir / "solution_balanced_quantiles.csv", sol_fields, balanced_solution_rows)
    write_csv(report_dir / "inverse_failure_rates_all_recovered.csv", failure_fields, failure_rows)
    if deterministic_rows:
        write_csv(report_dir / "deterministic_sequences.csv", det_fields, deterministic_rows)
    write_csv(report_dir / "refinement_effects.csv", refinement_fields, refinement_rows)
    write_csv(report_dir / "core_vs_balanced_sensitivity.csv", sensitivity_fields, sensitivity_rows)
    write_csv(report_dir / "inverse_power_fits_by_cell.csv", fit_fields, fit_rows)
    write_csv(report_dir / "inverse_power_fits_by_family.csv", family_fit_fields, family_fit_rows)
    write_csv(report_dir / "precision_consistency.csv", precision_fields, precision_rows)
    write_csv(report_dir / "censoring_audit.csv", censor_fields, censor_rows)

    # Bootstrap inputs, one file per stochastic parameter cell.
    bootstrap_dir = Path(args.bootstrap_input_dir)
    if bootstrap_dir.exists():
        shutil.rmtree(str(bootstrap_dir))
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    blocks_by_cell = defaultdict(list)
    for block in selected_blocks:
        meta = manifest_by_block[block]
        blocks_by_cell[meta["cell_id"]].append((meta["n"], block))
    bootstrap_index = []
    cell_items = sorted(blocks_by_cell.items())
    for cell_index, (cell_id, pairs) in enumerate(cell_items):
        pairs.sort()
        if len(pairs) != 13:
            raise RuntimeError("Bootstrap cell {} has {} dimensions".format(cell_id, len(pairs)))
        n_values = np.asarray([item[0] for item in pairs], dtype=np.int32)
        block_values = np.asarray([item[1] for item in pairs], dtype=np.int32)
        bindices = np.asarray([block_to_selected[item[1]] for item in pairs], dtype=np.int32)
        data = np.asarray(sel_inv[bindices, :, :, 0], dtype=np.float64)
        success = np.asarray(sel_inv_success[bindices, :, :], dtype=np.uint8)
        np.savez_compressed(
            str(bootstrap_dir / "cell_{:02d}.npz".format(cell_index)),
            n=n_values,
            block=block_values,
            rRu=data,
            success=success,
            cell_id=np.asarray(cell_id),
            family=np.asarray(manifest_by_block[pairs[0][1]]["family"]),
        )
        for combo in range(num_inv):
            bootstrap_index.append(
                {
                    "task_id": cell_index * num_inv + combo,
                    "cell_index": cell_index,
                    "cell_id": cell_id,
                    "family": manifest_by_block[pairs[0][1]]["family"],
                    "combo": combo,
                    "method": maps["inv_decode"][combo]["method"],
                    "dtype_name": maps["inv_decode"][combo]["dtype_name"],
                    "input_path": str(bootstrap_dir / "cell_{:02d}.npz".format(cell_index)),
                }
            )
    write_json(report_dir / "bootstrap_index.json", {"tasks": bootstrap_index})

    max_censor_numeric = 0.0
    for row in censor_rows:
        if row["metric"] == "mean_elapsed_seconds":
            continue
        value = abs(float(row["standardized_difference"]))
        if math.isfinite(value):
            max_censor_numeric = max(max_censor_numeric, value)

    sensitivity_abs = []
    for row in sensitivity_rows:
        for name in ("q50", "q90", "q95", "q99"):
            value = as_float(row["rRu_{}_relative_difference".format(name)])
            if math.isfinite(value):
                sensitivity_abs.append(abs(value))

    preliminary = {
        "status": "ANALYSIS_FINALIZE_OK",
        "partial_tasks": len(partial_paths),
        "core_inverse_rows": core_total,
        "balanced_blocks": bcount,
        "balanced_matrices": bcount * sample_size,
        "balanced_inverse_rows": bcount * sample_size * num_inv,
        "balanced_solution_rows": bcount * sample_size * num_sol,
        "fit_rows": len(fit_rows),
        "bootstrap_tasks": len(bootstrap_index),
        "maximum_abs_standardized_censoring_difference_excluding_runtime": max_censor_numeric,
        "median_abs_core_vs_balanced_relative_quantile_difference": (
            float(np.median(sensitivity_abs)) if sensitivity_abs else math.nan
        ),
        "maximum_abs_core_vs_balanced_relative_quantile_difference": (
            float(np.max(sensitivity_abs)) if sensitivity_abs else math.nan
        ),
        "outputs": {
            "inverse_core_exact_quantiles": str(report_dir / "inverse_core_exact_quantiles.csv"),
            "inverse_balanced_quantiles": str(report_dir / "inverse_balanced_quantiles.csv"),
            "solution_balanced_quantiles": str(report_dir / "solution_balanced_quantiles.csv"),
            "failure_rates": str(report_dir / "inverse_failure_rates_all_recovered.csv"),
            "fits_by_cell": str(report_dir / "inverse_power_fits_by_cell.csv"),
            "fits_by_family": str(report_dir / "inverse_power_fits_by_family.csv"),
            "censoring_audit": str(report_dir / "censoring_audit.csv"),
            "bootstrap_index": str(report_dir / "bootstrap_index.json"),
        },
    }
    write_json(report_dir / "analysis_preliminary_summary.json", preliminary)

    # Flush and remove large temporary memmaps after bootstrap inputs and summaries exist.
    core_rRu.flush()
    core_gamma.flush()
    sel_inv.flush()
    sel_inv_success.flush()
    sel_sol.flush()
    sel_sol_success.flush()
    del core_rRu, core_gamma, sel_inv, sel_inv_success, sel_sol, sel_sol_success
    shutil.rmtree(str(tmpdir))

    print(json.dumps(preliminary, indent=2, sort_keys=True))
    print("ANALYSIS_FINALIZE_OK")


def command_bootstrap(args):
    index = load_json(args.index)["tasks"]
    if args.task_id < 0 or args.task_id >= len(index):
        raise ValueError("Invalid bootstrap task id")
    item = index[args.task_id]
    data = np.load(item["input_path"], allow_pickle=False)
    n_values = data["n"].astype(np.float64)
    rRu = data["rRu"][:, :, int(item["combo"])]
    sample_size = rRu.shape[1]
    quantiles = np.asarray([value for value, _ in QUANTILE_NAMES], dtype=np.float64)

    original_curves = np.full((len(QUANTILE_NAMES), n_values.size), np.nan, dtype=np.float64)
    for index_n in range(n_values.size):
        values = rRu[index_n]
        values = values[np.isfinite(values)]
        if values.size:
            original_curves[:, index_n] = np.quantile(values, quantiles)
    original_slopes = np.full(len(QUANTILE_NAMES), np.nan, dtype=np.float64)
    for qindex in range(len(QUANTILE_NAMES)):
        fit = fit_curve(n_values, original_curves[qindex])
        if fit is not None:
            original_slopes[qindex] = fit["power_slope"]

    seed = hash64("castillo-bootstrap|{}".format(item["cell_id"]))
    rng = np.random.default_rng(seed)
    slopes = np.full((args.replicates, len(QUANTILE_NAMES)), np.nan, dtype=np.float64)

    for replicate in range(args.replicates):
        curves = np.full((len(QUANTILE_NAMES), n_values.size), np.nan, dtype=np.float64)
        for index_n in range(n_values.size):
            indices = rng.integers(0, sample_size, size=sample_size)
            values = rRu[index_n, indices]
            values = values[np.isfinite(values)]
            if values.size:
                curves[:, index_n] = np.quantile(values, quantiles)
        for qindex in range(len(QUANTILE_NAMES)):
            fit = fit_curve(n_values, curves[qindex])
            if fit is not None:
                slopes[replicate, qindex] = fit["power_slope"]

    rows = []
    for qindex, (_, qname) in enumerate(QUANTILE_NAMES):
        values = slopes[:, qindex]
        values = values[np.isfinite(values)]
        rows.append(
            {
                "quantile": qname,
                "original_slope": float(original_slopes[qindex]),
                "bootstrap_mean": float(np.mean(values)) if values.size else math.nan,
                "bootstrap_bias": (
                    float(np.mean(values) - original_slopes[qindex]) if values.size else math.nan
                ),
                "ci_low": float(np.quantile(values, 0.025)) if values.size else math.nan,
                "ci_high": float(np.quantile(values, 0.975)) if values.size else math.nan,
                "valid_bootstrap_replicates": int(values.size),
            }
        )

    report = dict(item)
    report.update(
        {
            "status": "ANALYSIS_BOOTSTRAP_OK",
            "bootstrap_replicates_requested": args.replicates,
            "ci_method": "percentile_cluster_bootstrap_stratified_by_n",
            "rows": rows,
        }
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "task_{:04d}.json".format(args.task_id), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ANALYSIS_BOOTSTRAP_OK={:04d}".format(args.task_id))


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def command_report(args):
    report_dir = Path(args.report_dir)
    index = load_json(report_dir / "bootstrap_index.json")["tasks"]
    bootstrap_paths = sorted(Path(args.bootstrap_dir).glob("task_*.json"))
    if len(bootstrap_paths) != len(index):
        raise RuntimeError(
            "Expected {} bootstrap outputs, found {}".format(len(index), len(bootstrap_paths))
        )

    bootstrap_rows = []
    for path in bootstrap_paths:
        report = load_json(path)
        if report.get("status") != "ANALYSIS_BOOTSTRAP_OK":
            raise RuntimeError("Non-OK bootstrap report: {}".format(path))
        for row in report["rows"]:
            item = {
                "task_id": report["task_id"],
                "cell_id": report["cell_id"],
                "family": report["family"],
                "method": report["method"],
                "dtype_name": report["dtype_name"],
                "quantile": row["quantile"],
                "original_slope": row["original_slope"],
                "bootstrap_mean": row["bootstrap_mean"],
                "bootstrap_bias": row["bootstrap_bias"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "valid_bootstrap_replicates": row["valid_bootstrap_replicates"],
                "ci_method": report["ci_method"],
            }
            bootstrap_rows.append(item)
    write_csv(report_dir / "bootstrap_power_slopes.csv", list(bootstrap_rows[0].keys()), bootstrap_rows)

    preliminary = load_json(report_dir / "analysis_preliminary_summary.json")
    fits = read_csv_rows(report_dir / "inverse_power_fits_by_cell.csv")
    refinement = read_csv_rows(report_dir / "refinement_effects.csv")
    censor = read_csv_rows(report_dir / "censoring_audit.csv")
    failure = read_csv_rows(report_dir / "inverse_failure_rates_all_recovered.csv")
    balanced = read_csv_rows(report_dir / "inverse_balanced_quantiles.csv")
    solution = read_csv_rows(report_dir / "solution_balanced_quantiles.csv")
    completion = read_csv_rows(args.completion_by_n)

    slope_summary = []
    grouped = defaultdict(list)
    for row in fits:
        if row["quantile"] != "q95":
            continue
        grouped[(row["method"], row["dtype_name"])].append(float(row["power_slope"]))
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        slope_summary.append(
            {
                "method": key[0],
                "dtype_name": key[1],
                "cell_count": int(array.size),
                "median_q95_slope": float(np.median(array)),
                "q25_q95_slope": float(np.quantile(array, 0.25)),
                "q75_q95_slope": float(np.quantile(array, 0.75)),
            }
        )
    write_csv(report_dir / "q95_slope_summary.csv", list(slope_summary[0].keys()), slope_summary)

    bootstrap_coverage = []
    for row in bootstrap_rows:
        original = float(row["original_slope"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        bootstrap_coverage.append(low <= original <= high)

    ratio_values = []
    for row in refinement:
        value = as_float(row["forward_error_ratio_step2_over_step0_q50"])
        if math.isfinite(value):
            ratio_values.append(value)

    max_censor_numeric = 0.0
    max_censor_runtime = 0.0
    for row in censor:
        value = abs(as_float(row["standardized_difference"]))
        if not math.isfinite(value):
            continue
        if row["metric"] == "mean_elapsed_seconds":
            max_censor_runtime = max(max_censor_runtime, value)
        else:
            max_censor_numeric = max(max_censor_numeric, value)

    total_records = sum(int(row["total"]) for row in failure)
    total_failures = sum(int(row["total"]) - int(row["success"]) for row in failure)

    summary = {
        "status": "CASTILLO_ANALYSIS_COMPLETE",
        "balanced_matrices": preliminary["balanced_matrices"],
        "core_inverse_rows": preliminary["core_inverse_rows"],
        "all_recovered_inverse_records_counted": total_records,
        "all_recovered_inverse_failures_counted": total_failures,
        "overall_inverse_failure_rate": (
            float(total_failures) / float(total_records) if total_records else math.nan
        ),
        "bootstrap_tasks": len(index),
        "bootstrap_rows": len(bootstrap_rows),
        "bootstrap_original_slope_inside_percentile_interval_fraction": (
            float(np.mean(bootstrap_coverage)) if bootstrap_coverage else math.nan
        ),
        "median_paired_forward_error_ratio_step2_over_step0": (
            float(np.median(ratio_values)) if ratio_values else math.nan
        ),
        "maximum_abs_standardized_censoring_difference_excluding_runtime": max_censor_numeric,
        "maximum_abs_standardized_runtime_difference": max_censor_runtime,
        "slope_summary": slope_summary,
        "limitations": [
            "The complete 20000-replica core ends at n=181.",
            "The exact balanced extension uses 2000 matrices per stochastic cell through n=512.",
            "n=724 is not included in the balanced inferential analysis.",
            "n=1024 is absent from the recovered campaign.",
            "Bootstrap intervals are percentile cluster-bootstrap intervals; BCa intervals were not used because a 2000-cluster jackknife for every cell/method/precision combination would add disproportionate cost without changing the primary inferential target.",
        ],
    }
    write_json(report_dir / "analysis_final_summary.json", summary)

    # Multipanel figure.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) coverage
    ax = axes[0, 0]
    n_cov = [int(row["n"]) for row in completion]
    frac = [float(row["completion_fraction"]) for row in completion]
    ax.plot(n_cov, frac, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("n")
    ax.set_ylabel("Recovered fraction")
    ax.set_title("(a) Campaign recovery")
    ax.grid(True, alpha=0.3)

    # (b) q95 inverse residual scaling, equal-cell median
    ax = axes[0, 1]
    curves = defaultdict(lambda: defaultdict(list))
    for row in balanced:
        curves[(row["method"], row["dtype_name"])][int(row["n"])].append(
            as_float(row["rRu_q95"])
        )
    for key, by_n in sorted(curves.items()):
        n_values = sorted(by_n)
        y_values = [float(np.nanmedian(np.asarray(by_n[n], dtype=np.float64))) for n in n_values]
        ax.plot(n_values, y_values, marker="o", label="{} {}".format(key[0], key[1]))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Equal-cell median of q95(r_R/u)")
    ax.set_title("(b) Inverse residual growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    # (c) slope distributions
    ax = axes[1, 0]
    labels = []
    values = []
    for row in slope_summary:
        key = (row["method"], row["dtype_name"])
        data_values = grouped[key]
        labels.append("{}\n{}".format(key[0], key[1]))
        values.append(data_values)
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Power-law slope s for q95")
    ax.set_title("(c) Cellwise scaling exponents")
    ax.grid(True, axis="y", alpha=0.3)

    # (d) refinement effect
    ax = axes[1, 1]
    refine_curves = defaultdict(lambda: defaultdict(list))
    for row in refinement:
        key = (row["method"], row["dtype_name"])
        refine_curves[key][int(row["n"])].append(
            as_float(row["forward_error_ratio_step2_over_step0_q50"])
        )
    for key, by_n in sorted(refine_curves.items()):
        n_values = sorted(by_n)
        y_values = [float(np.nanmedian(np.asarray(by_n[n], dtype=np.float64))) for n in n_values]
        ax.plot(n_values, y_values, marker="o", label="{} {}".format(key[0], key[1]))
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Median forward-error ratio: step 2 / step 0")
    ax.set_title("(d) Iterative refinement")
    ax.grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(report_dir / "castillo_recovered_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    md_path = report_dir / "analysis_final_report.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Recovered Castillo campaign: statistical analysis\n\n")
        handle.write("## Data basis\n\n")
        handle.write(
            "The analysis uses the recovered, matrix-complete dataset. The complete core contains 20,000 replicas per stochastic parameter cell for n=8,...,181. A deterministic hash-based sample of exactly 2,000 matrices per stochastic cell is used for every dimension through n=512. All five methods and both precisions remain paired within every selected matrix.\n\n"
        )
        handle.write("## Global accounting\n\n")
        handle.write("- Balanced matrices: `{}`\n".format(summary["balanced_matrices"]))
        handle.write("- Exact core inverse rows: `{}`\n".format(summary["core_inverse_rows"]))
        handle.write("- Recovered inverse records counted: `{}`\n".format(total_records))
        handle.write("- Recovered inverse failures counted: `{}`\n".format(total_failures))
        handle.write("- Overall recovered inverse-failure rate: `{:.8e}`\n".format(summary["overall_inverse_failure_rate"]))
        handle.write("\n## Scaling of q95(r_R/u)\n\n")
        handle.write("| method | precision | cells | median s | q25 | q75 |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in slope_summary:
            handle.write(
                "| {} | {} | {} | {:.6g} | {:.6g} | {:.6g} |\n".format(
                    row["method"],
                    row["dtype_name"],
                    row["cell_count"],
                    row["median_q95_slope"],
                    row["q25_q95_slope"],
                    row["q75_q95_slope"],
                )
            )
        handle.write("\n## Robustness checks\n\n")
        handle.write(
            "- Maximum absolute standardized censoring difference in numerical metrics (runtime excluded): `{:.6g}`.\n".format(
                max_censor_numeric
            )
        )
        handle.write(
            "- Maximum standardized runtime difference between low- and high-progress task quartiles: `{:.6g}`.\n".format(
                max_censor_runtime
            )
        )
        handle.write(
            "- Median paired forward-error ratio after two refinement steps versus no refinement: `{:.6g}`. Values below one indicate improvement.\n".format(
                summary["median_paired_forward_error_ratio_step2_over_step0"]
            )
        )
        handle.write(
            "- Bootstrap tasks completed: `{}`; percentile cluster bootstrap with `{}` replicates per task.\n".format(
                len(index), args.bootstrap_replicates
            )
        )
        handle.write("\n## Files\n\n")
        for name in (
            "inverse_core_exact_quantiles.csv",
            "inverse_balanced_quantiles.csv",
            "solution_balanced_quantiles.csv",
            "inverse_failure_rates_all_recovered.csv",
            "deterministic_sequences.csv",
            "refinement_effects.csv",
            "core_vs_balanced_sensitivity.csv",
            "inverse_power_fits_by_cell.csv",
            "inverse_power_fits_by_family.csv",
            "precision_consistency.csv",
            "censoring_audit.csv",
            "bootstrap_power_slopes.csv",
            "q95_slope_summary.csv",
            "castillo_recovered_analysis.png",
        ):
            path = report_dir / name
            if path.exists():
                handle.write("- `{}`\n".format(path))
        handle.write("\n## Scope limitations\n\n")
        for item in summary["limitations"]:
            handle.write("- {}\n".format(item))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("CASTILLO_ANALYSIS_COMPLETE")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    select = subparsers.add_parser("select")
    select.add_argument("--manifest", required=True)
    select.add_argument("--status-dir", required=True)
    select.add_argument("--outdir", required=True)
    select.add_argument("--summary", required=True)
    select.add_argument("--num-tasks", type=int, default=999)
    select.add_argument("--sample-size", type=int, default=2000)
    select.add_argument("--max-n", type=int, default=512)
    select.set_defaults(func=command_select)

    shard = subparsers.add_parser("shard")
    shard.add_argument("--task-id", type=int, required=True)
    shard.add_argument("--manifest", required=True)
    shard.add_argument("--config", required=True)
    shard.add_argument("--recovered-root", required=True)
    shard.add_argument("--selection-dir", required=True)
    shard.add_argument("--outdir", required=True)
    shard.add_argument("--sample-size", type=int, default=2000)
    shard.add_argument("--core-max-n", type=int, default=181)
    shard.add_argument("--censor-n", type=int, default=181)
    shard.set_defaults(func=command_shard)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--analysis-config", required=True)
    finalize.add_argument("--selection-summary", required=True)
    finalize.add_argument("--partial-dir", required=True)
    finalize.add_argument("--report-dir", required=True)
    finalize.add_argument("--tmpdir", required=True)
    finalize.add_argument("--bootstrap-input-dir", required=True)
    finalize.add_argument("--num-tasks", type=int, default=999)
    finalize.set_defaults(func=command_finalize)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--task-id", type=int, required=True)
    bootstrap.add_argument("--index", required=True)
    bootstrap.add_argument("--outdir", required=True)
    bootstrap.add_argument("--replicates", type=int, default=2000)
    bootstrap.set_defaults(func=command_bootstrap)

    report = subparsers.add_parser("report")
    report.add_argument("--report-dir", required=True)
    report.add_argument("--bootstrap-dir", required=True)
    report.add_argument("--completion-by-n", required=True)
    report.add_argument("--bootstrap-replicates", type=int, default=2000)
    report.set_defaults(func=command_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
