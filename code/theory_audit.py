#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["R0_C0", "R0_C1", "R0_C2", "R1_C1", "R2_C2"]
DTYPES = ["float32", "float64"]
DD_FAMILIES = {
    "row_diagonal_dominant_random",
    "row_diagonal_dominant_stochastic",
}

# Direct scope of the diagonal-dominance theorem:
# no row pivoting, with first-nonzero or maximum-modulus column selection.
# The remaining variants are retained as additional empirical DD checks.
DD_THEOREM_METHODS = {"R0_C0", "R0_C1"}
DD_EMPIRICAL_METHODS = set(METHODS) - DD_THEOREM_METHODS

SELECTED_FIELDS = [
    "task_id", "matrix_id", "block_index", "cell_id", "family", "n", "replica", "rank",
    "method", "dtype_name", "u", "success", "failure_class",
    "n_row_interchanges", "n_interchanges",
    "gamma_inf", "gamma_inf_numerator", "gamma_inf_denominator", "gamma_inf_max_local", "gamma_inf_max_tail",
    "right_inverse_scaled_residual_inf", "right_inverse_scaled_residual_2_est",
    "rinf_over_u", "r2_over_u",
    "inverse_backward_error_2_est", "eta_inv_reliable", "eta_inv_reliability_bound_u", "eta2_over_u", "rho_inv",
    "max_tableau_norm_inf", "max_inverse_tableau_norm_inf", "max_B_norm_inf", "max_multiplier",
    "min_abs_pivot", "min_scaled_pivot", "alpha", "alpha_measured",
]

INEQUALITIES = [
    "alpha_measured_ge_alpha",
    "min_abs_pivot_ge_alpha",
    "max_tableau_le_2_over_alpha",
    "max_inverse_tableau_le_1",
    "max_B_le_3_over_alpha2",
    "gamma_le_6n_over_alpha3",
    "rinf_le_gamma_n3_6n_over_alpha3",
]


def as_float(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def as_int(value, default=-1):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def gamma_k(k, u):
    ku = float(k) * float(u)
    if not math.isfinite(ku) or ku >= 1.0:
        return float("inf")
    return ku / (1.0 - ku)


def finite(v):
    return math.isfinite(v)


def check_state():
    return {"tested": 0, "strict_violations": 0, "tolerant_violations": 0,
            "worst_ratio": None, "worst": None}


def update_check(state, ratio, lower_bound, tol, row):
    if not finite(ratio):
        return
    state["tested"] += 1
    if lower_bound:
        strict = ratio < 1.0
        tolerant = ratio < (1.0 - tol)
        worse = state["worst_ratio"] is None or ratio < state["worst_ratio"]
    else:
        strict = ratio > 1.0
        tolerant = ratio > (1.0 + tol)
        worse = state["worst_ratio"] is None or ratio > state["worst_ratio"]
    if strict:
        state["strict_violations"] += 1
    if tolerant:
        state["tolerant_violations"] += 1
    if worse:
        state["worst_ratio"] = ratio
        state["worst"] = {
            "matrix_id": row.get("matrix_id", ""),
            "block_index": as_int(row.get("block_index")),
            "n": as_int(row.get("n")),
            "replica": as_int(row.get("replica")),
            "alpha": as_float(row.get("alpha")),
            "method": row.get("method", ""),
            "dtype_name": row.get("dtype_name", ""),
        }


def inverse_tableau_value(row):
    # Prefer an exact audit when present, then the raw value, then the accepted value.
    for name in (
        "max_inverse_tableau_norm_inf_exact_audit",
        "max_inverse_tableau_norm_inf_raw",
        "max_inverse_tableau_norm_inf",
    ):
        x = as_float(row.get(name))
        if finite(x):
            return x, name
    return float("nan"), "missing"


def command_shard(args):
    task = int(args.task_id)
    recovered = Path(args.recovered_root)
    selection_dir = Path(args.selection_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selection = np.load(str(selection_dir / ("task_%04d.npz" % task)), allow_pickle=False)
    selected = {}
    for b, r, rank in zip(selection["block"], selection["replica"], selection["rank"]):
        selected[(int(b), int(r))] = int(rank)

    inverse_path = recovered / "inverse" / ("inverse_%04d.csv.gz" % task)
    selected_path = outdir / ("selected_%04d.csv.gz" % task)
    summary_path = outdir / ("summary_%04d.json" % task)

    dd = {}
    global_interchanges = {}
    eta_counts = {}
    total_rows = 0
    selected_rows = 0

    with gzip.open(str(selected_path) + ".tmp", "wt", encoding="utf-8", newline="", compresslevel=1) as oh:
        writer = csv.DictWriter(oh, fieldnames=SELECTED_FIELDS, lineterminator="\n")
        writer.writeheader()
        with gzip.open(str(inverse_path), "rt", encoding="utf-8", newline="") as ih:
            for row in csv.DictReader(ih):
                total_rows += 1
                method = row.get("method", "")
                dtype_name = row.get("dtype_name", "")
                if method not in METHODS or dtype_name not in DTYPES:
                    continue
                success = as_bool(row.get("success"))
                n = as_int(row.get("n"), 0)
                u = as_float(row.get("u"))
                family = row.get("family", "")

                ikey = method + "|" + dtype_name
                ist = global_interchanges.setdefault(ikey, {
                    "attempts": 0, "successes": 0,
                    "column_sum_success": 0, "column_zero_success": 0, "column_max_success": 0,
                    "row_sum_success": 0, "row_zero_success": 0, "row_max_success": 0,
                })
                ist["attempts"] += 1
                if success:
                    ist["successes"] += 1
                    ci = as_int(row.get("n_interchanges"), -1)
                    ri = as_int(row.get("n_row_interchanges"), -1)
                    if ci >= 0:
                        ist["column_sum_success"] += ci
                        ist["column_zero_success"] += int(ci == 0)
                        ist["column_max_success"] = max(ist["column_max_success"], ci)
                    if ri >= 0:
                        ist["row_sum_success"] += ri
                        ist["row_zero_success"] += int(ri == 0)
                        ist["row_max_success"] = max(ist["row_max_success"], ri)

                est = eta_counts.setdefault(ikey, {"successes": 0, "reliable": 0})
                if success:
                    est["successes"] += 1
                    if as_bool(row.get("eta_inv_reliable")) and finite(as_float(row.get("inverse_backward_error_2_est"))):
                        est["reliable"] += 1

                if family in DD_FAMILIES and success:
                    alpha = as_float(row.get("alpha"))
                    if finite(alpha) and alpha > 0.0:
                        gkey = family + "|" + method + "|" + dtype_name
                        gst = dd.setdefault(gkey, {name: check_state() for name in INEQUALITIES})
                        tol = 128.0 * max(1, n) * u if finite(u) and u > 0 else 0.0

                        alpha_measured = as_float(row.get("alpha_measured"))
                        if finite(alpha_measured):
                            update_check(gst["alpha_measured_ge_alpha"], alpha_measured / alpha, True, tol, row)

                        pmin = as_float(row.get("min_abs_pivot"))
                        if finite(pmin):
                            update_check(gst["min_abs_pivot_ge_alpha"], pmin / alpha, True, tol, row)

                        maxv = as_float(row.get("max_tableau_norm_inf"))
                        if finite(maxv):
                            update_check(gst["max_tableau_le_2_over_alpha"], maxv * alpha / 2.0, False, tol, row)

                        maxvinv, src = inverse_tableau_value(row)
                        if finite(maxvinv):
                            update_check(gst["max_inverse_tableau_le_1"], maxvinv, False, tol, row)
                            gst["max_inverse_tableau_le_1"]["source_last"] = src

                        maxb = as_float(row.get("max_B_norm_inf"))
                        if finite(maxb):
                            update_check(gst["max_B_le_3_over_alpha2"], maxb * alpha * alpha / 3.0, False, tol, row)

                        gamma = as_float(row.get("gamma_inf"))
                        if finite(gamma):
                            update_check(gst["gamma_le_6n_over_alpha3"], gamma * alpha**3 / (6.0 * n), False, tol, row)

                        rinf = as_float(row.get("right_inverse_scaled_residual_inf"))
                        bound = gamma_k(n + 3, u) * (6.0 * n / alpha**3) if n > 0 else float("nan")
                        if finite(rinf) and finite(bound) and bound > 0.0:
                            update_check(gst["rinf_le_gamma_n3_6n_over_alpha3"], rinf / bound, False, tol, row)

                block = as_int(row.get("block_index"))
                replica = as_int(row.get("replica"))
                rank = selected.get((block, replica))
                if rank is None:
                    continue
                selected_rows += 1
                eta2 = as_float(row.get("inverse_backward_error_2_est"))
                eta_rel = as_bool(row.get("eta_inv_reliable"))
                eta_bound_u = as_float(row.get("eta_inv_reliability_bound_u"))
                eta2_over_u = eta2 / u if success and eta_rel and finite(eta2) and finite(u) and u > 0 else float("nan")
                rho = eta2 / eta_bound_u if success and eta_rel and finite(eta2) and finite(eta_bound_u) and eta_bound_u > 0 else float("nan")
                rinf = as_float(row.get("right_inverse_scaled_residual_inf"))
                r2 = as_float(row.get("right_inverse_scaled_residual_2_est"))

                out = {
                    "task_id": task,
                    "matrix_id": row.get("matrix_id", ""),
                    "block_index": block,
                    "cell_id": row.get("cell_id", ""),
                    "family": family,
                    "n": n,
                    "replica": replica,
                    "rank": rank,
                    "method": method,
                    "dtype_name": dtype_name,
                    "u": u,
                    "success": success,
                    "failure_class": row.get("failure_class", ""),
                    "n_row_interchanges": row.get("n_row_interchanges", ""),
                    "n_interchanges": row.get("n_interchanges", ""),
                    "gamma_inf": row.get("gamma_inf", ""),
                    "gamma_inf_numerator": row.get("gamma_inf_numerator", ""),
                    "gamma_inf_denominator": row.get("gamma_inf_denominator", ""),
                    "gamma_inf_max_local": row.get("gamma_inf_max_local", ""),
                    "gamma_inf_max_tail": row.get("gamma_inf_max_tail", ""),
                    "right_inverse_scaled_residual_inf": rinf,
                    "right_inverse_scaled_residual_2_est": r2,
                    "rinf_over_u": rinf / u if success and finite(rinf) and finite(u) and u > 0 else float("nan"),
                    "r2_over_u": r2 / u if success and finite(r2) and finite(u) and u > 0 else float("nan"),
                    "inverse_backward_error_2_est": eta2,
                    "eta_inv_reliable": eta_rel,
                    "eta_inv_reliability_bound_u": eta_bound_u,
                    "eta2_over_u": eta2_over_u,
                    "rho_inv": rho,
                    "max_tableau_norm_inf": row.get("max_tableau_norm_inf", ""),
                    "max_inverse_tableau_norm_inf": inverse_tableau_value(row)[0],
                    "max_B_norm_inf": row.get("max_B_norm_inf", ""),
                    "max_multiplier": row.get("max_multiplier", ""),
                    "min_abs_pivot": row.get("min_abs_pivot", ""),
                    "min_scaled_pivot": row.get("min_scaled_pivot", ""),
                    "alpha": row.get("alpha", ""),
                    "alpha_measured": row.get("alpha_measured", ""),
                }
                writer.writerow(out)

    os.replace(str(selected_path) + ".tmp", str(selected_path))
    summary = {
        "status": "THEORY_AUDIT_SHARD_OK",
        "task_id": task,
        "total_inverse_rows": total_rows,
        "selected_rows": selected_rows,
        "dd_checks": dd,
        "global_interchanges": global_interchanges,
        "eta_counts": eta_counts,
        "selected_path": str(selected_path),
    }
    write_json(summary_path, summary)
    print("THEORY_AUDIT_SHARD_OK=%04d" % task)


def q(arr, p):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, p))


def safe_log_ratio(a, b):
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    if not np.any(mask):
        return np.empty(0, dtype=np.float64), mask
    return np.log10(a[mask] / b[mask]), mask


def linear_fit_log_alpha(alpha, y):
    alpha = np.asarray(alpha, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(alpha) & np.isfinite(y) & (alpha > 0) & (y > 0)
    if np.sum(mask) < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.log10(alpha[mask])
    z = np.log10(y[mask])
    A = np.column_stack([np.ones(x.size), x])
    coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coef
    ssr = float(np.sum((z - pred)**2))
    sst = float(np.sum((z - np.mean(z))**2))
    r2 = 1.0 - ssr/sst if sst > 0 else float("nan")
    return float(-coef[1]), float(coef[0]), r2


def combo_index(method, dtype_name):
    return DTYPES.index(dtype_name) * len(METHODS) + METHODS.index(method)


def command_finalize(args):
    partial = Path(args.partial_dir)
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task in range(args.num_tasks):
        path = partial / ("summary_%04d.json" % task)
        if not path.is_file():
            raise RuntimeError("Missing summary %s" % path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "THEORY_AUDIT_SHARD_OK":
            raise RuntimeError("Non-OK summary %s" % path)
        summaries.append(obj)

    # Exact all-recovered DD inequality aggregation.
    agg_dd = {}
    for obj in summaries:
        for gkey, checks in obj["dd_checks"].items():
            gout = agg_dd.setdefault(gkey, {name: check_state() for name in INEQUALITIES})
            for name in INEQUALITIES:
                src = checks.get(name, {})
                dst = gout[name]
                dst["tested"] += int(src.get("tested", 0))
                dst["strict_violations"] += int(src.get("strict_violations", 0))
                dst["tolerant_violations"] += int(src.get("tolerant_violations", 0))
                ratio = src.get("worst_ratio")
                if ratio is None:
                    continue
                lower = name in {"alpha_measured_ge_alpha", "min_abs_pivot_ge_alpha"}
                worse = dst["worst_ratio"] is None or (ratio < dst["worst_ratio"] if lower else ratio > dst["worst_ratio"])
                if worse:
                    dst["worst_ratio"] = ratio
                    dst["worst"] = src.get("worst")

    with open(report / "dd_seven_inequalities.csv", "w", newline="", encoding="utf-8") as h:
        fields = ["family", "method", "dtype_name", "scope", "inequality", "tested", "strict_violations", "tolerant_violations", "worst_ratio", "worst_matrix_id", "worst_n", "worst_alpha"]
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for gkey in sorted(agg_dd):
            family, method, dtype_name = gkey.split("|")
            for name in INEQUALITIES:
                st = agg_dd[gkey][name]
                worst = st.get("worst") or {}
                w.writerow({
                    "family": family, "method": method, "dtype_name": dtype_name,
                    "scope": (
                        "theorem-covered"
                        if method in DD_THEOREM_METHODS
                        else "additional-empirical"
                    ),
                    "inequality": name,
                    "tested": st["tested"], "strict_violations": st["strict_violations"],
                    "tolerant_violations": st["tolerant_violations"], "worst_ratio": st["worst_ratio"],
                    "worst_matrix_id": worst.get("matrix_id", ""), "worst_n": worst.get("n", ""), "worst_alpha": worst.get("alpha", ""),
                })

    # Global all-recovered interchange and eta-reliability counts.
    inter = defaultdict(lambda: {"attempts":0,"successes":0,"column_sum_success":0,"column_zero_success":0,"column_max_success":0,"row_sum_success":0,"row_zero_success":0,"row_max_success":0})
    eta = defaultdict(lambda: {"successes":0,"reliable":0})
    for obj in summaries:
        for key, src in obj["global_interchanges"].items():
            dst = inter[key]
            for k in ["attempts","successes","column_sum_success","column_zero_success","row_sum_success","row_zero_success"]:
                dst[k] += int(src.get(k,0))
            dst["column_max_success"] = max(dst["column_max_success"], int(src.get("column_max_success",0)))
            dst["row_max_success"] = max(dst["row_max_success"], int(src.get("row_max_success",0)))
        for key, src in obj["eta_counts"].items():
            eta[key]["successes"] += int(src.get("successes",0))
            eta[key]["reliable"] += int(src.get("reliable",0))

    with open(report / "interchanges_all_recovered_global.csv", "w", newline="", encoding="utf-8") as h:
        fields=["method","dtype_name","attempts","successes","mean_column_interchanges_success","zero_column_fraction_success","max_column_interchanges_success","mean_row_interchanges_success","zero_row_fraction_success","max_row_interchanges_success"]
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for key in sorted(inter):
            method,dtype_name=key.split("|"); s=inter[key]; den=max(1,s["successes"])
            w.writerow({"method":method,"dtype_name":dtype_name,"attempts":s["attempts"],"successes":s["successes"],
                        "mean_column_interchanges_success":s["column_sum_success"]/den,"zero_column_fraction_success":s["column_zero_success"]/den,"max_column_interchanges_success":s["column_max_success"],
                        "mean_row_interchanges_success":s["row_sum_success"]/den,"zero_row_fraction_success":s["row_zero_success"]/den,"max_row_interchanges_success":s["row_max_success"]})
    with open(report / "eta_reliability_all_recovered.csv", "w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["method","dtype_name","successes","eta_reliable","reliable_fraction"]); w.writeheader()
        for key in sorted(eta):
            method,dtype_name=key.split("|"); s=eta[key]; den=max(1,s["successes"])
            w.writerow({"method":method,"dtype_name":dtype_name,"successes":s["successes"],"eta_reliable":s["reliable"],"reliable_fraction":s["reliable"]/den})

    selection_summary = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    selected_blocks = sorted(selection_summary["selected_blocks"], key=lambda x: int(x["block_index"]))
    block_pos = {int(x["block_index"]): i for i,x in enumerate(selected_blocks)}
    nb = len(selected_blocks); ns = int(args.sample_size); nc = len(METHODS)*len(DTYPES)

    tmpdir = Path(args.tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    cov = np.memmap(str(tmpdir/"cov.u1"), dtype=np.uint8, mode="w+", shape=(nb,ns,nc)); cov[:] = 0
    suc = np.memmap(str(tmpdir/"suc.u1"), dtype=np.uint8, mode="w+", shape=(nb,ns,nc)); suc[:] = 0
    erel = np.memmap(str(tmpdir/"erel.u1"), dtype=np.uint8, mode="w+", shape=(nb,ns,nc)); erel[:] = 0
    ints = np.memmap(str(tmpdir/"ints.i2"), dtype=np.int16, mode="w+", shape=(nb,ns,nc,2)); ints[:] = -1
    metric_names = ["gamma_over_n","gamma_num","gamma_den","gamma_local","gamma_tail","rinf_u","r2_u","eta2_u","rho","maxV","maxVinv","maxB","maxmult"]
    vals = np.memmap(str(tmpdir/"vals.f8"), dtype=np.float64, mode="w+", shape=(nb,ns,nc,len(metric_names))); vals[:] = np.nan
    taskarr = np.memmap(str(tmpdir/"task.i2"), dtype=np.int16, mode="w+", shape=(nb,ns,nc)); taskarr[:] = -1

    for task in range(args.num_tasks):
        sp = partial / ("selected_%04d.csv.gz" % task)
        with gzip.open(str(sp), "rt", encoding="utf-8", newline="") as h:
            for row in csv.DictReader(h):
                b=int(row["block_index"]); rank=int(row["rank"]); bp=block_pos.get(b)
                if bp is None: continue
                ci=combo_index(row["method"],row["dtype_name"])
                cov[bp,rank,ci]=1; suc[bp,rank,ci]=1 if as_bool(row["success"]) else 0; erel[bp,rank,ci]=1 if as_bool(row["eta_inv_reliable"]) else 0
                ints[bp,rank,ci,0]=as_int(row["n_interchanges"],-1); ints[bp,rank,ci,1]=as_int(row["n_row_interchanges"],-1); taskarr[bp,rank,ci]=task
                n=int(row["n"]); gamma=as_float(row["gamma_inf"])
                mvals=[gamma/n if finite(gamma) and n>0 else float("nan"), as_float(row["gamma_inf_numerator"]), as_float(row["gamma_inf_denominator"]), as_float(row["gamma_inf_max_local"]), as_float(row["gamma_inf_max_tail"]), as_float(row["rinf_over_u"]), as_float(row["r2_over_u"]), as_float(row["eta2_over_u"]), as_float(row["rho_inv"]), as_float(row["max_tableau_norm_inf"]), as_float(row["max_inverse_tableau_norm_inf"]), as_float(row["max_B_norm_inf"]), as_float(row["max_multiplier"])]
                vals[bp,rank,ci,:]=mvals

    expected=nb*ns*nc
    if int(np.sum(cov)) != expected:
        raise RuntimeError("Selected coverage %d != %d" % (int(np.sum(cov)), expected))

    # Per-block metrics, including Gamma pieces and eta/rho.
    block_metrics_path=report/"theory_balanced_block_metrics.csv"
    fields=["block_index","cell_id","family","n","parameters_json","method","dtype_name","success","failure_rate","eta_reliable"]
    for name in metric_names:
        fields += [name+"_q50",name+"_q95",name+"_q99"]
    fields += ["column_interchanges_q50","column_interchanges_q95","column_interchanges_max","column_interchanges_zero_fraction","row_interchanges_q50","row_interchanges_q95","row_interchanges_max","row_interchanges_zero_fraction"]
    block_rows=[]
    with open(block_metrics_path,"w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for bp,meta in enumerate(selected_blocks):
            for dtype_name in DTYPES:
                for method in METHODS:
                    ci=combo_index(method,dtype_name); sm=suc[bp,:,ci].astype(bool)
                    out={"block_index":meta["block_index"],"cell_id":meta["cell_id"],"family":meta["family"],"n":meta["n"],"parameters_json":meta["parameters_json"],"method":method,"dtype_name":dtype_name,"success":int(np.sum(sm)),"failure_rate":1.0-float(np.mean(sm)),"eta_reliable":int(np.sum(sm & erel[bp,:,ci].astype(bool)))}
                    for mi,name in enumerate(metric_names):
                        a=np.asarray(vals[bp,:,ci,mi]); mask=sm & np.isfinite(a)
                        if name in {"eta2_u","rho"}: mask &= erel[bp,:,ci].astype(bool)
                        aa=a[mask]
                        out[name+"_q50"]=q(aa,.5); out[name+"_q95"]=q(aa,.95); out[name+"_q99"]=q(aa,.99)
                    for k,prefix in [(0,"column_interchanges"),(1,"row_interchanges")]:
                        a=np.asarray(ints[bp,:,ci,k],dtype=np.float64); aa=a[sm & (a>=0)]
                        out[prefix+"_q50"]=q(aa,.5); out[prefix+"_q95"]=q(aa,.95); out[prefix+"_max"]=float(np.max(aa)) if aa.size else float("nan"); out[prefix+"_zero_fraction"]=float(np.mean(aa==0)) if aa.size else float("nan")
                    w.writerow(out); block_rows.append(out)

    # Balanced interchanges by family/n/method/dtype.
    with open(report/"interchanges_balanced_by_family_n.csv","w",newline="",encoding="utf-8") as h:
        fields=["family","n","method","dtype_name","success_count","column_q50","column_q90","column_q95","column_max","column_zero_fraction","row_q50","row_q90","row_q95","row_max","row_zero_fraction"]
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        famn=defaultdict(list)
        for bp,meta in enumerate(selected_blocks): famn[(meta["family"],int(meta["n"]))].append(bp)
        for (family,n),bps in sorted(famn.items()):
            for dtype_name in DTYPES:
                for method in METHODS:
                    ci=combo_index(method,dtype_name); ca=[]; ra=[]
                    for bp in bps:
                        sm=suc[bp,:,ci].astype(bool); c=np.asarray(ints[bp,:,ci,0]); r=np.asarray(ints[bp,:,ci,1]); ca.append(c[sm & (c>=0)]); ra.append(r[sm & (r>=0)])
                    ca=np.concatenate(ca) if ca else np.empty(0); ra=np.concatenate(ra) if ra else np.empty(0)
                    w.writerow({"family":family,"n":n,"method":method,"dtype_name":dtype_name,"success_count":int(ca.size),"column_q50":q(ca,.5),"column_q90":q(ca,.9),"column_q95":q(ca,.95),"column_max":float(np.max(ca)) if ca.size else float("nan"),"column_zero_fraction":float(np.mean(ca==0)) if ca.size else float("nan"),"row_q50":q(ra,.5),"row_q90":q(ra,.9),"row_q95":q(ra,.95),"row_max":float(np.max(ra)) if ra.size else float("nan"),"row_zero_fraction":float(np.mean(ra==0)) if ra.size else float("nan")})

    # Common-success paired comparison within every block and precision.
    with open(report/"paired_common_quantiles.csv","w",newline="",encoding="utf-8") as hq, open(report/"paired_common_method_ratios.csv","w",newline="",encoding="utf-8") as hp:
        qfields=["block_index","cell_id","family","n","dtype_name","method","common_success_n","common_eta_reliable_n","rinf_u_q50","rinf_u_q95","r2_u_q50","r2_u_q95","gamma_over_n_q50","gamma_over_n_q95","eta2_u_q50","eta2_u_q95","rho_q50","rho_q95"]
        pfields=["block_index","cell_id","family","n","dtype_name","metric","method_a","method_b","paired_n","median_log10_a_over_b","q25_log10_a_over_b","q75_log10_a_over_b","fraction_a_lt_b"]
        wq=csv.DictWriter(hq,fieldnames=qfields); wq.writeheader(); wp=csv.DictWriter(hp,fieldnames=pfields); wp.writeheader()
        midx={name:i for i,name in enumerate(metric_names)}
        for bp,meta in enumerate(selected_blocks):
            for dtype_name in DTYPES:
                cis=[combo_index(m,dtype_name) for m in METHODS]
                common=np.all(np.asarray(suc[bp,:,:],dtype=bool)[:,cis],axis=1)
                common_eta=common & np.all(np.asarray(erel[bp,:,:],dtype=bool)[:,cis],axis=1)
                for method,ci in zip(METHODS,cis):
                    out={"block_index":meta["block_index"],"cell_id":meta["cell_id"],"family":meta["family"],"n":meta["n"],"dtype_name":dtype_name,"method":method,"common_success_n":int(np.sum(common)),"common_eta_reliable_n":int(np.sum(common_eta))}
                    for nm,outnm,mask in [("rinf_u","rinf_u",common),("r2_u","r2_u",common),("gamma_over_n","gamma_over_n",common),("eta2_u","eta2_u",common_eta),("rho","rho",common_eta)]:
                        a=np.asarray(vals[bp,:,ci,midx[nm]]); aa=a[mask & np.isfinite(a)]; out[outnm+"_q50"]=q(aa,.5); out[outnm+"_q95"]=q(aa,.95)
                    wq.writerow(out)
                for ia in range(len(METHODS)):
                    for ib in range(ia+1,len(METHODS)):
                        ma,mb=METHODS[ia],METHODS[ib]; ca,cb=cis[ia],cis[ib]
                        for nm,mask in [("rinf_u",common),("r2_u",common),("gamma_over_n",common),("eta2_u",common_eta),("rho",common_eta)]:
                            a=np.asarray(vals[bp,:,ca,midx[nm]]); b=np.asarray(vals[bp,:,cb,midx[nm]]); mm=mask & np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0); lr=np.log10(a[mm]/b[mm])
                            wp.writerow({"block_index":meta["block_index"],"cell_id":meta["cell_id"],"family":meta["family"],"n":meta["n"],"dtype_name":dtype_name,"metric":nm,"method_a":ma,"method_b":mb,"paired_n":int(lr.size),"median_log10_a_over_b":q(lr,.5),"q25_log10_a_over_b":q(lr,.25),"q75_log10_a_over_b":q(lr,.75),"fraction_a_lt_b":float(np.mean(a[mm]<b[mm])) if lr.size else float("nan")})

    # Gamma/N scaling in alpha using block q50 and q95.
    brdf=block_rows
    with open(report/"gamma_over_n_alpha_fits.csv","w",newline="",encoding="utf-8") as h:
        fields=["family","method","dtype_name","n","quantile","alpha_points","alpha_exponent_p","intercept_log10","r2"]
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); fitrows=[]
        for family in sorted(DD_FAMILIES):
            for method in METHODS:
                for dtype_name in DTYPES:
                    for n in sorted(set(int(r["n"]) for r in brdf if r["family"]==family)):
                        rr=[r for r in brdf if r["family"]==family and r["method"]==method and r["dtype_name"]==dtype_name and int(r["n"])==n]
                        for qq in ["q50","q95"]:
                            aa=[]; yy=[]
                            for r in rr:
                                try: alpha=float(json.loads(r["parameters_json"])["alpha"])
                                except Exception: continue
                                y=float(r["gamma_over_n_"+qq])
                                aa.append(alpha); yy.append(y)
                            p,c,r2=linear_fit_log_alpha(aa,yy); out={"family":family,"method":method,"dtype_name":dtype_name,"n":n,"quantile":qq,"alpha_points":len(aa),"alpha_exponent_p":p,"intercept_log10":c,"r2":r2}; w.writerow(out); fitrows.append(out)

    # Summarize existing q95 s fit, quadratic sign, and bootstrap interval.
    fits=[]
    with open(args.power_fits, newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r.get("quantile")=="q95": fits.append(r)
    boots={}
    with open(args.bootstrap_slopes, newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r.get("quantile")=="q95": boots[(r["cell_id"],r["method"],r["dtype_name"])]=r
    with open(report/"s_q95_cell_audit.csv","w",newline="",encoding="utf-8") as h:
        fields=["cell_id","family","method","dtype_name","s","ci_low","ci_high","bootstrap_bias","quadratic_curvature","curvature_sign","delta_aic_quadratic_minus_power","quadratic_preferred"]
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); scells=[]
        for r in fits:
            b=boots.get((r["cell_id"],r["method"],r["dtype_name"]),{})
            cur=as_float(r.get("quadratic_curvature")); delta=as_float(r.get("quadratic_aic"))-as_float(r.get("power_aic"))
            out={"cell_id":r["cell_id"],"family":r["family"],"method":r["method"],"dtype_name":r["dtype_name"],"s":as_float(r.get("power_slope")),"ci_low":as_float(b.get("ci_low")),"ci_high":as_float(b.get("ci_high")),"bootstrap_bias":as_float(b.get("bootstrap_bias")),"quadratic_curvature":cur,"curvature_sign":"convex" if cur>0 else ("concave" if cur<0 else "zero"),"delta_aic_quadratic_minus_power":delta,"quadratic_preferred":delta<0}; w.writerow(out); scells.append(out)
    with open(report/"s_q95_summary.csv","w",newline="",encoding="utf-8") as h:
        fields=["method","dtype_name","cells","median_s","q25_s","q75_s","quadratic_preferred","convex_when_quadratic_preferred","concave_when_quadratic_preferred","median_quadratic_curvature_when_preferred","median_bootstrap_ci_width","positive_ci","negative_ci","ci_contains_zero"]
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for method in METHODS:
            for dtype_name in DTYPES:
                rr=[r for r in scells if r["method"]==method and r["dtype_name"]==dtype_name]; ss=np.array([r["s"] for r in rr],float); pref=[r for r in rr if r["quadratic_preferred"]]; widths=np.array([r["ci_high"]-r["ci_low"] for r in rr],float)
                w.writerow({"method":method,"dtype_name":dtype_name,"cells":len(rr),"median_s":q(ss,.5),"q25_s":q(ss,.25),"q75_s":q(ss,.75),"quadratic_preferred":len(pref),"convex_when_quadratic_preferred":sum(r["quadratic_curvature"]>0 for r in pref),"concave_when_quadratic_preferred":sum(r["quadratic_curvature"]<0 for r in pref),"median_quadratic_curvature_when_preferred":q([r["quadratic_curvature"] for r in pref],.5),"median_bootstrap_ci_width":q(widths,.5),"positive_ci":sum(r["ci_low"]>0 for r in rr),"negative_ci":sum(r["ci_high"]<0 for r in rr),"ci_contains_zero":sum(r["ci_low"]<=0<=r["ci_high"] for r in rr)})

    final={"status":"THEORY_AUDIT_COMPLETE","tasks":args.num_tasks,"selected_blocks":nb,"sample_size":ns,"selected_records":expected,
           "outputs":["dd_seven_inequalities.csv","interchanges_all_recovered_global.csv","eta_reliability_all_recovered.csv","theory_balanced_block_metrics.csv","interchanges_balanced_by_family_n.csv","paired_common_quantiles.csv","paired_common_method_ratios.csv","gamma_over_n_alpha_fits.csv","s_q95_cell_audit.csv","s_q95_summary.csv"]}
    write_json(report/"theory_audit_summary.json",final)
    # close and remove scratch memmaps
    del vals, ints, suc, erel, cov, taskarr
    for name in ["vals.f8","ints.i2","suc.u1","erel.u1","cov.u1","task.i2"]:
        try: os.remove(str(tmpdir/name))
        except OSError: pass
    print(json.dumps(final,indent=2,sort_keys=True)); print("THEORY_AUDIT_COMPLETE")


def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd",required=True)
    s=sp.add_parser("shard"); s.add_argument("--task-id",type=int,required=True); s.add_argument("--recovered-root",required=True); s.add_argument("--selection-dir",required=True); s.add_argument("--outdir",required=True); s.set_defaults(func=command_shard)
    f=sp.add_parser("finalize"); f.add_argument("--partial-dir",required=True); f.add_argument("--report-dir",required=True); f.add_argument("--selection-summary",required=True); f.add_argument("--power-fits",required=True); f.add_argument("--bootstrap-slopes",required=True); f.add_argument("--tmpdir",required=True); f.add_argument("--num-tasks",type=int,default=999); f.add_argument("--sample-size",type=int,default=2000); f.set_defaults(func=command_finalize)
    a=p.parse_args(); a.func(a)

if __name__=="__main__": main()
