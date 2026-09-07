#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv, json, math
import numpy as np

BASE = Path.home()/"castillo_lapack_baseline/results/spectral_estimator_validation"
TASKS = BASE/"tasks"
OUT = BASE/"final"

def q(a,p):
    x=np.asarray([v for v in a if math.isfinite(v)],dtype=float)
    return float(np.quantile(x,p)) if x.size else math.nan

def asf(s):
    try: return float(s)
    except: return math.nan

files=sorted(TASKS.glob("validation_task_*.csv"))
if not files: raise SystemExit("No task CSVs")
rows=[]
for p in files:
    with p.open(newline="",encoding="utf-8") as f: rows.extend(csv.DictReader(f))

metrics=("A_nu6_over_norm2","X_nu6_over_norm2","R_nu6_over_norm2","residual_nu6_over_norm2")
groups=defaultdict(list)
for r in rows:
    if str(r.get("success","")).lower() not in ("true","1"): continue
    groups[(r["dtype_name"],r["method"])].append(r)

OUT.mkdir(parents=True,exist_ok=True)
fields=["dtype_name","method","successful_records","metric","finite_positive_records",
        "min","q01","q05","median","q95","q99","max","fraction_within_1pct","fraction_within_5pct","fraction_within_10pct"]
summary=[]
for (dtype,method), rr in sorted(groups.items()):
    for metric in metrics:
        vals=[asf(r.get(metric,"")) for r in rr]
        vals=[v for v in vals if math.isfinite(v) and v>0]
        def within(t): return sum(abs(v-1)<=t for v in vals)/len(vals) if vals else math.nan
        summary.append({
            "dtype_name":dtype,"method":method,"successful_records":len(rr),"metric":metric,
            "finite_positive_records":len(vals),"min":min(vals) if vals else math.nan,
            "q01":q(vals,.01),"q05":q(vals,.05),"median":q(vals,.5),"q95":q(vals,.95),"q99":q(vals,.99),
            "max":max(vals) if vals else math.nan,"fraction_within_1pct":within(.01),
            "fraction_within_5pct":within(.05),"fraction_within_10pct":within(.10)})
with (OUT/"spectral_estimator_validation_summary.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)

case_indices=sorted({int(r["case_index"]) for r in rows})
status_counts=defaultdict(int)
for r in rows: status_counts[(r["dtype_name"],r["method"],r["status"])]+=1
obj={"task_files":len(files),"case_indices":case_indices,"records":len(rows),
     "successful_records":sum(str(r.get("success","")).lower() in ("true","1") for r in rows),
     "status_counts":{"|".join(k):v for k,v in sorted(status_counts.items())},
     "scientific_threshold_gate":None,
     "note":"No accuracy threshold is imposed a priori; this audit reports estimator/direct-norm agreement for interpretation."}
(OUT/"spectral_estimator_validation_summary.json").write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
print(json.dumps(obj,indent=2))
print("summary_csv =",OUT/"spectral_estimator_validation_summary.csv")
print("SPECTRAL_ESTIMATOR_VALIDATION_FINALIZE_OK")
