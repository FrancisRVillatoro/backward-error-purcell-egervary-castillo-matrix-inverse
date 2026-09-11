#!/usr/bin/env python3
from pathlib import Path
import csv,json,math,os,sys,time
import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

BASE=Path.home()/"castillo_lapack_baseline"
SOURCE=BASE/"source"
ROOT=BASE/"results"/"direct_svd_crosscheck"
MANIFEST=ROOT/"direct_svd_validation_manifest.csv"
TASKS=ROOT/"tasks"
METHODS=("R0_C0","R0_C1","R0_C2","R1_C1","R2_C2")
LAPACK="LAPACK_xGETRF_xGETRI"
ALL=METHODS+(LAPACK,)
DTYPES=(("float32",np.float32),("float64",np.float64))
ITS=6

sys.path.insert(0,str((SOURCE/"code").resolve()))
from families import build_matrix,stable_seed
from castillo import castillo_inverse,spectral_norm_estimate,unit_roundoff

FIELDS=["task_index","block_index","selection_rank","selection_task","replica",
"cell_id","family","n","seed","matrix_id","dtype_name","method","success","status",
"elapsed_seconds","A_nu6","A_svd","A_nu6_over_svd","X_nu6","X_svd",
"X_nu6_over_svd","R_nu6","R_svd","R_nu6_over_svd","rR2_nu6","rR2_svd",
"rR2_nu6_over_svd","u","rR2_nu6_over_u","rR2_svd_over_u"]

def task_specs(idx):
    rows=[]
    with MANIFEST.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["task_index"])!=idx: continue
            x=dict(r)
            for k in ("task_index","block_index","selection_rank","selection_task","replica","n"):
                x[k]=int(x[k])
            x["parameters"]=json.loads(x["parameters_json"])
            rows.append(x)
    rows.sort(key=lambda r:r["selection_rank"])
    if len(rows)!=3 or tuple(r["selection_rank"] for r in rows)!=(0,1000,1999):
        raise RuntimeError(f"task {idx}: bad 3-rank manifest slice")
    return rows

def svd2(a):
    a=np.asarray(a,dtype=np.float64)
    if not np.all(np.isfinite(a)): return math.nan
    try: s=np.linalg.svd(a,compute_uv=False,full_matrices=False)
    except np.linalg.LinAlgError: return math.nan
    return float(s[0]) if s.size else 0.0

def ratio(a,b):
    return a/b if math.isfinite(a) and math.isfinite(b) and b!=0 else math.nan

def lapack_inverse(a):
    af=np.array(a,order="F",copy=True)
    getrf,getri,getri_lwork=get_lapack_funcs(("getrf","getri","getri_lwork"),(af,))
    lu,piv,info=getrf(af,overwrite_a=True)
    if int(info)!=0: return None,f"getrf_info_{int(info)}"
    lwork,info=getri_lwork(af.shape[0])
    if int(info)!=0: return None,f"getri_lwork_info_{int(info)}"
    x,info=getri(lu,piv,lwork=max(1,int(float(np.real(lwork)))),overwrite_lu=True)
    if int(info)!=0: return None,f"getri_info_{int(info)}"
    if not np.all(np.isfinite(x)): return None,"nonfinite_inverse"
    return x,"completed"

def blank(u):
    d={k:math.nan for k in FIELDS[15:]}; d["u"]=u; return d

def metrics(x,a_res,A_nu6,A_svd,u):
    x64=np.asarray(x,dtype=np.float64)
    ar=np.asarray(a_res,dtype=np.float64)
    R=np.eye(ar.shape[0],dtype=np.float64)-ar@x64
    X_nu6=float(spectral_norm_estimate(x,ITS))
    R_nu6=float(spectral_norm_estimate(R,ITS))
    X_svd=svd2(x64); R_svd=svd2(R)
    dene=A_nu6*X_nu6; dens=A_svd*X_svd
    re=R_nu6/dene if dene>0 else math.nan
    rs=R_svd/dens if dens>0 else math.nan
    return {"A_nu6":A_nu6,"A_svd":A_svd,"A_nu6_over_svd":ratio(A_nu6,A_svd),
    "X_nu6":X_nu6,"X_svd":X_svd,"X_nu6_over_svd":ratio(X_nu6,X_svd),
    "R_nu6":R_nu6,"R_svd":R_svd,"R_nu6_over_svd":ratio(R_nu6,R_svd),
    "rR2_nu6":re,"rR2_svd":rs,"rR2_nu6_over_svd":ratio(re,rs),"u":u,
    "rR2_nu6_over_u":re/u if math.isfinite(re) else math.nan,
    "rR2_svd_over_u":rs/u if math.isfinite(rs) else math.nan}

def main():
    idx=int(os.environ.get("SLURM_ARRAY_TASK_ID",os.environ.get("TASK_INDEX","0")))
    specs=task_specs(idx)
    canonical=json.loads((SOURCE/"config/canonical.json").read_text(encoding="utf-8"))
    execution=json.loads((SOURCE/"config/execution.json").read_text(encoding="utf-8"))
    if int(execution["spectral_norm_iterations"])!=ITS:
        raise RuntimeError("frozen campaign does not use six spectral iterations")
    out=[]
    for s in specs:
        seed=stable_seed(canonical["seed_namespace"],s["cell_id"],s["n"],s["replica"])
        aref=build_matrix(family=s["family"],n=s["n"],parameters=s["parameters"],seed=seed)
        mid=f'{s["cell_id"]}|n={s["n"]}|rep={s["replica"]}|seed={seed}'
        common={k:s[k] for k in ("task_index","block_index","selection_rank","selection_task",
                                   "replica","cell_id","family","n")}
        common.update({"seed":seed,"matrix_id":mid})
        for dtype_name,dtype in DTYPES:
            u=float(unit_roundoff(dtype_name))
            with np.errstate(over="ignore",under="ignore",invalid="ignore"):
                aw=np.asarray(aref,dtype=dtype)
            if not np.all(np.isfinite(aw)):
                for method in ALL:
                    out.append({**common,"dtype_name":dtype_name,"method":method,"success":False,
                                "status":"nonrepresentable_input","elapsed_seconds":0.0,**blank(u)})
                continue
            A_nu6=float(spectral_norm_estimate(aw,ITS)); A_svd=svd2(aw)
            if not (math.isfinite(A_nu6) and A_nu6>0 and math.isfinite(A_svd) and A_svd>0):
                raise RuntimeError(f"invalid A norm task={idx}, block={s['block_index']}, dtype={dtype_name}")
            for method in METHODS:
                t=time.perf_counter()
                r=castillo_inverse(aref,method,dtype_name,spectral_iterations=ITS,audit=False)
                row={**common,"dtype_name":dtype_name,"method":method,"success":bool(r.success),
                     "status":r.failure_class,"elapsed_seconds":time.perf_counter()-t}
                if r.success and np.all(np.isfinite(r.inverse)):
                    aper=aw[np.asarray(r.row_order,dtype=int),:]
                    row.update(metrics(r.inverse,aper,A_nu6,A_svd,u))
                else: row.update(blank(u))
                out.append(row)
            t=time.perf_counter(); x,status=lapack_inverse(aw)
            row={**common,"dtype_name":dtype_name,"method":LAPACK,"success":x is not None,
                 "status":status,"elapsed_seconds":time.perf_counter()-t}
            if x is not None: row.update(metrics(x,aw,A_nu6,A_svd,u))
            else: row.update(blank(u))
            out.append(row)
    if len(out)!=36: raise RuntimeError(f"task {idx}: expected 36 rows, got {len(out)}")
    TASKS.mkdir(parents=True,exist_ok=True)
    final=TASKS/f"direct_svd_task_{idx:04d}.csv"
    tmp=TASKS/f".direct_svd_task_{idx:04d}.tmp.{os.getpid()}"
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(out)
    os.replace(tmp,final)
    print("task_index =",idx); print("output =",final); print("DIRECT_SVD_CROSSCHECK_TASK_OK")

if __name__=="__main__": main()
