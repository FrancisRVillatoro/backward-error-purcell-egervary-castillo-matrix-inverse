#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import sys
import time

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs


BASE = Path.home() / "castillo_lapack_baseline"
SOURCE = BASE / "source"
SELDIR = BASE / "exact_balanced_selection" / "selection"
OUTDIR = BASE / "results" / "benchmark"

sys.path.insert(
    0,
    str((SOURCE / "code").resolve()),
)

from families import build_matrix, stable_seed
from castillo import (
    spectral_norm_estimate,
    norm_inf,
    unit_roundoff,
)


def load_manifest(path):
    rows = []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            item = dict(row)
            item["block_index"] = int(
                row["block_index"]
            )
            item["n"] = int(
                row["n"]
            )
            item["stochastic"] = (
                str(
                    row["stochastic"]
                ).strip().lower()
                == "true"
            )
            item["parameters"] = json.loads(
                row["parameters_json"]
            )
            rows.append(item)

    return rows


def rank_zero_replicas(selection_dir):
    result = {}

    paths = sorted(
        selection_dir.glob(
            "task_????.npz"
        )
    )

    for task_id, path in enumerate(paths):
        z = np.load(
            str(path),
            allow_pickle=False,
        )

        blocks = np.asarray(
            z["block"],
            dtype=np.int64,
        )

        replicas = np.asarray(
            z["replica"],
            dtype=np.int64,
        )

        ranks = np.asarray(
            z["rank"],
            dtype=np.int64,
        )

        mask = ranks == 0

        for block, replica in zip(
            blocks[mask],
            replicas[mask],
        ):
            b = int(block)
            r = int(replica)

            if b in result:
                print(
                    "DUPLICATE_RANK_ZERO_BLOCK",
                    b,
                    result[b],
                    r,
                )
            else:
                result[b] = (
                    task_id,
                    r,
                )

    return result


def lapack_inverse(A):
    Af = np.array(
        A,
        order="F",
        copy=True,
    )

    getrf, getri, getri_lwork = (
        get_lapack_funcs(
            (
                "getrf",
                "getri",
                "getri_lwork",
            ),
            (Af,),
        )
    )

    t0 = time.perf_counter()

    lu, piv, info_rf = getrf(
        Af,
        overwrite_a=True,
    )

    t1 = time.perf_counter()

    lwork, info_lw = getri_lwork(
        Af.shape[0]
    )

    lw = max(
        1,
        int(
            float(
                np.real(lwork)
            )
        ),
    )

    X, info_ri = getri(
        lu,
        piv,
        lwork=lw,
        overwrite_lu=True,
    )

    t2 = time.perf_counter()

    return {
        "X": X,
        "info_rf": int(info_rf),
        "info_lw": int(info_lw),
        "info_ri": int(info_ri),
        "lwork": int(lw),
        "getrf_prefix": getattr(
            getrf,
            "prefix",
            None,
        ),
        "getri_prefix": getattr(
            getri,
            "prefix",
            None,
        ),
        "getrf_seconds": t1 - t0,
        "getri_seconds": t2 - t1,
        "inverse_seconds": t2 - t0,
    }


canonical = json.loads(
    (
        SOURCE
        / "config"
        / "canonical.json"
    ).read_text(
        encoding="utf-8"
    )
)

execution = json.loads(
    (
        SOURCE
        / "config"
        / "execution.json"
    ).read_text(
        encoding="utf-8"
    )
)

spectral_iterations = int(
    execution["spectral_norm_iterations"]
)

manifest = load_manifest(
    SOURCE
    / "reports"
    / "canonical"
    / "canonical_manifest.csv"
)

rank0 = rank_zero_replicas(
    SELDIR
)

print(
    "rank_zero_blocks =",
    len(rank0),
)

eligible = [
    row
    for row in manifest
    if (
        row["stochastic"]
        and row["n"] == 512
        and row["block_index"] in rank0
    )
]

by_family = defaultdict(list)

for row in eligible:
    by_family[
        row["family"]
    ].append(row)

selected = []

for family in sorted(by_family):
    rows = sorted(
        by_family[family],
        key=lambda x: (
            x["cell_id"],
            x["block_index"],
        ),
    )

    selected.append(
        rows[0]
    )


print(
    "n512_stochastic_blocks =",
    len(eligible),
)

print(
    "n512_stochastic_families =",
    len(by_family),
)

print(
    "benchmark_matrices =",
    len(selected),
)

for row in selected:
    task_id, replica = rank0[
        row["block_index"]
    ]

    print(
        "BENCHMARK_CASE",
        "family=",
        row["family"],
        "cell_id=",
        row["cell_id"],
        "block=",
        row["block_index"],
        "task=",
        task_id,
        "replica=",
        replica,
    )


fields = [
    "family",
    "cell_id",
    "block_index",
    "task_id",
    "n",
    "replica",
    "dtype_name",
    "seed",
    "representable",
    "success",
    "getrf_prefix",
    "getri_prefix",
    "getrf_info",
    "getri_lwork_info",
    "getri_info",
    "lwork",
    "generation_seconds",
    "getrf_seconds",
    "getri_seconds",
    "inverse_seconds",
    "metrics_seconds",
    "total_seconds",
    "rR2_over_u",
    "rL2_over_u",
    "rRinf_over_u",
    "rLinf_over_u",
]

rows_out = []

overall_start = time.perf_counter()

for case_index, row in enumerate(
    selected,
    1,
):
    block = row["block_index"]
    n = row["n"]
    task_id, replica = rank0[block]

    seed = stable_seed(
        canonical["seed_namespace"],
        row["cell_id"],
        n,
        replica,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "CASE",
        case_index,
        "OF",
        len(selected),
    )
    print(
        "family =",
        row["family"],
    )
    print(
        "cell_id =",
        row["cell_id"],
    )
    print(
        "block =",
        block,
    )
    print(
        "task =",
        task_id,
    )
    print(
        "replica =",
        replica,
    )
    print(
        "seed =",
        seed,
    )

    tg0 = time.perf_counter()

    Aref = build_matrix(
        family=row["family"],
        n=n,
        parameters=row["parameters"],
        seed=seed,
    )

    tg1 = time.perf_counter()

    generation_seconds = (
        tg1 - tg0
    )

    print(
        "generation_seconds =",
        generation_seconds,
    )

    print(
        "Aref_finite =",
        bool(
            np.all(
                np.isfinite(Aref)
            )
        ),
    )

    for dtype_name, dtype in (
        ("float32", np.float32),
        ("float64", np.float64),
    ):
        print()
        print(
            "dtype =",
            dtype_name,
        )

        case_start = time.perf_counter()

        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            A = np.asarray(
                Aref,
                dtype=dtype,
            )

        representable = bool(
            np.all(
                np.isfinite(A)
            )
        )

        output = {
            "family": row["family"],
            "cell_id": row["cell_id"],
            "block_index": block,
            "task_id": task_id,
            "n": n,
            "replica": replica,
            "dtype_name": dtype_name,
            "seed": seed,
            "representable": representable,
            "success": False,
            "getrf_prefix": "",
            "getri_prefix": "",
            "getrf_info": "",
            "getri_lwork_info": "",
            "getri_info": "",
            "lwork": "",
            "generation_seconds": generation_seconds,
            "getrf_seconds": math.nan,
            "getri_seconds": math.nan,
            "inverse_seconds": math.nan,
            "metrics_seconds": math.nan,
            "total_seconds": math.nan,
            "rR2_over_u": math.nan,
            "rL2_over_u": math.nan,
            "rRinf_over_u": math.nan,
            "rLinf_over_u": math.nan,
        }

        print(
            "representable =",
            representable,
        )

        if representable:
            try:
                inv = lapack_inverse(
                    A
                )

                output[
                    "getrf_prefix"
                ] = inv[
                    "getrf_prefix"
                ]

                output[
                    "getri_prefix"
                ] = inv[
                    "getri_prefix"
                ]

                output[
                    "getrf_info"
                ] = inv[
                    "info_rf"
                ]

                output[
                    "getri_lwork_info"
                ] = inv[
                    "info_lw"
                ]

                output[
                    "getri_info"
                ] = inv[
                    "info_ri"
                ]

                output[
                    "lwork"
                ] = inv[
                    "lwork"
                ]

                output[
                    "getrf_seconds"
                ] = inv[
                    "getrf_seconds"
                ]

                output[
                    "getri_seconds"
                ] = inv[
                    "getri_seconds"
                ]

                output[
                    "inverse_seconds"
                ] = inv[
                    "inverse_seconds"
                ]

                X = inv["X"]

                info_ok = (
                    inv["info_rf"] == 0
                    and inv["info_lw"] == 0
                    and inv["info_ri"] == 0
                )

                finite_X = bool(
                    np.all(
                        np.isfinite(X)
                    )
                )

                prefix_expected = (
                    "s"
                    if dtype_name == "float32"
                    else "d"
                )

                prefix_ok = (
                    inv["getrf_prefix"]
                    == prefix_expected
                    and inv["getri_prefix"]
                    == prefix_expected
                )

                dtype_ok = (
                    X.dtype
                    == np.dtype(dtype)
                )

                if (
                    info_ok
                    and finite_X
                    and prefix_ok
                    and dtype_ok
                ):
                    tm0 = time.perf_counter()

                    A64 = np.asarray(
                        A,
                        dtype=np.float64,
                    )

                    X64 = np.asarray(
                        X,
                        dtype=np.float64,
                    )

                    I = np.eye(
                        n,
                        dtype=np.float64,
                    )

                    Rright = (
                        I
                        - A64 @ X64
                    )

                    Rleft = (
                        I
                        - X64 @ A64
                    )

                    norm_A_2 = (
                        spectral_norm_estimate(
                            A,
                            spectral_iterations,
                        )
                    )

                    norm_X_2 = (
                        spectral_norm_estimate(
                            X,
                            spectral_iterations,
                        )
                    )

                    norm_RR_2 = (
                        spectral_norm_estimate(
                            Rright,
                            spectral_iterations,
                        )
                    )

                    norm_RL_2 = (
                        spectral_norm_estimate(
                            Rleft,
                            spectral_iterations,
                        )
                    )

                    tiny = np.finfo(
                        np.float64
                    ).tiny

                    denom_2 = max(
                        norm_A_2
                        * norm_X_2,
                        tiny,
                    )

                    denom_inf = max(
                        norm_inf(A64)
                        * norm_inf(X64),
                        tiny,
                    )

                    u = unit_roundoff(
                        dtype_name
                    )

                    rR2_over_u = (
                        norm_RR_2
                        / denom_2
                        / u
                    )

                    rL2_over_u = (
                        norm_RL_2
                        / denom_2
                        / u
                    )

                    rRinf_over_u = (
                        norm_inf(
                            Rright
                        )
                        / denom_inf
                        / u
                    )

                    rLinf_over_u = (
                        norm_inf(
                            Rleft
                        )
                        / denom_inf
                        / u
                    )

                    tm1 = time.perf_counter()

                    output[
                        "metrics_seconds"
                    ] = tm1 - tm0

                    output[
                        "rR2_over_u"
                    ] = rR2_over_u

                    output[
                        "rL2_over_u"
                    ] = rL2_over_u

                    output[
                        "rRinf_over_u"
                    ] = rRinf_over_u

                    output[
                        "rLinf_over_u"
                    ] = rLinf_over_u

                    output[
                        "success"
                    ] = all(
                        math.isfinite(x)
                        for x in (
                            rR2_over_u,
                            rL2_over_u,
                            rRinf_over_u,
                            rLinf_over_u,
                        )
                    )

                else:
                    print(
                        "LAPACK_VALIDATION_FAIL",
                        "info_ok=",
                        info_ok,
                        "finite_X=",
                        finite_X,
                        "prefix_ok=",
                        prefix_ok,
                        "dtype_ok=",
                        dtype_ok,
                    )

            except Exception as exc:
                print(
                    "LAPACK_CASE_EXCEPTION",
                    repr(exc),
                )

        case_end = time.perf_counter()

        output[
            "total_seconds"
        ] = (
            case_end
            - case_start
        )

        rows_out.append(
            output
        )

        print(
            "success =",
            output["success"],
        )

        print(
            "GETRF/GETRI =",
            output[
                "getrf_prefix"
            ],
            output[
                "getri_prefix"
            ],
        )

        print(
            "inverse_seconds =",
            output[
                "inverse_seconds"
            ],
        )

        print(
            "metrics_seconds =",
            output[
                "metrics_seconds"
            ],
        )

        print(
            "total_seconds =",
            output[
                "total_seconds"
            ],
        )

        print(
            "rR2_over_u =",
            output[
                "rR2_over_u"
            ],
        )

        print(
            "rRinf_over_u =",
            output[
                "rRinf_over_u"
            ],
        )


overall_end = time.perf_counter()

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

csv_path = (
    OUTDIR
    / "lapack_n512_family_benchmark.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for row in rows_out:
        writer.writerow(row)


print()
print(
    "============================================================"
)
print(
    "BENCHMARK SUMMARY"
)
print(
    "============================================================"
)

print(
    "benchmark_matrix_count =",
    len(selected),
)

print(
    "benchmark_precision_records =",
    len(rows_out),
)

print(
    "successful_records =",
    sum(
        bool(r["success"])
        for r in rows_out
    ),
)

print(
    "failed_records =",
    sum(
        not bool(r["success"])
        for r in rows_out
    ),
)

print(
    "wall_seconds =",
    overall_end
    - overall_start,
)

for dtype_name in (
    "float32",
    "float64",
):
    rr = [
        r
        for r in rows_out
        if (
            r["dtype_name"]
            == dtype_name
            and r["success"]
        )
    ]

    times = np.asarray(
        [
            float(
                r["total_seconds"]
            )
            for r in rr
        ],
        dtype=np.float64,
    )

    invtimes = np.asarray(
        [
            float(
                r["inverse_seconds"]
            )
            for r in rr
        ],
        dtype=np.float64,
    )

    mettimes = np.asarray(
        [
            float(
                r["metrics_seconds"]
            )
            for r in rr
        ],
        dtype=np.float64,
    )

    if times.size:
        print()
        print(
            "dtype =",
            dtype_name,
        )

        print(
            "median_total_seconds =",
            float(
                np.median(times)
            ),
        )

        print(
            "max_total_seconds =",
            float(
                np.max(times)
            ),
        )

        print(
            "median_inverse_seconds =",
            float(
                np.median(invtimes)
            ),
        )

        print(
            "median_metrics_seconds =",
            float(
                np.median(mettimes)
            ),
        )


success_gate = (
    len(selected) > 0
    and len(rows_out)
        == 2 * len(selected)
    and all(
        bool(r["success"])
        for r in rows_out
    )
)

print()
print(
    "benchmark_gate =",
    success_gate,
)

print(
    "csv =",
    csv_path,
)

if success_gate:
    print(
        "CASTILLO_LAPACK_N512_BENCHMARK=PASS"
    )
else:
    print(
        "CASTILLO_LAPACK_N512_BENCHMARK=FAIL"
    )
