#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs


BASE = Path.home() / "castillo_lapack_baseline"
SOURCE = BASE / "source"
SELDIR = BASE / "exact_balanced_selection" / "selection"

ROOT = BASE / "results" / "campaign"
DATADIR = ROOT / "data"
STATUSDIR = ROOT / "status"
TMPDIR = ROOT / "tmp"

sys.path.insert(
    0,
    str((SOURCE / "code").resolve()),
)

from families import (
    build_matrix,
    stable_seed,
    stable_matrix_id,
)

from castillo import (
    spectral_norm_estimate,
    norm_inf,
    unit_roundoff,
)


FIELDS = [
    "task_id",
    "matrix_id",
    "block_index",
    "cell_id",
    "family",
    "n",
    "replica",
    "rank",
    "seed",
    "dtype_name",
    "u",
    "input_representable",
    "success",
    "failure_class",
    "failure_reason",
    "getrf_prefix",
    "getri_prefix",
    "getrf_info",
    "getri_lwork_info",
    "getri_info",
    "lwork",
    "generation_seconds",
    "inverse_seconds",
    "metrics_seconds",
    "total_seconds",
    "norm_A_inf",
    "norm_A_2_est",
    "norm_X_inf",
    "norm_X_2_est",
    "right_defect_inf",
    "right_defect_2_est",
    "left_defect_inf",
    "left_defect_2_est",
    "right_scaled_inf",
    "right_scaled_2_est",
    "left_scaled_inf",
    "left_scaled_2_est",
    "rRinf_over_u",
    "rR2_over_u",
    "rLinf_over_u",
    "rL2_over_u",
]


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_json_atomic(path, obj):
    path = Path(path)

    tmp = (
        TMPDIR
        / (
            path.name
            + ".tmp."
            + str(os.getpid())
        )
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        str(tmp),
        str(path),
    )


def load_manifest(path):
    result = {}

    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            block = int(
                row["block_index"]
            )

            item = dict(row)

            item["block_index"] = block
            item["n"] = int(
                row["n"]
            )

            item["parameters"] = json.loads(
                row["parameters_json"]
            )

            result[block] = item

    return result


def blank_metrics():
    return {
        "norm_A_inf": math.nan,
        "norm_A_2_est": math.nan,
        "norm_X_inf": math.nan,
        "norm_X_2_est": math.nan,
        "right_defect_inf": math.nan,
        "right_defect_2_est": math.nan,
        "left_defect_inf": math.nan,
        "left_defect_2_est": math.nan,
        "right_scaled_inf": math.nan,
        "right_scaled_2_est": math.nan,
        "left_scaled_inf": math.nan,
        "left_scaled_2_est": math.nan,
        "rRinf_over_u": math.nan,
        "rR2_over_u": math.nan,
        "rLinf_over_u": math.nan,
        "rL2_over_u": math.nan,
    }


def main():
    task_id = int(
        sys.argv[1]
    )

    tag = f"{task_id:04d}"

    selection_path = (
        SELDIR
        / f"task_{tag}.npz"
    )

    output_path = (
        DATADIR
        / f"lapack_{tag}.csv.gz"
    )

    status_path = (
        STATUSDIR
        / f"task_{tag}.json"
    )

    if (
        output_path.is_file()
        and status_path.is_file()
    ):
        try:
            old = json.loads(
                status_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                old.get("status")
                == "TASK_OK"
            ):
                print(
                    f"TASK_ALREADY_COMPLETE={tag}"
                )
                return

        except Exception:
            pass

    canonical_path = (
        SOURCE
        / "config"
        / "canonical.json"
    )

    execution_path = (
        SOURCE
        / "config"
        / "execution.json"
    )

    manifest_path = (
        SOURCE
        / "reports"
        / "canonical"
        / "canonical_manifest.csv"
    )

    source_files = [
        SOURCE / "code" / "families.py",
        SOURCE / "code" / "castillo.py",
        canonical_path,
        execution_path,
        manifest_path,
        selection_path,
        Path(__file__),
    ]

    problems = []

    for p in source_files:
        if not p.is_file():
            problems.append(
                "missing:" + str(p)
            )

    if problems:
        status = {
            "status": "TASK_FAIL",
            "task_id": task_id,
            "problems": problems,
        }

        write_json_atomic(
            status_path,
            status,
        )

        print(
            f"LAPACK_TASK_FAIL={tag}"
        )
        print(
            json.dumps(
                status,
                indent=2,
                sort_keys=True,
            )
        )
        return

    canonical = json.loads(
        canonical_path.read_text(
            encoding="utf-8"
        )
    )

    execution = json.loads(
        execution_path.read_text(
            encoding="utf-8"
        )
    )

    spectral_iterations = int(
        execution[
            "spectral_norm_iterations"
        ]
    )

    manifest = load_manifest(
        manifest_path
    )

    sel = np.load(
        str(selection_path),
        allow_pickle=False,
    )

    blocks = np.asarray(
        sel["block"],
        dtype=np.int64,
    )

    replicas = np.asarray(
        sel["replica"],
        dtype=np.int64,
    )

    ranks = np.asarray(
        sel["rank"],
        dtype=np.int64,
    )

    schema_ok = (
        blocks.ndim == 1
        and replicas.ndim == 1
        and ranks.ndim == 1
        and blocks.size == replicas.size
        and blocks.size == ranks.size
    )

    if not schema_ok:
        status = {
            "status": "TASK_FAIL",
            "task_id": task_id,
            "problem": "selection_schema",
        }

        write_json_atomic(
            status_path,
            status,
        )

        print(
            f"LAPACK_TASK_FAIL={tag}"
        )
        return

    dtype_specs = (
        ("float32", np.float32, "s"),
        ("float64", np.float64, "d"),
    )

    lapack = {}

    for dtype_name, dtype, prefix in dtype_specs:
        probe = np.eye(
            2,
            dtype=dtype,
            order="F",
        )

        getrf, getri, getri_lwork = (
            get_lapack_funcs(
                (
                    "getrf",
                    "getri",
                    "getri_lwork",
                ),
                (probe,),
            )
        )

        lapack[dtype_name] = (
            getrf,
            getri,
            getri_lwork,
            prefix,
        )

    lwork_cache = {}
    identity_cache = {}

    failure_counts = Counter()
    dtype_counts = Counter()
    family_counts = Counter()

    records = 0
    successes = 0
    failures = 0
    build_failures = 0
    representability_failures = 0

    start_task = time.perf_counter()

    tmp_output = (
        TMPDIR
        / (
            output_path.name
            + ".tmp."
            + str(os.getpid())
        )
    )

    with gzip.open(
        str(tmp_output),
        mode="wt",
        encoding="utf-8",
        newline="",
        compresslevel=1,
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for block, replica, rank in zip(
            blocks.tolist(),
            replicas.tolist(),
            ranks.tolist(),
        ):
            block = int(block)
            replica = int(replica)
            rank = int(rank)

            meta = manifest.get(
                block
            )

            if meta is None:
                for dtype_name, _, _ in dtype_specs:
                    row = {
                        "task_id": task_id,
                        "matrix_id": "",
                        "block_index": block,
                        "cell_id": "",
                        "family": "",
                        "n": -1,
                        "replica": replica,
                        "rank": rank,
                        "seed": "",
                        "dtype_name": dtype_name,
                        "u": unit_roundoff(
                            dtype_name
                        ),
                        "input_representable": False,
                        "success": False,
                        "failure_class":
                            "manifest_block_missing",
                        "failure_reason":
                            "block_not_found",
                        "getrf_prefix": "",
                        "getri_prefix": "",
                        "getrf_info": "",
                        "getri_lwork_info": "",
                        "getri_info": "",
                        "lwork": "",
                        "generation_seconds":
                            math.nan,
                        "inverse_seconds":
                            math.nan,
                        "metrics_seconds":
                            math.nan,
                        "total_seconds":
                            math.nan,
                    }

                    row.update(
                        blank_metrics()
                    )

                    writer.writerow(row)

                    records += 1
                    failures += 1

                    failure_counts[
                        "manifest_block_missing"
                    ] += 1

                continue

            n = int(
                meta["n"]
            )

            family = str(
                meta["family"]
            )

            cell_id = str(
                meta["cell_id"]
            )

            parameters = meta[
                "parameters"
            ]

            seed = stable_seed(
                canonical[
                    "seed_namespace"
                ],
                cell_id,
                n,
                replica,
            )

            matrix_id = stable_matrix_id(
                canonical[
                    "seed_namespace"
                ],
                cell_id,
                n,
                replica,
            )

            family_counts[
                family
            ] += 1

            tg0 = time.perf_counter()

            build_ok = True
            build_reason = ""

            try:
                Aref = build_matrix(
                    family=family,
                    n=n,
                    parameters=parameters,
                    seed=seed,
                )

                if np.isnan(
                    Aref
                ).any():
                    build_ok = False
                    build_reason = (
                        "reference_contains_nan"
                    )

            except Exception as exc:
                build_ok = False
                build_reason = repr(
                    exc
                )
                Aref = None

            tg1 = time.perf_counter()

            generation_seconds = (
                tg1 - tg0
            )

            if not build_ok:
                build_failures += 1

            for (
                dtype_name,
                dtype,
                expected_prefix,
            ) in dtype_specs:

                dtype_counts[
                    dtype_name
                ] += 1

                u = unit_roundoff(
                    dtype_name
                )

                base = {
                    "task_id":
                        task_id,
                    "matrix_id":
                        matrix_id,
                    "block_index":
                        block,
                    "cell_id":
                        cell_id,
                    "family":
                        family,
                    "n":
                        n,
                    "replica":
                        replica,
                    "rank":
                        rank,
                    "seed":
                        seed,
                    "dtype_name":
                        dtype_name,
                    "u":
                        u,
                    "generation_seconds":
                        generation_seconds,
                }

                base.update(
                    blank_metrics()
                )

                td0 = time.perf_counter()

                if not build_ok:
                    base.update(
                        {
                            "input_representable":
                                False,
                            "success":
                                False,
                            "failure_class":
                                "matrix_build_failure",
                            "failure_reason":
                                build_reason,
                            "getrf_prefix":
                                "",
                            "getri_prefix":
                                "",
                            "getrf_info":
                                "",
                            "getri_lwork_info":
                                "",
                            "getri_info":
                                "",
                            "lwork":
                                "",
                            "inverse_seconds":
                                math.nan,
                            "metrics_seconds":
                                math.nan,
                            "total_seconds":
                                time.perf_counter()
                                - td0,
                        }
                    )

                    writer.writerow(base)

                    records += 1
                    failures += 1

                    failure_counts[
                        "matrix_build_failure"
                    ] += 1

                    continue

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

                base[
                    "input_representable"
                ] = representable

                if not representable:
                    representability_failures += 1

                    base.update(
                        {
                            "success":
                                False,
                            "failure_class":
                                "input_not_representable",
                            "failure_reason":
                                "nonfinite_in_working_precision",
                            "getrf_prefix":
                                "",
                            "getri_prefix":
                                "",
                            "getrf_info":
                                "",
                            "getri_lwork_info":
                                "",
                            "getri_info":
                                "",
                            "lwork":
                                "",
                            "inverse_seconds":
                                math.nan,
                            "metrics_seconds":
                                math.nan,
                            "total_seconds":
                                time.perf_counter()
                                - td0,
                        }
                    )

                    writer.writerow(base)

                    records += 1
                    failures += 1

                    failure_counts[
                        "input_not_representable"
                    ] += 1

                    continue

                A64 = np.asarray(
                    A,
                    dtype=np.float64,
                )

                norm_A_inf = norm_inf(
                    A64
                )

                norm_A_2 = (
                    spectral_norm_estimate(
                        A,
                        spectral_iterations,
                    )
                )

                base[
                    "norm_A_inf"
                ] = norm_A_inf

                base[
                    "norm_A_2_est"
                ] = norm_A_2

                getrf, getri, getri_lwork, _ = (
                    lapack[
                        dtype_name
                    ]
                )

                Af = np.array(
                    A,
                    dtype=dtype,
                    order="F",
                    copy=True,
                )

                ti0 = time.perf_counter()

                info_rf = -999
                info_lw = -999
                info_ri = -999
                lwork = -1
                X = None

                try:
                    lu, piv, info_rf = getrf(
                        Af,
                        overwrite_a=True,
                    )

                    key = (
                        dtype_name,
                        n,
                    )

                    if key not in lwork_cache:
                        lw_value, info_lw = (
                            getri_lwork(
                                n
                            )
                        )

                        lwork_cache[
                            key
                        ] = (
                            max(
                                1,
                                int(
                                    float(
                                        np.real(
                                            lw_value
                                        )
                                    )
                                ),
                            ),
                            int(
                                info_lw
                            ),
                        )

                    lwork, info_lw = (
                        lwork_cache[
                            key
                        ]
                    )

                    if (
                        int(info_rf) == 0
                        and int(info_lw) == 0
                    ):
                        X, info_ri = getri(
                            lu,
                            piv,
                            lwork=lwork,
                            overwrite_lu=True,
                        )

                except Exception as exc:
                    base[
                        "failure_reason"
                    ] = repr(
                        exc
                    )

                ti1 = time.perf_counter()

                inverse_seconds = (
                    ti1 - ti0
                )

                prefix_rf = getattr(
                    getrf,
                    "prefix",
                    None,
                )

                prefix_ri = getattr(
                    getri,
                    "prefix",
                    None,
                )

                base.update(
                    {
                        "getrf_prefix":
                            prefix_rf,
                        "getri_prefix":
                            prefix_ri,
                        "getrf_info":
                            int(info_rf),
                        "getri_lwork_info":
                            int(info_lw),
                        "getri_info":
                            int(info_ri),
                        "lwork":
                            int(lwork),
                        "inverse_seconds":
                            inverse_seconds,
                    }
                )

                lapack_ok = (
                    int(info_rf) == 0
                    and int(info_lw) == 0
                    and int(info_ri) == 0
                    and X is not None
                    and X.dtype
                        == np.dtype(dtype)
                    and prefix_rf
                        == expected_prefix
                    and prefix_ri
                        == expected_prefix
                    and np.all(
                        np.isfinite(X)
                    )
                )

                if not lapack_ok:
                    if (
                        int(info_rf) > 0
                    ):
                        failure_class = (
                            "singular_getrf"
                        )

                    elif (
                        int(info_rf) < 0
                        or int(info_lw) < 0
                        or int(info_ri) < 0
                    ):
                        failure_class = (
                            "lapack_argument_failure"
                        )

                    elif (
                        X is not None
                        and not np.all(
                            np.isfinite(X)
                        )
                    ):
                        failure_class = (
                            "nonfinite_inverse"
                        )

                    else:
                        failure_class = (
                            "lapack_failure"
                        )

                    base.update(
                        {
                            "success":
                                False,
                            "failure_class":
                                failure_class,
                            "failure_reason":
                                base.get(
                                    "failure_reason",
                                    "",
                                ),
                            "metrics_seconds":
                                math.nan,
                            "total_seconds":
                                time.perf_counter()
                                - td0,
                        }
                    )

                    writer.writerow(base)

                    records += 1
                    failures += 1

                    failure_counts[
                        failure_class
                    ] += 1

                    continue

                tm0 = time.perf_counter()

                X64 = np.asarray(
                    X,
                    dtype=np.float64,
                )

                if n not in identity_cache:
                    identity_cache[
                        n
                    ] = np.eye(
                        n,
                        dtype=np.float64,
                    )

                I = identity_cache[
                    n
                ]

                with np.errstate(
                    over="ignore",
                    invalid="ignore",
                ):
                    Rright = (
                        I
                        - A64 @ X64
                    )

                    Rleft = (
                        I
                        - X64 @ A64
                    )

                norm_X_inf = norm_inf(
                    X64
                )

                norm_X_2 = (
                    spectral_norm_estimate(
                        X64,
                        spectral_iterations,
                    )
                )

                right_inf = norm_inf(
                    Rright
                )

                left_inf = norm_inf(
                    Rleft
                )

                right_2 = (
                    spectral_norm_estimate(
                        Rright,
                        spectral_iterations,
                    )
                )

                left_2 = (
                    spectral_norm_estimate(
                        Rleft,
                        spectral_iterations,
                    )
                )

                tiny = np.finfo(
                    np.float64
                ).tiny

                denominator_inf = max(
                    norm_A_inf
                    * norm_X_inf,
                    tiny,
                )

                denominator_2 = max(
                    norm_A_2
                    * norm_X_2,
                    tiny,
                )

                right_scaled_inf = (
                    right_inf
                    / denominator_inf
                )

                left_scaled_inf = (
                    left_inf
                    / denominator_inf
                )

                right_scaled_2 = (
                    right_2
                    / denominator_2
                )

                left_scaled_2 = (
                    left_2
                    / denominator_2
                )

                tm1 = time.perf_counter()

                metrics_seconds = (
                    tm1 - tm0
                )

                values = [
                    norm_A_inf,
                    norm_A_2,
                    norm_X_inf,
                    norm_X_2,
                    right_inf,
                    right_2,
                    left_inf,
                    left_2,
                    right_scaled_inf,
                    right_scaled_2,
                    left_scaled_inf,
                    left_scaled_2,
                ]

                metrics_finite = all(
                    math.isfinite(
                        float(x)
                    )
                    for x in values
                )

                if metrics_finite:
                    success = True
                    failure_class = (
                        "completed"
                    )
                    failure_reason = ""
                    successes += 1

                else:
                    success = False
                    failure_class = (
                        "nonfinite_metric"
                    )
                    failure_reason = ""
                    failures += 1
                    failure_counts[
                        failure_class
                    ] += 1

                base.update(
                    {
                        "success":
                            success,
                        "failure_class":
                            failure_class,
                        "failure_reason":
                            failure_reason,
                        "metrics_seconds":
                            metrics_seconds,
                        "total_seconds":
                            time.perf_counter()
                            - td0,
                        "norm_X_inf":
                            norm_X_inf,
                        "norm_X_2_est":
                            norm_X_2,
                        "right_defect_inf":
                            right_inf,
                        "right_defect_2_est":
                            right_2,
                        "left_defect_inf":
                            left_inf,
                        "left_defect_2_est":
                            left_2,
                        "right_scaled_inf":
                            right_scaled_inf,
                        "right_scaled_2_est":
                            right_scaled_2,
                        "left_scaled_inf":
                            left_scaled_inf,
                        "left_scaled_2_est":
                            left_scaled_2,
                        "rRinf_over_u":
                            right_scaled_inf
                            / u,
                        "rR2_over_u":
                            right_scaled_2
                            / u,
                        "rLinf_over_u":
                            left_scaled_inf
                            / u,
                        "rL2_over_u":
                            left_scaled_2
                            / u,
                    }
                )

                writer.writerow(base)

                records += 1

    expected_records = (
        2 * int(blocks.size)
    )

    structural_ok = (
        records == expected_records
        and successes + failures
            == expected_records
    )

    if structural_ok:
        os.replace(
            str(tmp_output),
            str(output_path),
        )

        output_hash = sha256_file(
            output_path
        )

        status_value = (
            "TASK_OK"
        )

    else:
        output_hash = ""
        status_value = (
            "TASK_FAIL"
        )

    elapsed = (
        time.perf_counter()
        - start_task
    )

    status = {
        "status":
            status_value,
        "task_id":
            task_id,
        "selected_matrices":
            int(blocks.size),
        "expected_records":
            expected_records,
        "records":
            records,
        "successes":
            successes,
        "failures":
            failures,
        "build_failures":
            build_failures,
        "representability_failures":
            representability_failures,
        "failure_counts":
            dict(
                sorted(
                    failure_counts.items()
                )
            ),
        "dtype_counts":
            dict(
                sorted(
                    dtype_counts.items()
                )
            ),
        "family_matrix_counts":
            dict(
                sorted(
                    family_counts.items()
                )
            ),
        "elapsed_seconds":
            elapsed,
        "spectral_norm_iterations":
            spectral_iterations,
        "selection_sha256":
            sha256_file(
                selection_path
            ),
        "output_sha256":
            output_hash,
        "source_sha256": {
            str(
                p.relative_to(BASE)
                if p.is_relative_to(BASE)
                else p
            ):
                sha256_file(p)
            for p in source_files
        },
    }

    write_json_atomic(
        status_path,
        status,
    )

    print(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
    )

    if status_value == "TASK_OK":
        print(
            f"CASTILLO_LAPACK_TASK_OK={tag}"
        )
    else:
        print(
            f"CASTILLO_LAPACK_TASK_FAIL={tag}"
        )


try:
    main()

except Exception as exc:
    task_id = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else -1
    )

    tag = (
        f"{task_id:04d}"
        if task_id >= 0
        else "unknown"
    )

    STATUSDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        STATUSDIR
        / f"task_{tag}.json"
    )

    status = {
        "status":
            "TASK_FAIL",
        "task_id":
            task_id,
        "exception":
            repr(exc),
        "traceback":
            traceback.format_exc(),
    }

    write_json_atomic(
        status_path,
        status,
    )

    print(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"CASTILLO_LAPACK_TASK_FAIL={tag}"
    )
