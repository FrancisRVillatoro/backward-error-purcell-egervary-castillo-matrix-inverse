#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import gzip
import json
import math
import os
import sys
import time
import traceback

import numpy as np


BASE = Path.home() / "castillo_lapack_baseline"

SELECTION = (
    BASE
    / "exact_balanced_selection"
    / "selection"
)

CASTILLO = (
    Path.home()
    / "fscratch"
    / "castillo_lapack_paired"
    / "recovered_inverse"
)

LAPACK = (
    BASE
    / "results"
    / "campaign"
    / "data"
)

OUTROOT = (
    BASE
    / "results"
    / "paired"
)

PARTIAL = (
    OUTROOT
    / "partial"
)

STATUS = (
    OUTROOT
    / "status"
)

TMP = (
    OUTROOT
    / "tmp"
)


METHODS = [
    "R0_C0",
    "R0_C1",
    "R0_C2",
    "R1_C1",
    "R2_C2",
]

DTYPES = [
    "float32",
    "float64",
]

METHOD_INDEX = {
    name: i
    for i, name in enumerate(METHODS)
}

DTYPE_INDEX = {
    name: i
    for i, name in enumerate(DTYPES)
}


def as_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


def as_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def write_json_atomic(path, obj):
    path = Path(path)

    tmp = (
        TMP
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


def main():
    task = int(
        sys.argv[1]
    )

    tag = f"{task:04d}"

    selection_path = (
        SELECTION
        / f"task_{tag}.npz"
    )

    castillo_path = (
        CASTILLO
        / f"inverse_{tag}.csv.gz"
    )

    lapack_path = (
        LAPACK
        / f"lapack_{tag}.csv.gz"
    )

    output_path = (
        PARTIAL
        / f"paired_{tag}.npz"
    )

    status_path = (
        STATUS
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
                    f"PAIRED_TASK_ALREADY_COMPLETE={tag}"
                )
                return

        except Exception:
            pass


    required = [
        selection_path,
        castillo_path,
        lapack_path,
    ]

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    if missing:
        write_json_atomic(
            status_path,
            {
                "status":
                    "TASK_FAIL",
                "task_id":
                    task,
                "missing":
                    missing,
            },
        )

        print(
            f"PAIRED_TASK_FAIL={tag}"
        )
        return


    start = time.perf_counter()

    selected = np.load(
        str(selection_path),
        allow_pickle=False,
    )

    block = np.asarray(
        selected["block"],
        dtype=np.int32,
    )

    replica = np.asarray(
        selected["replica"],
        dtype=np.int32,
    )

    rank = np.asarray(
        selected["rank"],
        dtype=np.int32,
    )

    m = int(
        block.size
    )

    selection_schema_ok = (
        block.ndim == 1
        and replica.ndim == 1
        and rank.ndim == 1
        and replica.size == m
        and rank.size == m
    )

    key_to_index = {}

    selection_duplicates = 0

    for i in range(m):
        key = (
            int(block[i]),
            int(replica[i]),
        )

        if key in key_to_index:
            selection_duplicates += 1
        else:
            key_to_index[key] = i


    castillo_coverage = np.zeros(
        (
            m,
            len(DTYPES),
            len(METHODS),
        ),
        dtype=np.uint8,
    )

    castillo_success = np.zeros(
        castillo_coverage.shape,
        dtype=np.uint8,
    )

    castillo_r2u = np.full(
        castillo_coverage.shape,
        np.nan,
        dtype=np.float64,
    )

    castillo_rinfu = np.full(
        castillo_coverage.shape,
        np.nan,
        dtype=np.float64,
    )


    lapack_coverage = np.zeros(
        (
            m,
            len(DTYPES),
        ),
        dtype=np.uint8,
    )

    lapack_success = np.zeros(
        lapack_coverage.shape,
        dtype=np.uint8,
    )

    lapack_r2u = np.full(
        lapack_coverage.shape,
        np.nan,
        dtype=np.float64,
    )

    lapack_rinfu = np.full(
        lapack_coverage.shape,
        np.nan,
        dtype=np.float64,
    )


    matrix_ids = [
        None
        for _ in range(m)
    ]

    castillo_duplicate_records = 0
    castillo_selected_rows = 0
    castillo_unexpected_methods = Counter()
    castillo_unexpected_dtypes = Counter()

    with gzip.open(
        str(castillo_path),
        mode="rt",
        encoding="utf-8",
        newline="",
    ) as handle:

        for row in csv.DictReader(
            handle
        ):
            method = str(
                row.get(
                    "method",
                    "",
                )
            )

            dtype_name = str(
                row.get(
                    "dtype_name",
                    "",
                )
            )

            if method not in METHOD_INDEX:
                castillo_unexpected_methods[
                    method
                ] += 1
                continue

            if dtype_name not in DTYPE_INDEX:
                castillo_unexpected_dtypes[
                    dtype_name
                ] += 1
                continue

            try:
                key = (
                    int(
                        row[
                            "block_index"
                        ]
                    ),
                    int(
                        row[
                            "replica"
                        ]
                    ),
                )
            except Exception:
                continue

            i = key_to_index.get(
                key
            )

            if i is None:
                continue

            di = DTYPE_INDEX[
                dtype_name
            ]

            mi = METHOD_INDEX[
                method
            ]

            if (
                castillo_coverage[
                    i,
                    di,
                    mi,
                ]
                != 0
            ):
                castillo_duplicate_records += 1
                continue

            castillo_coverage[
                i,
                di,
                mi,
            ] = 1

            castillo_selected_rows += 1

            success = as_bool(
                row.get(
                    "success",
                    False,
                )
            )

            castillo_success[
                i,
                di,
                mi,
            ] = (
                1
                if success
                else 0
            )

            matrix_id = str(
                row.get(
                    "matrix_id",
                    "",
                )
            )

            if matrix_ids[i] is None:
                matrix_ids[i] = (
                    matrix_id
                )

            elif (
                matrix_ids[i]
                != matrix_id
            ):
                matrix_ids[i] = (
                    "__MISMATCH__"
                )

            if success:
                u = as_float(
                    row.get(
                        "u",
                        math.nan,
                    )
                )

                r2 = as_float(
                    row.get(
                        "right_inverse_scaled_residual_2_est",
                        math.nan,
                    )
                )

                rinf = as_float(
                    row.get(
                        "right_inverse_scaled_residual_inf",
                        math.nan,
                    )
                )

                if (
                    math.isfinite(u)
                    and u > 0.0
                ):
                    if math.isfinite(
                        r2
                    ):
                        castillo_r2u[
                            i,
                            di,
                            mi,
                        ] = (
                            r2 / u
                        )

                    if math.isfinite(
                        rinf
                    ):
                        castillo_rinfu[
                            i,
                            di,
                            mi,
                        ] = (
                            rinf / u
                        )


    lapack_duplicate_records = 0
    lapack_selected_rows = 0
    lapack_matrix_id_mismatches = 0
    lapack_rank_mismatches = 0
    lapack_unexpected_dtypes = Counter()

    with gzip.open(
        str(lapack_path),
        mode="rt",
        encoding="utf-8",
        newline="",
    ) as handle:

        for row in csv.DictReader(
            handle
        ):
            dtype_name = str(
                row.get(
                    "dtype_name",
                    "",
                )
            )

            if dtype_name not in DTYPE_INDEX:
                lapack_unexpected_dtypes[
                    dtype_name
                ] += 1
                continue

            try:
                key = (
                    int(
                        row[
                            "block_index"
                        ]
                    ),
                    int(
                        row[
                            "replica"
                        ]
                    ),
                )
            except Exception:
                continue

            i = key_to_index.get(
                key
            )

            if i is None:
                continue

            di = DTYPE_INDEX[
                dtype_name
            ]

            if (
                lapack_coverage[
                    i,
                    di,
                ]
                != 0
            ):
                lapack_duplicate_records += 1
                continue

            lapack_coverage[
                i,
                di,
            ] = 1

            lapack_selected_rows += 1

            success = as_bool(
                row.get(
                    "success",
                    False,
                )
            )

            lapack_success[
                i,
                di,
            ] = (
                1
                if success
                else 0
            )

            lapack_matrix_id = str(
                row.get(
                    "matrix_id",
                    "",
                )
            )

            if (
                matrix_ids[i]
                is not None
                and matrix_ids[i]
                != "__MISMATCH__"
                and lapack_matrix_id
                != matrix_ids[i]
            ):
                lapack_matrix_id_mismatches += 1

            try:
                lapack_rank = int(
                    row.get(
                        "rank",
                        -1,
                    )
                )
            except Exception:
                lapack_rank = -1

            if (
                lapack_rank
                != int(
                    rank[i]
                )
            ):
                lapack_rank_mismatches += 1

            if success:
                r2 = as_float(
                    row.get(
                        "rR2_over_u",
                        math.nan,
                    )
                )

                rinf = as_float(
                    row.get(
                        "rRinf_over_u",
                        math.nan,
                    )
                )

                if math.isfinite(
                    r2
                ):
                    lapack_r2u[
                        i,
                        di,
                    ] = r2

                if math.isfinite(
                    rinf
                ):
                    lapack_rinfu[
                        i,
                        di,
                    ] = rinf


    expected_castillo = (
        m
        * len(DTYPES)
        * len(METHODS)
    )

    expected_lapack = (
        m
        * len(DTYPES)
    )

    actual_castillo_coverage = int(
        np.sum(
            castillo_coverage
        )
    )

    actual_lapack_coverage = int(
        np.sum(
            lapack_coverage
        )
    )

    castillo_matrix_id_mismatches = sum(
        value == "__MISMATCH__"
        for value in matrix_ids
    )

    missing_matrix_ids = sum(
        value is None
        for value in matrix_ids
    )


    common5 = np.all(
        castillo_success.astype(
            bool
        ),
        axis=2,
    )

    common6 = (
        common5
        & lapack_success.astype(
            bool
        )
    )


    structural_gate = (
        selection_schema_ok
        and selection_duplicates == 0
        and actual_castillo_coverage
            == expected_castillo
        and actual_lapack_coverage
            == expected_lapack
        and castillo_duplicate_records
            == 0
        and lapack_duplicate_records
            == 0
        and castillo_matrix_id_mismatches
            == 0
        and missing_matrix_ids
            == 0
        and lapack_matrix_id_mismatches
            == 0
        and lapack_rank_mismatches
            == 0
    )


    tmp_output = (
        TMP
        / (
            output_path.name
            + ".tmp."
            + str(os.getpid())
        )
    )

    with tmp_output.open(
        "wb"
    ) as h:
        np.savez_compressed(
            h,
            block=block,
            replica=replica,
            rank=rank,

            castillo_coverage=
                castillo_coverage,
            castillo_success=
                castillo_success,
            castillo_r2u=
                castillo_r2u,
            castillo_rinfu=
                castillo_rinfu,

            lapack_coverage=
                lapack_coverage,
            lapack_success=
                lapack_success,
            lapack_r2u=
                lapack_r2u,
            lapack_rinfu=
                lapack_rinfu,

            common5=
                common5.astype(
                    np.uint8
                ),
            common6=
                common6.astype(
                    np.uint8
                ),
        )

    if structural_gate:
        os.replace(
            str(tmp_output),
            str(output_path),
        )

        task_status = (
            "TASK_OK"
        )

    else:
        task_status = (
            "TASK_FAIL"
        )


    elapsed = (
        time.perf_counter()
        - start
    )

    status = {
        "status":
            task_status,

        "task_id":
            task,

        "selected_matrices":
            m,

        "selection_schema_ok":
            selection_schema_ok,

        "selection_duplicates":
            selection_duplicates,

        "expected_castillo_records":
            expected_castillo,

        "actual_castillo_coverage":
            actual_castillo_coverage,

        "castillo_selected_rows":
            castillo_selected_rows,

        "castillo_duplicate_records":
            castillo_duplicate_records,

        "castillo_matrix_id_mismatches":
            castillo_matrix_id_mismatches,

        "missing_matrix_ids":
            missing_matrix_ids,

        "expected_lapack_records":
            expected_lapack,

        "actual_lapack_coverage":
            actual_lapack_coverage,

        "lapack_selected_rows":
            lapack_selected_rows,

        "lapack_duplicate_records":
            lapack_duplicate_records,

        "lapack_matrix_id_mismatches":
            lapack_matrix_id_mismatches,

        "lapack_rank_mismatches":
            lapack_rank_mismatches,

        "common5_float32":
            int(
                np.sum(
                    common5[:, 0]
                )
            ),

        "common5_float64":
            int(
                np.sum(
                    common5[:, 1]
                )
            ),

        "common6_float32":
            int(
                np.sum(
                    common6[:, 0]
                )
            ),

        "common6_float64":
            int(
                np.sum(
                    common6[:, 1]
                )
            ),

        "castillo_unexpected_methods":
            dict(
                castillo_unexpected_methods
            ),

        "castillo_unexpected_dtypes":
            dict(
                castillo_unexpected_dtypes
            ),

        "lapack_unexpected_dtypes":
            dict(
                lapack_unexpected_dtypes
            ),

        "elapsed_seconds":
            elapsed,

        "output":
            str(
                output_path
            ),
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

    if task_status == "TASK_OK":
        print(
            f"CASTILLO_SIX_METHOD_PAIR_TASK_OK={tag}"
        )
    else:
        print(
            f"CASTILLO_SIX_METHOD_PAIR_TASK_FAIL={tag}"
        )


try:
    main()

except Exception as exc:
    task = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else -1
    )

    tag = (
        f"{task:04d}"
        if task >= 0
        else "unknown"
    )

    status = {
        "status":
            "TASK_FAIL",

        "task_id":
            task,

        "exception":
            repr(exc),

        "traceback":
            traceback.format_exc(),
    }

    write_json_atomic(
        STATUS
        / f"task_{tag}.json",
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
        f"CASTILLO_SIX_METHOD_PAIR_TASK_FAIL={tag}"
    )
