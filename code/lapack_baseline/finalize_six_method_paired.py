#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json
import math
import traceback

import numpy as np


BASE = Path.home() / "castillo_lapack_baseline"

PAIRROOT = (
    BASE
    / "results"
    / "paired"
)

PARTIAL = (
    PAIRROOT
    / "partial"
)

STATUS = (
    PAIRROOT
    / "status"
)

OUT = (
    PAIRROOT
    / "final"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


METHODS = [
    "R0_C0",
    "R0_C1",
    "R0_C2",
    "R1_C1",
    "R2_C2",
    "LAPACK_xGETRF_xGETRI",
]

CASTILLO_METHODS = METHODS[:5]

DTYPES = [
    "float32",
    "float64",
]

NB = 832
NS = 2000
ND = 2
NM = 6


def finite_quantile(a, p):
    x = np.asarray(
        a,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return math.nan

    return float(
        np.quantile(
            x,
            p,
        )
    )


def finite_median(a):
    return finite_quantile(
        a,
        0.5,
    )


print(
    "============================================================"
)
print(
    "CASTILLO + LAPACK STRICT PAIRED FINALIZER"
)
print(
    "============================================================"
)


# ============================================================
# 1. AUDIT THE 999 TASKS BEFORE READING THE PARTIAL MATRICES
# ============================================================

status_paths = sorted(
    STATUS.glob(
        "task_????.json"
    )
)

partial_paths = sorted(
    PARTIAL.glob(
        "paired_????.npz"
    )
)

states = Counter()

seen_tasks = set()

selected_status_total = 0

castillo_coverage_status_total = 0
lapack_coverage_status_total = 0

common5_status = np.zeros(
    ND,
    dtype=np.int64,
)

common6_status = np.zeros(
    ND,
    dtype=np.int64,
)

bad_json = []
bad_status_structure = []

for p in status_paths:

    try:
        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        bad_json.append(
            (
                p.name,
                repr(exc),
            )
        )
        continue

    state = str(
        obj.get(
            "status",
            "MISSING",
        )
    )

    states[state] += 1

    task = int(
        obj.get(
            "task_id",
            -1,
        )
    )

    seen_tasks.add(
        task
    )

    selected = int(
        obj.get(
            "selected_matrices",
            0,
        )
    )

    selected_status_total += (
        selected
    )

    ccov = int(
        obj.get(
            "actual_castillo_coverage",
            0,
        )
    )

    lcov = int(
        obj.get(
            "actual_lapack_coverage",
            0,
        )
    )

    castillo_coverage_status_total += (
        ccov
    )

    lapack_coverage_status_total += (
        lcov
    )

    common5_status[0] += int(
        obj.get(
            "common5_float32",
            0,
        )
    )

    common5_status[1] += int(
        obj.get(
            "common5_float64",
            0,
        )
    )

    common6_status[0] += int(
        obj.get(
            "common6_float32",
            0,
        )
    )

    common6_status[1] += int(
        obj.get(
            "common6_float64",
            0,
        )
    )

    if state == "TASK_OK":

        expected_c = (
            selected
            * 2
            * 5
        )

        expected_l = (
            selected
            * 2
        )

        if (
            ccov != expected_c
            or lcov != expected_l
            or int(
                obj.get(
                    "selection_duplicates",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "castillo_duplicate_records",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "lapack_duplicate_records",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "castillo_matrix_id_mismatches",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "missing_matrix_ids",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "lapack_matrix_id_mismatches",
                    -1,
                )
            ) != 0
            or int(
                obj.get(
                    "lapack_rank_mismatches",
                    -1,
                )
            ) != 0
        ):
            bad_status_structure.append(
                task
            )


expected_tasks = set(
    range(999)
)

missing_tasks = sorted(
    expected_tasks
    - seen_tasks
)

unexpected_tasks = sorted(
    seen_tasks
    - expected_tasks
)


status_gate = (
    len(status_paths) == 999
    and len(partial_paths) == 999
    and states.get(
        "TASK_OK",
        0,
    ) == 999
    and len(states) == 1
    and not bad_json
    and not bad_status_structure
    and not missing_tasks
    and not unexpected_tasks
    and selected_status_total
        == 1664000
    and castillo_coverage_status_total
        == 16640000
    and lapack_coverage_status_total
        == 3328000
)


print()
print(
    "===== TASK AUDIT ====="
)

print(
    "status_files =",
    len(status_paths),
)

print(
    "partial_npz_files =",
    len(partial_paths),
)

print(
    "states =",
    dict(
        sorted(
            states.items()
        )
    ),
)

print(
    "selected_status_total =",
    selected_status_total,
)

print(
    "castillo_coverage_status_total =",
    castillo_coverage_status_total,
)

print(
    "lapack_coverage_status_total =",
    lapack_coverage_status_total,
)

print(
    "missing_tasks =",
    len(missing_tasks),
)

print(
    "unexpected_tasks =",
    len(unexpected_tasks),
)

print(
    "bad_json =",
    len(bad_json),
)

print(
    "bad_status_structure =",
    len(
        bad_status_structure
    ),
)

print(
    "status_gate =",
    status_gate,
)


# ============================================================
# 2. ONLY IF THE TASK AUDIT PASSES, RECONSTRUCT EVERYTHING
# ============================================================

final_gate = False

if status_gate:

    summary_path = (
        BASE
        / "exact_balanced_selection"
        / "summaries"
        / "selection_summary_release.json"
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    selected_blocks = sorted(
        summary[
            "selected_blocks"
        ],
        key=lambda x:
            int(
                x[
                    "block_index"
                ]
            ),
    )

    block_ids = [
        int(
            x[
                "block_index"
            ]
        )
        for x in selected_blocks
    ]

    if len(block_ids) != NB:
        print(
            "BAD_BLOCK_METADATA_COUNT =",
            len(block_ids),
        )

    block_pos = {
        block: i
        for i, block
        in enumerate(
            block_ids
        )
    }

    block_family = [
        str(
            x["family"]
        )
        for x in selected_blocks
    ]

    block_n = np.asarray(
        [
            int(
                x["n"]
            )
            for x in selected_blocks
        ],
        dtype=np.int32,
    )

    block_cell = [
        str(
            x["cell_id"]
        )
        for x in selected_blocks
    ]


    rank_coverage = np.zeros(
        (
            NB,
            NS,
        ),
        dtype=np.uint8,
    )

    replica = np.full(
        (
            NB,
            NS,
        ),
        -1,
        dtype=np.int32,
    )

    coverage = np.zeros(
        (
            NB,
            NS,
            ND,
            NM,
        ),
        dtype=np.uint8,
    )

    success = np.zeros(
        (
            NB,
            NS,
            ND,
            NM,
        ),
        dtype=np.uint8,
    )

    r2u = np.full(
        (
            NB,
            NS,
            ND,
            NM,
        ),
        np.nan,
        dtype=np.float64,
    )

    rinfu = np.full(
        (
            NB,
            NS,
            ND,
            NM,
        ),
        np.nan,
        dtype=np.float64,
    )


    global_rank_duplicates = 0
    unknown_blocks = 0

    stored_common5_mismatches = 0
    stored_common6_mismatches = 0

    total_partial_matrices = 0


    print()
    print(
        "===== RECONSTRUCT GLOBAL ARRAYS ====="
    )

    for file_index, path in enumerate(
        partial_paths,
        1,
    ):

        if (
            file_index == 1
            or file_index % 100 == 0
            or file_index
                == len(partial_paths)
        ):
            print(
                "loading",
                file_index,
                "of",
                len(partial_paths),
                path.name,
            )

        z = np.load(
            str(path),
            allow_pickle=False,
        )

        b = np.asarray(
            z["block"],
            dtype=np.int32,
        )

        rep = np.asarray(
            z["replica"],
            dtype=np.int32,
        )

        rank = np.asarray(
            z["rank"],
            dtype=np.int32,
        )

        c_cov = np.asarray(
            z["castillo_coverage"],
            dtype=np.uint8,
        )

        c_suc = np.asarray(
            z["castillo_success"],
            dtype=np.uint8,
        )

        c_r2 = np.asarray(
            z["castillo_r2u"],
            dtype=np.float64,
        )

        c_ri = np.asarray(
            z["castillo_rinfu"],
            dtype=np.float64,
        )

        l_cov = np.asarray(
            z["lapack_coverage"],
            dtype=np.uint8,
        )

        l_suc = np.asarray(
            z["lapack_success"],
            dtype=np.uint8,
        )

        l_r2 = np.asarray(
            z["lapack_r2u"],
            dtype=np.float64,
        )

        l_ri = np.asarray(
            z["lapack_rinfu"],
            dtype=np.float64,
        )

        stored5 = np.asarray(
            z["common5"],
            dtype=np.uint8,
        ).astype(bool)

        stored6 = np.asarray(
            z["common6"],
            dtype=np.uint8,
        ).astype(bool)

        m = int(
            b.size
        )

        total_partial_matrices += (
            m
        )

        calculated5 = np.all(
            c_suc.astype(bool),
            axis=2,
        )

        calculated6 = (
            calculated5
            & l_suc.astype(bool)
        )

        stored_common5_mismatches += int(
            np.sum(
                stored5
                != calculated5
            )
        )

        stored_common6_mismatches += int(
            np.sum(
                stored6
                != calculated6
            )
        )


        for ub in np.unique(
            b
        ):

            ub_int = int(
                ub
            )

            bp = block_pos.get(
                ub_int
            )

            if bp is None:
                unknown_blocks += int(
                    np.sum(
                        b == ub
                    )
                )
                continue

            mask = (
                b == ub
            )

            rr = rank[
                mask
            ].astype(
                np.int64
            )

            if np.any(
                (
                    rr < 0
                )
                | (
                    rr >= NS
                )
            ):
                unknown_blocks += int(
                    rr.size
                )
                continue

            duplicate_here = (
                rank_coverage[
                    bp,
                    rr,
                ]
                != 0
            )

            global_rank_duplicates += int(
                np.sum(
                    duplicate_here
                )
            )

            rank_coverage[
                bp,
                rr,
            ] = 1

            replica[
                bp,
                rr,
            ] = rep[
                mask
            ]

            coverage[
                bp,
                rr,
                :,
                :5,
            ] = c_cov[
                mask,
                :,
                :,
            ]

            success[
                bp,
                rr,
                :,
                :5,
            ] = c_suc[
                mask,
                :,
                :,
            ]

            r2u[
                bp,
                rr,
                :,
                :5,
            ] = c_r2[
                mask,
                :,
                :,
            ]

            rinfu[
                bp,
                rr,
                :,
                :5,
            ] = c_ri[
                mask,
                :,
                :,
            ]

            coverage[
                bp,
                rr,
                :,
                5,
            ] = l_cov[
                mask,
                :,
            ]

            success[
                bp,
                rr,
                :,
                5,
            ] = l_suc[
                mask,
                :,
            ]

            r2u[
                bp,
                rr,
                :,
                5,
            ] = l_r2[
                mask,
                :,
            ]

            rinfu[
                bp,
                rr,
                :,
                5,
            ] = l_ri[
                mask,
                :,
            ]


    rank_coverage_count = int(
        np.sum(
            rank_coverage
        )
    )

    method_coverage_count = int(
        np.sum(
            coverage
        )
    )

    expected_method_coverage = (
        NB
        * NS
        * ND
        * NM
    )

    reconstructed_common5 = np.all(
        success[
            :,
            :,
            :,
            :5,
        ].astype(bool),
        axis=3,
    )

    reconstructed_common6 = np.all(
        success.astype(bool),
        axis=3,
    )


    common5_counts = np.sum(
        reconstructed_common5,
        axis=(0, 1),
        dtype=np.int64,
    )

    common6_counts = np.sum(
        reconstructed_common6,
        axis=(0, 1),
        dtype=np.int64,
    )


    print()
    print(
        "===== GLOBAL COVERAGE AUDIT ====="
    )

    print(
        "total_partial_matrices =",
        total_partial_matrices,
    )

    print(
        "rank_coverage_count =",
        rank_coverage_count,
    )

    print(
        "expected_rank_coverage =",
        NB * NS,
    )

    print(
        "method_coverage_count =",
        method_coverage_count,
    )

    print(
        "expected_method_coverage =",
        expected_method_coverage,
    )

    print(
        "global_rank_duplicates =",
        global_rank_duplicates,
    )

    print(
        "unknown_blocks =",
        unknown_blocks,
    )

    print(
        "stored_common5_mismatches =",
        stored_common5_mismatches,
    )

    print(
        "stored_common6_mismatches =",
        stored_common6_mismatches,
    )

    print(
        "common5_float32 =",
        int(
            common5_counts[0]
        ),
    )

    print(
        "common5_float64 =",
        int(
            common5_counts[1]
        ),
    )

    print(
        "common6_float32 =",
        int(
            common6_counts[0]
        ),
    )

    print(
        "common6_float64 =",
        int(
            common6_counts[1]
        ),
    )


    reconstruction_gate = (
        len(block_ids) == NB
        and total_partial_matrices
            == NB * NS
        and rank_coverage_count
            == NB * NS
        and method_coverage_count
            == expected_method_coverage
        and global_rank_duplicates
            == 0
        and unknown_blocks
            == 0
        and stored_common5_mismatches
            == 0
        and stored_common6_mismatches
            == 0
        and int(
            common5_counts[0]
        )
            == int(
                common5_status[0]
            )
        and int(
            common5_counts[1]
        )
            == int(
                common5_status[1]
            )
        and int(
            common6_counts[0]
        )
            == int(
                common6_status[0]
            )
        and int(
            common6_counts[1]
        )
            == int(
                common6_status[1]
            )
    )


    print(
        "reconstruction_gate =",
        reconstruction_gate,
    )


    # ========================================================
    # 3. LAPACK FAILURE OVERLAP WITH THE FIVE-METHOD MASK
    # ========================================================

    print()
    print(
        "===== LAPACK FAILURE OVERLAP ====="
    )

    overlap_rows = []

    for di, dtype_name in enumerate(
        DTYPES
    ):

        lapack_success = (
            success[
                :,
                :,
                di,
                5,
            ].astype(bool)
        )

        lapack_failure = (
            rank_coverage.astype(bool)
            & (~lapack_success)
        )

        common5 = (
            reconstructed_common5[
                :,
                :,
                di,
            ]
        )

        inside_common5 = (
            common5
            & lapack_failure
        )

        outside_common5 = (
            lapack_failure
            & (~common5)
        )

        total_fail = int(
            np.sum(
                lapack_failure
            )
        )

        inside = int(
            np.sum(
                inside_common5
            )
        )

        outside = int(
            np.sum(
                outside_common5
            )
        )

        print()
        print(
            "dtype =",
            dtype_name,
        )

        print(
            "lapack_failures_total =",
            total_fail,
        )

        print(
            "lapack_failures_inside_common5 =",
            inside,
        )

        print(
            "lapack_failures_outside_common5 =",
            outside,
        )

        print(
            "fraction_lapack_failures_inside_common5 =",
            (
                inside
                / total_fail
                if total_fail
                else 0.0
            ),
        )

        print(
            "common5_count =",
            int(
                common5_counts[
                    di
                ]
            ),
        )

        print(
            "common6_count =",
            int(
                common6_counts[
                    di
                ]
            ),
        )

        print(
            "S6_retention_of_S5 =",
            (
                float(
                    common6_counts[
                        di
                    ]
                )
                / float(
                    common5_counts[
                        di
                    ]
                )
                if common5_counts[
                    di
                ]
                else math.nan
            ),
        )

        overlap_rows.append(
            {
                "dtype_name":
                    dtype_name,

                "lapack_failures_total":
                    total_fail,

                "lapack_failures_inside_common5":
                    inside,

                "lapack_failures_outside_common5":
                    outside,

                "common5_count":
                    int(
                        common5_counts[
                            di
                        ]
                    ),

                "common6_count":
                    int(
                        common6_counts[
                            di
                        ]
                    ),

                "S6_retention_of_S5":
                    (
                        float(
                            common6_counts[
                                di
                            ]
                        )
                        / float(
                            common5_counts[
                                di
                            ]
                        )
                        if common5_counts[
                            di
                        ]
                        else math.nan
                    ),
            }
        )


    overlap_path = (
        OUT
        / "six_method_failure_overlap.csv"
    )

    with overlap_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as h:

        writer = csv.DictWriter(
            h,
            fieldnames=list(
                overlap_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            overlap_rows
        )


    # ========================================================
    # 4. PER-BLOCK COMMON-SUCCESS QUANTILES
    # ========================================================

    block_rows = []

    for bp in range(NB):

        for di, dtype_name in enumerate(
            DTYPES
        ):

            mask5 = (
                reconstructed_common5[
                    bp,
                    :,
                    di,
                ]
            )

            mask6 = (
                reconstructed_common6[
                    bp,
                    :,
                    di,
                ]
            )

            for mi, method in enumerate(
                METHODS
            ):

                masks = []

                if mi < 5:
                    masks.append(
                        (
                            "S5",
                            mask5,
                        )
                    )

                masks.append(
                    (
                        "S6",
                        mask6,
                    )
                )

                for sample_name, mask in masks:

                    a2 = r2u[
                        bp,
                        :,
                        di,
                        mi,
                    ]

                    ai = rinfu[
                        bp,
                        :,
                        di,
                        mi,
                    ]

                    valid2 = (
                        mask
                        & np.isfinite(
                            a2
                        )
                    )

                    validi = (
                        mask
                        & np.isfinite(
                            ai
                        )
                    )

                    row = {
                        "sample":
                            sample_name,

                        "block_index":
                            block_ids[
                                bp
                            ],

                        "cell_id":
                            block_cell[
                                bp
                            ],

                        "family":
                            block_family[
                                bp
                            ],

                        "n":
                            int(
                                block_n[
                                    bp
                                ]
                            ),

                        "dtype_name":
                            dtype_name,

                        "method":
                            method,

                        "common_n":
                            int(
                                np.sum(
                                    mask
                                )
                            ),

                        "finite_r2_n":
                            int(
                                np.sum(
                                    valid2
                                )
                            ),

                        "finite_rinf_n":
                            int(
                                np.sum(
                                    validi
                                )
                            ),

                        "r2u_q50":
                            finite_quantile(
                                a2[
                                    valid2
                                ],
                                0.50,
                            ),

                        "r2u_q95":
                            finite_quantile(
                                a2[
                                    valid2
                                ],
                                0.95,
                            ),

                        "rinfu_q50":
                            finite_quantile(
                                ai[
                                    validi
                                ],
                                0.50,
                            ),

                        "rinfu_q95":
                            finite_quantile(
                                ai[
                                    validi
                                ],
                                0.95,
                            ),
                    }

                    block_rows.append(
                        row
                    )


    block_path = (
        OUT
        / "six_method_block_quantiles.csv"
    )

    with block_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as h:

        fields = [
            "sample",
            "block_index",
            "cell_id",
            "family",
            "n",
            "dtype_name",
            "method",
            "common_n",
            "finite_r2_n",
            "finite_rinf_n",
            "r2u_q50",
            "r2u_q95",
            "rinfu_q50",
            "rinfu_q95",
        ]

        writer = csv.DictWriter(
            h,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            block_rows
        )


    # ========================================================
    # 5. TABLE-2 STYLE MEDIANS ACROSS THE 832 BLOCKS
    # ========================================================

    table_rows = []

    for sample_name in (
        "S5",
        "S6",
    ):

        methods_here = (
            CASTILLO_METHODS
            if sample_name == "S5"
            else METHODS
        )

        for dtype_name in DTYPES:

            for method in methods_here:

                rr = [
                    x
                    for x in block_rows
                    if (
                        x["sample"]
                            == sample_name
                        and x[
                            "dtype_name"
                        ]
                            == dtype_name
                        and x[
                            "method"
                        ]
                            == method
                    )
                ]

                common_ns = np.asarray(
                    [
                        x[
                            "common_n"
                        ]
                        for x in rr
                    ],
                    dtype=np.int64,
                )

                out = {
                    "sample":
                        sample_name,

                    "method":
                        method,

                    "dtype_name":
                        dtype_name,

                    "blocks":
                        len(rr),

                    "common_n_min":
                        int(
                            np.min(
                                common_ns
                            )
                        )
                        if common_ns.size
                        else 0,

                    "common_n_q50":
                        finite_median(
                            common_ns
                        ),

                    "common_n_max":
                        int(
                            np.max(
                                common_ns
                            )
                        )
                        if common_ns.size
                        else 0,

                    "median_block_q50_rR2_over_u":
                        finite_median(
                            [
                                x[
                                    "r2u_q50"
                                ]
                                for x in rr
                            ]
                        ),

                    "median_block_q95_rR2_over_u":
                        finite_median(
                            [
                                x[
                                    "r2u_q95"
                                ]
                                for x in rr
                            ]
                        ),

                    "median_block_q50_rRinf_over_u":
                        finite_median(
                            [
                                x[
                                    "rinfu_q50"
                                ]
                                for x in rr
                            ]
                        ),

                    "median_block_q95_rRinf_over_u":
                        finite_median(
                            [
                                x[
                                    "rinfu_q95"
                                ]
                                for x in rr
                            ]
                        ),
                }

                table_rows.append(
                    out
                )


    table_path = (
        OUT
        / "six_method_table2_style.csv"
    )

    with table_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as h:

        writer = csv.DictWriter(
            h,
            fieldnames=list(
                table_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            table_rows
        )


    print()
    print(
        "============================================================"
    )
    print(
        "TABLE-2 STYLE -- ORIGINAL FIVE-METHOD MASK S5"
    )
    print(
        "============================================================"
    )

    for row in table_rows:

        if row[
            "sample"
        ] != "S5":
            continue

        print(
            row["method"],
            row["dtype_name"],
            "q50=",
            row[
                "median_block_q50_rR2_over_u"
            ],
            "q95=",
            row[
                "median_block_q95_rR2_over_u"
            ],
            "common_n[min,med,max]=",
            (
                row[
                    "common_n_min"
                ],
                row[
                    "common_n_q50"
                ],
                row[
                    "common_n_max"
                ],
            ),
        )


    print()
    print(
        "============================================================"
    )
    print(
        "TABLE-2 STYLE -- STRICT SIX-METHOD MASK S6"
    )
    print(
        "============================================================"
    )

    for row in table_rows:

        if row[
            "sample"
        ] != "S6":
            continue

        print(
            row["method"],
            row["dtype_name"],
            "q50=",
            row[
                "median_block_q50_rR2_over_u"
            ],
            "q95=",
            row[
                "median_block_q95_rR2_over_u"
            ],
            "common_n[min,med,max]=",
            (
                row[
                    "common_n_min"
                ],
                row[
                    "common_n_q50"
                ],
                row[
                    "common_n_max"
                ],
            ),
        )


    # ========================================================
    # 6. STRICTLY PAIRED METHOD/LAPACK RATIOS ON S6
    # ========================================================

    ratio_rows = []

    summary_ratio_rows = []

    for di, dtype_name in enumerate(
        DTYPES
    ):

        for mi, method in enumerate(
            CASTILLO_METHODS
        ):

            block_median_logs = []
            block_fractions = []
            block_q50_better = []
            block_q95_better = []

            pooled_better = 0
            pooled_n = 0

            for bp in range(NB):

                mask = (
                    reconstructed_common6[
                        bp,
                        :,
                        di,
                    ]
                )

                a = r2u[
                    bp,
                    :,
                    di,
                    mi,
                ]

                b = r2u[
                    bp,
                    :,
                    di,
                    5,
                ]

                valid = (
                    mask
                    & np.isfinite(a)
                    & np.isfinite(b)
                    & (a > 0.0)
                    & (b > 0.0)
                )

                aa = a[
                    valid
                ]

                bb = b[
                    valid
                ]

                if aa.size:
                    logs = np.log10(
                        aa / bb
                    )

                    medlog = float(
                        np.median(
                            logs
                        )
                    )

                    frac = float(
                        np.mean(
                            aa < bb
                        )
                    )

                    pooled_better += int(
                        np.sum(
                            aa < bb
                        )
                    )

                    pooled_n += int(
                        aa.size
                    )

                else:
                    medlog = math.nan
                    frac = math.nan

                q50_a = finite_quantile(
                    aa,
                    0.50,
                )

                q50_b = finite_quantile(
                    bb,
                    0.50,
                )

                q95_a = finite_quantile(
                    aa,
                    0.95,
                )

                q95_b = finite_quantile(
                    bb,
                    0.95,
                )

                if math.isfinite(
                    medlog
                ):
                    block_median_logs.append(
                        medlog
                    )

                if math.isfinite(
                    frac
                ):
                    block_fractions.append(
                        frac
                    )

                if (
                    math.isfinite(
                        q50_a
                    )
                    and math.isfinite(
                        q50_b
                    )
                ):
                    block_q50_better.append(
                        q50_a
                        < q50_b
                    )

                if (
                    math.isfinite(
                        q95_a
                    )
                    and math.isfinite(
                        q95_b
                    )
                ):
                    block_q95_better.append(
                        q95_a
                        < q95_b
                    )

                ratio_rows.append(
                    {
                        "block_index":
                            block_ids[
                                bp
                            ],

                        "cell_id":
                            block_cell[
                                bp
                            ],

                        "family":
                            block_family[
                                bp
                            ],

                        "n":
                            int(
                                block_n[
                                    bp
                                ]
                            ),

                        "dtype_name":
                            dtype_name,

                        "method":
                            method,

                        "paired_n":
                            int(
                                aa.size
                            ),

                        "median_log10_method_over_lapack":
                            medlog,

                        "fraction_method_lt_lapack":
                            frac,

                        "method_q50":
                            q50_a,

                        "lapack_q50":
                            q50_b,

                        "method_q95":
                            q95_a,

                        "lapack_q95":
                            q95_b,
                    }
                )


            summary_ratio_rows.append(
                {
                    "dtype_name":
                        dtype_name,

                    "method":
                        method,

                    "blocks":
                        NB,

                    "paired_n_total":
                        pooled_n,

                    "median_block_median_log10_method_over_lapack":
                        finite_median(
                            block_median_logs
                        ),

                    "median_block_fraction_method_lt_lapack":
                        finite_median(
                            block_fractions
                        ),

                    "pooled_fraction_method_lt_lapack":
                        (
                            pooled_better
                            / pooled_n
                            if pooled_n
                            else math.nan
                        ),

                    "fraction_blocks_q50_method_lt_lapack":
                        (
                            float(
                                np.mean(
                                    block_q50_better
                                )
                            )
                            if block_q50_better
                            else math.nan
                        ),

                    "fraction_blocks_q95_method_lt_lapack":
                        (
                            float(
                                np.mean(
                                    block_q95_better
                                )
                            )
                            if block_q95_better
                            else math.nan
                        ),
                }
            )


    ratio_path = (
        OUT
        / "six_method_vs_lapack_by_block.csv"
    )

    with ratio_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as h:

        writer = csv.DictWriter(
            h,
            fieldnames=list(
                ratio_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            ratio_rows
        )


    ratio_summary_path = (
        OUT
        / "six_method_vs_lapack_summary.csv"
    )

    with ratio_summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as h:

        writer = csv.DictWriter(
            h,
            fieldnames=list(
                summary_ratio_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_ratio_rows
        )


    print()
    print(
        "============================================================"
    )
    print(
        "STRICT PAIRED CASTILLO / LAPACK RATIOS ON S6"
    )
    print(
        "============================================================"
    )

    for row in summary_ratio_rows:

        print()
        print(
            row[
                "method"
            ],
            row[
                "dtype_name"
            ],
        )

        print(
            " paired_n_total =",
            row[
                "paired_n_total"
            ],
        )

        print(
            " median_block_median_log10_method_over_lapack =",
            row[
                "median_block_median_log10_method_over_lapack"
            ],
        )

        print(
            " pooled_fraction_method_lt_lapack =",
            row[
                "pooled_fraction_method_lt_lapack"
            ],
        )

        print(
            " fraction_blocks_q50_method_lt_lapack =",
            row[
                "fraction_blocks_q50_method_lt_lapack"
            ],
        )

        print(
            " fraction_blocks_q95_method_lt_lapack =",
            row[
                "fraction_blocks_q95_method_lt_lapack"
            ],
        )


    # ========================================================
    # 7. IMPORTANT INTERNAL CONTROL:
    #    BINARY64 S6 MUST EQUAL S5 BECAUSE LAPACK HAD ZERO FAILURES
    # ========================================================

    s5_f64 = {
        row["method"]: row
        for row in table_rows
        if (
            row["sample"]
                == "S5"
            and row[
                "dtype_name"
            ]
                == "float64"
        )
    }

    s6_f64 = {
        row["method"]: row
        for row in table_rows
        if (
            row["sample"]
                == "S6"
            and row[
                "dtype_name"
            ]
                == "float64"
            and row[
                "method"
            ]
                in CASTILLO_METHODS
        )
    }

    binary64_mask_identity = (
        int(
            common5_counts[
                1
            ]
        )
        == int(
            common6_counts[
                1
            ]
        )
    )

    binary64_table_identity = True

    for method in CASTILLO_METHODS:

        a = s5_f64[
            method
        ]

        b = s6_f64[
            method
        ]

        for key in (
            "median_block_q50_rR2_over_u",
            "median_block_q95_rR2_over_u",
            "median_block_q50_rRinf_over_u",
            "median_block_q95_rRinf_over_u",
        ):

            av = float(
                a[key]
            )

            bv = float(
                b[key]
            )

            if not (
                av == bv
                or (
                    math.isnan(av)
                    and math.isnan(bv)
                )
            ):
                binary64_table_identity = False


    print()
    print(
        "===== BINARY64 MASK CONTROL ====="
    )

    print(
        "binary64_mask_identity =",
        binary64_mask_identity,
    )

    print(
        "binary64_table_identity =",
        binary64_table_identity,
    )


    # ========================================================
    # 8. FINAL SUMMARY
    # ========================================================

    final_gate = (
        reconstruction_gate
        and binary64_mask_identity
        and binary64_table_identity
        and len(
            table_rows
        )
            == 22
        and len(
            summary_ratio_rows
        )
            == 10
    )

    final = {
        "status":
            (
                "SIX_METHOD_PAIRED_AUDIT_COMPLETE"
                if final_gate
                else "SIX_METHOD_PAIRED_AUDIT_FAIL"
            ),

        "task_status_gate":
            status_gate,

        "reconstruction_gate":
            reconstruction_gate,

        "binary64_mask_identity":
            binary64_mask_identity,

        "binary64_table_identity":
            binary64_table_identity,

        "selected_matrices":
            NB * NS,

        "blocks":
            NB,

        "sample_size":
            NS,

        "common5": {
            "float32":
                int(
                    common5_counts[
                        0
                    ]
                ),

            "float64":
                int(
                    common5_counts[
                        1
                    ]
                ),
        },

        "common6": {
            "float32":
                int(
                    common6_counts[
                        0
                    ]
                ),

            "float64":
                int(
                    common6_counts[
                        1
                    ]
                ),
        },

        "outputs": [
            str(
                overlap_path
            ),
            str(
                block_path
            ),
            str(
                table_path
            ),
            str(
                ratio_path
            ),
            str(
                ratio_summary_path
            ),
        ],
    }

    final_path = (
        OUT
        / "six_method_paired_summary.json"
    )

    final_path.write_text(
        json.dumps(
            final,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print(
        "============================================================"
    )
    print(
        "FINAL SIX-METHOD PAIRED GATE"
    )
    print(
        "============================================================"
    )

    print(
        "final_gate =",
        final_gate,
    )

    print(
        "summary =",
        final_path,
    )

    if final_gate:
        print(
            "CASTILLO_SIX_METHOD_PAIRED_AUDIT=PASS"
        )
    else:
        print(
            "CASTILLO_SIX_METHOD_PAIRED_AUDIT=FAIL"
        )


else:

    print()
    print(
        "============================================================"
    )
    print(
        "FINAL SIX-METHOD PAIRED GATE"
    )
    print(
        "============================================================"
    )

    print(
        "final_gate = False"
    )

    print(
        "PAIR_FINALIZATION_NOT_STARTED_STATUS_GATE_FAILED"
    )

    print(
        "CASTILLO_SIX_METHOD_PAIRED_AUDIT=FAIL"
    )
