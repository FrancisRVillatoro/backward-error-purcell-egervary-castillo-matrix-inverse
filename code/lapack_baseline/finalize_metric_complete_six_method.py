#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np


BASE = Path.home() / "castillo_lapack_baseline"

PARTIAL = (
    BASE
    / "results"
    / "paired"
    / "partial"
)

OUT = (
    BASE
    / "results"
    / "paired"
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

DTYPES = [
    "float32",
    "float64",
]

NB = 832
NS = 2000
ND = 2
NM = 6


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
    summary["selected_blocks"],
    key=lambda x:
        int(x["block_index"]),
)

block_ids = [
    int(x["block_index"])
    for x in selected_blocks
]

block_pos = {
    b: i
    for i, b in enumerate(block_ids)
}

block_family = [
    str(x["family"])
    for x in selected_blocks
]

block_cell = [
    str(x["cell_id"])
    for x in selected_blocks
]

block_n = np.asarray(
    [
        int(x["n"])
        for x in selected_blocks
    ],
    dtype=np.int32,
)


rank_coverage = np.zeros(
    (NB, NS),
    dtype=np.uint8,
)

success = np.zeros(
    (NB, NS, ND, NM),
    dtype=np.uint8,
)

r2u = np.full(
    (NB, NS, ND, NM),
    np.nan,
    dtype=np.float64,
)


duplicates = 0
unknown_blocks = 0
partial_matrices = 0


print(
    "============================================================"
)
print(
    "STRICT SIX-METHOD FINITE-rR2 FINALIZER"
)
print(
    "============================================================"
)


paths = sorted(
    PARTIAL.glob(
        "paired_????.npz"
    )
)

print(
    "partial_files =",
    len(paths),
)


for number, path in enumerate(
    paths,
    1,
):

    if (
        number == 1
        or number % 100 == 0
        or number == len(paths)
    ):
        print(
            "loading",
            number,
            "of",
            len(paths),
            path.name,
        )

    z = np.load(
        str(path),
        allow_pickle=False,
    )

    block = np.asarray(
        z["block"],
        dtype=np.int32,
    )

    rank = np.asarray(
        z["rank"],
        dtype=np.int32,
    )

    cs = np.asarray(
        z["castillo_success"],
        dtype=np.uint8,
    )

    cr = np.asarray(
        z["castillo_r2u"],
        dtype=np.float64,
    )

    ls = np.asarray(
        z["lapack_success"],
        dtype=np.uint8,
    )

    lr = np.asarray(
        z["lapack_r2u"],
        dtype=np.float64,
    )

    partial_matrices += int(
        block.size
    )

    for ub in np.unique(block):

        ub = int(ub)

        bp = block_pos.get(ub)

        mask = (
            block == ub
        )

        if bp is None:
            unknown_blocks += int(
                np.sum(mask)
            )
            continue

        rr = rank[
            mask
        ].astype(
            np.int64
        )

        if np.any(
            (rr < 0)
            | (rr >= NS)
        ):
            unknown_blocks += int(
                rr.size
            )
            continue

        duplicates += int(
            np.sum(
                rank_coverage[
                    bp,
                    rr,
                ]
                != 0
            )
        )

        rank_coverage[
            bp,
            rr,
        ] = 1

        success[
            bp,
            rr,
            :,
            :5,
        ] = cs[
            mask,
            :,
            :,
        ]

        r2u[
            bp,
            rr,
            :,
            :5,
        ] = cr[
            mask,
            :,
            :,
        ]

        success[
            bp,
            rr,
            :,
            5,
        ] = ls[
            mask,
            :,
        ]

        r2u[
            bp,
            rr,
            :,
            5,
        ] = lr[
            mask,
            :,
        ]


coverage_count = int(
    np.sum(
        rank_coverage
    )
)


common6_success = np.all(
    success.astype(bool),
    axis=3,
)

finite_nonnegative = (
    np.isfinite(r2u)
    & (r2u >= 0.0)
)

common6_metric = (
    common6_success
    & np.all(
        finite_nonnegative,
        axis=3,
    )
)


success_counts = np.sum(
    common6_success,
    axis=(0, 1),
    dtype=np.int64,
)

metric_counts = np.sum(
    common6_metric,
    axis=(0, 1),
    dtype=np.int64,
)


print()
print(
    "===== MASK ACCOUNTING ====="
)

for di, dtype_name in enumerate(
    DTYPES
):

    print()
    print(
        "dtype =",
        dtype_name,
    )

    print(
        "S6_success =",
        int(
            success_counts[di]
        ),
    )

    print(
        "S6_metric =",
        int(
            metric_counts[di]
        ),
    )

    print(
        "removed_nonfinite_metric =",
        int(
            success_counts[di]
            - metric_counts[di]
        ),
    )

    print(
        "retention =",
        float(
            metric_counts[di]
            / success_counts[di]
        ),
    )


# ------------------------------------------------------------
# Identify every removed matrix/method.
# ------------------------------------------------------------

bad_rows = []

for bp in range(NB):

    for rank in range(NS):

        for di, dtype_name in enumerate(
            DTYPES
        ):

            if not common6_success[
                bp,
                rank,
                di,
            ]:
                continue

            if common6_metric[
                bp,
                rank,
                di,
            ]:
                continue

            bad_methods = [
                METHODS[mi]
                for mi in range(NM)
                if not finite_nonnegative[
                    bp,
                    rank,
                    di,
                    mi,
                ]
            ]

            bad_rows.append(
                {
                    "block_index":
                        block_ids[bp],

                    "cell_id":
                        block_cell[bp],

                    "family":
                        block_family[bp],

                    "n":
                        int(
                            block_n[bp]
                        ),

                    "rank":
                        rank,

                    "dtype_name":
                        dtype_name,

                    "bad_methods":
                        ";".join(
                            bad_methods
                        ),
                }
            )


bad_path = (
    OUT
    / "six_method_metric_invalid_records.csv"
)

with bad_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    fields = [
        "block_index",
        "cell_id",
        "family",
        "n",
        "rank",
        "dtype_name",
        "bad_methods",
    ]

    writer = csv.DictWriter(
        h,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        bad_rows
    )


print()
print(
    "metric_invalid_records =",
    len(bad_rows),
)


# ------------------------------------------------------------
# Per-block quantiles on exactly S6_metric.
# ------------------------------------------------------------

def q(x, p):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.size == 0:
        return math.nan

    return float(
        np.quantile(
            x,
            p,
        )
    )


block_rows = []

for bp in range(NB):

    for di, dtype_name in enumerate(
        DTYPES
    ):

        mask = common6_metric[
            bp,
            :,
            di,
        ]

        common_n = int(
            np.sum(mask)
        )

        for mi, method in enumerate(
            METHODS
        ):

            x = r2u[
                bp,
                :,
                di,
                mi,
            ][mask]

            block_rows.append(
                {
                    "block_index":
                        block_ids[bp],

                    "cell_id":
                        block_cell[bp],

                    "family":
                        block_family[bp],

                    "n":
                        int(
                            block_n[bp]
                        ),

                    "dtype_name":
                        dtype_name,

                    "method":
                        method,

                    "common_n":
                        common_n,

                    "r2u_q50":
                        q(
                            x,
                            0.50,
                        ),

                    "r2u_q95":
                        q(
                            x,
                            0.95,
                        ),
                }
            )


block_path = (
    OUT
    / "six_method_metric_complete_block_quantiles.csv"
)

with block_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    writer = csv.DictWriter(
        h,
        fieldnames=list(
            block_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        block_rows
    )


# ------------------------------------------------------------
# Table-2-style medians across the 832 blocks.
# ------------------------------------------------------------

table_rows = []

for di, dtype_name in enumerate(
    DTYPES
):

    common_n_by_block = np.sum(
        common6_metric[
            :,
            :,
            di,
        ],
        axis=1,
        dtype=np.int64,
    )

    for method in METHODS:

        rows = [
            x
            for x in block_rows
            if (
                x["dtype_name"]
                == dtype_name
                and x["method"]
                == method
            )
        ]

        med_q50 = float(
            np.median(
                [
                    x["r2u_q50"]
                    for x in rows
                ]
            )
        )

        med_q95 = float(
            np.median(
                [
                    x["r2u_q95"]
                    for x in rows
                ]
            )
        )

        table_rows.append(
            {
                "dtype_name":
                    dtype_name,

                "method":
                    method,

                "blocks":
                    NB,

                "common_n_total":
                    int(
                        metric_counts[di]
                    ),

                "common_n_min":
                    int(
                        np.min(
                            common_n_by_block
                        )
                    ),

                "common_n_median":
                    float(
                        np.median(
                            common_n_by_block
                        )
                    ),

                "common_n_max":
                    int(
                        np.max(
                            common_n_by_block
                        )
                    ),

                "median_block_q50_rR2_over_u":
                    med_q50,

                "median_block_q95_rR2_over_u":
                    med_q95,
            }
        )


table_path = (
    OUT
    / "six_method_metric_complete_table2_style.csv"
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
    "TABLE-2 STYLE ON STRICT FINITE-METRIC MASK"
)
print(
    "============================================================"
)

for row in table_rows:

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
        "common_n_total=",
        row[
            "common_n_total"
        ],
        "block_n[min,med,max]=",
        (
            row[
                "common_n_min"
            ],
            row[
                "common_n_median"
            ],
            row[
                "common_n_max"
            ],
        ),
    )


# ------------------------------------------------------------
# Strict paired Castillo-vs-LAPACK comparisons.
# Zeros are INCLUDED in <, =, > accounting.
# log ratio is reported only for positive/positive pairs.
# ------------------------------------------------------------

ratio_rows = []

for di, dtype_name in enumerate(
    DTYPES
):

    mask = common6_metric[
        :,
        :,
        di,
    ]

    lapack = r2u[
        :,
        :,
        di,
        5,
    ][mask]

    for mi, method in enumerate(
        METHODS[:5]
    ):

        castillo = r2u[
            :,
            :,
            di,
            mi,
        ][mask]

        lt = int(
            np.sum(
                castillo < lapack
            )
        )

        eq = int(
            np.sum(
                castillo == lapack
            )
        )

        gt = int(
            np.sum(
                castillo > lapack
            )
        )

        n = int(
            castillo.size
        )

        pp = (
            (castillo > 0.0)
            & (lapack > 0.0)
        )

        if np.any(pp):

            pooled_median_log10 = float(
                np.median(
                    np.log10(
                        castillo[pp]
                        / lapack[pp]
                    )
                )
            )

        else:
            pooled_median_log10 = math.nan


        block_q50_better = []
        block_q95_better = []

        for bp in range(NB):

            bm = common6_metric[
                bp,
                :,
                di,
            ]

            c = r2u[
                bp,
                :,
                di,
                mi,
            ][bm]

            l = r2u[
                bp,
                :,
                di,
                5,
            ][bm]

            if c.size == 0:
                continue

            block_q50_better.append(
                np.quantile(
                    c,
                    0.50,
                )
                <
                np.quantile(
                    l,
                    0.50,
                )
            )

            block_q95_better.append(
                np.quantile(
                    c,
                    0.95,
                )
                <
                np.quantile(
                    l,
                    0.95,
                )
            )


        ratio_rows.append(
            {
                "dtype_name":
                    dtype_name,

                "method":
                    method,

                "paired_n":
                    n,

                "method_lt_lapack":
                    lt,

                "method_eq_lapack":
                    eq,

                "method_gt_lapack":
                    gt,

                "fraction_method_lt_lapack":
                    (
                        lt / n
                        if n
                        else math.nan
                    ),

                "fraction_method_le_lapack":
                    (
                        (lt + eq) / n
                        if n
                        else math.nan
                    ),

                "pooled_median_log10_method_over_lapack_positive_pairs":
                    pooled_median_log10,

                "fraction_blocks_q50_method_lt_lapack":
                    float(
                        np.mean(
                            block_q50_better
                        )
                    ),

                "fraction_blocks_q95_method_lt_lapack":
                    float(
                        np.mean(
                            block_q95_better
                        )
                    ),
            }
        )


ratio_path = (
    OUT
    / "six_method_metric_complete_vs_lapack.csv"
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


print()
print(
    "============================================================"
)
print(
    "STRICT PAIRED CASTILLO vs LAPACK"
)
print(
    "============================================================"
)

for row in ratio_rows:

    print()
    print(
        row["method"],
        row["dtype_name"],
    )

    print(
        " paired_n =",
        row["paired_n"],
    )

    print(
        " fraction_method_lt_lapack =",
        row[
            "fraction_method_lt_lapack"
        ],
    )

    print(
        " fraction_method_le_lapack =",
        row[
            "fraction_method_le_lapack"
        ],
    )

    print(
        " pooled_median_log10_method_over_lapack_positive_pairs =",
        row[
            "pooled_median_log10_method_over_lapack_positive_pairs"
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


gate = (
    len(paths) == 999
    and len(block_ids) == 832
    and partial_matrices == 1664000
    and coverage_count == 1664000
    and duplicates == 0
    and unknown_blocks == 0
    and int(
        success_counts[0]
    ) == 1629883
    and int(
        success_counts[1]
    ) == 1663999
    and int(
        metric_counts[0]
    ) == 1629883
    and int(
        metric_counts[1]
    ) == 1663993
    and len(bad_rows) == 6
)


summary_out = {
    "gate":
        gate,

    "selected_matrices":
        1664000,

    "S6_success": {
        "float32":
            int(
                success_counts[0]
            ),

        "float64":
            int(
                success_counts[1]
            ),
    },

    "S6_metric_complete": {
        "float32":
            int(
                metric_counts[0]
            ),

        "float64":
            int(
                metric_counts[1]
            ),
    },

    "metric_invalid_success_records":
        len(bad_rows),

    "outputs": [
        str(
            bad_path
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
    ],
}

summary_out_path = (
    OUT
    / "six_method_metric_complete_summary.json"
)

summary_out_path.write_text(
    json.dumps(
        summary_out,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "============================================================"
)
print(
    "FINAL METRIC-COMPLETE GATE"
)
print(
    "============================================================"
)

print(
    "partial_matrices =",
    partial_matrices,
)

print(
    "coverage_count =",
    coverage_count,
)

print(
    "duplicates =",
    duplicates,
)

print(
    "unknown_blocks =",
    unknown_blocks,
)

print(
    "metric_invalid_success_records =",
    len(bad_rows),
)

print(
    "final_gate =",
    gate,
)

print(
    "summary =",
    summary_out_path,
)

if gate:
    print(
        "CASTILLO_SIX_METHOD_METRIC_COMPLETE_AUDIT=PASS"
    )
else:
    print(
        "CASTILLO_SIX_METHOD_METRIC_COMPLETE_AUDIT=FAIL"
    )
