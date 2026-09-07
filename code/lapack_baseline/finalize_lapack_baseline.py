#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict, Counter
import csv
import gzip
import json
import math
import statistics

import numpy as np


BASE = Path.home() / "castillo_lapack_baseline"

DATADIR = (
    BASE
    / "results"
    / "campaign"
    / "data"
)

STATUSDIR = (
    BASE
    / "results"
    / "campaign"
    / "status"
)

OUTDIR = (
    BASE
    / "results"
    / "final"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


QUANTILES = (
    (0.50, "q50"),
    (0.90, "q90"),
    (0.95, "q95"),
    (0.99, "q99"),
)

METRICS = (
    "rR2_over_u",
    "rRinf_over_u",
    "rL2_over_u",
    "rLinf_over_u",
    "inverse_seconds",
    "metrics_seconds",
    "total_seconds",
)


def as_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def as_int(x):
    try:
        return int(x)
    except Exception:
        return -1


def as_bool(x):
    return (
        str(x)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def qvalues(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    result = {
        "finite_count":
            int(x.size)
    }

    if x.size == 0:
        for _, name in QUANTILES:
            result[name] = math.nan
        return result

    qq = np.quantile(
        x,
        [
            p
            for p, _ in QUANTILES
        ],
    )

    for i, (_, name) in enumerate(
        QUANTILES
    ):
        result[name] = float(
            qq[i]
        )

    return result


def median_finite(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return math.nan

    return float(
        np.median(x)
    )


print(
    "============================================================"
)
print(
    "CASTILLO LAPACK BASELINE -- FINAL AGGREGATION"
)
print(
    "============================================================"
)

# ----------------------------------------------------------------
# Authoritative completion status
# ----------------------------------------------------------------

status_paths = sorted(
    STATUSDIR.glob(
        "task_????.json"
    )
)

data_paths = sorted(
    DATADIR.glob(
        "lapack_????.csv.gz"
    )
)

statuses_ok = True

for p in status_paths:
    try:
        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        if (
            obj.get("status")
            != "TASK_OK"
        ):
            statuses_ok = False

    except Exception:
        statuses_ok = False


print()
print(
    "status_files =",
    len(status_paths),
)

print(
    "data_files =",
    len(data_paths),
)

print(
    "statuses_ok =",
    statuses_ok,
)


# ----------------------------------------------------------------
# Streaming aggregation
# ----------------------------------------------------------------

overall = defaultdict(
    lambda: {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "failure_classes": Counter(),
        "metrics": defaultdict(list),
    }
)

by_family = defaultdict(
    lambda: {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "failure_classes": Counter(),
    }
)

by_n = defaultdict(
    lambda: {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "failure_classes": Counter(),
        "metrics": defaultdict(list),
    }
)

block_data = defaultdict(
    lambda: {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "failure_classes": Counter(),
        "metrics": defaultdict(list),
    }
)

seen_matrix_precision = set()

duplicate_records = 0

total_rows = 0

for file_index, path in enumerate(
    data_paths,
    1,
):
    if (
        file_index == 1
        or file_index % 100 == 0
        or file_index == len(data_paths)
    ):
        print(
            "reading_file",
            file_index,
            "of",
            len(data_paths),
            path.name,
        )

    with gzip.open(
        str(path),
        "rt",
        encoding="utf-8",
        newline="",
    ) as h:

        for row in csv.DictReader(h):
            total_rows += 1

            dtype_name = str(
                row["dtype_name"]
            )

            family = str(
                row["family"]
            )

            n = as_int(
                row["n"]
            )

            block = as_int(
                row["block_index"]
            )

            replica = as_int(
                row["replica"]
            )

            rank = as_int(
                row["rank"]
            )

            matrix_id = str(
                row["matrix_id"]
            )

            success = as_bool(
                row["success"]
            )

            failure_class = str(
                row[
                    "failure_class"
                ]
            )

            unique_key = (
                block,
                replica,
                dtype_name,
            )

            if unique_key in seen_matrix_precision:
                duplicate_records += 1
            else:
                seen_matrix_precision.add(
                    unique_key
                )

            key_overall = (
                dtype_name
            )

            key_family = (
                dtype_name,
                family,
            )

            key_n = (
                dtype_name,
                n,
            )

            key_block = (
                dtype_name,
                block,
                family,
                n,
            )

            containers = [
                overall[
                    key_overall
                ],
                by_family[
                    key_family
                ],
                by_n[
                    key_n
                ],
                block_data[
                    key_block
                ],
            ]

            for d in containers:
                d["attempts"] += 1

                if success:
                    d["successes"] += 1
                else:
                    d["failures"] += 1

                    d[
                        "failure_classes"
                    ][
                        failure_class
                    ] += 1

            if success:
                for metric in METRICS:
                    value = as_float(
                        row.get(
                            metric,
                            math.nan,
                        )
                    )

                    if math.isfinite(
                        value
                    ):
                        overall[
                            key_overall
                        ][
                            "metrics"
                        ][metric].append(
                            value
                        )

                        by_n[
                            key_n
                        ][
                            "metrics"
                        ][metric].append(
                            value
                        )

                        block_data[
                            key_block
                        ][
                            "metrics"
                        ][metric].append(
                            value
                        )


print()
print(
    "total_rows =",
    total_rows,
)

print(
    "distinct_matrix_precision =",
    len(
        seen_matrix_precision
    ),
)

print(
    "duplicate_records =",
    duplicate_records,
)


# ----------------------------------------------------------------
# Precision summary
# ----------------------------------------------------------------

precision_path = (
    OUTDIR
    / "lapack_precision_summary.csv"
)

precision_fields = [
    "dtype_name",
    "attempts",
    "successes",
    "failures",
    "failure_rate",
]

for metric in METRICS:
    for _, qname in QUANTILES:
        precision_fields.append(
            metric
            + "_"
            + qname
        )


with precision_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    writer = csv.DictWriter(
        h,
        fieldnames=precision_fields,
    )

    writer.writeheader()

    for dtype_name in (
        "float32",
        "float64",
    ):
        d = overall[
            dtype_name
        ]

        row = {
            "dtype_name":
                dtype_name,
            "attempts":
                d["attempts"],
            "successes":
                d["successes"],
            "failures":
                d["failures"],
            "failure_rate":
                (
                    d["failures"]
                    / d["attempts"]
                    if d["attempts"]
                    else math.nan
                ),
        }

        for metric in METRICS:
            qq = qvalues(
                d[
                    "metrics"
                ][metric]
            )

            for _, qname in QUANTILES:
                row[
                    metric
                    + "_"
                    + qname
                ] = qq[
                    qname
                ]

        writer.writerow(
            row
        )


# ----------------------------------------------------------------
# Failure breakdown by family
# ----------------------------------------------------------------

family_path = (
    OUTDIR
    / "lapack_failure_by_family.csv"
)

with family_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    fields = [
        "dtype_name",
        "family",
        "attempts",
        "successes",
        "failures",
        "failure_rate",
        "singular_getrf",
    ]

    writer = csv.DictWriter(
        h,
        fieldnames=fields,
    )

    writer.writeheader()

    for (
        dtype_name,
        family,
    ), d in sorted(
        by_family.items()
    ):
        writer.writerow(
            {
                "dtype_name":
                    dtype_name,
                "family":
                    family,
                "attempts":
                    d["attempts"],
                "successes":
                    d["successes"],
                "failures":
                    d["failures"],
                "failure_rate":
                    (
                        d["failures"]
                        / d["attempts"]
                        if d["attempts"]
                        else math.nan
                    ),
                "singular_getrf":
                    d[
                        "failure_classes"
                    ].get(
                        "singular_getrf",
                        0,
                    ),
            }
        )


# ----------------------------------------------------------------
# Dimension summary
# ----------------------------------------------------------------

n_path = (
    OUTDIR
    / "lapack_by_dimension.csv"
)

n_fields = [
    "dtype_name",
    "n",
    "attempts",
    "successes",
    "failures",
    "failure_rate",
]

for metric in (
    "rR2_over_u",
    "rRinf_over_u",
):
    for _, qname in QUANTILES:
        n_fields.append(
            metric
            + "_"
            + qname
        )


with n_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    writer = csv.DictWriter(
        h,
        fieldnames=n_fields,
    )

    writer.writeheader()

    for (
        dtype_name,
        n,
    ), d in sorted(
        by_n.items()
    ):
        row = {
            "dtype_name":
                dtype_name,
            "n":
                n,
            "attempts":
                d["attempts"],
            "successes":
                d["successes"],
            "failures":
                d["failures"],
            "failure_rate":
                (
                    d["failures"]
                    / d["attempts"]
                    if d["attempts"]
                    else math.nan
                ),
        }

        for metric in (
            "rR2_over_u",
            "rRinf_over_u",
        ):
            qq = qvalues(
                d[
                    "metrics"
                ][metric]
            )

            for _, qname in QUANTILES:
                row[
                    metric
                    + "_"
                    + qname
                ] = qq[
                    qname
                ]

        writer.writerow(
            row
        )


# ----------------------------------------------------------------
# Per-block quantiles
# ----------------------------------------------------------------

block_path = (
    OUTDIR
    / "lapack_block_quantiles.csv"
)

block_fields = [
    "dtype_name",
    "block_index",
    "family",
    "n",
    "attempts",
    "successes",
    "failures",
    "failure_rate",
]

for metric in (
    "rR2_over_u",
    "rRinf_over_u",
    "rL2_over_u",
    "rLinf_over_u",
):
    for _, qname in QUANTILES:
        block_fields.append(
            metric
            + "_"
            + qname
        )


block_rows = []

with block_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    writer = csv.DictWriter(
        h,
        fieldnames=block_fields,
    )

    writer.writeheader()

    for (
        dtype_name,
        block,
        family,
        n,
    ), d in sorted(
        block_data.items()
    ):
        row = {
            "dtype_name":
                dtype_name,
            "block_index":
                block,
            "family":
                family,
            "n":
                n,
            "attempts":
                d["attempts"],
            "successes":
                d["successes"],
            "failures":
                d["failures"],
            "failure_rate":
                (
                    d["failures"]
                    / d["attempts"]
                    if d["attempts"]
                    else math.nan
                ),
        }

        for metric in (
            "rR2_over_u",
            "rRinf_over_u",
            "rL2_over_u",
            "rLinf_over_u",
        ):
            qq = qvalues(
                d[
                    "metrics"
                ][metric]
            )

            for _, qname in QUANTILES:
                row[
                    metric
                    + "_"
                    + qname
                ] = qq[
                    qname
                ]

        writer.writerow(
            row
        )

        block_rows.append(
            row
        )


# ----------------------------------------------------------------
# Table-2 style summary:
# median across the 832 stochastic blocks
# ----------------------------------------------------------------

table_path = (
    OUTDIR
    / "lapack_table2_style.csv"
)

table_rows = []

with table_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    fields = [
        "method",
        "dtype_name",
        "blocks",
        "attempts",
        "successes",
        "failures",
        "failure_rate",
        "median_block_q50_rR2_over_u",
        "median_block_q95_rR2_over_u",
        "median_block_q50_rRinf_over_u",
        "median_block_q95_rRinf_over_u",
        "median_block_q50_rL2_over_u",
        "median_block_q95_rL2_over_u",
    ]

    writer = csv.DictWriter(
        h,
        fieldnames=fields,
    )

    writer.writeheader()

    for dtype_name in (
        "float32",
        "float64",
    ):
        rr = [
            row
            for row in block_rows
            if row["dtype_name"]
            == dtype_name
        ]

        d = overall[
            dtype_name
        ]

        out = {
            "method":
                "LAPACK_xGETRF_xGETRI",
            "dtype_name":
                dtype_name,
            "blocks":
                len(rr),
            "attempts":
                d["attempts"],
            "successes":
                d["successes"],
            "failures":
                d["failures"],
            "failure_rate":
                (
                    d["failures"]
                    / d["attempts"]
                    if d["attempts"]
                    else math.nan
                ),
            "median_block_q50_rR2_over_u":
                median_finite(
                    [
                        x[
                            "rR2_over_u_q50"
                        ]
                        for x in rr
                    ]
                ),
            "median_block_q95_rR2_over_u":
                median_finite(
                    [
                        x[
                            "rR2_over_u_q95"
                        ]
                        for x in rr
                    ]
                ),
            "median_block_q50_rRinf_over_u":
                median_finite(
                    [
                        x[
                            "rRinf_over_u_q50"
                        ]
                        for x in rr
                    ]
                ),
            "median_block_q95_rRinf_over_u":
                median_finite(
                    [
                        x[
                            "rRinf_over_u_q95"
                        ]
                        for x in rr
                    ]
                ),
            "median_block_q50_rL2_over_u":
                median_finite(
                    [
                        x[
                            "rL2_over_u_q50"
                        ]
                        for x in rr
                    ]
                ),
            "median_block_q95_rL2_over_u":
                median_finite(
                    [
                        x[
                            "rL2_over_u_q95"
                        ]
                        for x in rr
                    ]
                ),
        }

        writer.writerow(
            out
        )

        table_rows.append(
            out
        )


# ----------------------------------------------------------------
# Left/right asymmetry on block medians
# ----------------------------------------------------------------

asym_path = (
    OUTDIR
    / "lapack_left_right_asymmetry.csv"
)

with asym_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as h:

    fields = [
        "dtype_name",
        "blocks",
        "median_log10_q50_right_over_left_2",
        "median_log10_q95_right_over_left_2",
        "fraction_q50_right_lt_left_2",
        "fraction_q95_right_lt_left_2",
    ]

    writer = csv.DictWriter(
        h,
        fieldnames=fields,
    )

    writer.writeheader()

    for dtype_name in (
        "float32",
        "float64",
    ):
        rr = [
            row
            for row in block_rows
            if row["dtype_name"]
            == dtype_name
        ]

        lr50 = []
        lr95 = []

        right_lt_left_50 = []
        right_lt_left_95 = []

        for row in rr:
            r50 = as_float(
                row[
                    "rR2_over_u_q50"
                ]
            )

            l50 = as_float(
                row[
                    "rL2_over_u_q50"
                ]
            )

            r95 = as_float(
                row[
                    "rR2_over_u_q95"
                ]
            )

            l95 = as_float(
                row[
                    "rL2_over_u_q95"
                ]
            )

            if (
                math.isfinite(r50)
                and math.isfinite(l50)
                and r50 > 0.0
                and l50 > 0.0
            ):
                lr50.append(
                    math.log10(
                        r50 / l50
                    )
                )

                right_lt_left_50.append(
                    r50 < l50
                )

            if (
                math.isfinite(r95)
                and math.isfinite(l95)
                and r95 > 0.0
                and l95 > 0.0
            ):
                lr95.append(
                    math.log10(
                        r95 / l95
                    )
                )

                right_lt_left_95.append(
                    r95 < l95
                )

        writer.writerow(
            {
                "dtype_name":
                    dtype_name,
                "blocks":
                    len(rr),
                "median_log10_q50_right_over_left_2":
                    median_finite(
                        lr50
                    ),
                "median_log10_q95_right_over_left_2":
                    median_finite(
                        lr95
                    ),
                "fraction_q50_right_lt_left_2":
                    (
                        float(
                            np.mean(
                                right_lt_left_50
                            )
                        )
                        if right_lt_left_50
                        else math.nan
                    ),
                "fraction_q95_right_lt_left_2":
                    (
                        float(
                            np.mean(
                                right_lt_left_95
                            )
                        )
                        if right_lt_left_95
                        else math.nan
                    ),
            }
        )


# ----------------------------------------------------------------
# Final JSON
# ----------------------------------------------------------------

final = {
    "status":
        "LAPACK_BASELINE_FINALIZED",
    "tasks":
        len(status_paths),
    "data_files":
        len(data_paths),
    "records":
        total_rows,
    "distinct_matrix_precision_records":
        len(
            seen_matrix_precision
        ),
    "duplicate_records":
        duplicate_records,
    "expected_records":
        3328000,
    "expected_blocks_per_precision":
        832,
    "precision": {},
    "outputs": [
        precision_path.name,
        family_path.name,
        n_path.name,
        block_path.name,
        table_path.name,
        asym_path.name,
    ],
}

for dtype_name in (
    "float32",
    "float64",
):
    d = overall[
        dtype_name
    ]

    final[
        "precision"
    ][dtype_name] = {
        "attempts":
            d["attempts"],
        "successes":
            d["successes"],
        "failures":
            d["failures"],
        "failure_rate":
            (
                d["failures"]
                / d["attempts"]
                if d["attempts"]
                else math.nan
            ),
        "failure_classes":
            dict(
                d[
                    "failure_classes"
                ]
            ),
    }


gate = (
    statuses_ok
    and len(status_paths)
        == 999
    and len(data_paths)
        == 999
    and total_rows
        == 3328000
    and len(
        seen_matrix_precision
    )
        == 3328000
    and duplicate_records
        == 0
    and all(
        sum(
            1
            for r in block_rows
            if r["dtype_name"]
            == dtype_name
        )
        == 832
        for dtype_name
        in (
            "float32",
            "float64",
        )
    )
)

final[
    "aggregation_gate"
] = gate

json_path = (
    OUTDIR
    / "lapack_final_summary.json"
)

json_path.write_text(
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
    "TABLE-2 STYLE LAPACK RESULTS"
)
print(
    "============================================================"
)

for row in table_rows:
    print()
    for key, value in row.items():
        print(
            key,
            "=",
            value,
        )


print()
print(
    "============================================================"
)
print(
    "FAILURES BY PRECISION"
)
print(
    "============================================================"
)

for dtype_name in (
    "float32",
    "float64",
):
    print(
        dtype_name,
        json.dumps(
            final[
                "precision"
            ][dtype_name],
            sort_keys=True,
        ),
    )


print()
print(
    "============================================================"
)
print(
    "FINAL GATE"
)
print(
    "============================================================"
)

print(
    "aggregation_gate =",
    gate,
)

print(
    "summary_json =",
    json_path,
)

if gate:
    print(
        "CASTILLO_LAPACK_BASELINE_FINALIZED=PASS"
    )
else:
    print(
        "CASTILLO_LAPACK_BASELINE_FINALIZED=FAIL"
    )
