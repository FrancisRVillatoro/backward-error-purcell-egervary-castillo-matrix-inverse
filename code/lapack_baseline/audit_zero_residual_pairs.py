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


summary = {}

print(
    "============================================================"
)
print(
    "ZERO-RESIDUAL / STRICT PAIR AUDIT"
)
print(
    "============================================================"
)


for di, dtype_name in enumerate(DTYPES):

    summary[dtype_name] = {}

    for mi, method in enumerate(METHODS):

        total_s6 = 0

        finite_pairs = 0

        positive_positive = 0

        method_zero = 0
        lapack_zero = 0
        both_zero = 0

        method_lt = 0
        method_eq = 0
        method_gt = 0

        log_values = []

        for path in sorted(
            PARTIAL.glob(
                "paired_????.npz"
            )
        ):

            z = np.load(
                str(path),
                allow_pickle=False,
            )

            common6 = np.asarray(
                z["common6"],
                dtype=np.uint8,
            ).astype(bool)

            c = np.asarray(
                z["castillo_r2u"],
                dtype=np.float64,
            )[:, di, mi]

            l = np.asarray(
                z["lapack_r2u"],
                dtype=np.float64,
            )[:, di]

            mask = common6[:, di]

            total_s6 += int(
                np.sum(mask)
            )

            finite = (
                mask
                & np.isfinite(c)
                & np.isfinite(l)
                & (c >= 0.0)
                & (l >= 0.0)
            )

            cc = c[finite]
            ll = l[finite]

            finite_pairs += int(
                cc.size
            )

            cz = (
                cc == 0.0
            )

            lz = (
                ll == 0.0
            )

            both_zero += int(
                np.sum(
                    cz & lz
                )
            )

            method_zero += int(
                np.sum(cz)
            )

            lapack_zero += int(
                np.sum(lz)
            )

            method_lt += int(
                np.sum(
                    cc < ll
                )
            )

            method_eq += int(
                np.sum(
                    cc == ll
                )
            )

            method_gt += int(
                np.sum(
                    cc > ll
                )
            )

            pp = (
                (cc > 0.0)
                & (ll > 0.0)
            )

            positive_positive += int(
                np.sum(pp)
            )

            if np.any(pp):
                log_values.append(
                    np.log10(
                        cc[pp]
                        / ll[pp]
                    )
                )


        if log_values:
            logs = np.concatenate(
                log_values
            )

            median_log = float(
                np.median(logs)
            )

        else:
            median_log = math.nan


        result = {
            "S6_count":
                total_s6,

            "finite_nonnegative_pairs":
                finite_pairs,

            "positive_positive_pairs":
                positive_positive,

            "method_zero":
                method_zero,

            "lapack_zero":
                lapack_zero,

            "both_zero":
                both_zero,

            "method_lt_lapack":
                method_lt,

            "method_eq_lapack":
                method_eq,

            "method_gt_lapack":
                method_gt,

            "fraction_method_lt_lapack_all_finite":
                (
                    method_lt
                    / finite_pairs
                    if finite_pairs
                    else math.nan
                ),

            "fraction_method_le_lapack_all_finite":
                (
                    (
                        method_lt
                        + method_eq
                    )
                    / finite_pairs
                    if finite_pairs
                    else math.nan
                ),

            "median_log10_method_over_lapack_positive_pairs":
                median_log,
        }

        summary[
            dtype_name
        ][method] = result


        print()
        print(
            method,
            dtype_name,
        )

        for key, value in result.items():
            print(
                key,
                "=",
                value,
            )


gate = True

for dtype_name in DTYPES:

    expected = (
        1629883
        if dtype_name == "float32"
        else 1663999
    )

    for method in METHODS:

        r = summary[
            dtype_name
        ][method]

        if (
            r[
                "S6_count"
            ]
            != expected
        ):
            gate = False

        if (
            r[
                "finite_nonnegative_pairs"
            ]
            != expected
        ):
            gate = False

        if (
            r[
                "method_lt_lapack"
            ]
            + r[
                "method_eq_lapack"
            ]
            + r[
                "method_gt_lapack"
            ]
            != expected
        ):
            gate = False


out = (
    OUT
    / "zero_residual_pair_audit.json"
)

out.write_text(
    json.dumps(
        {
            "gate":
                gate,
            "results":
                summary,
        },
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
    "FINAL ZERO-PAIR GATE"
)
print(
    "============================================================"
)

print(
    "zero_pair_gate =",
    gate,
)

print(
    "output =",
    out,
)

if gate:
    print(
        "CASTILLO_ZERO_RESIDUAL_PAIR_AUDIT=PASS"
    )
else:
    print(
        "CASTILLO_ZERO_RESIDUAL_PAIR_AUDIT=FAIL"
    )
