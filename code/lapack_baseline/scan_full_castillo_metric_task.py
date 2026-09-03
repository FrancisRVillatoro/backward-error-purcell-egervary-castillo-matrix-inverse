#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import gzip
import json
import math
import os
import sys
import traceback


ROOT = (
    Path.home()
    / "fscratch"
    / "castillo_lapack_paired"
)

SOURCE = (
    ROOT
    / "recovered_inverse"
)

OUT = (
    ROOT
    / "full_metric_scan"
    / "partial"
)


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


def metric_state(value):
    raw = str(
        value
        if value is not None
        else ""
    ).strip()

    if raw == "":
        return "blank", None

    try:
        x = float(raw)
    except Exception:
        return "parse_error", None

    if math.isnan(x):
        return "nan", x

    if math.isinf(x):
        if x > 0:
            return "posinf", x
        return "neginf", x

    if x < 0.0:
        return "negative", x

    return "ok", x


def main():
    task = int(
        sys.argv[1]
    )

    tag = f"{task:04d}"

    source = (
        SOURCE
        / f"inverse_{tag}.csv.gz"
    )

    output = (
        OUT
        / f"metric_scan_{tag}.json"
    )

    result = {
        "task_id":
            task,

        "source":
            str(source),

        "status":
            "TASK_FAIL",

        "rows_total":
            0,

        "success_total":
            0,

        "failure_total":
            0,

        "success_r2_invalid_total":
            0,

        "success_rinf_invalid_total":
            0,

        "success_either_invalid_total":
            0,

        "r2_states":
            {},

        "rinf_states":
            {},

        "success_by_dtype_method":
            {},

        "invalid_r2_by_dtype_method":
            {},

        "invalid_rinf_by_dtype_method":
            {},

        "invalid_r2_by_family":
            {},

        "invalid_r2_by_cell":
            {},

        "invalid_r2_by_dimension":
            {},

        "anomalies":
            [],
    }


    if not source.is_file():
        result[
            "failure_reason"
        ] = "source_missing"

        output.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )

        return


    r2_states = Counter()
    rinf_states = Counter()

    success_by_dtype_method = Counter()

    invalid_r2_by_dtype_method = Counter()
    invalid_rinf_by_dtype_method = Counter()

    invalid_r2_by_family = Counter()
    invalid_r2_by_cell = Counter()
    invalid_r2_by_dimension = Counter()

    anomalies = []


    with gzip.open(
        str(source),
        mode="rt",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:

            result[
                "rows_total"
            ] += 1

            success = as_bool(
                row.get(
                    "success",
                    False,
                )
            )

            if not success:
                result[
                    "failure_total"
                ] += 1
                continue

            result[
                "success_total"
            ] += 1

            dtype_name = str(
                row.get(
                    "dtype_name",
                    "",
                )
            )

            method = str(
                row.get(
                    "method",
                    "",
                )
            )

            family = str(
                row.get(
                    "family",
                    "",
                )
            )

            cell_id = str(
                row.get(
                    "cell_id",
                    "",
                )
            )

            n = str(
                row.get(
                    "n",
                    "",
                )
            )

            dm = (
                dtype_name
                + "|"
                + method
            )

            success_by_dtype_method[
                dm
            ] += 1


            r2_raw = row.get(
                "right_inverse_scaled_residual_2_est",
                "",
            )

            rinf_raw = row.get(
                "right_inverse_scaled_residual_inf",
                "",
            )

            r2_state, _ = metric_state(
                r2_raw
            )

            rinf_state, _ = metric_state(
                rinf_raw
            )

            r2_states[
                r2_state
            ] += 1

            rinf_states[
                rinf_state
            ] += 1


            bad_r2 = (
                r2_state
                != "ok"
            )

            bad_rinf = (
                rinf_state
                != "ok"
            )


            if bad_r2:

                result[
                    "success_r2_invalid_total"
                ] += 1

                invalid_r2_by_dtype_method[
                    dm
                ] += 1

                invalid_r2_by_family[
                    family
                ] += 1

                invalid_r2_by_cell[
                    cell_id
                ] += 1

                invalid_r2_by_dimension[
                    dtype_name
                    + "|n="
                    + n
                ] += 1


            if bad_rinf:

                result[
                    "success_rinf_invalid_total"
                ] += 1

                invalid_rinf_by_dtype_method[
                    dm
                ] += 1


            if (
                bad_r2
                or bad_rinf
            ):

                result[
                    "success_either_invalid_total"
                ] += 1

                anomalies.append(
                    {
                        "task_id":
                            task,

                        "matrix_id":
                            row.get(
                                "matrix_id",
                                "",
                            ),

                        "case_index":
                            row.get(
                                "case_index",
                                "",
                            ),

                        "block_index":
                            row.get(
                                "block_index",
                                "",
                            ),

                        "cell_id":
                            cell_id,

                        "family":
                            family,

                        "parameters_json":
                            row.get(
                                "parameters_json",
                                "",
                            ),

                        "n":
                            n,

                        "replica":
                            row.get(
                                "replica",
                                "",
                            ),

                        "method":
                            method,

                        "dtype_name":
                            dtype_name,

                        "failure_class":
                            row.get(
                                "failure_class",
                                "",
                            ),

                        "r2_state":
                            r2_state,

                        "r2_raw":
                            str(
                                r2_raw
                            ),

                        "rinf_state":
                            rinf_state,

                        "rinf_raw":
                            str(
                                rinf_raw
                            ),

                        "right_inverse_defect_2_est":
                            row.get(
                                "right_inverse_defect_2_est",
                                "",
                            ),

                        "right_inverse_defect_inf":
                            row.get(
                                "right_inverse_defect_inf",
                                "",
                            ),

                        "norm_A_2_est":
                            row.get(
                                "norm_A_2_est",
                                "",
                            ),

                        "norm_V_2_est":
                            row.get(
                                "norm_V_2_est",
                                "",
                            ),

                        "norm_A_inf":
                            row.get(
                                "norm_A_inf",
                                "",
                            ),

                        "norm_V_inf":
                            row.get(
                                "norm_V_inf",
                                "",
                            ),

                        "inverse_tableau_norm_reliable":
                            row.get(
                                "inverse_tableau_norm_reliable",
                                "",
                            ),
                    }
                )


    result[
        "r2_states"
    ] = dict(
        sorted(
            r2_states.items()
        )
    )

    result[
        "rinf_states"
    ] = dict(
        sorted(
            rinf_states.items()
        )
    )

    result[
        "success_by_dtype_method"
    ] = dict(
        sorted(
            success_by_dtype_method.items()
        )
    )

    result[
        "invalid_r2_by_dtype_method"
    ] = dict(
        sorted(
            invalid_r2_by_dtype_method.items()
        )
    )

    result[
        "invalid_rinf_by_dtype_method"
    ] = dict(
        sorted(
            invalid_rinf_by_dtype_method.items()
        )
    )

    result[
        "invalid_r2_by_family"
    ] = dict(
        sorted(
            invalid_r2_by_family.items()
        )
    )

    result[
        "invalid_r2_by_cell"
    ] = dict(
        sorted(
            invalid_r2_by_cell.items()
        )
    )

    result[
        "invalid_r2_by_dimension"
    ] = dict(
        sorted(
            invalid_r2_by_dimension.items()
        )
    )

    result[
        "anomalies"
    ] = anomalies

    result[
        "status"
    ] = "TASK_OK"


    tmp = output.with_name(
        output.name
        + ".tmp."
        + str(
            os.getpid()
        )
    )

    tmp.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        str(tmp),
        str(output),
    )


    print(
        "task_id =",
        task,
    )

    print(
        "rows_total =",
        result[
            "rows_total"
        ],
    )

    print(
        "success_total =",
        result[
            "success_total"
        ],
    )

    print(
        "success_r2_invalid_total =",
        result[
            "success_r2_invalid_total"
        ],
    )

    print(
        "success_rinf_invalid_total =",
        result[
            "success_rinf_invalid_total"
        ],
    )

    print(
        "success_either_invalid_total =",
        result[
            "success_either_invalid_total"
        ],
    )

    print(
        "FULL_METRIC_SCAN_TASK_OK =",
        tag,
    )


try:
    main()

except Exception as exc:

    task = (
        int(
            sys.argv[1]
        )
        if len(
            sys.argv
        ) > 1
        else -1
    )

    tag = (
        f"{task:04d}"
        if task >= 0
        else "unknown"
    )

    output = (
        OUT
        / f"metric_scan_{tag}.json"
    )

    result = {
        "task_id":
            task,

        "status":
            "TASK_FAIL",

        "exception":
            repr(exc),

        "traceback":
            traceback.format_exc(),
    }

    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "FULL_METRIC_SCAN_TASK_FAIL =",
        tag,
    )
