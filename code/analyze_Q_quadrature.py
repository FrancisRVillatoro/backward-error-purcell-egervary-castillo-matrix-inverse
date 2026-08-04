#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path.home() / "castillo_stability_campaign"
SCRATCH = (
    Path.home()
    / "fscratch"
    / "castillo_stability_campaign"
)

RAW = (
    SCRATCH
    / "results"
    / "defect_replay"
    / "defect_replay_records.csv.gz"
)

REPORT = ROOT / "reports" / "defect_replay"

METHOD0 = "R0_C1"
METHOD1 = "R1_C1"


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "1", "true", "yes", "success"
    }


def as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def qvalue(row: dict[str, str]) -> float:
    d = as_float(row["D_2_est"])
    r = as_float(row["R_2_est_ld"])
    ae = as_float(row["AE_2_est"])

    if not (
        math.isfinite(d)
        and math.isfinite(r)
        and math.isfinite(ae)
        and d >= 0.0
        and r >= 0.0
        and ae >= 0.0
    ):
        return math.nan

    denominator = math.hypot(d, ae)

    if denominator == 0.0:
        return (
            0.0
            if r == 0.0
            else math.inf
        )

    return r / denominator


def valid_row(row: dict[str, str]) -> bool:
    if not as_bool(row["success"]):
        return False

    ratio = as_float(
        row["closure_abs_ratio_bound"]
    )

    return (
        math.isfinite(ratio)
        and ratio <= 1.0
    )


class Stat:
    def __init__(self):
        self.q0 = array("d")
        self.q1 = array("d")
        self.ratio = array("d")

        self.q1_lt_q0 = 0
        self.q1_le_090_q0 = 0
        self.q1_le_080_q0 = 0
        self.q1_ge_110_q0 = 0

    def add(
        self,
        q0: float,
        q1: float,
    ) -> None:
        if not (
            math.isfinite(q0)
            and math.isfinite(q1)
            and q0 > 0.0
            and q1 >= 0.0
        ):
            return

        ratio = q1 / q0

        self.q0.append(q0)
        self.q1.append(q1)
        self.ratio.append(ratio)

        if q1 < q0:
            self.q1_lt_q0 += 1

        if ratio <= 0.90:
            self.q1_le_090_q0 += 1

        if ratio <= 0.80:
            self.q1_le_080_q0 += 1

        if ratio >= 1.10:
            self.q1_ge_110_q0 += 1


def quantile(values: array, p: float) -> float:
    if len(values) == 0:
        return math.nan

    x = np.frombuffer(
        values,
        dtype=np.float64,
    )

    return float(
        np.quantile(
            x,
            p,
            method="linear",
        )
    )


def mean_value(values: array) -> float:
    if len(values) == 0:
        return math.nan

    x = np.frombuffer(
        values,
        dtype=np.float64,
    )

    return float(np.mean(x))


def row_from_stat(
    labels: dict,
    stat: Stat,
) -> dict:
    n = len(stat.ratio)

    if n == 0:
        frac_lt = math.nan
        frac_090 = math.nan
        frac_080 = math.nan
        frac_110 = math.nan
    else:
        frac_lt = stat.q1_lt_q0 / n
        frac_090 = stat.q1_le_090_q0 / n
        frac_080 = stat.q1_le_080_q0 / n
        frac_110 = stat.q1_ge_110_q0 / n

    result = dict(labels)

    result.update(
        {
            "pairs": n,

            "Q_R0_q10":
                quantile(stat.q0, 0.10),
            "Q_R0_q50":
                quantile(stat.q0, 0.50),
            "Q_R0_q90":
                quantile(stat.q0, 0.90),
            "Q_R0_q95":
                quantile(stat.q0, 0.95),

            "Q_R1_q10":
                quantile(stat.q1, 0.10),
            "Q_R1_q50":
                quantile(stat.q1, 0.50),
            "Q_R1_q90":
                quantile(stat.q1, 0.90),
            "Q_R1_q95":
                quantile(stat.q1, 0.95),

            "Qratio_R1_over_R0_q10":
                quantile(stat.ratio, 0.10),
            "Qratio_R1_over_R0_q50":
                quantile(stat.ratio, 0.50),
            "Qratio_R1_over_R0_q90":
                quantile(stat.ratio, 0.90),
            "Qratio_R1_over_R0_q95":
                quantile(stat.ratio, 0.95),

            "Qratio_R1_over_R0_mean":
                mean_value(stat.ratio),

            "fraction_Q_R1_lt_Q_R0":
                frac_lt,

            "fraction_Q_R1_le_0p90_Q_R0":
                frac_090,

            "fraction_Q_R1_le_0p80_Q_R0":
                frac_080,

            "fraction_Q_R1_ge_1p10_Q_R0":
                frac_110,
        }
    )

    return result


def add_to_groups(
    groups,
    meta,
    q0,
    q1,
):
    family = meta["family"]
    dtype_name = meta["dtype_name"]
    n = int(meta["n"])
    alpha = float(meta["alpha"])

    keys = (
        (
            "family",
            family,
            dtype_name,
        ),
        (
            "n",
            family,
            dtype_name,
            n,
        ),
        (
            "alpha",
            family,
            dtype_name,
            alpha,
        ),
        (
            "block",
            family,
            dtype_name,
            n,
            alpha,
        ),
    )

    for key in keys:
        groups[key].add(q0, q1)


def write_level(
    path: Path,
    groups,
    level: str,
) -> None:

    rows = []

    for key, stat in groups.items():
        if key[0] != level:
            continue

        if level == "family":
            _, family, dtype_name = key

            labels = {
                "family": family,
                "dtype_name": dtype_name,
            }

        elif level == "n":
            _, family, dtype_name, n = key

            labels = {
                "family": family,
                "dtype_name": dtype_name,
                "n": n,
            }

        elif level == "alpha":
            _, family, dtype_name, alpha = key

            labels = {
                "family": family,
                "dtype_name": dtype_name,
                "alpha": alpha,
            }

        elif level == "block":
            (
                _,
                family,
                dtype_name,
                n,
                alpha,
            ) = key

            labels = {
                "family": family,
                "dtype_name": dtype_name,
                "n": n,
                "alpha": alpha,
            }

        else:
            raise ValueError(level)

        rows.append(
            row_from_stat(
                labels,
                stat,
            )
        )

    def sort_key(row):
        return (
            row["family"],
            row["dtype_name"],
            int(row.get("n", -1)),
            float(row.get("alpha", -1.0)),
        )

    rows.sort(key=sort_key)

    if not rows:
        raise RuntimeError(
            f"No rows for level {level}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = defaultdict(Stat)

    pending0 = {}
    pending1 = {}
    invalid = set()

    total_rows = 0
    target_rows = 0
    invalid_target_rows = 0
    paired = 0

    with gzip.open(
        RAW,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        required = {
            "matrix_id",
            "family",
            "n",
            "alpha",
            "dtype_name",
            "method",
            "success",
            "D_2_est",
            "R_2_est_ld",
            "AE_2_est",
            "closure_abs_ratio_bound",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "Missing raw fields: "
                + repr(sorted(missing))
            )

        for row in reader:
            total_rows += 1

            method = row["method"]

            if method not in {
                METHOD0,
                METHOD1,
            }:
                continue

            target_rows += 1

            key = (
                row["matrix_id"],
                row["dtype_name"],
            )

            if not valid_row(row):
                invalid_target_rows += 1
                invalid.add(key)
                pending0.pop(key, None)
                pending1.pop(key, None)
                continue

            if key in invalid:
                continue

            q = qvalue(row)

            if not math.isfinite(q):
                invalid_target_rows += 1
                invalid.add(key)
                pending0.pop(key, None)
                pending1.pop(key, None)
                continue

            meta = {
                "family":
                    row["family"],
                "dtype_name":
                    row["dtype_name"],
                "n":
                    int(row["n"]),
                "alpha":
                    float(row["alpha"]),
            }

            if method == METHOD0:

                if key in pending1:
                    q1, meta1 = pending1.pop(
                        key
                    )

                    if meta != meta1:
                        raise RuntimeError(
                            "Metadata mismatch "
                            f"for {key}"
                        )

                    add_to_groups(
                        groups,
                        meta,
                        q,
                        q1,
                    )

                    paired += 1

                else:
                    pending0[key] = (
                        q,
                        meta,
                    )

            else:

                if key in pending0:
                    q0, meta0 = pending0.pop(
                        key
                    )

                    if meta != meta0:
                        raise RuntimeError(
                            "Metadata mismatch "
                            f"for {key}"
                        )

                    add_to_groups(
                        groups,
                        meta,
                        q0,
                        q,
                    )

                    paired += 1

                else:
                    pending1[key] = (
                        q,
                        meta,
                    )

    unmatched = (
        len(pending0)
        + len(pending1)
    )

    print(
        "RAW_ROWS=",
        total_rows,
        sep="",
    )
    print(
        "TARGET_ROWS=",
        target_rows,
        sep="",
    )
    print(
        "INVALID_TARGET_ROWS=",
        invalid_target_rows,
        sep="",
    )
    print(
        "PAIRS=",
        paired,
        sep="",
    )
    print(
        "UNMATCHED_VALID_ROWS=",
        unmatched,
        sep="",
    )

    if unmatched:
        raise RuntimeError(
            "Unmatched valid R0/R1 rows: "
            f"{unmatched}"
        )

    paths = {
        "family":
            REPORT
            / "Q_quadrature_family.csv",

        "n":
            REPORT
            / "Q_quadrature_by_n.csv",

        "alpha":
            REPORT
            / "Q_quadrature_by_alpha.csv",

        "block":
            REPORT
            / "Q_quadrature_by_block.csv",
    }

    for level, path in paths.items():
        write_level(
            path,
            groups,
            level,
        )

    family_rows = []

    with paths["family"].open(
        newline="",
        encoding="utf-8",
    ) as handle:
        family_rows = list(
            csv.DictReader(handle)
        )

    summary = {
        "raw_file": str(RAW),
        "definition": (
            "Q = R_2_est_ld / "
            "hypot(D_2_est, AE_2_est)"
        ),
        "methods": [
            METHOD0,
            METHOD1,
        ],
        "admissibility": (
            "success=True and "
            "closure_abs_ratio_bound<=1"
        ),
        "raw_rows": total_rows,
        "target_rows": target_rows,
        "invalid_target_rows":
            invalid_target_rows,
        "paired_matrices_dtype":
            paired,
        "unmatched_valid_rows":
            unmatched,
        "family_summary":
            family_rows,
        "interpretation_note": (
            "Q is a spectral-norm cancellation "
            "proxy, not a Frobenius-angle or "
            "orthogonality measure."
        ),
    }

    summary_path = (
        REPORT
        / "Q_quadrature_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== FAMILY SUMMARY =====")

    with paths["family"].open(
        encoding="utf-8"
    ) as handle:
        print(handle.read(), end="")

    print(
        "Q_QUADRATURE_ANALYSIS_OK"
    )


if __name__ == "__main__":
    main()
