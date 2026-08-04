#!/usr/bin/env python3

import csv
import gzip
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path.home() / "castillo_stability_campaign"

RAW = (
    Path.home()
    / "fscratch"
    / "castillo_stability_campaign"
    / "results"
    / "defect_replay"
    / "defect_replay_records.csv.gz"
)

REPORT = ROOT / "reports" / "defect_replay"

M0 = "R0_C1"
M1 = "R1_C1"


def as_bool(x):
    return str(x).strip().lower() in {
        "1", "true", "yes", "success"
    }


def as_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def valid_row(row):
    if not as_bool(row["success"]):
        return False

    g = as_float(
        row["closure_abs_ratio_bound"]
    )

    return (
        math.isfinite(g)
        and g <= 1.0
    )


def get_values(row):
    d = as_float(row["D_2_est"])
    r = as_float(row["R_2_est_ld"])
    ae = as_float(row["AE_2_est"])

    if not all(
        math.isfinite(x)
        and x >= 0.0
        for x in (d, r, ae)
    ):
        return None

    return d, r, ae


def relerr(a, b):
    return (
        abs(a - b)
        / max(
            abs(a),
            abs(b),
            1.0e-300,
        )
    )


class Stat:
    def __init__(self):
        self.d0ae0 = array("d")
        self.wd0 = array("d")
        self.dratio = array("d")
        self.aeratio = array("d")
        self.hratio = array("d")
        self.qratio = array("d")
        self.rratio = array("d")

        self.d0_lt_ae0 = 0
        self.h_up = 0
        self.q_down = 0
        self.r_up = 0
        self.h_up_q_down = 0
        self.h_log_dominates_q = 0

    def add(
        self,
        d0ae0,
        wd0,
        dratio,
        aeratio,
        hratio,
        qratio,
        rratio,
    ):
        self.d0ae0.append(d0ae0)
        self.wd0.append(wd0)
        self.dratio.append(dratio)
        self.aeratio.append(aeratio)
        self.hratio.append(hratio)
        self.qratio.append(qratio)
        self.rratio.append(rratio)

        if d0ae0 < 1.0:
            self.d0_lt_ae0 += 1

        if hratio > 1.0:
            self.h_up += 1

        if qratio < 1.0:
            self.q_down += 1

        if rratio > 1.0:
            self.r_up += 1

        if (
            hratio > 1.0
            and qratio < 1.0
        ):
            self.h_up_q_down += 1

        if (
            abs(math.log(hratio))
            >
            abs(math.log(qratio))
        ):
            self.h_log_dominates_q += 1


def np_view(a):
    return np.frombuffer(
        a,
        dtype=np.float64,
    )


def q(a, p):
    if len(a) == 0:
        return math.nan

    return float(
        np.quantile(
            np_view(a),
            p,
            method="linear",
        )
    )


def corr(a, b):
    if (
        len(a) < 3
        or len(b) != len(a)
    ):
        return math.nan

    x = np.log(np_view(a))
    y = np.log(np_view(b))

    sx = float(np.std(x))
    sy = float(np.std(y))

    if sx == 0.0 or sy == 0.0:
        return math.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def summarize(labels, s):
    n = len(s.rratio)

    def frac(count):
        return (
            count / n
            if n
            else math.nan
        )

    out = dict(labels)

    out.update({
        "pairs": n,

        "D0_over_AE0_q10":
            q(s.d0ae0, 0.10),
        "D0_over_AE0_q50":
            q(s.d0ae0, 0.50),
        "D0_over_AE0_q90":
            q(s.d0ae0, 0.90),

        "wD0_q10":
            q(s.wd0, 0.10),
        "wD0_q50":
            q(s.wd0, 0.50),
        "wD0_q90":
            q(s.wd0, 0.90),

        "D_ratio_q50":
            q(s.dratio, 0.50),
        "D_ratio_q90":
            q(s.dratio, 0.90),
        "D_ratio_q95":
            q(s.dratio, 0.95),

        "AE_ratio_q50":
            q(s.aeratio, 0.50),
        "AE_ratio_q90":
            q(s.aeratio, 0.90),
        "AE_ratio_q95":
            q(s.aeratio, 0.95),

        "H_ratio_q50":
            q(s.hratio, 0.50),
        "H_ratio_q90":
            q(s.hratio, 0.90),
        "H_ratio_q95":
            q(s.hratio, 0.95),

        "Q_ratio_q50":
            q(s.qratio, 0.50),
        "Q_ratio_q90":
            q(s.qratio, 0.90),
        "Q_ratio_q95":
            q(s.qratio, 0.95),

        "R_ratio_q50":
            q(s.rratio, 0.50),
        "R_ratio_q90":
            q(s.rratio, 0.90),
        "R_ratio_q95":
            q(s.rratio, 0.95),

        "fraction_D0_lt_AE0":
            frac(s.d0_lt_ae0),

        "fraction_H_ratio_gt_1":
            frac(s.h_up),

        "fraction_Q_ratio_lt_1":
            frac(s.q_down),

        "fraction_R_ratio_gt_1":
            frac(s.r_up),

        "fraction_H_up_Q_down":
            frac(s.h_up_q_down),

        "fraction_abs_logH_gt_abs_logQ":
            frac(
                s.h_log_dominates_q
            ),

        "pearson_logR_logH":
            corr(
                s.rratio,
                s.hratio,
            ),

        "pearson_logR_logQ":
            corr(
                s.rratio,
                s.qratio,
            ),

        "pearson_logH_logD":
            corr(
                s.hratio,
                s.dratio,
            ),
    })

    return out


def add_groups(
    groups,
    meta,
    metrics,
):
    family = meta["family"]
    dtype_name = meta["dtype_name"]
    n = meta["n"]
    alpha = meta["alpha"]

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
        groups[key].add(*metrics)


def write_level(
    groups,
    level,
    path,
):
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
            (
                _,
                family,
                dtype_name,
                alpha,
            ) = key

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
            summarize(
                labels,
                stat,
            )
        )

    rows.sort(
        key=lambda r: (
            r["family"],
            r["dtype_name"],
            int(r.get("n", -1)),
            float(r.get("alpha", -1.0)),
        )
    )

    if not rows:
        raise RuntimeError(
            "No rows for " + level
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = defaultdict(Stat)

    pending0 = {}
    pending1 = {}
    invalid_keys = set()

    raw_rows = 0
    target_rows = 0
    invalid_rows = 0
    pairs = 0
    metric_invalid_pairs = 0

    max_identity_R = 0.0
    max_identity_H = 0.0

    identity_R_failures = 0
    identity_H_failures = 0

    with gzip.open(
        RAW,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

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
                "Missing fields: "
                + repr(sorted(missing))
            )

        for row in reader:
            raw_rows += 1

            if row["method"] not in {
                M0,
                M1,
            }:
                continue

            target_rows += 1

            key = (
                row["matrix_id"],
                row["dtype_name"],
            )

            if not valid_row(row):
                invalid_rows += 1
                invalid_keys.add(key)
                pending0.pop(key, None)
                pending1.pop(key, None)
                continue

            if key in invalid_keys:
                continue

            vals = get_values(row)

            if vals is None:
                invalid_rows += 1
                invalid_keys.add(key)
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

            item = (vals, meta)

            if row["method"] == M0:
                if key in pending1:
                    item1 = pending1.pop(key)
                    item0 = item
                else:
                    pending0[key] = item
                    continue
            else:
                if key in pending0:
                    item0 = pending0.pop(key)
                    item1 = item
                else:
                    pending1[key] = item
                    continue

            (d0, r0, ae0), meta0 = item0
            (d1, r1, ae1), meta1 = item1

            if meta0 != meta1:
                raise RuntimeError(
                    "Metadata mismatch: "
                    + repr(key)
                )

            h0 = math.hypot(
                d0,
                ae0,
            )

            h1 = math.hypot(
                d1,
                ae1,
            )

            if not all(
                x > 0.0
                and math.isfinite(x)
                for x in (
                    d0,
                    r0,
                    ae0,
                    d1,
                    r1,
                    ae1,
                    h0,
                    h1,
                )
            ):
                metric_invalid_pairs += 1
                continue

            d0ae0 = d0 / ae0

            wd0 = (
                d0 * d0
                / (
                    d0 * d0
                    + ae0 * ae0
                )
            )

            dratio = d1 / d0
            aeratio = ae1 / ae0
            hratio = h1 / h0

            q0 = r0 / h0
            q1 = r1 / h1

            qratio = q1 / q0
            rratio = r1 / r0

            # Exact scalar identity:
            #
            # R1/R0 =
            # (Q1/Q0)(H1/H0)
            #
            pred_r = (
                qratio
                * hratio
            )

            err_r = relerr(
                rratio,
                pred_r,
            )

            max_identity_R = max(
                max_identity_R,
                err_r,
            )

            if err_r > 1.0e-12:
                identity_R_failures += 1

            # Second exact scalar identity:
            #
            # (H1/H0)^2 =
            # w D_ratio^2
            # + (1-w) AE_ratio^2
            #
            pred_h = math.sqrt(
                wd0
                * dratio
                * dratio
                +
                (1.0 - wd0)
                * aeratio
                * aeratio
            )

            err_h = relerr(
                hratio,
                pred_h,
            )

            max_identity_H = max(
                max_identity_H,
                err_h,
            )

            if err_h > 1.0e-12:
                identity_H_failures += 1

            metrics = (
                d0ae0,
                wd0,
                dratio,
                aeratio,
                hratio,
                qratio,
                rratio,
            )

            add_groups(
                groups,
                meta0,
                metrics,
            )

            pairs += 1

    unmatched = (
        len(pending0)
        + len(pending1)
    )

    paths = {
        "family":
            REPORT
            / "DAE_mechanism_family.csv",

        "n":
            REPORT
            / "DAE_mechanism_by_n.csv",

        "alpha":
            REPORT
            / "DAE_mechanism_by_alpha.csv",

        "block":
            REPORT
            / "DAE_mechanism_by_block.csv",
    }

    for level, path in paths.items():
        write_level(
            groups,
            level,
            path,
        )

    with paths["family"].open(
        newline="",
        encoding="utf-8",
    ) as f:
        family_rows = list(
            csv.DictReader(f)
        )

    summary = {
        "raw_file": str(RAW),
        "methods": [M0, M1],
        "raw_rows": raw_rows,
        "target_rows": target_rows,
        "invalid_target_rows":
            invalid_rows,
        "paired_matrices_dtype":
            pairs,
        "metric_invalid_pairs":
            metric_invalid_pairs,
        "unmatched_valid_rows":
            unmatched,
        "max_identity_R_relative_error":
            max_identity_R,
        "identity_R_failures_gt_1e-12":
            identity_R_failures,
        "max_identity_H_relative_error":
            max_identity_H,
        "identity_H_failures_gt_1e-12":
            identity_H_failures,
        "definitions": {
            "H":
                "hypot(D_2_est, AE_2_est)",
            "Q":
                "R_2_est_ld / H",
            "wD0":
                "D0^2 / (D0^2 + AE0^2)",
            "R_ratio_identity":
                "R1/R0 = (Q1/Q0)(H1/H0)",
            "H_ratio_identity":
                "(H1/H0)^2 = "
                "wD0*(D1/D0)^2 + "
                "(1-wD0)*(AE1/AE0)^2",
        },
        "interpretation_note": (
            "H and wD0 are scalar constructs "
            "from spectral norms. They are not "
            "Frobenius-energy decompositions "
            "and do not establish orthogonality."
        ),
        "family_summary":
            family_rows,
    }

    (
        REPORT
        / "DAE_mechanism_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "RAW_ROWS=",
        raw_rows,
        sep="",
    )
    print(
        "TARGET_ROWS=",
        target_rows,
        sep="",
    )
    print(
        "PAIRS=",
        pairs,
        sep="",
    )
    print(
        "INVALID_TARGET_ROWS=",
        invalid_rows,
        sep="",
    )
    print(
        "METRIC_INVALID_PAIRS=",
        metric_invalid_pairs,
        sep="",
    )
    print(
        "UNMATCHED_VALID_ROWS=",
        unmatched,
        sep="",
    )

    print(
        "MAX_R_IDENTITY_RELERR=",
        max_identity_R,
        sep="",
    )
    print(
        "R_IDENTITY_FAILURES_GT_1E-12=",
        identity_R_failures,
        sep="",
    )

    print(
        "MAX_H_IDENTITY_RELERR=",
        max_identity_H,
        sep="",
    )
    print(
        "H_IDENTITY_FAILURES_GT_1E-12=",
        identity_H_failures,
        sep="",
    )

    print()
    print(
        "===== FAMILY MECHANISM SUMMARY ====="
    )

    with paths["family"].open(
        encoding="utf-8",
    ) as f:
        print(f.read(), end="")

    print(
        "DAE_MECHANISM_ANALYSIS_OK"
    )


if __name__ == "__main__":
    main()
