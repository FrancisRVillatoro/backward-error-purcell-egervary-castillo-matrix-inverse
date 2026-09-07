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
    vals = (
        as_float(row["D_2_est"]),
        as_float(row["R_2_est_ld"]),
        as_float(row["AE_2_est"]),
    )

    if not all(
        math.isfinite(x) and x > 0.0
        for x in vals
    ):
        return None

    return vals


def relerr(a, b):
    return abs(a - b) / max(
        abs(a),
        abs(b),
        1.0e-300,
    )


class Stat:
    def __init__(self):
        self.w = array("d")
        self.d = array("d")
        self.ae = array("d")
        self.h = array("d")
        self.q = array("d")
        self.r = array("d")
        self.mask = array("d")
        self.mask_weight = array("d")
        self.ae_effect = array("d")

    def add(
        self,
        w,
        d,
        ae,
        h,
        q,
        r,
        mask,
        mask_weight,
        ae_effect,
    ):
        self.w.append(w)
        self.d.append(d)
        self.ae.append(ae)
        self.h.append(h)
        self.q.append(q)
        self.r.append(r)
        self.mask.append(mask)
        self.mask_weight.append(
            mask_weight
        )
        self.ae_effect.append(
            ae_effect
        )


def arr(a):
    return np.frombuffer(
        a,
        dtype=np.float64,
    )


def quant(a, p):
    if not a:
        return math.nan

    return float(
        np.quantile(
            arr(a),
            p,
            method="linear",
        )
    )


def geometric_mean(a):
    if not a:
        return math.nan

    x = arr(a)

    if np.any(x <= 0.0):
        return math.nan

    return float(
        np.exp(
            np.mean(np.log(x))
        )
    )


def corr_log(a, b):
    if len(a) < 3 or len(a) != len(b):
        return math.nan

    x = np.log(arr(a))
    y = np.log(arr(b))

    if (
        np.std(x) == 0.0
        or np.std(y) == 0.0
    ):
        return math.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def summarize(labels, s):
    w = arr(s.w)
    d = arr(s.d)
    mask = arr(s.mask)

    n = len(w)

    out = dict(labels)

    out.update({
        "pairs": n,

        "wD0_q10":
            quant(s.w, 0.10),
        "wD0_q50":
            quant(s.w, 0.50),
        "wD0_q90":
            quant(s.w, 0.90),

        "D_ratio_q50":
            quant(s.d, 0.50),
        "D_ratio_q90":
            quant(s.d, 0.90),

        "AE_ratio_q50":
            quant(s.ae, 0.50),

        "H_ratio_q50":
            quant(s.h, 0.50),

        "Q_ratio_q50":
            quant(s.q, 0.50),

        "R_ratio_q50":
            quant(s.r, 0.50),

        "Mscale_q10":
            quant(s.mask, 0.10),
        "Mscale_q50":
            quant(s.mask, 0.50),
        "Mscale_q90":
            quant(s.mask, 0.90),

        "Mweight_q50":
            quant(
                s.mask_weight,
                0.50,
            ),

        "MAE_q50":
            quant(
                s.ae_effect,
                0.50,
            ),

        # Geometric means are especially useful
        # because the multiplicative decomposition
        # remains exact after taking mean logs.
        "GM_D_ratio":
            geometric_mean(s.d),

        "GM_Mscale":
            geometric_mean(s.mask),

        "GM_Mweight":
            geometric_mean(
                s.mask_weight
            ),

        "GM_MAE":
            geometric_mean(
                s.ae_effect
            ),

        "GM_Q_ratio":
            geometric_mean(s.q),

        "GM_R_ratio":
            geometric_mean(s.r),

        "corr_logD_logwD0":
            corr_log(
                s.d,
                s.w,
            ),

        "corr_logD_logMscale":
            corr_log(
                s.d,
                s.mask,
            ),

        "corr_logwD0_logMweight":
            corr_log(
                s.w,
                s.mask_weight,
            ),

        "corr_logR_logD":
            corr_log(
                s.r,
                s.d,
            ),

        "corr_logR_logMscale":
            corr_log(
                s.r,
                s.mask,
            ),

        "fraction_D_up":
            (
                float(np.mean(d > 1.0))
                if n else math.nan
            ),

        "fraction_Mscale_lt_1":
            (
                float(
                    np.mean(mask < 1.0)
                )
                if n else math.nan
            ),

        "fraction_D_up_Mscale_down":
            (
                float(
                    np.mean(
                        (d > 1.0)
                        & (mask < 1.0)
                    )
                )
                if n else math.nan
            ),

        "fraction_wD0_lt_0p1":
            (
                float(
                    np.mean(w < 0.1)
                )
                if n else math.nan
            ),
    })

    # Exact geometric-mean check:
    #
    # GM(R) = GM(D) GM(Mscale) GM(Q)
    #
    pred = (
        out["GM_D_ratio"]
        * out["GM_Mscale"]
        * out["GM_Q_ratio"]
    )

    out[
        "GM_factorization_relerr"
    ] = relerr(
        out["GM_R_ratio"],
        pred,
    )

    # And:
    #
    # GM(Mscale) =
    # GM(Mweight) GM(MAE)
    #
    pred2 = (
        out["GM_Mweight"]
        * out["GM_MAE"]
    )

    out[
        "GM_mask_factorization_relerr"
    ] = relerr(
        out["GM_Mscale"],
        pred2,
    )

    return out


def add_groups(
    groups,
    meta,
    metrics,
):
    family = meta["family"]
    dtype = meta["dtype_name"]
    n = meta["n"]
    alpha = meta["alpha"]

    for key in (
        (
            "family",
            family,
            dtype,
        ),
        (
            "n",
            family,
            dtype,
            n,
        ),
        (
            "alpha",
            family,
            dtype,
            alpha,
        ),
        (
            "block",
            family,
            dtype,
            n,
            alpha,
        ),
    ):
        groups[key].add(*metrics)


def labels_for_key(key):
    if key[0] == "family":
        _, family, dtype = key

        return {
            "family": family,
            "dtype_name": dtype,
        }

    if key[0] == "n":
        _, family, dtype, n = key

        return {
            "family": family,
            "dtype_name": dtype,
            "n": n,
        }

    if key[0] == "alpha":
        _, family, dtype, alpha = key

        return {
            "family": family,
            "dtype_name": dtype,
            "alpha": alpha,
        }

    if key[0] == "block":
        (
            _,
            family,
            dtype,
            n,
            alpha,
        ) = key

        return {
            "family": family,
            "dtype_name": dtype,
            "n": n,
            "alpha": alpha,
        }

    raise ValueError(key)


def write_level(
    groups,
    level,
    path,
):
    rows = []

    for key, stat in groups.items():
        if key[0] != level:
            continue

        rows.append(
            summarize(
                labels_for_key(key),
                stat,
            )
        )

    rows.sort(
        key=lambda r: (
            r["family"],
            r["dtype_name"],
            int(r.get("n", -1)),
            float(
                r.get("alpha", -1.0)
            ),
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


def write_deciles(
    groups,
    path,
):
    rows = []

    for key, stat in groups.items():
        if key[0] != "n":
            continue

        (
            _,
            family,
            dtype,
            n,
        ) = key

        w = arr(stat.w)
        d = arr(stat.d)
        mask = arr(stat.mask)
        r = arr(stat.r)

        if len(w) == 0:
            continue

        # Quantile bins of baseline weight.
        edges = np.quantile(
            w,
            np.linspace(
                0.0,
                1.0,
                11,
            ),
        )

        # digitize against internal boundaries.
        bins = np.digitize(
            w,
            edges[1:-1],
            right=True,
        )

        for decile in range(10):
            sel = bins == decile

            if not np.any(sel):
                continue

            rows.append({
                "family":
                    family,
                "dtype_name":
                    dtype,
                "n":
                    n,
                "w_decile":
                    decile + 1,
                "pairs":
                    int(np.sum(sel)),
                "wD0_q50":
                    float(
                        np.median(
                            w[sel]
                        )
                    ),
                "D_ratio_q50":
                    float(
                        np.median(
                            d[sel]
                        )
                    ),
                "Mscale_q50":
                    float(
                        np.median(
                            mask[sel]
                        )
                    ),
                "R_ratio_q50":
                    float(
                        np.median(
                            r[sel]
                        )
                    ),
            })

    rows.sort(
        key=lambda r: (
            r["family"],
            r["dtype_name"],
            r["n"],
            r["w_decile"],
        )
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
    invalid = set()

    raw_rows = 0
    target_rows = 0
    pairs = 0
    invalid_rows = 0

    max_r_identity = 0.0
    max_mask_identity = 0.0

    for row in gzip.open(
        RAW,
        "rt",
        encoding="utf-8",
        newline="",
    ):
        # We need DictReader, so handled below.
        break

    with gzip.open(
        RAW,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            raw_rows += 1

            method = row["method"]

            if method not in {
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
                invalid.add(key)
                pending0.pop(
                    key,
                    None,
                )
                pending1.pop(
                    key,
                    None,
                )
                continue

            if key in invalid:
                continue

            vals = get_values(row)

            if vals is None:
                invalid_rows += 1
                invalid.add(key)
                pending0.pop(
                    key,
                    None,
                )
                pending1.pop(
                    key,
                    None,
                )
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

            item = (
                vals,
                meta,
            )

            if method == M0:

                if key not in pending1:
                    pending0[key] = item
                    continue

                item0 = item
                item1 = pending1.pop(key)

            else:

                if key not in pending0:
                    pending1[key] = item
                    continue

                item0 = pending0.pop(key)
                item1 = item

            (
                (d0, r0, ae0),
                meta0,
            ) = item0

            (
                (d1, r1, ae1),
                meta1,
            ) = item1

            if meta0 != meta1:
                raise RuntimeError(
                    "Metadata mismatch "
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

            w = (
                d0 * d0
                / (
                    d0 * d0
                    + ae0 * ae0
                )
            )

            d_ratio = d1 / d0
            ae_ratio = ae1 / ae0
            h_ratio = h1 / h0

            q0 = r0 / h0
            q1 = r1 / h1

            q_ratio = q1 / q0
            r_ratio = r1 / r0

            # Total masking of D amplification.
            mask = (
                h_ratio
                / d_ratio
            )

            # Counterfactual H ratio if AE did
            # not change at all.
            h_ae_fixed = math.sqrt(
                w
                * d_ratio
                * d_ratio
                + (1.0 - w)
            )

            mask_weight = (
                h_ae_fixed
                / d_ratio
            )

            # Remaining change caused by the
            # actual AE_ratio rather than 1.
            ae_effect = (
                h_ratio
                / h_ae_fixed
            )

            # Exact:
            #
            # Rratio = Dratio * mask * Qratio
            pred_r = (
                d_ratio
                * mask
                * q_ratio
            )

            max_r_identity = max(
                max_r_identity,
                relerr(
                    r_ratio,
                    pred_r,
                ),
            )

            # Exact:
            #
            # mask = mask_weight * ae_effect
            pred_mask = (
                mask_weight
                * ae_effect
            )

            max_mask_identity = max(
                max_mask_identity,
                relerr(
                    mask,
                    pred_mask,
                ),
            )

            add_groups(
                groups,
                meta0,
                (
                    w,
                    d_ratio,
                    ae_ratio,
                    h_ratio,
                    q_ratio,
                    r_ratio,
                    mask,
                    mask_weight,
                    ae_effect,
                ),
            )

            pairs += 1

    unmatched = (
        len(pending0)
        + len(pending1)
    )

    paths = {
        "family":
            REPORT
            / "compensation_family.csv",

        "n":
            REPORT
            / "compensation_by_n.csv",

        "alpha":
            REPORT
            / "compensation_by_alpha.csv",

        "block":
            REPORT
            / "compensation_by_block.csv",
    }

    for level, path in paths.items():
        write_level(
            groups,
            level,
            path,
        )

    decile_path = (
        REPORT
        / "compensation_w_deciles_by_n.csv"
    )

    write_deciles(
        groups,
        decile_path,
    )

    with paths["family"].open(
        newline="",
        encoding="utf-8",
    ) as f:
        family_rows = list(
            csv.DictReader(f)
        )

    summary = {
        "raw_rows": raw_rows,
        "target_rows": target_rows,
        "pairs": pairs,
        "invalid_target_rows":
            invalid_rows,
        "unmatched_valid_rows":
            unmatched,
        "max_R_factorization_relerr":
            max_r_identity,
        "max_mask_factorization_relerr":
            max_mask_identity,
        "definitions": {
            "wD0":
                "D0^2/(D0^2+AE0^2)",
            "Mscale":
                "(H1/H0)/(D1/D0)",
            "Mweight":
                "sqrt(wD0*Dratio^2+1-wD0)"
                "/Dratio",
            "MAE":
                "Hratio/"
                "sqrt(wD0*Dratio^2+1-wD0)",
            "identity_1":
                "Rratio = "
                "Dratio*Mscale*Qratio",
            "identity_2":
                "Mscale = Mweight*MAE",
        },
        "family_summary":
            family_rows,
    }

    (
        REPORT
        / "compensation_summary.json"
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
        f"RAW_ROWS={raw_rows}"
    )
    print(
        f"TARGET_ROWS={target_rows}"
    )
    print(
        f"PAIRS={pairs}"
    )
    print(
        "INVALID_TARGET_ROWS="
        f"{invalid_rows}"
    )
    print(
        "UNMATCHED_VALID_ROWS="
        f"{unmatched}"
    )
    print(
        "MAX_R_FACTORIZATION_RELERR="
        f"{max_r_identity}"
    )
    print(
        "MAX_MASK_FACTORIZATION_RELERR="
        f"{max_mask_identity}"
    )

    print()
    print(
        "===== FAMILY COMPENSATION ====="
    )

    with paths["family"].open(
        encoding="utf-8",
    ) as f:
        print(f.read(), end="")

    print(
        "COMPENSATION_ANALYSIS_OK"
    )


if __name__ == "__main__":
    main()
