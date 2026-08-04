#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import json
import math
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT = (
    Path.home()
    / "castillo_stability_campaign"
)

SCRATCH = (
    Path.home()
    / "fscratch"
    / "castillo_stability_campaign"
)

WORK = (
    SCRATCH
    / "results"
    / "defect_replay"
)

PARTIAL = WORK / "partial"

REPORT = (
    PROJECT
    / "reports"
    / "defect_replay"
)

MERGED = (
    WORK
    / "defect_replay_records.csv.gz"
)

METHODS = (
    "R0_C0",
    "R0_C1",
    "R1_C1",
    "R2_C2",
)

DTYPES = (
    "float32",
    "float64",
)

COMBOS = [
    (method, dtype_name)
    for method in METHODS
    for dtype_name in DTYPES
]

COMBO_INDEX = {
    combo: i
    for i, combo in enumerate(COMBOS)
}

PAIRS = (
    ("R1_C1", "R0_C1"),
    ("R2_C2", "R0_C1"),
    ("R0_C0", "R0_C1"),
)


def as_bool(value) -> bool:
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "success",
    }


def as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def q(values, probability):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        return math.nan

    return float(
        np.quantile(
            x,
            probability,
        )
    )


def ratios(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    a = np.asarray(
        numerator,
        dtype=np.float64,
    )

    b = np.asarray(
        denominator,
        dtype=np.float64,
    )

    out = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a >= 0.0)
        & (b >= 0.0)
    )

    both_zero = (
        finite
        & (a == 0.0)
        & (b == 0.0)
    )

    out[both_zero] = 1.0

    positive_denominator = (
        finite
        & (b > 0.0)
    )

    out[
        positive_denominator
    ] = (
        a[positive_denominator]
        / b[positive_denominator]
    )

    return out


def correlation_summary(
    d_ratio: np.ndarray,
    r_ratio: np.ndarray,
):
    mask = (
        np.isfinite(d_ratio)
        & np.isfinite(r_ratio)
        & (d_ratio > 0.0)
        & (r_ratio > 0.0)
    )

    if int(np.sum(mask)) < 3:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
        )

    ld = np.log10(
        d_ratio[mask]
    )

    lr = np.log10(
        r_ratio[mask]
    )

    if (
        np.std(ld) > 0.0
        and np.std(lr) > 0.0
    ):
        pearson = float(
            np.corrcoef(
                ld,
                lr,
            )[0, 1]
        )
    else:
        pearson = math.nan

    try:
        spearman = float(
            spearmanr(
                ld,
                lr,
            ).correlation
        )
    except Exception:
        spearman = math.nan

    log2_disagreement = np.abs(
        np.log2(
            d_ratio[mask]
            / r_ratio[mask]
        )
    )

    median_disagreement = q(
        log2_disagreement,
        0.50,
    )

    q95_disagreement = q(
        log2_disagreement,
        0.95,
    )

    factor2_fraction = float(
        np.mean(
            log2_disagreement > 1.0
        )
    )

    return (
        pearson,
        spearman,
        median_disagreement,
        q95_disagreement,
        int(np.sum(mask)),
        factor2_fraction,
    )


def relative_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    scale = np.maximum(
        np.maximum(
            np.abs(a),
            np.abs(b),
        ),
        1.0e-300,
    )

    return np.abs(
        a - b
    ) / scale


def main() -> None:
    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        PARTIAL
        / f"task_{task:04d}.csv.gz"
        for task in range(1000)
    ]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    reuse_merged = False

    if missing:
        if MERGED.exists():
            # Reentrant finalization: the original
            # partial shards were intentionally
            # removed after the first successful
            # merge.  Reuse the retained canonical
            # merged replay bank as the sole input.
            paths = [MERGED]
            reuse_merged = True

            print(
                "REUSING_EXISTING_MERGED_REPLAY="
                f"{MERGED}"
            )
        else:
            raise RuntimeError(
                "Missing replay task outputs and "
                "no merged replay bank exists: "
                + ", ".join(
                    missing[:20]
                )
            )

    block_replicas = defaultdict(set)
    block_meta = {}
    array_job_ids = set()
    source_tasks = set()

    records = 0
    matrices = 0
    header = None

    temp_merged = MERGED.with_name(
        MERGED.name + ".tmp"
    )

    with gzip.open(
        temp_merged,
        "wt",
        encoding="utf-8",
        newline="",
    ) as out_handle:
        writer = None

        for task, path in enumerate(paths):
            with gzip.open(
                path,
                "rt",
                encoding="utf-8",
                newline="",
            ) as in_handle:
                reader = csv.DictReader(
                    in_handle
                )

                current_header = (
                    reader.fieldnames
                )

                if header is None:
                    header = current_header
                    writer = csv.DictWriter(
                        out_handle,
                        fieldnames=header,
                    )
                    writer.writeheader()

                elif current_header != header:
                    raise RuntimeError(
                        "Header mismatch in "
                        f"{path}"
                    )

                for row in reader:
                    writer.writerow(row)
                    records += 1

                    if row[
                        "array_job_id"
                    ]:
                        array_job_ids.add(
                            row[
                                "array_job_id"
                            ]
                        )

                    source_tasks.add(
                        int(
                            row[
                                "source_task"
                            ]
                        )
                    )

                    if (
                        row["method"]
                        == "R0_C0"
                        and row["dtype_name"]
                        == "float64"
                    ):
                        block = int(
                            row[
                                "block_index"
                            ]
                        )

                        replica = int(
                            row["replica"]
                        )

                        block_replicas[
                            block
                        ].add(replica)

                        matrices += 1

                        block_meta[
                            block
                        ] = {
                            "cell_id":
                                row["cell_id"],
                            "family":
                                row["family"],
                            "n":
                                int(row["n"]),
                            "alpha":
                                as_float(
                                    row["alpha"]
                                ),
                            "parameters_json":
                                row[
                                    "parameters_json"
                                ],
                        }

    temp_merged.replace(MERGED)

    if array_job_ids and len(
        array_job_ids
    ) != 1:
        raise RuntimeError(
            "Partial files come from "
            f"multiple array jobs: "
            f"{sorted(array_job_ids)}"
        )

    if source_tasks != set(
        range(999)
    ):
        missing_source = (
            set(range(999))
            - source_tasks
        )

        raise RuntimeError(
            "Recovered source-task coverage "
            "is not 0..998; missing "
            f"{sorted(missing_source)[:20]}"
        )

    if matrices != 520000:
        raise RuntimeError(
            "Expected 520000 selected "
            f"matrices, found {matrices}"
        )

    if records != 4160000:
        raise RuntimeError(
            "Expected 4160000 replay "
            f"records, found {records}"
        )

    if len(block_replicas) != 260:
        raise RuntimeError(
            "Expected 260 B1/B2 blocks, "
            f"found {len(block_replicas)}"
        )

    for block, replicas in (
        block_replicas.items()
    ):
        if len(replicas) != 2000:
            raise RuntimeError(
                f"Block {block} has "
                f"{len(replicas)} matrices, "
                "expected 2000"
            )

    blocks = sorted(
        block_replicas
    )

    block_position = {
        block: i
        for i, block in enumerate(
            blocks
        )
    }

    replica_rank = {}

    for block in blocks:
        replica_rank[block] = {
            replica: rank
            for rank, replica in enumerate(
                sorted(
                    block_replicas[
                        block
                    ]
                )
            )
        }

    nb = len(blocks)
    ns = 2000
    nc = len(COMBOS)

    # Metrics:
    # 0 D2
    # 1 R2
    # 2 AE2
    # 3 Dinf
    # 4 Rinf
    # 5 AEinf
    values = np.full(
        (
            nb,
            ns,
            nc,
            6,
        ),
        np.nan,
        dtype=np.float64,
    )

    success = np.zeros(
        (
            nb,
            ns,
            nc,
        ),
        dtype=bool,
    )

    precision_gate = np.zeros(
        (
            nb,
            ns,
            nc,
        ),
        dtype=bool,
    )

    filled = np.zeros(
        (
            nb,
            ns,
            nc,
        ),
        dtype=bool,
    )

    path_hash = np.zeros(
        (
            nb,
            ns,
            nc,
        ),
        dtype=np.uint64,
    )

    escalation_path = (
        REPORT
        / "defect_precision_gate_failures.csv"
    )

    escalation_fields = [
        "matrix_id",
        "block_index",
        "family",
        "n",
        "replica",
        "alpha",
        "method",
        "dtype_name",
        "closure_inf_ld",
        "closure_abs_allowed_ld",
        "closure_abs_ratio_bound",
        "closure_rel_sum",
        "closure_rel_R",
    ]

    precision_failures = 0

    with open(
        escalation_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as escalation_handle:
        escalation_writer = (
            csv.DictWriter(
                escalation_handle,
                fieldnames=
                    escalation_fields,
            )
        )

        escalation_writer.writeheader()

        with gzip.open(
            MERGED,
            "rt",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for row in reader:
                block = int(
                    row["block_index"]
                )

                replica = int(
                    row["replica"]
                )

                bp = block_position[
                    block
                ]

                rp = replica_rank[
                    block
                ][replica]

                ci = COMBO_INDEX[
                    (
                        row["method"],
                        row[
                            "dtype_name"
                        ],
                    )
                ]

                if filled[
                    bp,
                    rp,
                    ci,
                ]:
                    raise RuntimeError(
                        "Duplicate replay record "
                        f"for block={block}, "
                        f"replica={replica}, "
                        f"combo={COMBOS[ci]}"
                    )

                filled[
                    bp,
                    rp,
                    ci,
                ] = True

                success[
                    bp,
                    rp,
                    ci,
                ] = as_bool(
                    row["success"]
                )

                # Canonical replay precision gate:
                # use the absolute long-double closure bound.
                # closure_rel_R is retained as a diagnostic
                # only; R is a difference and is not a
                # suitable normalization scale for this gate.
                abs_ratio = float(
                    row[
                        "closure_abs_ratio_bound"
                    ]
                )

                precision_gate[
                    bp,
                    rp,
                    ci,
                ] = bool(
                    np.isfinite(abs_ratio)
                    and abs_ratio <= 1.0
                )

                if row["path_hash64"]:
                    path_hash[
                        bp,
                        rp,
                        ci,
                    ] = np.uint64(
                        int(
                            row[
                                "path_hash64"
                            ]
                        )
                    )

                values[
                    bp,
                    rp,
                    ci,
                    0,
                ] = as_float(
                    row["D_2_est"]
                )

                values[
                    bp,
                    rp,
                    ci,
                    1,
                ] = as_float(
                    row[
                        "R_2_est_ld"
                    ]
                )

                values[
                    bp,
                    rp,
                    ci,
                    2,
                ] = as_float(
                    row["AE_2_est"]
                )

                values[
                    bp,
                    rp,
                    ci,
                    3,
                ] = as_float(
                    row["D_inf_ld"]
                )

                values[
                    bp,
                    rp,
                    ci,
                    4,
                ] = as_float(
                    row["R_inf_ld"]
                )

                values[
                    bp,
                    rp,
                    ci,
                    5,
                ] = as_float(
                    row["AE_inf_ld"]
                )

                if (
                    success[
                        bp,
                        rp,
                        ci,
                    ]
                    and not precision_gate[
                        bp,
                        rp,
                        ci,
                    ]
                ):
                    precision_failures += 1

                    escalation_writer.writerow(
                        {
                            field:
                                row[field]
                            for field
                            in escalation_fields
                        }
                    )

    if int(
        np.sum(filled)
    ) != 4160000:
        raise RuntimeError(
            "Final coverage mismatch: "
            f"{int(np.sum(filled))}"
        )

    # ----------------------------------------------------------
    # Per-block quantiles.
    # ----------------------------------------------------------
    block_quantile_path = (
        REPORT
        / "defect_block_quantiles.csv"
    )

    block_fields = [
        "block_index",
        "cell_id",
        "family",
        "n",
        "alpha",
        "method",
        "dtype_name",
        "usable_count",
        "failure_count",
        "precision_gate_failure_count",
        "D2_q50",
        "D2_q90",
        "D2_q95",
        "R2_q50",
        "R2_q90",
        "R2_q95",
        "AE2_q50",
        "AE2_q90",
        "AE2_q95",
    ]

    with open(
        block_quantile_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=block_fields,
        )

        writer.writeheader()

        for block in blocks:
            bp = block_position[
                block
            ]

            meta = block_meta[
                block
            ]

            for ci, (
                method,
                dtype_name,
            ) in enumerate(COMBOS):
                usable = (
                    success[
                        bp,
                        :,
                        ci,
                    ]
                    & precision_gate[
                        bp,
                        :,
                        ci,
                    ]
                )

                d = values[
                    bp,
                    usable,
                    ci,
                    0,
                ]

                r = values[
                    bp,
                    usable,
                    ci,
                    1,
                ]

                ae = values[
                    bp,
                    usable,
                    ci,
                    2,
                ]

                writer.writerow(
                    {
                        "block_index":
                            block,
                        "cell_id":
                            meta[
                                "cell_id"
                            ],
                        "family":
                            meta[
                                "family"
                            ],
                        "n":
                            meta["n"],
                        "alpha":
                            meta[
                                "alpha"
                            ],
                        "method":
                            method,
                        "dtype_name":
                            dtype_name,
                        "usable_count":
                            int(
                                np.sum(
                                    usable
                                )
                            ),
                        "failure_count":
                            int(
                                np.sum(
                                    ~success[
                                        bp,
                                        :,
                                        ci,
                                    ]
                                )
                            ),
                        "precision_gate_failure_count":
                            int(
                                np.sum(
                                    success[
                                        bp,
                                        :,
                                        ci,
                                    ]
                                    & ~precision_gate[
                                        bp,
                                        :,
                                        ci,
                                    ]
                                )
                            ),
                        "D2_q50":
                            q(d, 0.50),
                        "D2_q90":
                            q(d, 0.90),
                        "D2_q95":
                            q(d, 0.95),
                        "R2_q50":
                            q(r, 0.50),
                        "R2_q90":
                            q(r, 0.90),
                        "R2_q95":
                            q(r, 0.95),
                        "AE2_q50":
                            q(ae, 0.50),
                        "AE2_q90":
                            q(ae, 0.90),
                        "AE2_q95":
                            q(ae, 0.95),
                    }
                )

    # ----------------------------------------------------------
    # Paired mechanistic comparison.
    # ----------------------------------------------------------
    paired_path = (
        REPORT
        / "defect_paired_summary.csv"
    )

    paired_fields = [
        "block_index",
        "cell_id",
        "family",
        "n",
        "alpha",
        "dtype_name",
        "numerator_method",
        "denominator_method",
        "common_usable",
        "positive_ratio_pairs",
        "D_ratio_q50",
        "D_ratio_q90",
        "D_ratio_q95",
        "R_ratio_q50",
        "R_ratio_q90",
        "R_ratio_q95",
        "AE_ratio_q50",
        "AE_ratio_q90",
        "AE_ratio_q95",
        "pearson_logD_logR",
        "spearman_logD_logR",
        "median_abs_log2_disagreement",
        "q95_abs_log2_disagreement",
        "fraction_disagreement_factor_gt_2",
    ]

    family_accumulator = defaultdict(
        lambda: {
            "D": [],
            "R": [],
            "AE": [],
        }
    )

    with open(
        paired_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=paired_fields,
        )

        writer.writeheader()

        for block in blocks:
            bp = block_position[
                block
            ]

            meta = block_meta[
                block
            ]

            for dtype_name in DTYPES:
                for (
                    numerator_method,
                    denominator_method,
                ) in PAIRS:
                    ni = COMBO_INDEX[
                        (
                            numerator_method,
                            dtype_name,
                        )
                    ]

                    di = COMBO_INDEX[
                        (
                            denominator_method,
                            dtype_name,
                        )
                    ]

                    common = (
                        success[
                            bp,
                            :,
                            ni,
                        ]
                        & success[
                            bp,
                            :,
                            di,
                        ]
                        & precision_gate[
                            bp,
                            :,
                            ni,
                        ]
                        & precision_gate[
                            bp,
                            :,
                            di,
                        ]
                    )

                    dn = values[
                        bp,
                        :,
                        ni,
                        0,
                    ]

                    dd = values[
                        bp,
                        :,
                        di,
                        0,
                    ]

                    rn = values[
                        bp,
                        :,
                        ni,
                        1,
                    ]

                    rd = values[
                        bp,
                        :,
                        di,
                        1,
                    ]

                    an = values[
                        bp,
                        :,
                        ni,
                        2,
                    ]

                    ad = values[
                        bp,
                        :,
                        di,
                        2,
                    ]

                    d_ratio = ratios(
                        dn,
                        dd,
                    )

                    r_ratio = ratios(
                        rn,
                        rd,
                    )

                    ae_ratio = ratios(
                        an,
                        ad,
                    )

                    d_ratio[~common] = (
                        np.nan
                    )

                    r_ratio[~common] = (
                        np.nan
                    )

                    ae_ratio[~common] = (
                        np.nan
                    )

                    corr = (
                        correlation_summary(
                            d_ratio,
                            r_ratio,
                        )
                    )

                    (
                        pearson,
                        spearman,
                        median_dis,
                        q95_dis,
                        positive_pairs,
                        factor2_fraction,
                    ) = corr

                    writer.writerow(
                        {
                            "block_index":
                                block,
                            "cell_id":
                                meta[
                                    "cell_id"
                                ],
                            "family":
                                meta[
                                    "family"
                                ],
                            "n":
                                meta["n"],
                            "alpha":
                                meta[
                                    "alpha"
                                ],
                            "dtype_name":
                                dtype_name,
                            "numerator_method":
                                numerator_method,
                            "denominator_method":
                                denominator_method,
                            "common_usable":
                                int(
                                    np.sum(
                                        common
                                    )
                                ),
                            "positive_ratio_pairs":
                                positive_pairs,
                            "D_ratio_q50":
                                q(
                                    d_ratio,
                                    0.50,
                                ),
                            "D_ratio_q90":
                                q(
                                    d_ratio,
                                    0.90,
                                ),
                            "D_ratio_q95":
                                q(
                                    d_ratio,
                                    0.95,
                                ),
                            "R_ratio_q50":
                                q(
                                    r_ratio,
                                    0.50,
                                ),
                            "R_ratio_q90":
                                q(
                                    r_ratio,
                                    0.90,
                                ),
                            "R_ratio_q95":
                                q(
                                    r_ratio,
                                    0.95,
                                ),
                            "AE_ratio_q50":
                                q(
                                    ae_ratio,
                                    0.50,
                                ),
                            "AE_ratio_q90":
                                q(
                                    ae_ratio,
                                    0.90,
                                ),
                            "AE_ratio_q95":
                                q(
                                    ae_ratio,
                                    0.95,
                                ),
                            "pearson_logD_logR":
                                pearson,
                            "spearman_logD_logR":
                                spearman,
                            "median_abs_log2_disagreement":
                                median_dis,
                            "q95_abs_log2_disagreement":
                                q95_dis,
                            "fraction_disagreement_factor_gt_2":
                                factor2_fraction,
                        }
                    )

                    key = (
                        meta["family"],
                        dtype_name,
                        numerator_method,
                        denominator_method,
                    )

                    family_accumulator[
                        key
                    ]["D"].append(
                        d_ratio[
                            np.isfinite(
                                d_ratio
                            )
                        ]
                    )

                    family_accumulator[
                        key
                    ]["R"].append(
                        r_ratio[
                            np.isfinite(
                                r_ratio
                            )
                        ]
                    )

                    family_accumulator[
                        key
                    ]["AE"].append(
                        ae_ratio[
                            np.isfinite(
                                ae_ratio
                            )
                        ]
                    )

    # ----------------------------------------------------------
    # Family-wide paired summaries.
    # ----------------------------------------------------------
    family_path = (
        REPORT
        / "defect_paired_family_summary.csv"
    )

    family_fields = [
        "family",
        "dtype_name",
        "numerator_method",
        "denominator_method",
        "pairs",
        "D_ratio_q50",
        "D_ratio_q90",
        "D_ratio_q95",
        "R_ratio_q50",
        "R_ratio_q90",
        "R_ratio_q95",
        "AE_ratio_q50",
        "AE_ratio_q90",
        "AE_ratio_q95",
        "pearson_logD_logR",
        "spearman_logD_logR",
        "median_abs_log2_disagreement",
        "q95_abs_log2_disagreement",
        "fraction_disagreement_factor_gt_2",
    ]

    with open(
        family_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=family_fields,
        )

        writer.writeheader()

        for key in sorted(
            family_accumulator
        ):
            (
                family,
                dtype_name,
                numerator_method,
                denominator_method,
            ) = key

            acc = family_accumulator[
                key
            ]

            d = (
                np.concatenate(
                    acc["D"]
                )
                if acc["D"]
                else np.array([])
            )

            r = (
                np.concatenate(
                    acc["R"]
                )
                if acc["R"]
                else np.array([])
            )

            ae = (
                np.concatenate(
                    acc["AE"]
                )
                if acc["AE"]
                else np.array([])
            )

            corr = (
                correlation_summary(
                    d,
                    r,
                )
            )

            (
                pearson,
                spearman,
                median_dis,
                q95_dis,
                positive_pairs,
                factor2_fraction,
            ) = corr

            writer.writerow(
                {
                    "family":
                        family,
                    "dtype_name":
                        dtype_name,
                    "numerator_method":
                        numerator_method,
                    "denominator_method":
                        denominator_method,
                    "pairs":
                        positive_pairs,
                    "D_ratio_q50":
                        q(d, 0.50),
                    "D_ratio_q90":
                        q(d, 0.90),
                    "D_ratio_q95":
                        q(d, 0.95),
                    "R_ratio_q50":
                        q(r, 0.50),
                    "R_ratio_q90":
                        q(r, 0.90),
                    "R_ratio_q95":
                        q(r, 0.95),
                    "AE_ratio_q50":
                        q(ae, 0.50),
                    "AE_ratio_q90":
                        q(ae, 0.90),
                    "AE_ratio_q95":
                        q(ae, 0.95),
                    "pearson_logD_logR":
                        pearson,
                    "spearman_logD_logR":
                        spearman,
                    "median_abs_log2_disagreement":
                        median_dis,
                    "q95_abs_log2_disagreement":
                        q95_dis,
                    "fraction_disagreement_factor_gt_2":
                        factor2_fraction,
                }
            )

    # ----------------------------------------------------------
    # R0_C0 / R0_C1 internal control.
    # ----------------------------------------------------------
    control_path = (
        REPORT
        / "defect_control_R0C0_vs_R0C1.csv"
    )

    control_fields = [
        "block_index",
        "family",
        "n",
        "alpha",
        "dtype_name",
        "common_usable",
        "path_hash_mismatches",
        "D_exact_equal_fraction",
        "R_exact_equal_fraction",
        "AE_exact_equal_fraction",
        "D_max_relative_difference",
        "R_max_relative_difference",
        "AE_max_relative_difference",
    ]

    total_control_path_mismatches = 0

    with open(
        control_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=control_fields,
        )

        writer.writeheader()

        for block in blocks:
            bp = block_position[
                block
            ]

            meta = block_meta[
                block
            ]

            for dtype_name in DTYPES:
                i0 = COMBO_INDEX[
                    (
                        "R0_C0",
                        dtype_name,
                    )
                ]

                i1 = COMBO_INDEX[
                    (
                        "R0_C1",
                        dtype_name,
                    )
                ]

                common = (
                    success[
                        bp,
                        :,
                        i0,
                    ]
                    & success[
                        bp,
                        :,
                        i1,
                    ]
                    & precision_gate[
                        bp,
                        :,
                        i0,
                    ]
                    & precision_gate[
                        bp,
                        :,
                        i1,
                    ]
                )

                paths_differ = (
                    path_hash[
                        bp,
                        :,
                        i0,
                    ]
                    != path_hash[
                        bp,
                        :,
                        i1,
                    ]
                )

                mismatch_count = int(
                    np.sum(
                        common
                        & paths_differ
                    )
                )

                total_control_path_mismatches += (
                    mismatch_count
                )

                selected = np.where(
                    common
                )[0]

                if selected.size:
                    d0 = values[
                        bp,
                        selected,
                        i0,
                        0,
                    ]

                    d1 = values[
                        bp,
                        selected,
                        i1,
                        0,
                    ]

                    r0 = values[
                        bp,
                        selected,
                        i0,
                        1,
                    ]

                    r1 = values[
                        bp,
                        selected,
                        i1,
                        1,
                    ]

                    a0 = values[
                        bp,
                        selected,
                        i0,
                        2,
                    ]

                    a1 = values[
                        bp,
                        selected,
                        i1,
                        2,
                    ]

                    drel = (
                        relative_difference(
                            d0,
                            d1,
                        )
                    )

                    rrel = (
                        relative_difference(
                            r0,
                            r1,
                        )
                    )

                    arel = (
                        relative_difference(
                            a0,
                            a1,
                        )
                    )

                    d_equal = float(
                        np.mean(
                            d0 == d1
                        )
                    )

                    r_equal = float(
                        np.mean(
                            r0 == r1
                        )
                    )

                    a_equal = float(
                        np.mean(
                            a0 == a1
                        )
                    )

                    dmax = float(
                        np.nanmax(drel)
                    )

                    rmax = float(
                        np.nanmax(rrel)
                    )

                    amax = float(
                        np.nanmax(arel)
                    )

                else:
                    d_equal = math.nan
                    r_equal = math.nan
                    a_equal = math.nan
                    dmax = math.nan
                    rmax = math.nan
                    amax = math.nan

                writer.writerow(
                    {
                        "block_index":
                            block,
                        "family":
                            meta[
                                "family"
                            ],
                        "n":
                            meta["n"],
                        "alpha":
                            meta[
                                "alpha"
                            ],
                        "dtype_name":
                            dtype_name,
                        "common_usable":
                            int(
                                selected.size
                            ),
                        "path_hash_mismatches":
                            mismatch_count,
                        "D_exact_equal_fraction":
                            d_equal,
                        "R_exact_equal_fraction":
                            r_equal,
                        "AE_exact_equal_fraction":
                            a_equal,
                        "D_max_relative_difference":
                            dmax,
                        "R_max_relative_difference":
                            rmax,
                        "AE_max_relative_difference":
                            amax,
                    }
                )

    hashes = (
        PROJECT
        / "config"
        / "defect_replay_source_hashes.sha256"
    )

    if hashes.exists():
        shutil.copy2(
            hashes,
            REPORT
            / hashes.name,
        )

    preflight = (
        WORK
        / "preflight.csv.gz"
    )

    if preflight.exists():
        shutil.copy2(
            preflight,
            REPORT
            / "preflight.csv.gz",
        )

    if total_control_path_mismatches:
        status = (
            "CONTROL_FAILURE"
        )
    elif precision_failures:
        status = (
            "NEEDS_MPMATH_ESCALATION"
        )
    else:
        status = (
            "DEFECT_REPLAY_COMPLETE"
        )

    summary = {
        "status": status,
        "selected_matrices": 520000,
        "method_dtype_records":
            4160000,
        "blocks": 260,
        "sample_size_per_block":
            2000,
        "methods": list(METHODS),
        "dtypes": list(DTYPES),
        "array_job_ids":
            sorted(array_job_ids),
        "precision_gate_failures":
            precision_failures,
        "R0_C0_R0_C1_path_mismatches":
            total_control_path_mismatches,
        "merged_raw_file":
            str(MERGED),
        "outputs": [
            "defect_block_quantiles.csv",
            "defect_paired_summary.csv",
            "defect_paired_family_summary.csv",
            "defect_control_R0C0_vs_R0C1.csv",
            "defect_precision_gate_failures.csv",
        ],
    }

    summary_path = (
        REPORT
        / "defect_replay_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    tar_path = (
        Path.home()
        / "castillo_defect_replay_reports.tar.gz"
    )

    with tarfile.open(
        tar_path,
        "w:gz",
    ) as archive:
        for path in sorted(
            REPORT.iterdir()
        ):
            archive.add(
                path,
                arcname=path.name,
            )

    # The merged raw file is retained as
    # the single replay record bank.
    # Remove inputs only when they are the
    # original temporary partial shards.
    # In reentrant mode paths == [MERGED],
    # so the canonical merged bank must
    # never be unlinked.
    if not reuse_merged:
        for path in paths:
            path.unlink()

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    if total_control_path_mismatches:
        raise RuntimeError(
            "R0_C0/R0_C1 path control "
            "failed; reports were retained"
        )

    if precision_failures:
        print(
            "DEFECT_REPLAY_FINAL_"
            "NEEDS_MPMATH"
        )
    else:
        print(
            "DEFECT_REPLAY_FINAL_OK"
        )


if __name__ == "__main__":
    main()
