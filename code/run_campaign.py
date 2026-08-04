#!/usr/bin/env python3
"""Sharded worker for the canonical Castillo campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from castillo import (
    CastilloResult,
    castillo_inverse,
    spectral_norm_estimate,
    unit_roundoff,
    working_dtype,
)

from families import (
    build_matrix,
    diagonal_dominance_gap,
    stable_matrix_id,
    stable_seed,
)


PARAMETER_COLUMNS = [
    "condition_number",
    "profile",
    "node_type",
    "cluster_parameter",
    "column_scaling",
    "theta",
    "epsilon",
    "decay",
    "scale_range",
    "alpha",
    "kappa",
]


RESULT_COLUMNS = [
    field.name
    for field in fields(CastilloResult)
    if field.name not in {
        "inverse",
        "row_order",
    }
]


INVERSE_FIELDS = [
    "matrix_id",
    "case_index",
    "campaign",
    "block_index",
    "cell_id",
    "family",
    "stochastic",
    "n",
    "replica",
    "seed",
    "parameters_json",
] + PARAMETER_COLUMNS + [
    "alpha_measured",
    "method",
    "dtype_name",
    "u",
    "input_representable",
    "row_score_update",
    "elapsed_seconds",
] + RESULT_COLUMNS


SOLUTION_FIELDS = [
    "matrix_id",
    "method",
    "dtype_name",
    "u",
    "rhs_type",
    "refinement_steps",
    "success",
    "failure_class",
    "failure_reason",
    "residual_norm_inf",
    "residual_norm_2",
    "relative_residual_inf",
    "relative_residual_2",
    "normwise_backward_error_inf",
    "normwise_backward_error_2",
    "componentwise_backward_error",
    "forward_error_2",
    "last_correction_norm_2",
    "elapsed_seconds",
]


def load_config(
    path: Path,
) -> Dict[str, Any]:
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    base_name = raw.get(
        "base_config"
    )

    if base_name is None:
        return raw

    base = load_config(
        path.parent / str(base_name)
    )

    merged = dict(base)

    for key, value in raw.items():
        if key != "base_config":
            merged[key] = value

    return merged


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def fingerprint(
    paths: List[Path],
    extra: Dict[str, Any],
) -> str:
    digest = hashlib.sha256()

    for path in paths:
        digest.update(
            str(path).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            path.read_bytes()
        )
        digest.update(b"\0")

    digest.update(
        canonical_json(extra).encode(
            "utf-8"
        )
    )

    return digest.hexdigest()


def rhs_seed(
    matrix_id: str,
    rhs_type: str,
) -> int:
    payload = (
        matrix_id
        + "|rhs|"
        + rhs_type
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    )


def make_exact_solutions(
    a_reference: np.ndarray,
    matrix_id: str,
    rhs_types: List[str],
) -> Dict[str, np.ndarray]:
    n = a_reference.shape[0]

    result: Dict[
        str,
        np.ndarray,
    ] = {}

    singular_vector: Optional[
        np.ndarray
    ] = None

    for rhs_type in rhs_types:
        if rhs_type == "random":
            rng = np.random.default_rng(
                rhs_seed(
                    matrix_id,
                    rhs_type,
                )
            )

            x = rng.standard_normal(n)

        elif rhs_type == "alternating":
            x = (
                (-1.0)
                ** np.arange(
                    n,
                    dtype=np.float64,
                )
            )

        elif rhs_type == "right_singular_min":
            if singular_vector is None:
                _, _, vh = np.linalg.svd(
                    np.asarray(
                        a_reference,
                        dtype=np.float64,
                    ),
                    full_matrices=False,
                )

                singular_vector = (
                    vh[-1, :].copy()
                )

            x = singular_vector.copy()

        else:
            raise ValueError(
                "unknown rhs type: {}".format(
                    rhs_type
                )
            )

        value = float(
            np.linalg.norm(x)
        )

        if (
            value == 0.0
            or not np.isfinite(value)
        ):
            raise RuntimeError(
                "invalid exact solution "
                "for {}".format(rhs_type)
            )

        result[rhs_type] = np.asarray(
            x / value,
            dtype=np.float64,
        )

    return result


def csv_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (np.bool_, bool),
    ):
        return bool(value)

    if isinstance(
        value,
        (np.integer, int),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating, float),
    ):
        x = float(value)

        if math.isnan(x):
            return "NaN"

        if math.isinf(x):
            return (
                "Infinity"
                if x > 0.0
                else "-Infinity"
            )

        return repr(x)

    return value


def assigned_replicas(
    case_start: int,
    replicas: int,
    task_id: int,
    num_tasks: int,
) -> Iterable[Tuple[int, int]]:
    """
    Balanced deterministic partition:

        global_case_index mod num_tasks = task_id.
    """
    first = (
        task_id
        - case_start
    ) % num_tasks

    for replica in range(
        first,
        replicas,
        num_tasks,
    ):
        yield (
            replica,
            case_start + replica,
        )


def matrix_metadata(
    block: Dict[str, str],
    replica: int,
    case_index: int,
    seed: int,
    matrix_id: str,
) -> Dict[str, Any]:
    parameters = json.loads(
        block["parameters_json"]
    )

    record: Dict[str, Any] = {
        "matrix_id": matrix_id,
        "case_index": case_index,
        "campaign": block["campaign"],
        "block_index": int(
            block["block_index"]
        ),
        "cell_id": block["cell_id"],
        "family": block["family"],
        "stochastic": (
            str(
                block["stochastic"]
            ).lower()
            == "true"
        ),
        "n": int(block["n"]),
        "replica": replica,
        "seed": seed,
        "parameters_json":
            block["parameters_json"],
    }

    for name in PARAMETER_COLUMNS:
        record[name] = parameters.get(
            name,
            "",
        )

    return record


def solution_metrics(
    a: np.ndarray,
    norm_a_inf: float,
    norm_a_2_est: float,
    b: np.ndarray,
    x_hat: np.ndarray,
    x_true: np.ndarray,
) -> Dict[str, float]:
    a64 = np.asarray(
        a,
        dtype=np.float64,
    )

    b64 = np.asarray(
        b,
        dtype=np.float64,
    )

    x64 = np.asarray(
        x_hat,
        dtype=np.float64,
    )

    xtrue64 = np.asarray(
        x_true,
        dtype=np.float64,
    )

    residual = (
        b64
        - a64 @ x64
    )

    residual_inf = float(
        np.linalg.norm(
            residual,
            ord=np.inf,
        )
    )

    residual_2 = float(
        np.linalg.norm(residual)
    )

    norm_x_inf = float(
        np.linalg.norm(
            x64,
            ord=np.inf,
        )
    )

    norm_x_2 = float(
        np.linalg.norm(x64)
    )

    norm_b_inf = float(
        np.linalg.norm(
            b64,
            ord=np.inf,
        )
    )

    norm_b_2 = float(
        np.linalg.norm(b64)
    )

    tiny = np.finfo(
        np.float64
    ).tiny

    denominator = (
        np.abs(a64) @ np.abs(x64)
        + np.abs(b64)
    )

    ratios = np.zeros_like(
        residual
    )

    positive = denominator > 0.0

    ratios[positive] = (
        np.abs(residual[positive])
        / denominator[positive]
    )

    impossible = (
        (~positive)
        & (residual != 0.0)
    )

    ratios[impossible] = math.inf

    return {
        "residual_norm_inf":
            residual_inf,

        "residual_norm_2":
            residual_2,

        "relative_residual_inf":
            residual_inf
            / max(norm_b_inf, tiny),

        "relative_residual_2":
            residual_2
            / max(norm_b_2, tiny),

        "normwise_backward_error_inf":
            residual_inf
            / max(
                norm_a_inf * norm_x_inf
                + norm_b_inf,
                tiny,
            ),

        "normwise_backward_error_2":
            residual_2
            / max(
                norm_a_2_est * norm_x_2
                + norm_b_2,
                tiny,
            ),

        "componentwise_backward_error":
            float(np.max(ratios)),

        "forward_error_2":
            float(
                np.linalg.norm(
                    x64 - xtrue64
                )
            )
            / max(
                float(
                    np.linalg.norm(
                        xtrue64
                    )
                ),
                tiny,
            ),
    }


def write_failed_solution_grid(
    writer: csv.DictWriter,
    matrix_id: str,
    method: str,
    dtype_name: str,
    u: float,
    rhs_types: List[str],
    refinement_steps: List[int],
    failure_class: str,
    failure_reason: str,
) -> int:
    count = 0

    for rhs_type in rhs_types:
        for steps in refinement_steps:
            writer.writerow(
                {
                    "matrix_id":
                        matrix_id,

                    "method":
                        method,

                    "dtype_name":
                        dtype_name,

                    "u":
                        repr(u),

                    "rhs_type":
                        rhs_type,

                    "refinement_steps":
                        steps,

                    "success":
                        False,

                    "failure_class":
                        failure_class,

                    "failure_reason":
                        failure_reason,

                    "residual_norm_inf":
                        "NaN",

                    "residual_norm_2":
                        "NaN",

                    "relative_residual_inf":
                        "NaN",

                    "relative_residual_2":
                        "NaN",

                    "normwise_backward_error_inf":
                        "NaN",

                    "normwise_backward_error_2":
                        "NaN",

                    "componentwise_backward_error":
                        "NaN",

                    "forward_error_2":
                        "NaN",

                    "last_correction_norm_2":
                        "NaN",

                    "elapsed_seconds":
                        "0.0",
                }
            )

            count += 1

    return count


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--execution",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--audit",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.task_id < 0
        or args.task_id >= args.num_tasks
    ):
        raise ValueError(
            "invalid task id"
        )

    config_path = Path(
        args.config
    )

    manifest_path = Path(
        args.manifest
    )

    execution_path = Path(
        args.execution
    )

    config = load_config(
        config_path
    )

    execution = json.loads(
        execution_path.read_text(
            encoding="utf-8"
        )
    )

    methods = [
        str(value)
        for value in config["methods"]
    ]

    dtypes = [
        str(value)
        for value in config["dtypes"]
    ]

    rhs_types = [
        str(value)
        for value
        in config["solution"]["rhs_types"]
    ]

    refinement_steps = sorted(
        int(value)
        for value
        in config["solution"][
            "refinement_steps"
        ]
    )

    max_refinement = max(
        refinement_steps
    )

    residual_dtype = (
        np.longdouble
        if config["solution"][
            "residual_dtype"
        ] == "longdouble"
        else np.float64
    )

    seed_namespace = str(
        config["seed_namespace"]
    )

    spectral_iterations = int(
        execution[
            "spectral_norm_iterations"
        ]
    )

    compression_level = int(
        execution[
            "gzip_compresslevel"
        ]
    )

    code_root = (
        Path(__file__).resolve().parent
    )

    fingerprint_paths = [
        config_path,
        manifest_path,
        execution_path,
        code_root / "families.py",
        code_root / "castillo.py",
        code_root / "run_campaign.py",
    ]

    campaign_fingerprint = fingerprint(
        fingerprint_paths,
        {
            "num_tasks":
                args.num_tasks,

            "audit":
                bool(args.audit),
        },
    )

    run_fingerprint = hashlib.sha256(
        (
            campaign_fingerprint
            + "|task_id="
            + str(args.task_id)
        ).encode("utf-8")
    ).hexdigest()

    outdir = Path(args.outdir)

    inverse_dir = (
        outdir / "inverse"
    )

    solution_dir = (
        outdir / "solution"
    )

    status_dir = (
        outdir / "status"
    )

    tmp_dir = (
        outdir / "tmp"
    )

    for directory in (
        inverse_dir,
        solution_dir,
        status_dir,
        tmp_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    tag = "{:04d}".format(
        args.task_id
    )

    inverse_path = (
        inverse_dir
        / (
            "inverse_"
            + tag
            + ".csv.gz"
        )
    )

    solution_path = (
        solution_dir
        / (
            "solution_"
            + tag
            + ".csv.gz"
        )
    )

    status_path = (
        status_dir
        / (
            "task_"
            + tag
            + ".json"
        )
    )

    if (
        status_path.exists()
        and not args.force
    ):
        old = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            old.get("status")
            == "TASK_OK"
            and old.get(
                "run_fingerprint"
            ) == run_fingerprint
            and inverse_path.exists()
            and solution_path.exists()
        ):
            print(
                json.dumps(
                    {
                        "status":
                            "TASK_ALREADY_COMPLETE",

                        "task_id":
                            args.task_id,
                    },
                    indent=2,
                )
            )
            return

    tmp_suffix = (
        ".tmp."
        + str(os.getpid())
        + "."
        + str(int(time.time()))
    )

    inverse_tmp = (
        tmp_dir
        / (
            inverse_path.name
            + tmp_suffix
        )
    )

    solution_tmp = (
        tmp_dir
        / (
            solution_path.name
            + tmp_suffix
        )
    )

    matrix_cases = 0
    inverse_records = 0
    solution_records = 0

    inverse_failures = 0
    solution_failures = 0

    failure_counts: Counter = Counter()
    family_case_counts: Counter = Counter()
    dtype_method_counts: Counter = Counter()

    first_case: Optional[int] = None
    last_case: Optional[int] = None

    start_all = time.perf_counter()

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as manifest_handle, \
            gzip.open(
                str(inverse_tmp),
                mode="wt",
                encoding="utf-8",
                newline="",
                compresslevel=compression_level,
            ) as inverse_handle, \
            gzip.open(
                str(solution_tmp),
                mode="wt",
                encoding="utf-8",
                newline="",
                compresslevel=compression_level,
            ) as solution_handle:

        manifest_reader = csv.DictReader(
            manifest_handle
        )

        inverse_writer = csv.DictWriter(
            inverse_handle,
            fieldnames=INVERSE_FIELDS,
            extrasaction="ignore",
        )

        solution_writer = csv.DictWriter(
            solution_handle,
            fieldnames=SOLUTION_FIELDS,
            extrasaction="ignore",
        )

        inverse_writer.writeheader()
        solution_writer.writeheader()

        for block in manifest_reader:
            case_start = int(
                block["case_start"]
            )

            replicas = int(
                block["replicas"]
            )

            n = int(block["n"])

            family = str(
                block["family"]
            )

            parameters = json.loads(
                block["parameters_json"]
            )

            for replica, case_index in assigned_replicas(
                case_start,
                replicas,
                args.task_id,
                args.num_tasks,
            ):
                matrix_cases += 1
                family_case_counts[family] += 1

                first_case = (
                    case_index
                    if first_case is None
                    else min(
                        first_case,
                        case_index,
                    )
                )

                last_case = (
                    case_index
                    if last_case is None
                    else max(
                        last_case,
                        case_index,
                    )
                )

                seed = stable_seed(
                    seed_namespace,
                    block["cell_id"],
                    n,
                    replica,
                )

                matrix_id = stable_matrix_id(
                    seed_namespace,
                    block["cell_id"],
                    n,
                    replica,
                )

                base = matrix_metadata(
                    block,
                    replica,
                    case_index,
                    seed,
                    matrix_id,
                )

                try:
                    a_reference = build_matrix(
                        family=family,
                        n=n,
                        parameters=parameters,
                        seed=seed,
                    )

                    if np.isnan(
                        a_reference
                    ).any():
                        raise FloatingPointError(
                            "reference matrix "
                            "contains NaN"
                        )

                except Exception as exc:
                    reason = (
                        "matrix_build_failed:"
                        + repr(exc).replace(
                            ",",
                            ";",
                        )
                    )

                    for dtype_name in dtypes:
                        u = unit_roundoff(
                            dtype_name
                        )

                        for method in methods:
                            record = dict(base)

                            record.update(
                                {
                                    "alpha_measured":
                                        "NaN",

                                    "method":
                                        method,

                                    "dtype_name":
                                        dtype_name,

                                    "u":
                                        repr(u),

                                    "input_representable":
                                        False,

                                    "row_score_update":
                                        "recursive_AV_for_R1_R2",

                                    "elapsed_seconds":
                                        "0.0",

                                    "success":
                                        False,

                                    "failure_class":
                                        "input_build_failure",

                                    "failure_reason":
                                        reason,

                                    "failure_step":
                                        -1,
                                }
                            )

                            inverse_writer.writerow(
                                record
                            )

                            inverse_records += 1
                            inverse_failures += 1

                            failure_counts[
                                (
                                    family,
                                    dtype_name,
                                    method,
                                    "input_build_failure",
                                    reason,
                                )
                            ] += 1

                            written = (
                                write_failed_solution_grid(
                                    solution_writer,
                                    matrix_id,
                                    method,
                                    dtype_name,
                                    u,
                                    rhs_types,
                                    refinement_steps,
                                    "input_build_failure",
                                    reason,
                                )
                            )

                            solution_records += written
                            solution_failures += written

                    continue

                exact_solutions: Dict[
                    str,
                    np.ndarray,
                ] = {}

                rhs_build_failures: Dict[
                    str,
                    str,
                ] = {}

                for rhs_type in rhs_types:
                    try:
                        exact_solutions.update(
                            make_exact_solutions(
                                a_reference,
                                matrix_id,
                                [rhs_type],
                            )
                        )

                    except Exception as exc:
                        rhs_build_failures[
                            rhs_type
                        ] = (
                            "rhs_reference_build_failed:"
                            + repr(exc).replace(
                                ",",
                                ";",
                            )
                        )

                alpha_measured: Any = ""

                if family.startswith(
                    "row_diagonal_dominant_"
                ):
                    alpha_measured = (
                        diagonal_dominance_gap(
                            a_reference
                        )
                    )

                for dtype_name in dtypes:
                    dtype = working_dtype(
                        dtype_name
                    )

                    u = unit_roundoff(
                        dtype_name
                    )

                    with np.errstate(
                        over="ignore",
                        under="ignore",
                        invalid="ignore",
                        divide="ignore",
                    ):
                        a_work = np.asarray(
                            a_reference,
                            dtype=dtype,
                        )

                    representable = bool(
                        np.all(
                            np.isfinite(a_work)
                        )
                    )

                    norm_a_inf = (
                        float(
                            np.linalg.norm(
                                a_work.astype(
                                    np.float64
                                ),
                                ord=np.inf,
                            )
                        )
                        if representable
                        else math.inf
                    )

                    norm_a_2_est = (
                        spectral_norm_estimate(
                            a_work,
                            spectral_iterations,
                        )
                        if representable
                        else math.inf
                    )

                    for method in methods:
                        dtype_method_counts[
                            (
                                dtype_name,
                                method,
                            )
                        ] += 1

                        method_start = (
                            time.perf_counter()
                        )

                        if not representable:
                            failure_class = (
                                "input_not_representable"
                            )

                            failure_reason = (
                                "input_nonfinite_in_"
                                "working_precision"
                            )

                            record = dict(base)

                            record.update(
                                {
                                    "alpha_measured":
                                        alpha_measured,

                                    "method":
                                        method,

                                    "dtype_name":
                                        dtype_name,

                                    "u":
                                        repr(u),

                                    "input_representable":
                                        False,

                                    "row_score_update":
                                        "recursive_AV_for_R1_R2",

                                    "elapsed_seconds":
                                        repr(
                                            time.perf_counter()
                                            - method_start
                                        ),

                                    "success":
                                        False,

                                    "failure_class":
                                        failure_class,

                                    "failure_reason":
                                        failure_reason,

                                    "failure_step":
                                        -1,
                                }
                            )

                            inverse_writer.writerow(
                                record
                            )

                            inverse_records += 1
                            inverse_failures += 1

                            failure_counts[
                                (
                                    family,
                                    dtype_name,
                                    method,
                                    failure_class,
                                    failure_reason,
                                )
                            ] += 1

                            written = (
                                write_failed_solution_grid(
                                    solution_writer,
                                    matrix_id,
                                    method,
                                    dtype_name,
                                    u,
                                    rhs_types,
                                    refinement_steps,
                                    failure_class,
                                    failure_reason,
                                )
                            )

                            solution_records += written
                            solution_failures += written
                            continue

                        try:
                            result = castillo_inverse(
                                a_input=a_work,
                                method=method,
                                dtype_name=dtype_name,
                                norm_A_2_est=norm_a_2_est,
                                spectral_iterations=spectral_iterations,
                                audit=bool(args.audit),
                            )

                        except Exception as exc:
                            failure_class = (
                                "kernel_exception"
                            )

                            failure_reason = (
                                repr(exc).replace(
                                    ",",
                                    ";",
                                )
                            )

                            record = dict(base)

                            record.update(
                                {
                                    "alpha_measured":
                                        alpha_measured,

                                    "method":
                                        method,

                                    "dtype_name":
                                        dtype_name,

                                    "u":
                                        repr(u),

                                    "input_representable":
                                        True,

                                    "row_score_update":
                                        "recursive_AV_for_R1_R2",

                                    "elapsed_seconds":
                                        repr(
                                            time.perf_counter()
                                            - method_start
                                        ),

                                    "success":
                                        False,

                                    "failure_class":
                                        failure_class,

                                    "failure_reason":
                                        failure_reason,

                                    "failure_step":
                                        -1,
                                }
                            )

                            inverse_writer.writerow(
                                record
                            )

                            inverse_records += 1
                            inverse_failures += 1

                            failure_counts[
                                (
                                    family,
                                    dtype_name,
                                    method,
                                    failure_class,
                                    failure_reason,
                                )
                            ] += 1

                            written = (
                                write_failed_solution_grid(
                                    solution_writer,
                                    matrix_id,
                                    method,
                                    dtype_name,
                                    u,
                                    rhs_types,
                                    refinement_steps,
                                    failure_class,
                                    failure_reason,
                                )
                            )

                            solution_records += written
                            solution_failures += written
                            continue

                        record = dict(base)

                        record.update(
                            {
                                "alpha_measured":
                                    alpha_measured,

                                "method":
                                    method,

                                "dtype_name":
                                    dtype_name,

                                "u":
                                    repr(u),

                                "input_representable":
                                    True,

                                "row_score_update":
                                    "recursive_AV_for_R1_R2",

                                "elapsed_seconds":
                                    repr(
                                        time.perf_counter()
                                        - method_start
                                    ),
                            }
                        )

                        for name in RESULT_COLUMNS:
                            record[name] = csv_value(
                                getattr(
                                    result,
                                    name,
                                )
                            )

                        inverse_writer.writerow(
                            record
                        )

                        inverse_records += 1

                        if not result.success:
                            inverse_failures += 1

                            failure_counts[
                                (
                                    family,
                                    dtype_name,
                                    method,
                                    result.failure_class,
                                    result.failure_reason,
                                )
                            ] += 1

                            written = (
                                write_failed_solution_grid(
                                    solution_writer,
                                    matrix_id,
                                    method,
                                    dtype_name,
                                    u,
                                    rhs_types,
                                    refinement_steps,
                                    result.failure_class,
                                    result.failure_reason,
                                )
                            )

                            solution_records += written
                            solution_failures += written
                            continue

                        for rhs_type in rhs_types:
                            solution_start = (
                                time.perf_counter()
                            )

                            if rhs_type in rhs_build_failures:
                                written = (
                                    write_failed_solution_grid(
                                        solution_writer,
                                        matrix_id,
                                        method,
                                        dtype_name,
                                        u,
                                        [rhs_type],
                                        refinement_steps,
                                        "rhs_reference_failure",
                                        rhs_build_failures[
                                            rhs_type
                                        ],
                                    )
                                )

                                solution_records += written
                                solution_failures += written
                                continue

                            x_true = np.asarray(
                                exact_solutions[
                                    rhs_type
                                ],
                                dtype=dtype,
                            )

                            with np.errstate(
                                over="ignore",
                                under="ignore",
                                invalid="ignore",
                                divide="ignore",
                            ):
                                b = np.asarray(
                                    a_work @ x_true,
                                    dtype=dtype,
                                )

                            if not np.all(
                                np.isfinite(b)
                            ):
                                written = (
                                    write_failed_solution_grid(
                                        solution_writer,
                                        matrix_id,
                                        method,
                                        dtype_name,
                                        u,
                                        [rhs_type],
                                        refinement_steps,
                                        "rhs_nonfinite",
                                        "nonfinite_rhs",
                                    )
                                )

                                solution_records += written
                                solution_failures += written
                                continue

                            b_permuted = (
                                b[result.row_order]
                            )

                            with np.errstate(
                                over="ignore",
                                under="ignore",
                                invalid="ignore",
                                divide="ignore",
                            ):
                                x_current = np.asarray(
                                    result.inverse
                                    @ b_permuted,
                                    dtype=dtype,
                                )

                            last_correction_norm = 0.0

                            for current_step in range(
                                max_refinement + 1
                            ):
                                if (
                                    current_step
                                    in refinement_steps
                                ):
                                    solution_record: Dict[
                                        str,
                                        Any,
                                    ] = {
                                        "matrix_id":
                                            matrix_id,

                                        "method":
                                            method,

                                        "dtype_name":
                                            dtype_name,

                                        "u":
                                            repr(u),

                                        "rhs_type":
                                            rhs_type,

                                        "refinement_steps":
                                            current_step,

                                        "elapsed_seconds":
                                            repr(
                                                time.perf_counter()
                                                - solution_start
                                            ),
                                    }

                                    if np.all(
                                        np.isfinite(
                                            x_current
                                        )
                                    ):
                                        metrics = solution_metrics(
                                            a_work,
                                            norm_a_inf,
                                            norm_a_2_est,
                                            b,
                                            x_current,
                                            x_true,
                                        )

                                        solution_record.update(
                                            metrics
                                        )

                                        solution_record.update(
                                            {
                                                "success":
                                                    True,

                                                "failure_class":
                                                    "completed",

                                                "failure_reason":
                                                    "",

                                                "last_correction_norm_2":
                                                    last_correction_norm,
                                            }
                                        )

                                    else:
                                        solution_record.update(
                                            {
                                                "success":
                                                    False,

                                                "failure_class":
                                                    "solution_nonfinite",

                                                "failure_reason":
                                                    "nonfinite_solution",

                                                "residual_norm_inf":
                                                    "NaN",

                                                "residual_norm_2":
                                                    "NaN",

                                                "relative_residual_inf":
                                                    "NaN",

                                                "relative_residual_2":
                                                    "NaN",

                                                "normwise_backward_error_inf":
                                                    "NaN",

                                                "normwise_backward_error_2":
                                                    "NaN",

                                                "componentwise_backward_error":
                                                    "NaN",

                                                "forward_error_2":
                                                    "NaN",

                                                "last_correction_norm_2":
                                                    "NaN",
                                            }
                                        )

                                        solution_failures += 1

                                    solution_writer.writerow(
                                        {
                                            key: csv_value(
                                                value
                                            )
                                            for key, value
                                            in solution_record.items()
                                        }
                                    )

                                    solution_records += 1

                                if (
                                    current_step
                                    == max_refinement
                                ):
                                    break

                                if not np.all(
                                    np.isfinite(
                                        x_current
                                    )
                                ):
                                    continue

                                a_high = np.asarray(
                                    a_work,
                                    dtype=residual_dtype,
                                )

                                b_high = np.asarray(
                                    b,
                                    dtype=residual_dtype,
                                )

                                x_high = np.asarray(
                                    x_current,
                                    dtype=residual_dtype,
                                )

                                residual_high = (
                                    b_high
                                    - a_high @ x_high
                                )

                                residual_work = np.asarray(
                                    residual_high,
                                    dtype=dtype,
                                )

                                residual_permuted = (
                                    residual_work[
                                        result.row_order
                                    ]
                                )

                                with np.errstate(
                                    over="ignore",
                                    under="ignore",
                                    invalid="ignore",
                                    divide="ignore",
                                ):
                                    correction = np.asarray(
                                        result.inverse
                                        @ residual_permuted,
                                        dtype=dtype,
                                    )

                                    x_current = np.asarray(
                                        x_current
                                        + correction,
                                        dtype=dtype,
                                    )

                                last_correction_norm = float(
                                    np.linalg.norm(
                                        correction.astype(
                                            np.float64
                                        )
                                    )
                                )

    os.replace(
        str(inverse_tmp),
        str(inverse_path),
    )

    os.replace(
        str(solution_tmp),
        str(solution_path),
    )

    status = {
        "status":
            "TASK_OK",

        "task_id":
            args.task_id,

        "num_tasks":
            args.num_tasks,

        "run_fingerprint":
            run_fingerprint,

        "campaign_fingerprint":
            campaign_fingerprint,

        "audit":
            bool(args.audit),

        "matrix_cases":
            matrix_cases,

        "inverse_records":
            inverse_records,

        "solution_records":
            solution_records,

        "inverse_failures":
            inverse_failures,

        "solution_failures":
            solution_failures,

        "first_case_index":
            first_case,

        "last_case_index":
            last_case,

        "family_case_counts": {
            str(key): int(value)
            for key, value
            in sorted(
                family_case_counts.items()
            )
        },

        "dtype_method_counts": {
            "|".join(key): int(value)
            for key, value
            in sorted(
                dtype_method_counts.items()
            )
        },

        "failure_counts": {
            "|".join(key): int(value)
            for key, value
            in sorted(
                failure_counts.items()
            )
        },

        "inverse_path":
            str(inverse_path),

        "solution_path":
            str(solution_path),

        "inverse_size_bytes":
            inverse_path.stat().st_size,

        "solution_size_bytes":
            solution_path.stat().st_size,

        "inverse_sha256":
            sha256_file(inverse_path),

        "solution_sha256":
            sha256_file(solution_path),

        "elapsed_seconds":
            time.perf_counter()
            - start_all,

        "spectral_norm_iterations":
            spectral_iterations,

        "row_assignment":
            "global_case_index_mod_num_tasks",
    }

    tmp_status = status_path.with_suffix(
        ".json.tmp"
    )

    tmp_status.write_text(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        str(tmp_status),
        str(status_path),
    )

    print(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
    )

    print("TASK_OK")


if __name__ == "__main__":
    main()
