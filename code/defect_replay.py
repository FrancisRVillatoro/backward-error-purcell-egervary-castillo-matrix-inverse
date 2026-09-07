#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT = Path.home() / "castillo_stability_campaign"
SCRATCH = (
    Path.home()
    / "fscratch"
    / "castillo_stability_campaign"
)

CODE = PROJECT / "code"

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import castillo
import families


EXPECTED_CASTILLO_SHA256 = (
    "1497d253ef10c7cfe95eaf2594774e99e"
    "d2c152b9d153486d8b1369c2dc7c1db"
)

DD_FAMILIES = {
    "row_diagonal_dominant_random",
    "row_diagonal_dominant_stochastic",
}

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

EXPECTED_COMBINATIONS = {
    (method, dtype_name)
    for method in METHODS
    for dtype_name in DTYPES
}

OUTPUT_FIELDS = [
    "array_job_id",
    "source_task",
    "matrix_id",
    "case_index",
    "block_index",
    "cell_id",
    "family",
    "n",
    "replica",
    "seed",
    "alpha",
    "parameters_json",
    "dtype_name",
    "method",
    "success",
    "failure_class",
    "failure_reason",
    "failure_step",
    "n_interchanges",
    "n_row_interchanges",
    "path_hash64",
    "norm_A_inf",
    "norm_A_2_est",
    "gamma_inf",
    "D_inf_ld",
    "R_inf_ld",
    "AE_inf_ld",
    "E_inf_ld",
    "closure_inf_ld",
    "closure_abs_allowed_ld",
    "closure_abs_ratio_bound",
    "D_2_est",
    "R_2_est_ld",
    "AE_2_est",
    "D_scaled_2_est",
    "R_scaled_2_est_ld",
    "AE_scaled_2_est",
    "canonical_R_inf",
    "canonical_R_2_est",
    "expected_R_inf",
    "expected_R_2_est",
    "source_match_rel_Rinf",
    "source_match_rel_R2",
    "source_match",
    "R_unpermuted_inf_ld",
    "closure_rel_sum",
    "closure_rel_R",
    "precision_gate_pass",
    "elapsed_seconds",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def check_canonical_source() -> None:
    actual = sha256_file(CODE / "castillo.py")

    if actual != EXPECTED_CASTILLO_SHA256:
        raise RuntimeError(
            "Canonical castillo.py hash mismatch: "
            f"{actual}"
        )


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

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


def as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def relerr(a: float, b: float) -> float:
    if not (
        math.isfinite(a)
        and math.isfinite(b)
    ):
        return math.inf

    if a == b:
        return 0.0

    scale = max(
        abs(a),
        abs(b),
        1.0e-300,
    )

    return abs(a - b) / scale


def norm_inf_ld(a: np.ndarray) -> float:
    x = np.asarray(
        a,
        dtype=np.longdouble,
    )

    if x.size == 0:
        return 0.0

    rows = np.sum(
        np.abs(x),
        axis=1,
        dtype=np.longdouble,
    )

    return float(np.max(rows))


def spectral_estimate(a: np.ndarray) -> float:
    x64 = np.asarray(
        a,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(x64)):
        return math.nan

    return float(
        castillo.spectral_norm_estimate(
            x64,
            6,
        )
    )


def selection_pairs(path: Path) -> set[tuple[int, int]]:
    """
    Recover the selected (block_index, replica) pairs.

    The existing theory audit uses a mapping keyed by exactly
    (block, replica).  This reader deliberately accepts several
    ordinary npz encodings so that it does not depend on one
    spelling of the saved array names.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in data.files
        }

    # ----------------------------------------------------------
    # 1. Separate arrays containing block and replica.
    # ----------------------------------------------------------
    block_candidates = []
    replica_candidates = []

    for key, arr in arrays.items():
        low = key.lower()

        if "block" in low:
            block_candidates.append(
                (key, arr)
            )

        if (
            "replica" in low
            or low.endswith("reps")
            or low == "reps"
        ):
            replica_candidates.append(
                (key, arr)
            )

    for _, blocks in block_candidates:
        for _, replicas in replica_candidates:
            b = np.asarray(blocks)
            r = np.asarray(replicas)

            if (
                b.ndim == 1
                and r.ndim == 1
                and b.size == r.size
                and b.size > 0
            ):
                return {
                    (int(bb), int(rr))
                    for bb, rr in zip(
                        b.tolist(),
                        r.tolist(),
                    )
                }

            if (
                b.ndim == 1
                and r.ndim == 2
                and b.size == r.shape[0]
                and r.size > 0
            ):
                pairs = set()

                for i, bb in enumerate(
                    b.tolist()
                ):
                    for rr in r[i, :].tolist():
                        if int(rr) >= 0:
                            pairs.add(
                                (int(bb), int(rr))
                            )

                if pairs:
                    return pairs

            if (
                b.shape == r.shape
                and b.ndim >= 1
                and b.size > 0
            ):
                return {
                    (int(bb), int(rr))
                    for bb, rr in zip(
                        b.reshape(-1).tolist(),
                        r.reshape(-1).tolist(),
                    )
                    if int(rr) >= 0
                }

    # ----------------------------------------------------------
    # 2. Structured array.
    # ----------------------------------------------------------
    for arr in arrays.values():
        names = (
            arr.dtype.names
            if arr.dtype.names
            else ()
        )

        if not names:
            continue

        block_name = next(
            (
                name
                for name in names
                if "block" in name.lower()
            ),
            None,
        )

        replica_name = next(
            (
                name
                for name in names
                if "replica" in name.lower()
                or name.lower() == "rep"
            ),
            None,
        )

        if (
            block_name is not None
            and replica_name is not None
        ):
            return {
                (
                    int(row[block_name]),
                    int(row[replica_name]),
                )
                for row in arr.reshape(-1)
            }

    # ----------------------------------------------------------
    # 3. Pickled dictionary or list of tuple keys.
    # ----------------------------------------------------------
    for arr in arrays.values():
        if arr.dtype != object:
            continue

        objects = arr.reshape(-1).tolist()

        for obj in objects:
            if isinstance(obj, dict):
                pairs = set()

                for key in obj.keys():
                    if (
                        isinstance(
                            key,
                            (tuple, list),
                        )
                        and len(key) >= 2
                    ):
                        pairs.add(
                            (
                                int(key[0]),
                                int(key[1]),
                            )
                        )

                if pairs:
                    return pairs

        pairs = set()

        for obj in objects:
            if (
                isinstance(
                    obj,
                    (tuple, list, np.ndarray),
                )
                and len(obj) >= 2
            ):
                try:
                    pairs.add(
                        (
                            int(obj[0]),
                            int(obj[1]),
                        )
                    )
                except Exception:
                    pass

        if pairs:
            return pairs

    # ----------------------------------------------------------
    # 4. Generic Nx2/Nx3 array whose name indicates selected
    #    keys or pairs.
    # ----------------------------------------------------------
    for key, arr in arrays.items():
        low = key.lower()

        if not (
            "select" in low
            or "key" in low
            or "pair" in low
        ):
            continue

        a = np.asarray(arr)

        if (
            a.ndim == 2
            and a.shape[1] >= 2
            and a.shape[0] > 0
        ):
            try:
                return {
                    (
                        int(row[0]),
                        int(row[1]),
                    )
                    for row in a
                }
            except Exception:
                pass

    description = {
        key: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        for key, arr in arrays.items()
    }

    raise RuntimeError(
        "Unsupported selection npz layout: "
        + json.dumps(
            description,
            sort_keys=True,
        )
    )


def expected_from_row(
    row: dict[str, str],
) -> dict[str, str]:
    fields = [
        "success",
        "failure_class",
        "failure_reason",
        "failure_step",
        "n_interchanges",
        "n_row_interchanges",
        "norm_A_inf",
        "norm_A_2_est",
        "gamma_inf",
        "right_inverse_defect_inf",
        "right_inverse_defect_2_est",
    ]

    return {
        field: row.get(field, "")
        for field in fields
    }


def meta_from_row(
    row: dict[str, str],
    source_task: int,
) -> dict:
    return {
        "source_task": source_task,
        "matrix_id": row["matrix_id"],
        "case_index": int(
            row["case_index"]
        ),
        "block_index": int(
            row["block_index"]
        ),
        "cell_id": row["cell_id"],
        "family": row["family"],
        "n": int(row["n"]),
        "replica": int(row["replica"]),
        "seed": int(row["seed"]),
        "alpha": as_float(
            row.get("alpha")
        ),
        "parameters_json":
            row["parameters_json"],
        "expected": {},
    }


def collect_selected_cases(
    source_task: int,
) -> list[dict]:
    selection_path = (
        SCRATCH
        / "results"
        / "analysis"
        / "selection"
        / f"task_{source_task:04d}.npz"
    )

    recovered_path = (
        SCRATCH
        / "results"
        / "recovered"
        / "inverse"
        / f"inverse_{source_task:04d}.csv.gz"
    )

    selected = selection_pairs(
        selection_path
    )

    cases: dict[
        tuple[int, int],
        dict,
    ] = {}

    with gzip.open(
        recovered_path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            family = row["family"]

            if family not in DD_FAMILIES:
                continue

            key = (
                int(row["block_index"]),
                int(row["replica"]),
            )

            if key not in selected:
                continue

            method = row["method"]
            dtype_name = row["dtype_name"]

            if (
                method,
                dtype_name,
            ) not in EXPECTED_COMBINATIONS:
                continue

            if key not in cases:
                cases[key] = meta_from_row(
                    row,
                    source_task,
                )

            case = cases[key]

            if (
                case["matrix_id"]
                != row["matrix_id"]
            ):
                raise RuntimeError(
                    "matrix_id changed within "
                    f"selected key {key}"
                )

            combination = (
                method,
                dtype_name,
            )

            if (
                combination
                in case["expected"]
            ):
                raise RuntimeError(
                    "Duplicate recovered inverse "
                    f"record for {key} "
                    f"{combination}"
                )

            case["expected"][
                combination
            ] = expected_from_row(row)

    result = sorted(
        cases.values(),
        key=lambda item: (
            item["block_index"],
            item["replica"],
        ),
    )

    for case in result:
        found = set(
            case["expected"].keys()
        )

        if found != EXPECTED_COMBINATIONS:
            raise RuntimeError(
                "Incomplete recovered method/dtype "
                "group for "
                f"block={case['block_index']} "
                f"replica={case['replica']}: "
                f"{sorted(found)}"
            )

    return result


class Trace:
    def __init__(self):
        self.steps = {}
        self.rows = {}


def run_canonical_with_trace(
    a: np.ndarray,
    method: str,
    dtype_name: str,
    norm_A_2_est: float,
):
    trace = Trace()

    original_right = (
        castillo.compact_right_update
    )

    original_row = (
        castillo.choose_row_from_table
    )

    def traced_right(
        v,
        j,
        p,
        b_row,
    ):
        jj = int(j)

        if jj not in trace.steps:
            trace.steps[jj] = (
                int(p),
                np.asarray(
                    b_row
                ).copy(),
            )

        return original_right(
            v,
            j,
            p,
            b_row,
        )

    def traced_row(
        transformed_rows,
        a_local,
        row_order,
        v,
        j,
        row_rule,
    ):
        position = original_row(
            transformed_rows,
            a_local,
            row_order,
            v,
            j,
            row_rule,
        )

        jj = int(j)

        trace.rows[jj] = int(
            row_order[int(position)]
        )

        return position

    castillo.compact_right_update = (
        traced_right
    )

    castillo.choose_row_from_table = (
        traced_row
    )

    try:
        result = castillo.castillo_inverse(
            a_input=a.copy(),
            method=method,
            dtype_name=dtype_name,
            norm_A_2_est=norm_A_2_est,
            spectral_iterations=6,
            audit=False,
        )
    finally:
        castillo.compact_right_update = (
            original_right
        )

        castillo.choose_row_from_table = (
            original_row
        )

    return result, trace


def get_inverse(result):
    for name in (
        "inverse",
        "v",
        "tableau",
        "V",
    ):
        if hasattr(result, name):
            value = getattr(
                result,
                name,
            )

            if value is not None:
                return np.asarray(value)

    raise RuntimeError(
        "Cannot locate final inverse/tableau "
        f"in CastilloResult fields: "
        f"{sorted(vars(result).keys())}"
    )


def processed_rows(
    trace: Trace,
    method: str,
    n: int,
) -> list[int]:
    if len(trace.rows) == n:
        return [
            int(trace.rows[j])
            for j in range(n)
        ]

    if (
        method in {"R0_C0", "R0_C1"}
        and len(trace.rows) == 0
    ):
        return list(range(n))

    raise RuntimeError(
        f"Captured {len(trace.rows)} "
        f"row decisions for {method}, "
        f"expected {n}"
    )


def reconstruct_vtilde(
    trace: Trace,
    n: int,
) -> np.ndarray:
    if len(trace.steps) != n:
        raise RuntimeError(
            f"Captured {len(trace.steps)} "
            f"transformations, expected {n}"
        )

    v = np.eye(
        n,
        dtype=np.longdouble,
    )

    for j in range(n):
        p, b_work = trace.steps[j]

        if p != j:
            temp = v[:, j].copy()
            v[:, j] = v[:, p]
            v[:, p] = temp

        pivot_column = (
            v[:, j].copy()
        )

        b = np.asarray(
            b_work,
            dtype=np.longdouble,
        )

        # V <- (V P) B.
        #
        # B is identity except row j,
        # which is the already-rounded b_row.
        v += (
            pivot_column[:, None]
            * b[None, :]
        )

        # For column j the identity
        # contribution is absent.
        v[:, j] -= pivot_column

    return v


def path_hash64(
    trace: Trace,
    rows: list[int],
    n: int,
) -> int:
    h = hashlib.blake2b(
        digest_size=8
    )

    for row_index in rows:
        h.update(
            int(row_index).to_bytes(
                8,
                byteorder="little",
                signed=True,
            )
        )

    for j in range(n):
        p, b = trace.steps[j]

        h.update(
            int(j).to_bytes(
                8,
                byteorder="little",
                signed=True,
            )
        )

        h.update(
            int(p).to_bytes(
                8,
                byteorder="little",
                signed=True,
            )
        )

        h.update(
            np.asarray(b).dtype.str.encode(
                "ascii"
            )
        )

        h.update(
            np.ascontiguousarray(
                b
            ).tobytes()
        )

    return int.from_bytes(
        h.digest(),
        byteorder="little",
        signed=False,
    )


def build_working_matrix(
    case: dict,
    dtype_name: str,
) -> np.ndarray:
    parameters = json.loads(
        case["parameters_json"]
    )

    # Reproduce the canonical campaign construction:
    # build the reference matrix once in float64, then
    # cast it to the requested working arithmetic.
    a_reference = families.build_matrix(
        family=case["family"],
        n=int(case["n"]),
        parameters=parameters,
        seed=int(case["seed"]),
    )

    return np.asarray(
        a_reference,
        dtype=castillo.working_dtype(
            dtype_name
        ),
    )


def blank_output(
    case: dict,
    method: str,
    dtype_name: str,
) -> dict:
    row = {
        field: ""
        for field in OUTPUT_FIELDS
    }

    row.update(
        {
            "array_job_id":
                os.environ.get(
                    "SLURM_ARRAY_JOB_ID",
                    os.environ.get(
                        "SLURM_JOB_ID",
                        "",
                    ),
                ),
            "source_task":
                case["source_task"],
            "matrix_id":
                case["matrix_id"],
            "case_index":
                case["case_index"],
            "block_index":
                case["block_index"],
            "cell_id":
                case["cell_id"],
            "family":
                case["family"],
            "n":
                case["n"],
            "replica":
                case["replica"],
            "seed":
                case["seed"],
            "alpha":
                case["alpha"],
            "parameters_json":
                case["parameters_json"],
            "dtype_name":
                dtype_name,
            "method":
                method,
        }
    )

    return row


def replay_one(
    case: dict,
    a: np.ndarray,
    method: str,
    dtype_name: str,
) -> dict:
    start = time.perf_counter()

    expected = case["expected"][
        (method, dtype_name)
    ]

    expected_success = as_bool(
        expected["success"]
    )

    # New replay, same matrix and same production algorithm.
    # Recompute the spectral estimate from the current working-
    # precision matrix, exactly as run_campaign.py does.
    norm_A_2_est = float(
        castillo.spectral_norm_estimate(
            a,
            6,
        )
    )

    if not math.isfinite(
        norm_A_2_est
    ):
        raise RuntimeError(
            "Nonfinite replay norm_A_2_est"
        )

    output = blank_output(
        case,
        method,
        dtype_name,
    )

    result, trace = (
        run_canonical_with_trace(
            a=a,
            method=method,
            dtype_name=dtype_name,
            norm_A_2_est=norm_A_2_est,
        )
    )

    success = bool(
        getattr(
            result,
            "success",
            False,
        )
    )

    output["success"] = success

    output["failure_class"] = getattr(
        result,
        "failure_class",
        "",
    )

    output["failure_reason"] = getattr(
        result,
        "failure_reason",
        "",
    )

    output["failure_step"] = getattr(
        result,
        "failure_step",
        -1,
    )

    n_interchanges = int(
        getattr(
            result,
            "n_interchanges",
            -1,
        )
    )

    n_row_interchanges = int(
        getattr(
            result,
            "n_row_interchanges",
            -1,
        )
    )

    output[
        "n_interchanges"
    ] = n_interchanges

    output[
        "n_row_interchanges"
    ] = n_row_interchanges

    source_match = (
        success == expected_success
    )

    if n_interchanges != as_int(
        expected["n_interchanges"]
    ):
        source_match = False

    if n_row_interchanges != as_int(
        expected["n_row_interchanges"]
    ):
        source_match = False

    if success:
        canonical_R_inf = float(
            getattr(
                result,
                "right_inverse_defect_inf",
            )
        )

        canonical_R_2 = float(
            getattr(
                result,
                "right_inverse_defect_2_est",
            )
        )

        expected_R_inf = as_float(
            expected[
                "right_inverse_defect_inf"
            ]
        )

        expected_R_2 = as_float(
            expected[
                "right_inverse_defect_2_est"
            ]
        )

        match_Rinf = relerr(
            canonical_R_inf,
            expected_R_inf,
        )

        match_R2 = relerr(
            canonical_R_2,
            expected_R_2,
        )

        # The recovered campaign and the replay may run on
        # different CPU nodes/BLAS reduction paths.  When R_inf
        # itself is only O(u), a relative comparison is meaningless:
        # a sub-ulp absolute change can look like a percent-level
        # relative difference.
        #
        # Require exact agreement of the discrete algorithmic
        # outcomes above, and use a mixed absolute/relative
        # compatibility test only for the final residual diagnostic.
        replay_u = castillo.unit_roundoff(
            dtype_name
        )

        rinf_abs_difference = abs(
            canonical_R_inf
            - expected_R_inf
        )

        rinf_allowed_difference = (
            8.0 * replay_u
            + 1.0e-6
            * max(
                abs(canonical_R_inf),
                abs(expected_R_inf),
            )
        )

        if (
            not math.isfinite(
                rinf_abs_difference
            )
            or rinf_abs_difference
            > rinf_allowed_difference
        ):
            source_match = False

        # R2 is an iterative spectral estimate.
        # We record its reproduction error but
        # do not make it the hard reproduction
        # gate.
        output[
            "source_match_rel_Rinf"
        ] = match_Rinf

        output[
            "source_match_rel_R2"
        ] = match_R2

        output[
            "canonical_R_inf"
        ] = canonical_R_inf

        output[
            "canonical_R_2_est"
        ] = canonical_R_2

        output[
            "expected_R_inf"
        ] = expected_R_inf

        output[
            "expected_R_2_est"
        ] = expected_R_2

    output[
        "source_match"
    ] = source_match

    # Historical agreement is retained in source_match as a
    # diagnostic only.  A new run may make a different row/column
    # decision at a floating-point tie or near-tie.  What must be
    # internally exact is the trajectory and defect decomposition
    # of THIS replay.

    output["norm_A_inf"] = float(
        getattr(
            result,
            "norm_A_inf",
            as_float(
                expected["norm_A_inf"]
            ),
        )
    )

    output[
        "norm_A_2_est"
    ] = norm_A_2_est

    output["gamma_inf"] = float(
        getattr(
            result,
            "gamma_inf",
            as_float(
                expected["gamma_inf"]
            ),
        )
    )

    if not success:
        output[
            "precision_gate_pass"
        ] = False

        output[
            "elapsed_seconds"
        ] = (
            time.perf_counter()
            - start
        )

        return output

    n = int(case["n"])

    captured_rows = processed_rows(
        trace,
        method,
        n,
    )

    result_rows = [
        int(value)
        for value in np.asarray(
            result.row_order,
            dtype=int,
        ).tolist()
    ]

    if captured_rows != result_rows:
        raise RuntimeError(
            "TRACE_ROW_ORDER_MISMATCH: "
            f"matrix={case['matrix_id']} "
            f"method={method} "
            f"dtype={dtype_name}; "
            f"captured={captured_rows}; "
            f"result={result_rows}"
        )

    # Use the permutation returned by the production kernel itself
    # when forming Pi A in R_N = D_N - Pi A E_{N+1}.
    rows = result_rows

    output["path_hash64"] = (
        path_hash64(
            trace,
            rows,
            n,
        )
    )

    vhat_work = get_inverse(
        result
    )

    vhat = np.asarray(
        vhat_work,
        dtype=np.longdouble,
    )

    vtilde = reconstruct_vtilde(
        trace,
        n,
    )

    a_ld = np.asarray(
        a,
        dtype=np.longdouble,
    )

    api = a_ld[
        np.asarray(
            rows,
            dtype=int,
        ),
        :,
    ]

    identity = np.eye(
        n,
        dtype=np.longdouble,
    )

    e = vhat - vtilde

    # Independent evaluations of all three
    # matrices in
    #
    # R_N = D_N - Pi A E_{N+1}.
    r = identity - api @ vhat
    d = identity - api @ vtilde
    ae = api @ e

    closure = r - (d - ae)

    # Deliberately wrong row convention,
    # retained only as a diagnostic.
    r_unpermuted = (
        identity
        - a_ld @ vhat
    )

    D_inf = norm_inf_ld(d)
    R_inf = norm_inf_ld(r)
    AE_inf = norm_inf_ld(ae)
    E_inf = norm_inf_ld(e)
    closure_inf = norm_inf_ld(
        closure
    )

    R_unpermuted_inf = (
        norm_inf_ld(
            r_unpermuted
        )
    )

    D_2 = spectral_estimate(d)
    R_2 = spectral_estimate(r)
    AE_2 = spectral_estimate(ae)

    A_2 = norm_A_2_est

    output.update(
        {
            "D_inf_ld": D_inf,
            "R_inf_ld": R_inf,
            "AE_inf_ld": AE_inf,
            "E_inf_ld": E_inf,
            "closure_inf_ld":
                closure_inf,
            "D_2_est": D_2,
            "R_2_est_ld": R_2,
            "AE_2_est": AE_2,
            "D_scaled_2_est":
                D_2 / A_2,
            "R_scaled_2_est_ld":
                R_2 / A_2,
            "AE_scaled_2_est":
                AE_2 / A_2,
            "R_unpermuted_inf_ld":
                R_unpermuted_inf,
        }
    )

    sum_scale = max(
        D_inf + R_inf + AE_inf,
        float(
            np.finfo(
                np.longdouble
            ).tiny
        ),
    )

    closure_rel_sum = (
        closure_inf / sum_scale
    )

    if R_inf > 0.0:
        closure_rel_R = (
            closure_inf / R_inf
        )
    elif closure_inf == 0.0:
        closure_rel_R = 0.0
    else:
        closure_rel_R = math.inf

    eps_ld = float(
        np.finfo(
            np.longdouble
        ).eps
    )

    # Unit roundoff of the extended arithmetic.
    u_ld = 0.5 * eps_ld

    # A scale-aware absolute allowance for evaluating
    #
    #   R = D - A E
    #
    # with several longdouble matrix products and
    # subtractions.  The reference scale is intentionally
    # based on the operands, not on the tiny residual after
    # cancellation.
    api_inf = norm_inf_ld(api)
    vhat_inf = norm_inf_ld(vhat)
    vtilde_inf = norm_inf_ld(vtilde)

    closure_reference_scale = max(
        1.0,
        api_inf
        * (
            vhat_inf
            + vtilde_inf
            + E_inf
        ),
    )

    gamma_ld = castillo.gamma_k(
        n,
        u_ld,
    )

    closure_abs_allowed = (
        64.0
        * (
            gamma_ld
            + u_ld
        )
        * closure_reference_scale
    )

    closure_abs_ok = bool(
        math.isfinite(closure_inf)
        and math.isfinite(
            closure_abs_allowed
        )
        and closure_inf
        <= closure_abs_allowed
    )

    closure_abs_ratio_bound = (
        closure_inf
        / closure_abs_allowed
        if closure_abs_allowed > 0.0
        else math.inf
    )

    # When R itself lies at the extended-arithmetic
    # noise floor, closure/R is not informative.
    # Otherwise require the reconstructed identity to
    # agree to better than 0.1% of the actual R_N signal.
    r_resolved = (
        R_inf
        >
        1.0e4
        * eps_ld
        * max(
            1.0,
            D_inf + AE_inf,
        )
    )

    relative_R_ok = (
        (not r_resolved)
        or closure_rel_R
        <= 1.0e-3
    )

    precision_gate_pass = bool(
        closure_abs_ok
        and relative_R_ok
    )

    output[
        "closure_abs_allowed_ld"
    ] = closure_abs_allowed

    output[
        "closure_abs_ratio_bound"
    ] = closure_abs_ratio_bound

    # Retained as diagnostics, but closure_rel_sum is no
    # longer a hard gate because D, R and AE may themselves
    # be O(u_work), particularly in binary64.
    output[
        "closure_rel_sum"
    ] = closure_rel_sum

    output[
        "closure_rel_R"
    ] = closure_rel_R

    output[
        "precision_gate_pass"
    ] = precision_gate_pass

    output[
        "elapsed_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    return output


def preflight_subset(
    cases: list[dict],
) -> list[dict]:
    selected = []

    for family in sorted(
        DD_FAMILIES
    ):
        family_cases = [
            case
            for case in cases
            if case["family"] == family
        ]

        if not family_cases:
            raise RuntimeError(
                "Preflight task contains no "
                f"selected {family} matrices"
            )

        dimensions = sorted(
            {
                int(case["n"])
                for case in family_cases
            }
        )

        wanted = {
            dimensions[0],
            dimensions[
                len(dimensions) // 2
            ],
            dimensions[-1],
        }

        for n in sorted(wanted):
            candidates = [
                case
                for case in family_cases
                if int(case["n"]) == n
            ]

            selected.append(
                candidates[0]
            )

    return selected


def write_task(
    task_id: int,
    preflight: bool,
) -> None:
    eps_ld = float(
        np.finfo(
            np.longdouble
        ).eps
    )

    eps64 = float(
        np.finfo(
            np.float64
        ).eps
    )

    if not (
        eps_ld
        < eps64 / 10.0
    ):
        raise RuntimeError(
            "np.longdouble is not extended "
            "beyond float64 on this node: "
            f"eps_ld={eps_ld}, "
            f"eps64={eps64}"
        )

    root = (
        SCRATCH
        / "results"
        / "defect_replay"
    )

    if preflight:
        output_path = (
            root
            / "preflight.csv.gz"
        )
    else:
        output_path = (
            root
            / "partial"
            / f"task_{task_id:04d}.csv.gz"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        output_path.name
        + ".tmp."
        + os.environ.get(
            "SLURM_JOB_ID",
            str(os.getpid()),
        )
    )

    # Requested array has indices 0..999,
    # whereas the recovered campaign has
    # source tasks 0..998.
    if (
        task_id == 999
        and not preflight
    ):
        with gzip.open(
            temp_path,
            "wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_FIELDS,
            )
            writer.writeheader()

        os.replace(
            temp_path,
            output_path,
        )

        print(
            "DEFECT_REPLAY_EMPTY_"
            "TASK_999_OK"
        )
        return

    if not (
        0 <= task_id <= 998
    ):
        raise ValueError(
            "source task must be 0..998"
        )

    cases = collect_selected_cases(
        task_id
    )

    if preflight:
        cases = preflight_subset(
            cases
        )

    records_for_control = []
    matrices_done = 0
    records_done = 0
    gate_failures = 0

    with gzip.open(
        temp_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()

        for case in cases:
            for dtype_name in DTYPES:
                a = build_working_matrix(
                    case,
                    dtype_name,
                )

                expected_A_inf = as_float(
                    case["expected"][
                        (
                            "R0_C0",
                            dtype_name,
                        )
                    ][
                        "norm_A_inf"
                    ]
                )

                regenerated_A_inf = (
                    norm_inf_ld(a)
                )

                if (
                    relerr(
                        regenerated_A_inf,
                        expected_A_inf,
                    )
                    > 1.0e-12
                ):
                    raise RuntimeError(
                        "Regenerated matrix does "
                        "not reproduce canonical "
                        "norm_A_inf: "
                        f"matrix={case['matrix_id']} "
                        f"dtype={dtype_name}"
                    )

                for method in METHODS:
                    record = replay_one(
                        case=case,
                        a=a,
                        method=method,
                        dtype_name=dtype_name,
                    )

                    writer.writerow(record)
                    records_done += 1

                    if (
                        as_bool(
                            record[
                                "success"
                            ]
                        )
                        and not as_bool(
                            record[
                                "precision_gate_pass"
                            ]
                        )
                    ):
                        gate_failures += 1

                    if preflight:
                        records_for_control.append(
                            record
                        )

            matrices_done += 1

    os.replace(
        temp_path,
        output_path,
    )

    if preflight:
        grouped = {}

        for row in records_for_control:
            key = (
                row["matrix_id"],
                row["dtype_name"],
            )

            grouped.setdefault(
                key,
                {},
            )[row["method"]] = row

        for key, methods in grouped.items():
            c0 = methods["R0_C0"]
            c1 = methods["R0_C1"]

            if (
                not as_bool(c0["success"])
                or not as_bool(c1["success"])
            ):
                raise RuntimeError(
                    "R0_C0/R0_C1 control "
                    f"failed on {key}"
                )

            if (
                int(c0["path_hash64"])
                != int(c1["path_hash64"])
            ):
                raise RuntimeError(
                    "R0_C0 and R0_C1 do "
                    "not have identical paths "
                    f"in preflight: {key}"
                )

            for metric in (
                "D_inf_ld",
                "R_inf_ld",
                "AE_inf_ld",
                "D_2_est",
                "R_2_est_ld",
                "AE_2_est",
            ):
                if relerr(
                    as_float(c0[metric]),
                    as_float(c1[metric]),
                ) > 1.0e-13:
                    raise RuntimeError(
                        "R0_C0/R0_C1 control "
                        "metric mismatch: "
                        f"{key} {metric}"
                    )

        if gate_failures:
            raise RuntimeError(
                "Long-double preflight "
                f"precision gate failed in "
                f"{gate_failures} records"
            )

        print(
            "DEFECT_REPLAY_PREFLIGHT_OK "
            f"matrices={matrices_done} "
            f"records={records_done} "
            f"eps_longdouble={eps_ld:.6e}"
        )

    else:
        print(
            "DEFECT_REPLAY_TASK_OK "
            f"task={task_id} "
            f"matrices={matrices_done} "
            f"records={records_done} "
            f"precision_gate_failures="
            f"{gate_failures}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    check_canonical_source()

    write_task(
        task_id=args.task_id,
        preflight=args.preflight,
    )


if __name__ == "__main__":
    main()
