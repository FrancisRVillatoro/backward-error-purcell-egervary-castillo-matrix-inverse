#!/usr/bin/env python3
"""O(n^3) Castillo inverse kernel for the canonical campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


METHOD_RULES = {
    "R0_C0": ("none", "first_nonzero"),
    "R0_C1": ("none", "max_abs"),
    "R0_C2": ("none", "max_scaled"),
    "R1_C1": ("max_abs", "max_abs"),
    "R2_C2": ("max_scaled", "max_scaled"),
}


@dataclass
class CastilloResult:
    inverse: np.ndarray
    row_order: np.ndarray

    success: bool
    failure_class: str
    failure_reason: str
    failure_step: int

    n_row_interchanges: int
    n_interchanges: int

    norm_A_inf: float
    norm_A_2_est: float
    norm_V_inf: float
    norm_V_2_est: float

    min_abs_pivot: float
    min_scaled_pivot: float

    max_tableau_norm_inf: float

    max_inverse_tableau_norm_inf_raw: float
    max_inverse_tableau_norm_inf: float
    inverse_tableau_norm_reliable: bool
    inverse_tableau_candidate_step: int
    inverse_tableau_reliability_bound_u: float
    inverse_tableau_inverse_residual_bound_f: float

    max_B_norm_inf: float

    max_multiplier: float
    max_step_sum_abs_multipliers: float
    total_sum_abs_multipliers: float

    gamma_inf: float
    gamma_inf_numerator: float
    gamma_inf_denominator: float
    gamma_inf_max_local: float
    gamma_inf_max_tail: float

    gamma_2_est: float
    gamma_2_est_iterations: int
    gamma_2_exact_audit: float

    right_inverse_defect_inf: float
    right_inverse_defect_2_est: float
    left_inverse_defect_inf: float
    left_inverse_defect_2_est: float

    right_inverse_scaled_residual_inf: float
    right_inverse_scaled_residual_2_est: float
    left_inverse_scaled_residual_inf: float
    left_inverse_scaled_residual_2_est: float

    inverse_backward_error_inf: float
    inverse_backward_error_2_est: float
    eta_inv_reliable: bool
    eta_inv_reliability_bound_u: float

    local_defect_max_ratio_audit: float
    local_defect_allowed_ratio_audit: float
    local_defect_zero_violations_audit: int

    max_inverse_tableau_norm_inf_exact_audit: float


def unit_roundoff(dtype_name: str) -> float:
    if dtype_name == "float32":
        return 2.0 ** -24

    if dtype_name == "float64":
        return 2.0 ** -53

    raise ValueError(
        "unsupported dtype: {}".format(dtype_name)
    )


def working_dtype(dtype_name: str) -> Any:
    if dtype_name == "float32":
        return np.float32

    if dtype_name == "float64":
        return np.float64

    raise ValueError(
        "unsupported dtype: {}".format(dtype_name)
    )


def gamma_k(k: int, u: float) -> float:
    ku = float(k) * float(u)

    if ku >= 1.0:
        return math.inf

    return ku / (1.0 - ku)


def norm_inf(a: np.ndarray) -> float:
    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        return float(
            np.max(
                np.sum(
                    np.abs(a),
                    axis=1,
                )
            )
        )


def fro_norm(a: np.ndarray) -> float:
    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        return float(
            np.linalg.norm(
                a,
                ord="fro",
            )
        )


def method_rules(
    method: str,
) -> Tuple[str, str]:
    if method not in METHOD_RULES:
        raise ValueError(
            "unknown Castillo method: {}".format(method)
        )

    return METHOD_RULES[method]


def deterministic_start_vector(
    n: int,
) -> np.ndarray:
    indices = np.arange(
        1,
        n + 1,
        dtype=np.float64,
    )

    x = (
        np.sin(indices)
        + np.cos(np.sqrt(2.0) * indices)
    )

    value = float(
        np.linalg.norm(x)
    )

    if value == 0.0 or not np.isfinite(value):
        x = np.ones(
            n,
            dtype=np.float64,
        )
        value = math.sqrt(float(n))

    return x / value


def spectral_norm_estimate(
    a: np.ndarray,
    iterations: int = 6,
) -> float:
    """
    Deterministic fixed-iteration estimate of ||a||_2.

    This quantity is deliberately named an estimate.  It is not used
    in the theorem-backed infinity-norm assertions.
    """
    matrix = np.asarray(
        a,
        dtype=np.float64,
    )

    if matrix.size == 0:
        return 0.0

    scale = norm_inf(matrix)

    if scale == 0.0:
        return 0.0

    if not np.isfinite(scale):
        return math.inf

    scaled = matrix / scale

    x = deterministic_start_vector(
        matrix.shape[1]
    )

    for _ in range(int(iterations)):
        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            y = scaled @ x
            z = scaled.T @ y

        value = float(
            np.linalg.norm(z)
        )

        if value == 0.0:
            return (
                float(np.linalg.norm(y))
                * scale
            )

        if not np.isfinite(value):
            return math.inf

        x = z / value

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        estimate = (
            float(
                np.linalg.norm(
                    scaled @ x
                )
            )
            * scale
        )

    return estimate


def choose_row(
    a: np.ndarray,
    v: np.ndarray,
    row_order: np.ndarray,
    j: int,
    row_rule: str,
) -> int:
    """
    Fresh-dot implementation retained for independent smoke replay.

    The production R1/R2 implementation uses
    choose_row_from_table(), which updates A V recursively.
    """
    if row_rule == "none":
        return j

    active = np.arange(
        j,
        a.shape[0],
    )

    column_norms = np.linalg.norm(
        v[:, active].astype(np.float64),
        axis=0,
    )

    best_position = j
    best_score = -1.0

    for position in range(j, a.shape[0]):
        row = a[row_order[position], :]
        t = row @ v

        if row_rule == "max_abs":
            values = np.abs(t[active])

        elif row_rule == "max_scaled":
            denominator = (
                float(
                    np.linalg.norm(
                        row.astype(np.float64)
                    )
                )
                * column_norms
            )

            values = np.full(
                active.size,
                -1.0,
                dtype=np.float64,
            )

            valid = (
                np.isfinite(denominator)
                & (denominator > 0.0)
            )

            values[valid] = (
                np.abs(t[active][valid])
                / denominator[valid]
            )

        else:
            raise ValueError(
                "unknown row rule: {}".format(
                    row_rule
                )
            )

        finite = np.isfinite(values)

        score = (
            float(np.max(values[finite]))
            if np.any(finite)
            else -1.0
        )

        if score > best_score:
            best_score = score
            best_position = position

    return best_position


def choose_row_from_table(
    transformed_rows: np.ndarray,
    a: np.ndarray,
    row_order: np.ndarray,
    v: np.ndarray,
    j: int,
    row_rule: str,
) -> int:
    """
    O(n^2)-per-step row selection from recursively updated A V.

    In exact arithmetic transformed_rows equals A[row_order,:] V.
    This avoids recomputing all remaining row-tableau products at
    every step, which would make R1/R2 quartic.
    """
    if row_rule == "none":
        return j

    active = np.arange(
        j,
        a.shape[0],
    )

    block = transformed_rows[j:, j:]

    if row_rule == "max_abs":
        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            scores = np.max(
                np.abs(block),
                axis=1,
            )

    elif row_rule == "max_scaled":
        row_norms = np.linalg.norm(
            a[row_order[j:], :].astype(
                np.float64
            ),
            axis=1,
        )

        column_norms = np.linalg.norm(
            v[:, active].astype(np.float64),
            axis=0,
        )

        denominator = (
            row_norms[:, None]
            * column_norms[None, :]
        )

        ratios = np.full(
            block.shape,
            -1.0,
            dtype=np.float64,
        )

        valid = (
            np.isfinite(block)
            & np.isfinite(denominator)
            & (denominator > 0.0)
        )

        ratios[valid] = (
            np.abs(block[valid])
            / denominator[valid]
        )

        scores = np.max(
            ratios,
            axis=1,
        )

    else:
        raise ValueError(
            "unknown row rule: {}".format(
                row_rule
            )
        )

    finite = np.isfinite(scores)

    if not np.any(finite):
        return j

    safe = np.where(
        finite,
        scores,
        -1.0,
    )

    return j + int(np.argmax(safe))


def choose_pivot(
    row: np.ndarray,
    v: np.ndarray,
    j: int,
    pivot_rule: str,
) -> Tuple[Optional[int], np.ndarray]:
    t = row @ v

    active = np.arange(
        j,
        v.shape[1],
    )

    admissible = active[
        np.isfinite(t[active])
        & (t[active] != 0.0)
    ]

    if admissible.size == 0:
        return None, t

    if pivot_rule == "first_nonzero":
        return int(admissible[0]), t

    if pivot_rule == "max_abs":
        local = int(
            np.argmax(
                np.abs(t[admissible])
            )
        )
        return int(admissible[local]), t

    if pivot_rule == "max_scaled":
        row_norm = float(
            np.linalg.norm(
                row.astype(np.float64)
            )
        )

        column_norms = np.linalg.norm(
            v[:, admissible].astype(
                np.float64
            ),
            axis=0,
        )

        denominator = (
            row_norm
            * column_norms
        )

        scores = np.full(
            admissible.size,
            -1.0,
            dtype=np.float64,
        )

        valid = (
            np.isfinite(denominator)
            & (denominator > 0.0)
        )

        scores[valid] = (
            np.abs(t[admissible][valid])
            / denominator[valid]
        )

        local = int(
            np.argmax(scores)
        )

        return int(admissible[local]), t

    raise ValueError(
        "unknown pivot rule: {}".format(
            pivot_rule
        )
    )



def compact_right_update(
    v: np.ndarray,
    j: int,
    p: int,
    b_row: np.ndarray,
) -> np.ndarray:
    """
    Compute V_new = V P B without forming P or B.

    B is the identity except for row j.  The active column is evaluated
    directly as

        V_new[:,j] = (V P)[:,j] * B[j,j],

    rather than through the algebraically equivalent but numerically
    different expression

        (V P)[:,j] + (V P)[:,j] * (B[j,j]-1).

    Thus each non-active entry uses one multiplication and one addition,
    while each active-column entry uses one multiplication.  This is the
    operation sequence assumed by the local gamma_2 defect bound.
    """
    vp = v.copy()

    if p != j:
        vp[:, [j, p]] = vp[:, [p, j]]

    column_j = vp[:, j].copy()
    new_v = vp.copy()

    with np.errstate(
        over="ignore",
        invalid="ignore",
        under="ignore",
    ):
        if j > 0:
            new_v[:, :j] = (
                vp[:, :j]
                + column_j[:, None]
                * b_row[None, :j]
            )

        new_v[:, j] = (
            column_j * b_row[j]
        )

        if j + 1 < v.shape[1]:
            new_v[:, j + 1:] = (
                vp[:, j + 1:]
                + column_j[:, None]
                * b_row[None, j + 1:]
            )

    return new_v


def compact_transform_inverse_update(
    w: np.ndarray,
    j: int,
    p: int,
    t_after_swap: np.ndarray,
) -> np.ndarray:
    """
    Update W=(computed transformation product)^{-1}:

        W_new = B^{-1} P W.
    """
    new_w = w.copy()

    if p != j:
        new_w[[j, p], :] = (
            new_w[[p, j], :]
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        new_w[j, :] = (
            np.asarray(
                t_after_swap,
                dtype=np.float64,
            )
            @ new_w
        )

    return new_w


def compact_tail_left_update(
    tail: np.ndarray,
    j: int,
    p: int,
    b_row: np.ndarray,
) -> np.ndarray:
    """
    Compute

        tail_new = P B tail

    using the compact representation of B.
    """
    new_tail = tail.copy()

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        new_tail[j, :] = (
            np.asarray(
                b_row,
                dtype=np.float64,
            )
            @ new_tail
        )

    if p != j:
        new_tail[[j, p], :] = (
            new_tail[[p, j], :]
        )

    return new_tail



def local_absolute_matrix(
    vp: np.ndarray,
    j: int,
    b_row: np.ndarray,
) -> np.ndarray:
    """
    Form |V P| |B| without cancellation.

    For k != j,

        (|VP||B|)[:,k]
        = |VP|[:,k] + |VP|[:,j] |B[j,k]|,

    while

        (|VP||B|)[:,j]
        = |VP|[:,j] |B[j,j]|.
    """
    absolute_vp = np.abs(
        np.asarray(
            vp,
            dtype=np.float64,
        )
    )

    absolute_b_row = np.abs(
        np.asarray(
            b_row,
            dtype=np.float64,
        )
    )

    result = absolute_vp.copy()
    column_j = absolute_vp[:, j].copy()

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        if j > 0:
            result[:, :j] = (
                absolute_vp[:, :j]
                + column_j[:, None]
                * absolute_b_row[None, :j]
            )

        result[:, j] = (
            column_j * absolute_b_row[j]
        )

        if j + 1 < vp.shape[1]:
            result[:, j + 1:] = (
                absolute_vp[:, j + 1:]
                + column_j[:, None]
                * absolute_b_row[None, j + 1:]
            )

    return result



def local_inf_factor(
    vp: np.ndarray,
    j: int,
    b_row: np.ndarray,
) -> float:
    """
    Exact infinity norm of |V P| |B|, evaluated as a sum of
    nonnegative terms and therefore without subtractive cancellation.
    """
    absolute_vp = np.abs(
        np.asarray(
            vp,
            dtype=np.float64,
        )
    )

    absolute_b_row = np.abs(
        np.asarray(
            b_row,
            dtype=np.float64,
        )
    )

    n = vp.shape[1]

    left = (
        np.sum(
            absolute_vp[:, :j],
            axis=1,
        )
        if j > 0
        else np.zeros(
            vp.shape[0],
            dtype=np.float64,
        )
    )

    right = (
        np.sum(
            absolute_vp[:, j + 1:],
            axis=1,
        )
        if j + 1 < n
        else np.zeros(
            vp.shape[0],
            dtype=np.float64,
        )
    )

    sum_absolute_b_row = float(
        np.sum(absolute_b_row)
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        row_sums = (
            left
            + right
            + absolute_vp[:, j]
            * sum_absolute_b_row
        )

    return float(np.max(row_sums))


def local_defect_audit(
    vp: np.ndarray,
    j: int,
    b_row: np.ndarray,
    v_after: np.ndarray,
    u: float,
) -> Dict[str, float]:
    """
    Verify in extended precision

        |F_l|
        <= gamma_2 |V_l P_l| |B_l|.

    Entries with zero majorant are asserted separately.
    """
    n = vp.shape[0]

    b = np.eye(
        n,
        dtype=np.longdouble,
    )

    b[j, :] = np.asarray(
        b_row,
        dtype=np.longdouble,
    )

    reference = (
        np.asarray(
            vp,
            dtype=np.longdouble,
        )
        @ b
    )

    defect = (
        np.asarray(
            v_after,
            dtype=np.longdouble,
        )
        - reference
    )

    scale = (
        np.abs(
            np.asarray(
                vp,
                dtype=np.longdouble,
            )
        )
        @ np.abs(b)
    )

    positive = (
        scale
        > np.longdouble(0.0)
    )

    zero_violations = int(
        np.count_nonzero(
            (~positive)
            & (defect != 0.0)
        )
    )

    ratio = 0.0

    if np.any(positive):
        ratios = np.zeros(
            scale.shape,
            dtype=np.longdouble,
        )

        ratios[positive] = (
            np.abs(defect[positive])
            / (
                np.longdouble(
                    gamma_k(2, u)
                )
                * scale[positive]
            )
        )

        ratio = float(
            np.max(ratios)
        )

    u_extended = (
        0.5
        * float(
            np.finfo(
                np.longdouble
            ).eps
        )
    )

    allowance = (
        1.0
        + 2.0
        * gamma_k(n, u_extended)
        / gamma_k(2, u)
        + 1.0e-12
    )

    return {
        "ratio": ratio,
        "allowed": allowance,
        "zero_violations": zero_violations,
    }


def reliability_certificate(
    v: np.ndarray,
    w: np.ndarray,
    working_u: float,
) -> Dict[str, float]:
    """
    Conservative sufficient certificate for

        kappa_2(V) working_u < 1e-2.

    If E=I-WV and ||E||_2<1,

        ||V^{-1}||_2
        <= ||W||_2/(1-||E||_2).

    Frobenius norms and a multiplication-roundoff allowance provide
    a conservative computable upper bound.
    """
    v64 = np.asarray(
        v,
        dtype=np.float64,
    )

    w64 = np.asarray(
        w,
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        product = w64 @ v64

    residual_f = fro_norm(
        np.eye(v64.shape[0])
        - product
    )

    v_f = fro_norm(v64)
    w_f = fro_norm(w64)

    roundoff = (
        gamma_k(
            v64.shape[0],
            2.0 ** -53,
        )
        * v_f
        * w_f
    )

    residual_bound = (
        residual_f
        + roundoff
    )

    if (
        residual_bound < 1.0
        and np.isfinite(residual_bound)
    ):
        kappa_upper = (
            v_f
            * w_f
            / (1.0 - residual_bound)
        )
    else:
        kappa_upper = math.inf

    measure = (
        kappa_upper
        * working_u
    )

    return {
        "inverse_residual_bound_f":
            residual_bound,
        "kappa_2_upper":
            kappa_upper,
        "kappa_2_upper_u":
            measure,
        "reliable": bool(
            np.isfinite(measure)
            and measure < 1.0e-2
        ),
    }


def _blank_result(
    a: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    row_order: np.ndarray,
    failure_class: str,
    failure_reason: str,
    failure_step: int,
    state: Dict[str, Any],
    u: float,
    norm_A_2_est: float,
) -> CastilloResult:
    reliability = reliability_certificate(
        v,
        w,
        u,
    )

    return CastilloResult(
        inverse=v.copy(),
        row_order=row_order.copy(),

        success=False,
        failure_class=failure_class,
        failure_reason=failure_reason,
        failure_step=int(failure_step),

        n_row_interchanges=int(
            state["n_row_interchanges"]
        ),
        n_interchanges=int(
            state["n_interchanges"]
        ),

        norm_A_inf=norm_inf(
            np.asarray(
                a,
                dtype=np.float64,
            )
        ),
        norm_A_2_est=float(
            norm_A_2_est
        ),
        norm_V_inf=norm_inf(
            np.asarray(
                v,
                dtype=np.float64,
            )
        ),
        norm_V_2_est=spectral_norm_estimate(v),

        min_abs_pivot=float(
            state["min_abs_pivot"]
        ),
        min_scaled_pivot=float(
            state["min_scaled_pivot"]
        ),

        max_tableau_norm_inf=float(
            state["max_tableau_norm_inf"]
        ),

        max_inverse_tableau_norm_inf_raw=float(
            state["max_inverse_raw"]
        ),
        max_inverse_tableau_norm_inf=math.nan,
        inverse_tableau_norm_reliable=False,
        inverse_tableau_candidate_step=int(
            state["inverse_candidate_step"]
        ),
        inverse_tableau_reliability_bound_u=float(
            reliability["kappa_2_upper_u"]
        ),
        inverse_tableau_inverse_residual_bound_f=float(
            reliability[
                "inverse_residual_bound_f"
            ]
        ),

        max_B_norm_inf=float(
            state["max_B_norm_inf"]
        ),

        max_multiplier=float(
            state["max_multiplier"]
        ),
        max_step_sum_abs_multipliers=float(
            state["max_step_sum"]
        ),
        total_sum_abs_multipliers=float(
            state["total_sum"]
        ),

        gamma_inf=math.nan,
        gamma_inf_numerator=math.nan,
        gamma_inf_denominator=math.nan,
        gamma_inf_max_local=math.nan,
        gamma_inf_max_tail=math.nan,

        gamma_2_est=math.nan,
        gamma_2_est_iterations=int(
            state["norm_iterations"]
        ),
        gamma_2_exact_audit=math.nan,

        right_inverse_defect_inf=math.nan,
        right_inverse_defect_2_est=math.nan,
        left_inverse_defect_inf=math.nan,
        left_inverse_defect_2_est=math.nan,

        right_inverse_scaled_residual_inf=math.nan,
        right_inverse_scaled_residual_2_est=math.nan,
        left_inverse_scaled_residual_inf=math.nan,
        left_inverse_scaled_residual_2_est=math.nan,

        inverse_backward_error_inf=math.nan,
        inverse_backward_error_2_est=math.nan,
        eta_inv_reliable=False,
        eta_inv_reliability_bound_u=float(
            reliability["kappa_2_upper_u"]
        ),

        local_defect_max_ratio_audit=float(
            state["local_defect_ratio"]
        ),
        local_defect_allowed_ratio_audit=float(
            state["local_defect_allowed"]
        ),
        local_defect_zero_violations_audit=int(
            state["local_zero_violations"]
        ),

        max_inverse_tableau_norm_inf_exact_audit=float(
            state["max_inverse_exact_audit"]
        ),
    )


def castillo_inverse(
    a_input: np.ndarray,
    method: str,
    dtype_name: str,
    norm_A_2_est: Optional[float] = None,
    spectral_iterations: int = 6,
    audit: bool = False,
) -> CastilloResult:
    dtype = working_dtype(dtype_name)
    u = unit_roundoff(dtype_name)

    a = np.asarray(
        a_input,
        dtype=dtype,
    )

    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
    ):
        raise ValueError("A must be square")

    if not np.all(np.isfinite(a)):
        raise ValueError(
            "A contains NaN or Inf "
            "in working precision"
        )

    n = a.shape[0]

    if norm_A_2_est is None:
        norm_A_2_est = spectral_norm_estimate(
            a,
            spectral_iterations,
        )

    row_rule, pivot_rule = method_rules(method)

    v = np.eye(
        n,
        dtype=dtype,
    )

    w = np.eye(
        n,
        dtype=np.float64,
    )

    row_order = np.arange(
        n,
        dtype=int,
    )

    transformed_rows = (
        a.copy()
        if row_rule != "none"
        else None
    )

    state: Dict[str, Any] = {
        "n_row_interchanges": 0,
        "n_interchanges": 0,

        "min_abs_pivot": math.inf,
        "min_scaled_pivot": math.inf,

        "max_tableau_norm_inf": 1.0,

        "max_inverse_raw": 1.0,
        "inverse_candidate_step": 0,

        "max_B_norm_inf": 1.0,

        "max_multiplier": 0.0,
        "max_step_sum": 0.0,
        "total_sum": 0.0,

        "norm_iterations":
            int(spectral_iterations),

        "local_defect_ratio": (
            0.0 if audit else math.nan
        ),
        "local_defect_allowed": (
            0.0 if audit else math.nan
        ),
        "local_zero_violations": (
            0 if audit else -1
        ),

        "max_inverse_exact_audit": (
            1.0 if audit else math.nan
        ),
    }

    candidate_v = (
        v.astype(np.float64).copy()
    )

    candidate_w = w.copy()

    steps: List[
        Tuple[
            int,
            int,
            np.ndarray,
        ]
    ] = []

    local_inf_values: List[float] = []
    local_2_values: List[float] = []

    local_matrices_audit: List[
        np.ndarray
    ] = []

    for j in range(n):
        if row_rule == "none":
            row_position = j
        else:
            row_position = choose_row_from_table(
                transformed_rows,
                a,
                row_order,
                v,
                j,
                row_rule,
            )

        if row_position != j:
            row_order[[j, row_position]] = (
                row_order[[row_position, j]]
            )

            transformed_rows[
                [j, row_position],
                :
            ] = transformed_rows[
                [row_position, j],
                :
            ]

            state[
                "n_row_interchanges"
            ] += 1

        row = a[row_order[j], :]

        pivot_column, t_before = choose_pivot(
            row,
            v,
            j,
            pivot_rule,
        )

        if pivot_column is None:
            return _blank_result(
                a,
                v,
                w,
                row_order,
                "method_breakdown_no_pivot",
                "no_finite_nonzero_pivot",
                j,
                state,
                u,
                float(norm_A_2_est),
            )

        p = int(pivot_column)

        vp = v.copy()
        t = t_before.copy()

        if p != j:
            vp[:, [j, p]] = (
                vp[:, [p, j]]
            )

            t[[j, p]] = t[[p, j]]

            state[
                "n_interchanges"
            ] += 1

        pivot = t[j]

        if (
            not np.isfinite(pivot)
            or pivot == 0.0
        ):
            return _blank_result(
                a,
                v,
                w,
                row_order,
                "method_breakdown_invalid_pivot",
                "invalid_pivot_after_interchange",
                j,
                state,
                u,
                float(norm_A_2_est),
            )

        absolute_pivot = float(abs(pivot))

        state["min_abs_pivot"] = min(
            state["min_abs_pivot"],
            absolute_pivot,
        )

        row_norm = float(
            np.linalg.norm(
                row.astype(np.float64)
            )
        )

        column_norm = float(
            np.linalg.norm(
                vp[:, j].astype(np.float64)
            )
        )

        if (
            row_norm > 0.0
            and column_norm > 0.0
        ):
            state[
                "min_scaled_pivot"
            ] = min(
                state["min_scaled_pivot"],
                absolute_pivot
                / (
                    row_norm
                    * column_norm
                ),
            )

        b_row = np.zeros(
            n,
            dtype=dtype,
        )

        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore",
            under="ignore",
        ):
            b_row[j] = (
                dtype(1.0)
                / pivot
            )

            b_row[:j] = (
                -t[:j]
                / pivot
            )

            b_row[j + 1:] = (
                -t[j + 1:]
                / pivot
            )

        if not np.all(np.isfinite(b_row)):
            return _blank_result(
                a,
                v,
                w,
                row_order,
                "method_breakdown_nonfinite_B",
                "nonfinite_elementary_transformation",
                j,
                state,
                u,
                float(norm_A_2_est),
            )

        off = np.delete(
            np.abs(
                b_row.astype(np.float64)
            ),
            j,
        )

        step_sum = float(
            np.sum(off)
        )

        state["max_multiplier"] = max(
            state["max_multiplier"],
            (
                float(np.max(off))
                if off.size
                else 0.0
            ),
        )

        state["max_step_sum"] = max(
            state["max_step_sum"],
            step_sum,
        )

        state["total_sum"] += step_sum

        state["max_B_norm_inf"] = max(
            state["max_B_norm_inf"],
            max(
                1.0,
                float(
                    np.sum(
                        np.abs(
                            b_row.astype(
                                np.float64
                            )
                        )
                    )
                ),
            ),
        )

        local_inf = local_inf_factor(
            vp,
            j,
            b_row,
        )

        local_matrix = local_absolute_matrix(
            vp,
            j,
            b_row,
        )

        local_2 = spectral_norm_estimate(
            local_matrix,
            spectral_iterations,
        )

        local_inf_values.append(
            local_inf
        )

        local_2_values.append(
            local_2
        )

        if audit:
            local_matrices_audit.append(
                local_matrix
            )

        v_after = compact_right_update(
            v,
            j,
            p,
            b_row,
        )

        if not np.all(np.isfinite(v_after)):
            return _blank_result(
                a,
                v_after,
                w,
                row_order,
                "method_breakdown_nonfinite_tableau",
                "nonfinite_tableau",
                j,
                state,
                u,
                float(norm_A_2_est),
            )

        if audit:
            local_audit = local_defect_audit(
                vp,
                j,
                b_row,
                v_after,
                u,
            )

            state[
                "local_defect_ratio"
            ] = max(
                state[
                    "local_defect_ratio"
                ],
                local_audit["ratio"],
            )

            state[
                "local_defect_allowed"
            ] = max(
                state[
                    "local_defect_allowed"
                ],
                local_audit["allowed"],
            )

            state[
                "local_zero_violations"
            ] += int(
                local_audit[
                    "zero_violations"
                ]
            )

        w = compact_transform_inverse_update(
            w,
            j,
            p,
            t,
        )

        v = v_after

        if transformed_rows is not None:
            transformed_rows = (
                compact_right_update(
                    transformed_rows,
                    j,
                    p,
                    b_row,
                )
            )

        state[
            "max_tableau_norm_inf"
        ] = max(
            state[
                "max_tableau_norm_inf"
            ],
            norm_inf(
                v.astype(np.float64)
            ),
        )

        raw_inverse_norm = norm_inf(w)

        if (
            raw_inverse_norm
            > state["max_inverse_raw"]
        ):
            state["max_inverse_raw"] = (
                raw_inverse_norm
            )

            state[
                "inverse_candidate_step"
            ] = j + 1

            candidate_v = (
                v.astype(np.float64).copy()
            )

            candidate_w = w.copy()

        if audit:
            try:
                state[
                    "max_inverse_exact_audit"
                ] = max(
                    state[
                        "max_inverse_exact_audit"
                    ],
                    norm_inf(
                        np.linalg.inv(
                            v.astype(np.float64)
                        )
                    ),
                )
            except np.linalg.LinAlgError:
                state[
                    "max_inverse_exact_audit"
                ] = math.inf

        steps.append(
            (
                j,
                p,
                b_row.astype(
                    np.float64
                ).copy(),
            )
        )

    candidate_reliability = (
        reliability_certificate(
            candidate_v,
            candidate_w,
            u,
        )
    )

    reliable_inverse_max = (
        state["max_inverse_raw"]
        if candidate_reliability["reliable"]
        else math.nan
    )

    tail = np.eye(
        n,
        dtype=np.float64,
    )

    gamma_inf_numerator = 0.0
    gamma_2_numerator = 0.0
    gamma_2_exact_numerator = 0.0

    max_tail_inf = 1.0

    max_local_inf = (
        max(local_inf_values)
        if local_inf_values
        else 0.0
    )

    for index in range(
        n - 1,
        -1,
        -1,
    ):
        tail_inf = norm_inf(tail)

        tail_2 = spectral_norm_estimate(
            tail,
            spectral_iterations,
        )

        gamma_inf_numerator += (
            local_inf_values[index]
            * tail_inf
        )

        gamma_2_numerator += (
            local_2_values[index]
            * tail_2
        )

        if audit:
            gamma_2_exact_numerator += (
                float(
                    np.linalg.norm(
                        local_matrices_audit[
                            index
                        ],
                        2,
                    )
                )
                * float(
                    np.linalg.norm(
                        tail,
                        2,
                    )
                )
            )

        max_tail_inf = max(
            max_tail_inf,
            tail_inf,
        )

        j, p, b_row64 = steps[index]

        tail = compact_tail_left_update(
            tail,
            j,
            p,
            b_row64,
        )

    v64 = v.astype(np.float64)

    a64 = (
        a.astype(np.float64)[
            row_order,
            :
        ]
    )

    norm_v_inf = norm_inf(v64)

    norm_v_2_est = spectral_norm_estimate(
        v64,
        spectral_iterations,
    )

    gamma_inf = (
        gamma_inf_numerator
        / max(
            norm_v_inf,
            np.finfo(np.float64).tiny,
        )
    )

    gamma_2_est = (
        gamma_2_numerator
        / max(
            norm_v_2_est,
            np.finfo(np.float64).tiny,
        )
    )

    gamma_2_exact_audit = math.nan

    if audit:
        gamma_2_exact_audit = (
            gamma_2_exact_numerator
            / max(
                float(
                    np.linalg.norm(
                        v64,
                        2,
                    )
                ),
                np.finfo(np.float64).tiny,
            )
        )

    identity = np.eye(
        n,
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        right_defect = (
            identity
            - a64 @ v64
        )

        left_defect = (
            identity
            - v64 @ a64
        )

    norm_a_inf = norm_inf(a64)
    norm_a_2 = float(norm_A_2_est)

    right_inf = norm_inf(
        right_defect
    )

    left_inf = norm_inf(
        left_defect
    )

    right_2 = spectral_norm_estimate(
        right_defect,
        spectral_iterations,
    )

    left_2 = spectral_norm_estimate(
        left_defect,
        spectral_iterations,
    )

    denominator_inf = max(
        norm_a_inf * norm_v_inf,
        np.finfo(np.float64).tiny,
    )

    denominator_2 = max(
        norm_a_2 * norm_v_2_est,
        np.finfo(np.float64).tiny,
    )

    final_reliability = (
        reliability_certificate(
            v64,
            w,
            u,
        )
    )

    eta_inf = math.nan
    eta_2 = math.nan

    if final_reliability["reliable"]:
        try:
            backward_matrix = (
                np.linalg.solve(
                    v64.T,
                    right_defect.T,
                ).T
            )

            eta_inf = (
                norm_inf(backward_matrix)
                / max(
                    norm_a_inf,
                    np.finfo(np.float64).tiny,
                )
            )

            eta_2 = (
                spectral_norm_estimate(
                    backward_matrix,
                    spectral_iterations,
                )
                / max(
                    norm_a_2,
                    np.finfo(np.float64).tiny,
                )
            )

        except np.linalg.LinAlgError:
            pass

    return CastilloResult(
        inverse=v.copy(),
        row_order=row_order.copy(),

        success=True,
        failure_class="completed",
        failure_reason="",
        failure_step=-1,

        n_row_interchanges=int(
            state["n_row_interchanges"]
        ),
        n_interchanges=int(
            state["n_interchanges"]
        ),

        norm_A_inf=norm_a_inf,
        norm_A_2_est=norm_a_2,

        norm_V_inf=norm_v_inf,
        norm_V_2_est=norm_v_2_est,

        min_abs_pivot=float(
            state["min_abs_pivot"]
        ),
        min_scaled_pivot=float(
            state["min_scaled_pivot"]
        ),

        max_tableau_norm_inf=float(
            state["max_tableau_norm_inf"]
        ),

        max_inverse_tableau_norm_inf_raw=float(
            state["max_inverse_raw"]
        ),
        max_inverse_tableau_norm_inf=float(
            reliable_inverse_max
        ),
        inverse_tableau_norm_reliable=bool(
            candidate_reliability["reliable"]
        ),
        inverse_tableau_candidate_step=int(
            state["inverse_candidate_step"]
        ),
        inverse_tableau_reliability_bound_u=float(
            candidate_reliability[
                "kappa_2_upper_u"
            ]
        ),
        inverse_tableau_inverse_residual_bound_f=float(
            candidate_reliability[
                "inverse_residual_bound_f"
            ]
        ),

        max_B_norm_inf=float(
            state["max_B_norm_inf"]
        ),

        max_multiplier=float(
            state["max_multiplier"]
        ),
        max_step_sum_abs_multipliers=float(
            state["max_step_sum"]
        ),
        total_sum_abs_multipliers=float(
            state["total_sum"]
        ),

        gamma_inf=float(gamma_inf),
        gamma_inf_numerator=float(
            gamma_inf_numerator
        ),
        gamma_inf_denominator=float(
            norm_v_inf
        ),
        gamma_inf_max_local=float(
            max_local_inf
        ),
        gamma_inf_max_tail=float(
            max_tail_inf
        ),

        gamma_2_est=float(
            gamma_2_est
        ),
        gamma_2_est_iterations=int(
            spectral_iterations
        ),
        gamma_2_exact_audit=float(
            gamma_2_exact_audit
        ),

        right_inverse_defect_inf=float(
            right_inf
        ),
        right_inverse_defect_2_est=float(
            right_2
        ),
        left_inverse_defect_inf=float(
            left_inf
        ),
        left_inverse_defect_2_est=float(
            left_2
        ),

        right_inverse_scaled_residual_inf=float(
            right_inf
            / denominator_inf
        ),
        right_inverse_scaled_residual_2_est=float(
            right_2
            / denominator_2
        ),
        left_inverse_scaled_residual_inf=float(
            left_inf
            / denominator_inf
        ),
        left_inverse_scaled_residual_2_est=float(
            left_2
            / denominator_2
        ),

        inverse_backward_error_inf=float(
            eta_inf
        ),
        inverse_backward_error_2_est=float(
            eta_2
        ),
        eta_inv_reliable=bool(
            final_reliability["reliable"]
            and np.isfinite(eta_inf)
        ),
        eta_inv_reliability_bound_u=float(
            final_reliability[
                "kappa_2_upper_u"
            ]
        ),

        local_defect_max_ratio_audit=float(
            state["local_defect_ratio"]
        ),
        local_defect_allowed_ratio_audit=float(
            state["local_defect_allowed"]
        ),
        local_defect_zero_violations_audit=int(
            state["local_zero_violations"]
        ),

        max_inverse_tableau_norm_inf_exact_audit=float(
            state["max_inverse_exact_audit"]
        ),
    )
