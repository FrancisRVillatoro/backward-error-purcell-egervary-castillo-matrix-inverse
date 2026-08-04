#!/usr/bin/env python3
"""
Canonical matrix families for the Castillo stability campaign.

Matrices are first constructed in float64.  Working-precision conversion
is performed later by the campaign driver, so float32 and float64 use the
same underlying mathematical realization.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_seed(
    seed_namespace: str,
    cell_id: str,
    n: int,
    replica: int,
) -> int:
    payload = canonical_json(
        {
            "seed_namespace": seed_namespace,
            "cell_id": cell_id,
            "n": int(n),
            "replica": int(replica),
        }
    )

    digest = hashlib.sha256(payload.encode("utf-8")).digest()

    # A 64-bit seed avoids the non-negligible collision probability of
    # 32-bit seeds when millions of matrices are generated.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_matrix_id(
    seed_namespace: str,
    cell_id: str,
    n: int,
    replica: int,
) -> str:
    payload = canonical_json(
        {
            "seed_namespace": seed_namespace,
            "cell_id": cell_id,
            "n": int(n),
            "replica": int(replica),
        }
    )

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:24]


def infinity_norm(a: np.ndarray) -> float:
    return float(np.abs(a).sum(axis=1).max())


def normalize_infinity(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    value = infinity_norm(a)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("matrix has invalid infinity norm")

    return a / value


def diagonal_dominance_gap(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    diagonal = np.abs(np.diag(a))
    off_diagonal = np.abs(a).sum(axis=1) - diagonal
    return float(np.min(diagonal - off_diagonal))


def random_orthogonal(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))

    diagonal = np.diag(r)
    signs = np.sign(diagonal)
    signs[signs == 0.0] = 1.0

    return q * signs[np.newaxis, :]


def prescribed_singular_values(
    n: int,
    condition_number: float,
    profile: str,
) -> np.ndarray:
    kappa = float(condition_number)

    if kappa < 1.0:
        raise ValueError("condition_number must be at least one")

    if profile == "orthogonal":
        if kappa != 1.0:
            raise ValueError(
                "the orthogonal profile requires condition_number=1"
            )
        return np.ones(n, dtype=np.float64)

    if profile == "geometric":
        return np.geomspace(1.0, 1.0 / kappa, n)

    if profile == "one_small":
        values = np.ones(n, dtype=np.float64)
        values[-1] = 1.0 / kappa
        return values

    if profile == "clustered":
        values = np.ones(n, dtype=np.float64)

        first = max(1, n // 3)
        second = max(first + 1, 2 * n // 3)

        values[first:second] = kappa ** (-0.5)
        values[second:] = kappa ** (-1.0)

        return values

    raise ValueError(f"unknown singular-value profile: {profile}")


def random_conditioned(
    n: int,
    condition_number: float,
    profile: str,
    rng: np.random.Generator,
) -> np.ndarray:
    left = random_orthogonal(n, rng)
    right = random_orthogonal(n, rng)

    singular_values = prescribed_singular_values(
        n=n,
        condition_number=condition_number,
        profile=profile,
    )

    return (left * singular_values[np.newaxis, :]) @ right.T


def hilbert_matrix(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    j = np.arange(n, dtype=np.float64)

    return 1.0 / (i[:, None] + j[None, :] + 1.0)


def vandermonde_nodes(
    n: int,
    node_type: str,
    rng: np.random.Generator,
    cluster_parameter: float,
) -> np.ndarray:
    if node_type == "equispaced":
        return np.linspace(-1.0, 1.0, n)

    if node_type == "chebyshev":
        j = np.arange(n, dtype=np.float64)
        return np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))

    if node_type == "clustered_zero":
        raw = np.linspace(-1.0, 1.0, n)
        return (
            np.sign(raw)
            * np.abs(raw) ** float(cluster_parameter)
        )

    if node_type == "near_one":
        exponents = np.linspace(
            1.0,
            float(cluster_parameter),
            n,
        )
        return 1.0 - 10.0 ** (-exponents)

    if node_type == "random":
        return np.sort(rng.uniform(-1.0, 1.0, size=n))

    if node_type == "equispaced_positive":
        return np.linspace(
            1.0 / (n + 1.0),
            n / (n + 1.0),
            n,
        )

    if node_type == "clustered_positive":
        raw = np.linspace(
            1.0 / (n + 1.0),
            n / (n + 1.0),
            n,
        )
        return raw ** float(cluster_parameter)

    raise ValueError(f"unknown Vandermonde node type: {node_type}")


def vandermonde_matrix(
    n: int,
    node_type: str,
    rng: np.random.Generator,
    cluster_parameter: float,
    column_scaling: bool,
) -> np.ndarray:
    nodes = vandermonde_nodes(
        n=n,
        node_type=node_type,
        rng=rng,
        cluster_parameter=cluster_parameter,
    )

    a = np.vander(nodes, N=n, increasing=True)

    if column_scaling:
        column_norms = np.linalg.norm(a, axis=0)
        column_norms[column_norms == 0.0] = 1.0
        a = a / column_norms[np.newaxis, :]

    return a


def kahan_type_matrix(
    n: int,
    theta: float,
    column_scaling: bool,
) -> np.ndarray:
    """
    Upper-triangular Kahan-type matrix.

    With s=sin(theta), c=cos(theta),

        A[i,i] = s^(i+1),
        A[i,j] = -c s^(i+1),  j>i.

    Optional right scaling divides column j by s^(j+1).  This retains the
    growth mechanism while delaying complete underflow.
    """
    theta = float(theta)

    if theta <= 0.0:
        raise ValueError("theta must be positive")

    s = float(np.sin(theta))
    c = float(np.cos(theta))

    a = np.zeros((n, n), dtype=np.float64)
    powers = np.power(s, np.arange(1, n + 1, dtype=np.float64))

    for i in range(n):
        a[i, i] = powers[i]

        if i + 1 < n:
            a[i, i + 1:] = -c * powers[i]

    if column_scaling:
        safe = powers.copy()
        safe[safe == 0.0] = np.finfo(np.float64).tiny

        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            a = a / safe[np.newaxis, :]

    return a


def adversarial_rotation_blocks(
    n: int,
    epsilon: float,
) -> np.ndarray:
    """
    Orthogonal 2x2 blocks with a deliberately small first pivot.
    """
    epsilon = float(epsilon)

    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0,1)")

    a = np.eye(n, dtype=np.float64)
    q = float(np.sqrt(max(0.0, 1.0 - epsilon * epsilon)))

    block = np.array(
        [
            [epsilon, q],
            [q, -epsilon],
        ],
        dtype=np.float64,
    )

    for i in range(0, n - 1, 2):
        a[i:i + 2, i:i + 2] = block

    return a


def repeated_bad_pivots(
    n: int,
    epsilon: float,
    decay: float,
) -> np.ndarray:
    """
    Sequence of orthogonal 2x2 blocks with progressively smaller pivots.
    """
    epsilon = float(epsilon)
    decay = float(decay)

    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0,1)")

    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must lie in (0,1]")

    a = np.eye(n, dtype=np.float64)
    safe_epsilon = 10.0 * np.finfo(np.float64).eps

    block_index = 0

    for i in range(0, n - 1, 2):
        epsilon_k = max(
            epsilon * decay**block_index,
            safe_epsilon,
        )
        q = float(
            np.sqrt(max(0.0, 1.0 - epsilon_k * epsilon_k))
        )

        a[i:i + 2, i:i + 2] = np.array(
            [
                [epsilon_k, q],
                [q, -epsilon_k],
            ],
            dtype=np.float64,
        )

        block_index += 1

    return a


def scaled_adversarial(
    n: int,
    epsilon: float,
    scale_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base = adversarial_rotation_blocks(n, epsilon)

    row_exponents = rng.uniform(
        -float(scale_range),
        float(scale_range),
        size=n,
    )
    column_exponents = rng.uniform(
        -float(scale_range),
        float(scale_range),
        size=n,
    )

    row_scaling = 10.0 ** row_exponents
    column_scaling = 10.0 ** column_exponents

    return (
        row_scaling[:, None]
        * base
        * column_scaling[None, :]
    )


def sdd_random(
    n: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dense random matrix with ||A||_inf=1 and exact DD gap alpha.
    """
    alpha = float(alpha)
    g = 2.0 * alpha / (1.0 - alpha)

    a = rng.standard_normal((n, n))
    np.fill_diagonal(a, 0.0)

    for i in range(n):
        sigma = float(np.abs(a[i]).sum())

        if sigma == 0.0:
            a[i, (i + 1) % n] = 1.0
            sigma = 1.0

        a[i] /= sigma

        diagonal_sign = (
            -1.0
            if rng.standard_normal() < 0.0
            else 1.0
        )
        a[i, i] = diagonal_sign * (1.0 + g)

    return a / (2.0 + g)


def sdd_tridiagonal(
    n: int,
    alpha: float,
) -> np.ndarray:
    """
    Tridiagonal matrix with ||A||_inf=1 and exact DD gap alpha.
    """
    alpha = float(alpha)
    g = 4.0 * alpha / (1.0 - alpha)

    a = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        a[i, i] = 2.0 + g

        if i > 0:
            a[i, i - 1] = -1.0

        if i < n - 1:
            a[i, i + 1] = -1.0

    a[0, 0] = 1.0 + g
    a[-1, -1] = 1.0 + g

    return a / (4.0 + g)


def sdd_rank_one(
    n: int,
    alpha: float,
) -> np.ndarray:
    """
    Normalized I-beta*J/n family with exact DD gap alpha.
    """
    alpha = float(alpha)

    c = float(n - 2) / float(n)
    beta = (1.0 - alpha) / (1.0 + alpha * c)

    jmat = np.ones((n, n), dtype=np.float64) / float(n)
    a = np.eye(n, dtype=np.float64) - beta * jmat

    return normalize_infinity(a)


def sdd_stochastic(
    n: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    A=I-beta*P with random positive zero-diagonal row-stochastic P.

    beta=(1-alpha)/(1+alpha) gives normalized DD gap alpha.
    """
    alpha = float(alpha)

    p = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(p, 0.0)

    row_sums = p.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0.0):
        raise RuntimeError("zero stochastic row")

    p /= row_sums

    beta = (1.0 - alpha) / (1.0 + alpha)
    a = np.eye(n, dtype=np.float64) - beta * p

    return a / (1.0 + beta)


def spd_prescribed_spectrum(
    n: int,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = random_orthogonal(n, rng)
    eigenvalues = np.geomspace(1.0, 1.0 / float(kappa), n)

    a = (q * eigenvalues[np.newaxis, :]) @ q.T
    a = 0.5 * (a + a.T)

    return normalize_infinity(a)


def pascal_matrix(n: int) -> np.ndarray:
    """
    Symmetric Pascal matrix, globally scaled before exponentiation.

    The positive scalar normalization preserves total positivity while
    avoiding immediate overflow of the binomial coefficients.
    """
    logarithms = np.empty((n, n), dtype=np.float64)

    maximum = -np.inf

    for i in range(n):
        for j in range(n):
            value = (
                math.lgamma(i + j + 1.0)
                - math.lgamma(i + 1.0)
                - math.lgamma(j + 1.0)
            )

            logarithms[i, j] = value
            maximum = max(maximum, value)

    with np.errstate(under="ignore"):
        a = np.exp(logarithms - maximum)

    return normalize_infinity(a)


def build_matrix(
    family: str,
    n: int,
    parameters: dict[str, Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))

    if family == "random_conditioned":
        return random_conditioned(
            n=n,
            condition_number=float(parameters["condition_number"]),
            profile=str(parameters["profile"]),
            rng=rng,
        )

    if family == "hilbert":
        return hilbert_matrix(n)

    if family == "vandermonde":
        return vandermonde_matrix(
            n=n,
            node_type=str(parameters["node_type"]),
            rng=rng,
            cluster_parameter=float(
                parameters.get("cluster_parameter", 10.0)
            ),
            column_scaling=bool(
                parameters.get("column_scaling", False)
            ),
        )

    if family == "kahan_type":
        return kahan_type_matrix(
            n=n,
            theta=float(parameters["theta"]),
            column_scaling=bool(
                parameters.get("column_scaling", False)
            ),
        )

    if family == "adversarial_rotation":
        return adversarial_rotation_blocks(
            n=n,
            epsilon=float(parameters["epsilon"]),
        )

    if family == "repeated_bad_pivots":
        return repeated_bad_pivots(
            n=n,
            epsilon=float(parameters["epsilon"]),
            decay=float(parameters["decay"]),
        )

    if family == "scaled_adversarial":
        return scaled_adversarial(
            n=n,
            epsilon=float(parameters["epsilon"]),
            scale_range=float(parameters["scale_range"]),
            rng=rng,
        )

    if family == "row_diagonal_dominant_random":
        return sdd_random(
            n=n,
            alpha=float(parameters["alpha"]),
            rng=rng,
        )

    if family == "row_diagonal_dominant_tridiagonal":
        return sdd_tridiagonal(
            n=n,
            alpha=float(parameters["alpha"]),
        )

    if family == "row_diagonal_dominant_rankone":
        return sdd_rank_one(
            n=n,
            alpha=float(parameters["alpha"]),
        )

    if family == "row_diagonal_dominant_stochastic":
        return sdd_stochastic(
            n=n,
            alpha=float(parameters["alpha"]),
            rng=rng,
        )

    if family == "spd_prescribed_spectrum":
        return spd_prescribed_spectrum(
            n=n,
            kappa=float(parameters["kappa"]),
            rng=rng,
        )

    if family == "tnn_vandermonde_positive":
        return normalize_infinity(
            vandermonde_matrix(
                n=n,
                node_type=str(parameters["node_type"]),
                rng=rng,
                cluster_parameter=float(
                    parameters.get("cluster_parameter", 2.0)
                ),
                column_scaling=False,
            )
        )

    if family == "pascal":
        return pascal_matrix(n)

    raise ValueError(f"unknown matrix family: {family}")
