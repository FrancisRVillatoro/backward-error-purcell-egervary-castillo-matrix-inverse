#!/usr/bin/env python3
"""
Numerical verification of the diagonal-dominance theorem and of the
propositions defining the transformation growth factor.

The matrices are normalized so that ||A||_inf = 1 and their measured
row-diagonal-dominance gap is the requested alpha.

Run through Slurm with:

    sbatch slurm/verify_dd_theorem.slurm
"""

import math
import sys
from collections import defaultdict

import numpy as np


U64 = 2.0 ** -53


def inf_norm(x):
    return float(np.abs(x).sum(axis=1).max())


def diagonal_dominance_gap(a):
    diagonal = np.abs(np.diag(a))
    off_diagonal = np.abs(a).sum(axis=1) - diagonal
    return float(np.min(diagonal - off_diagonal))


def gamma_k(k, u):
    ku = float(k) * float(u)
    if ku >= 1.0:
        raise ValueError(f"gamma_{k} is undefined because k*u >= 1")
    return ku / (1.0 - ku)


def normalize_inf(a):
    value = inf_norm(a)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("matrix has invalid infinity norm")
    return np.asarray(a, dtype=np.float64) / value


def sdd_random(n, alpha, rng):
    """
    Dense random matrix with ||A||_inf = 1 and exact row-DD gap alpha.

    Before normalization, every off-diagonal row has absolute sum one,
    the diagonal magnitude is 1+g, and g/(2+g)=alpha.
    """
    g = 2.0 * alpha / (1.0 - alpha)

    a = rng.standard_normal((n, n))
    np.fill_diagonal(a, 0.0)

    for i in range(n):
        sigma = float(np.abs(a[i]).sum())
        if sigma == 0.0:
            a[i, (i + 1) % n] = 1.0
            sigma = 1.0

        a[i] /= sigma
        diagonal_sign = -1.0 if rng.standard_normal() < 0.0 else 1.0
        a[i, i] = diagonal_sign * (1.0 + g)

    return a / (2.0 + g)


def sdd_tridiagonal(n, alpha):
    """
    Tridiagonal matrix with ||A||_inf = 1 and exact gap alpha.

    The unnormalized interior rows have diagonal 2+g and off-diagonal
    sum two. Boundary rows have diagonal 1+g and off-diagonal sum one.
    Thus every row has gap g, while ||A||_inf=4+g.
    """
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


def sdd_rank_one(n, alpha):
    """
    Normalized rank-one family based on I-beta*J/n.

    beta is chosen so that the gap after infinity-norm normalization
    is exactly alpha.
    """
    c = float(n - 2) / float(n)
    beta = (1.0 - alpha) / (1.0 + alpha * c)

    jmat = np.ones((n, n), dtype=np.float64) / float(n)
    a = np.eye(n, dtype=np.float64) - beta * jmat

    return normalize_inf(a)


def sdd_varah(n, alpha, rng):
    """
    Random positive row-stochastic family with zero diagonal.

        A = I - beta P,

    where beta=(1-alpha)/(1+alpha).  Before normalization each row has
    norm 1+beta and gap 1-beta, hence the normalized gap is alpha.
    """
    p = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(p, 0.0)

    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0.0):
        raise RuntimeError("zero stochastic row")

    p /= row_sums

    beta = (1.0 - alpha) / (1.0 + alpha)
    a = np.eye(n, dtype=np.float64) - beta * p

    return a / (1.0 + beta)


def castillo_unpivoted(a):
    """
    Castillo transformation with V^1=I and rows in natural order.

    The first admissible column is checked explicitly.  For the
    diagonally dominant families it must always be the active column,
    so n_interchanges must remain zero.
    """
    n = a.shape[0]
    v = np.eye(n, dtype=np.float64)

    tableaux = [v.copy()]
    elementary = []
    pivots = []
    n_interchanges = 0

    for j in range(n):
        t = a[j] @ v

        admissible = np.flatnonzero(t[j:] != 0.0)
        if admissible.size == 0:
            raise RuntimeError(f"no admissible pivot at step {j}")

        p = j + int(admissible[0])

        if p != j:
            v[:, [j, p]] = v[:, [p, j]]
            t[[j, p]] = t[[p, j]]
            n_interchanges += 1

        pivot = float(t[j])
        if pivot == 0.0:
            raise RuntimeError(f"zero pivot at step {j}")

        pivots.append(pivot)

        b = np.eye(n, dtype=np.float64)
        row = -t / pivot
        row[j] = 1.0 / pivot
        b[j, :] = row

        elementary.append(b.copy())
        v = v @ b
        tableaux.append(v.copy())

    return tableaux, elementary, np.asarray(pivots), n_interchanges


def growth_factor(tableaux, elementary):
    """
    Gamma_N in infinity norm.

    tableaux[l] is V_{l+1} in zero-based Python indexing before
    transformation l, and tails[l+1] is B_{l+1}...B_N.
    """
    n = len(elementary)

    tail = np.eye(n, dtype=np.float64)
    tails = [None] * (n + 1)
    tails[n] = tail.copy()

    for l in range(n - 1, -1, -1):
        tail = elementary[l] @ tail
        tails[l] = tail.copy()

    numerator = 0.0

    for l in range(n):
        local = inf_norm(np.abs(tableaux[l]) @ np.abs(elementary[l]))
        trailing = inf_norm(tails[l + 1])
        numerator += local * trailing

    return numerator / inf_norm(tableaux[n])


def explicit_tableau(a, j):
    """
    Explicit V^{j+1} for j completed steps:

        [ A_j^{-1}   -A_j^{-1} A_j' ]
        [     0                I     ]
    """
    n = a.shape[0]
    a11 = a[:j, :j]
    a12 = a[:j, j:]

    inverse_a11 = np.linalg.solve(a11, np.eye(j))

    if j < n:
        upper = np.hstack((inverse_a11, -inverse_a11 @ a12))
        lower = np.hstack((
            np.zeros((n - j, j), dtype=np.float64),
            np.eye(n - j, dtype=np.float64),
        ))
        return np.vstack((upper, lower))

    return inverse_a11


def add_check(checks, name, ratio, metadata):
    checks[name].append((float(ratio), dict(metadata)))


def main():
    rng = np.random.default_rng(20260728)

    families = [
        ("sdd_random", lambda n, alpha: sdd_random(n, alpha, rng)),
        ("tridiagonal", sdd_tridiagonal),
        ("rank_one", sdd_rank_one),
        ("varah", lambda n, alpha: sdd_varah(n, alpha, rng)),
    ]

    dimensions = [8, 16, 32, 64]
    alphas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    checks = defaultdict(list)
    empirical_gamma_ge_n = []
    cases = 0

    for family_name, generator in families:
        for n in dimensions:
            for requested_alpha in alphas:
                a = generator(n, requested_alpha)
                measured_alpha = diagonal_dominance_gap(a)

                metadata = {
                    "family": family_name,
                    "n": n,
                    "requested_alpha": requested_alpha,
                    "measured_alpha": measured_alpha,
                }

                norm_a = inf_norm(a)

                add_check(
                    checks,
                    "||A||_inf=1",
                    abs(norm_a - 1.0) / (1000.0 * U64),
                    metadata,
                )

                add_check(
                    checks,
                    "gap=alpha",
                    abs(measured_alpha - requested_alpha)
                    / (10000.0 * U64 * max(1.0, abs(requested_alpha))),
                    metadata,
                )

                if measured_alpha <= 0.0:
                    raise RuntimeError(
                        f"nonpositive DD gap for {metadata}"
                    )

                tableaux, elementary, pivots, n_interchanges = (
                    castillo_unpivoted(a)
                )

                gamma_n3 = gamma_k(n + 3, U64)
                gamma_2 = gamma_k(2, U64)
                growth = growth_factor(tableaux, elementary)

                residual = (
                    inf_norm(np.eye(n) - a @ tableaux[n])
                    / (inf_norm(a) * inf_norm(tableaux[n]))
                )

                add_check(
                    checks,
                    "no_interchanges",
                    float(n_interchanges),
                    metadata,
                )

                add_check(
                    checks,
                    "V<=2/alpha",
                    max(inf_norm(v) for v in tableaux)
                    / (2.0 / measured_alpha),
                    metadata,
                )

                add_check(
                    checks,
                    "Vinv<=1",
                    max(
                        inf_norm(np.linalg.inv(v))
                        for v in tableaux
                    ),
                    metadata,
                )

                add_check(
                    checks,
                    "B<=3/alpha^2",
                    max(inf_norm(b) for b in elementary)
                    / (3.0 / measured_alpha**2),
                    metadata,
                )

                add_check(
                    checks,
                    "|pivot|>=alpha",
                    measured_alpha / np.min(np.abs(pivots)),
                    metadata,
                )

                proved_floor = n / (1.0 + n * gamma_2)

                add_check(
                    checks,
                    "Gamma>=proved_floor",
                    proved_floor / growth,
                    metadata,
                )

                empirical_gamma_ge_n.append(
                    (n / growth, dict(metadata))
                )

                add_check(
                    checks,
                    "Gamma<=6n/alpha^3",
                    growth / (6.0 * n / measured_alpha**3),
                    metadata,
                )

                add_check(
                    checks,
                    "rR<=gamma_{n+3}*Gamma",
                    residual / (gamma_n3 * growth),
                    metadata,
                )

                worst_tableau_error = 0.0

                for j in range(1, n + 1):
                    reference = explicit_tableau(a, j)
                    denominator = max(
                        np.linalg.norm(reference),
                        np.finfo(np.float64).tiny,
                    )
                    relative_error = (
                        np.linalg.norm(tableaux[j] - reference)
                        / denominator
                    )
                    scaled_error = (
                        relative_error
                        / (10000.0 * n * n * U64)
                    )
                    worst_tableau_error = max(
                        worst_tableau_error,
                        scaled_error,
                    )

                add_check(
                    checks,
                    "explicit_tableau",
                    worst_tableau_error,
                    metadata,
                )

                cases += 1

    print(f"CASES={cases}")
    print(
        f"{'check':>25}  {'worst ratio':>14}  "
        f"{'status':>8}  worst case"
    )

    all_ok = True

    for name in sorted(checks):
        worst_ratio, metadata = max(
            checks[name],
            key=lambda item: item[0],
        )

        passed = worst_ratio <= 1.0 + 1.0e-12
        all_ok = all_ok and passed
        status = "OK" if passed else "FAIL"

        print(
            f"{name:>25}  {worst_ratio:14.6e}  "
            f"{status:>8}  {metadata}"
        )

    empirical_worst, empirical_metadata = max(
        empirical_gamma_ge_n,
        key=lambda item: item[0],
    )

    print()
    print(
        "EMPIRICAL_N_OVER_GAMMA_MAX="
        f"{empirical_worst:.16e} {empirical_metadata}"
    )
    print(
        "The empirical Gamma>=N check is reported but is not used "
        "as a theorem assertion."
    )

    if not all_ok:
        print("DD_THEOREM_VERIFICATION_FAILED")
        sys.exit(1)

    print("DD_THEOREM_VERIFICATION_OK")


if __name__ == "__main__":
    main()
