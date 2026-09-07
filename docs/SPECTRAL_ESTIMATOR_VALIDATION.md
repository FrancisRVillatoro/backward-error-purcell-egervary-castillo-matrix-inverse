# Validation of the six-iteration spectral-norm diagnostic

The production campaign does not compute exact spectral norms. It uses the
frozen deterministic estimator `spectral_norm_estimate(..., iterations=6)`,
denoted by $\nu_6$. The empirical normalized right residual is therefore

    rhat_R,2^(6) = nu_6(I - A Xhat) / (nu_6(A) nu_6(Xhat)).

This is a deterministic diagnostic, not a certified upper or lower bound for
the exact spectral-norm residual.

## Independent stratified audit

Picasso Slurm job **2207042** executed 48/48 array tasks successfully
(`COMPLETED`, `ExitCode 0:0`). The audit reconstructed 48 matrices from the
balanced design, used both binary32 and binary64, and evaluated the five
Castillo rules plus the LAPACK `xGETRF+xGETRI` baseline. Direct SVD norms were
used only for this validation.

For each finite positive case,

    Q_R = rhat_R,2^(6) / r_R,2^exact.

Across the twelve method--precision groups, the median is between 1.000075 and
1.002349. The groupwise q95 ranges from 1.114111 to 1.264617 and the observed
maximum from 1.292344 to 1.532479. Depending on group, 70.2--81.2% of finite
positive ratios are within 5% of one and 87.5--91.7% are within 10%.

The estimator therefore has negligible median bias in this stratified audit,
but it is not an individually high-accuracy surrogate for the spectral norm.
Large-scale tables must be interpreted as distributions of the common
six-iteration diagnostic, not as exact spectral-norm quantiles. Close rankings
among magnitude-based Castillo variants are not interpreted as exact two-norm
rankings. Rigorous theorem-backed residual statements use the infinity norm and
are independent of this estimator.

The machine-readable residual-ratio summary is
`reports/spectral_estimator_validation/residual_ratio_summary.csv` and the
frozen validation/finalization programs are under `code/validation/`.
