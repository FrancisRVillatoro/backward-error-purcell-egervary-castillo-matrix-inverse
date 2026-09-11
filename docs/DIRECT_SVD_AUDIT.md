# Direct-SVD follow-up audit

This directory contains the reproducibility material for the prespecified
2496-matrix direct-SVD follow-up audit associated with:

**Growth, Pivoting, and Backward Error in the Purcell–Egerváry–Castillo Matrix Inverse**

Francisco R. Villatoro

## Dependency

The audit reconstructs its tested matrices from the frozen balanced campaign
archived in reproducibility release v1.0.0:

- Zenodo v1.0.0 DOI: `10.5281/zenodo.22648665`
- v1.0.0 archive SHA-256:
  `5b54ecbbd667c0a6ef595cb7f6824ea9b3b2863204b07d404c0f6f169d9c5f39`

## Prespecified population

Three fixed balanced-sample ranks, `0`, `1000`, and `1999`, are reconstructed
from each of 832 stochastic blocks, giving 2496 matrices. Five Castillo variants
and LAPACK `xGETRF+xGETRI` are recomputed in binary32 and binary64.

## Direct-SVD metric

Direct binary64 SVD replaces the deterministic six-iteration production norm
estimator in this follow-up comparison. Residual matrices are formed in binary64
before the SVD. Thus the audit evaluates the spectral norm of the binary64
residual matrix actually formed; it is not an interval certification of the
exact residual of the stored operands.

## Paired inference

For each Castillo variant, the LAPACK comparison uses methodwise common-success
pairs. Pointwise percentile 95% intervals use 2000 block-bootstrap replications
with `block_index` as the resampling unit. They are not simultaneous intervals
and do not include floating-point residual-formation uncertainty.

Favorable majority-win fractions and medians do not imply uniform or tailwise
dominance. The upper paired-ratio tail is substantially larger for `R0_C1` and
`R0_C2` than for the row-pivoted variants.

## Files

- `code/direct_svd/build_direct_svd_manifest.py`
- `code/direct_svd/direct_svd_crosscheck.py`
- `code/direct_svd/direct_svd_crosscheck_finalize.py`
- `code/direct_svd/direct_svd_methodwise_pair_bootstrap.py`
- `slurm/direct_svd/direct_svd_crosscheck.slurm`
- `slurm/direct_svd/direct_svd_crosscheck_finalize.slurm`
- `reports/direct_svd/direct_svd_estimator_accuracy.csv`
- `reports/direct_svd/direct_svd_methodwise_pair_bootstrap.csv`

The 832 intermediate task CSV files are generated reproducibly from the frozen
v1.0.0 campaign and are not stored in GitHub.
