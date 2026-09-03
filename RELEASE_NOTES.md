# Release 1.1.0

## Title

**Growth, Pivoting, and Backward Error in the Purcell–Egerváry–Castillo Matrix Inverse**

## Scope

Version 1.1.0 extends the original reproducibility release with the production
explicit-inverse benchmark used in the revised manuscript.

The original numerical campaign and the version 1.0.0 tag remain unchanged.

## New in version 1.1.0

- LAPACK `SGETRF+SGETRI` binary32 baseline.
- LAPACK `DGETRF+DGETRI` binary64 baseline.
- Exact reconstruction of the deterministic balanced sample.
- Strict six-method common-success comparison.
- Strict common finite-metric comparison.
- LAPACK failure localization by family and dimension.
- Full recovered-population audit of unavailable residual diagnostics.
- Compact machine-readable final reports.
- Frozen Picasso execution and post-processing scripts.

## Balanced baseline

Each precision contains 1,664,000 matrices in 832 stochastic blocks.

LAPACK binary32:

- 1,655,277 successes;
- 8,723 `SGETRF` zero-pivot failures;
- failure rate 0.52421875%.

LAPACK binary64:

- 1,664,000 successes;
- zero failures.

The strict finite-metric six-method comparison contains 1,629,883 binary32
matrices and 1,663,993 binary64 matrices.

## Interpretation

The first-nonzero Castillo variant `R0_C0` is clearly worse than LAPACK in
the normalized inverse-residual diagnostic. The four magnitude-based variants
are competitive with LAPACK and have smaller blockwise q95 values in both
precisions. The data do not support a uniform matrix-by-matrix superiority
claim.

## Provenance

The complete 999-shard recovered campaign is not duplicated in this software
release. Version 1.1.0 contains compact derived reports and the exact analysis
programs. The original campaign population remains tied to the version 1.0.0
reproducibility dataset DOI:

    10.5281/zenodo.21794454

See `docs/LAPACK_BASELINE.md`.
