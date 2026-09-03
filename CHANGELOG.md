# Changelog

## 1.1.0 — LAPACK baseline and strict paired audit

- Adds an explicit LAPACK `xGETRF+xGETRI` production baseline on the exact
  deterministic balanced sample.
- Adds strict paired comparisons of LAPACK with the five Castillo variants.
- Adds common finite-metric accounting for the residual comparison.
- Documents 8,723 binary32 `SGETRF` zero-pivot events and zero binary64
  LAPACK failures in the balanced sample.
- Documents the 40 successful recovered Castillo inverse records for which
  the requested two-norm residual diagnostic is unavailable.
- Adds compact LAPACK and paired-audit reports and Picasso execution scripts.
- Updates software metadata to version 1.1.0.


## 1.0.0 — archival release

- Reproducibility release accompanying the SIMAX manuscript.
- Includes the canonical numerical implementation, campaign configuration,
  Slurm scripts, analysis/audit code, compact reports, figures, and paper
  sources.
- Synchronizes the diagonal-dominance residual audit with `gamma_(n+3)`.
- Includes the independent 96-case DD regression verifier.
- Full raw and recovered campaign outputs are archived in the linked Zenodo
  dataset record rather than GitHub.
