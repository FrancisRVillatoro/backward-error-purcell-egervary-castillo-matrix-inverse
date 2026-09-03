# Growth, Pivoting, and Backward Error in the Purcell–Egerváry–Castillo Matrix Inverse

Reproducibility repository for the manuscript **“Growth, Pivoting, and Backward Error in the Purcell–Egerváry–Castillo Matrix Inverse”**
by Francisco R. Villatoro.

Repository:

    https://github.com/FrancisRVillatoro/backward-error-purcell-egervary-castillo-matrix-inverse

## Scope

This project studies the backward error and numerical stability of explicitly
formed matrix inverses produced by the Purcell–Egerváry–Castillo pivoting
transformation.

The repository contains:

- the frozen numerical implementation;
- matrix-family and deterministic campaign generators;
- canonical configuration and manifests;
- Slurm launchers used on the Picasso HPC system;
- recovery and finalization programs;
- theory, diagonal-dominance, defect-replay, and validation audits;
- post-processing programs and compact final reports;
- manuscript and supplementary LaTeX sources;
- provenance and SHA-256 manifests.

Version 1.0.0 preserves the original archival computational record.
Version 1.1.0 adds the LAPACK `xGETRF+xGETRI` explicit-inverse baseline,
the exact deterministic balanced-sample reconstruction, the strict
six-method paired comparison, and the associated metric-validity audit.

## Computational record

The reported numerical experiments include:

- a mechanistic campaign with 18,744 matrix/right-hand-side configurations;
- a recovered large-scale campaign with 14,686,763 complete matrices;
- 146,867,630 inverse records;
- 1,321,808,670 solution records;
- targeted diagonal-dominance, defect-replay, compensation, quadrature,
  and high-precision validation audits.

## LAPACK explicit-inverse baseline — version 1.1.0

The revised numerical comparison adds a production explicit-inverse baseline
using LAPACK `SGETRF+SGETRI` in binary32 and `DGETRF+DGETRI` in binary64.

The baseline evaluates exactly 1,664,000 balanced matrices per precision.
LAPACK completes all binary64 cases. In binary32, `SGETRF` reports an exactly
zero factorization pivot in 8,723 cases (0.52421875%); 99.31% of those events
belong to the `random_conditioned` family.

The strict common-success, common-finite-metric comparison of the five Castillo
variants and LAPACK contains 1,629,883 binary32 matrices and 1,663,993
binary64 matrices.

See `docs/LAPACK_BASELINE.md` and `reports/lapack_baseline/`.

## Public reproducibility dataset

Dataset title:

**Backward Error of the Purcell–Egerváry–Castillo Matrix Inverse:
Reproducibility Dataset**

Version:

    1.0.0

Public release asset:

    castillo_stability_reproducibility_v1.0.0.tar.zst

Exact asset size:

    869,259,584 bytes

SHA-256:

    7e2865fa6897883db8b49a04e2dbe6be3d51b4394218b66de1a8cb150ad3eadf

Reserved Zenodo dataset DOI:

    10.5281/zenodo.21794454

The DOI becomes active when the Zenodo record is published.

The public asset contains:

- the frozen scientific code and configurations;
- the canonical campaign manifest;
- all 999 recovered-task status records;
- the recovered-population prefix certificate;
- compact analysis data;
- the complete defect-replay record;
- final reports and audit summaries;
- software-environment information;
- content and provenance manifests.

## Retained inverse and solution outputs

The complete recovered inverse and solution collection is not included in the
public version 1.0.0 asset.

It consists of:

- 999 inverse gzip CSV shards;
- 999 solution gzip CSV shards;
- 1,998 files;
- 136,127,339,793 bytes.

These working data are retained by the author during peer review so that
additional statistics or reviewer-requested calculations can be performed
without rerunning the complete campaign.

The exact population used in the manuscript remains independently identifiable
from:

- `reports/canonical/canonical_manifest.csv`;
- `config/canonical.json`;
- `config/execution.json`;
- the 999 recovered-task status JSON files in the public dataset;
- `reports/recovered/recovered_prefix_certificate.json`;
- `reports/recovered/recovered_prefix_certificate.csv`;
- `reports/recovered/recovered_bulk_sha256.txt`;
- `reports/recovered/recovered_bulk_sizes.txt`.

The prefix certificate verifies that every recovered task is the deterministic
prefix of complete paired matrix groups generated from the frozen manifest,
task partition, seed namespace, and matrix-identifier rule.

## Repository layout

- `code/` — frozen numerical algorithms, campaign drivers, and analyses.
- `config/` — canonical campaign and analysis configurations.
- `slurm/` — Picasso/Slurm scripts used for the computational record.
- `reports/` — compact final reports, certificates, and manifests.
- `release_tools/` — independent release-verification tools.
- `paper/` — manuscript and supplementary LaTeX sources.
- `docs/` — data-archive and reproducibility documentation.

Absolute Picasso paths in historical Slurm scripts and reports are preserved
intentionally as part of the computational record. They must be adapted when
running the software on another system.

## Verification

Verify the public asset with:

    sha256sum -c castillo_stability_reproducibility_v1.0.0.tar.zst.sha256

Test the compressed archive with:

    zstd -t castillo_stability_reproducibility_v1.0.0.tar.zst

Inspect its contents without extracting:

    zstd -dc castillo_stability_reproducibility_v1.0.0.tar.zst | tar -tf -

See `docs/DATA_ARCHIVE.md` and `docs/REPRODUCIBILITY.md` for further details.

## Citation

GitHub uses `CITATION.cff` to display the software citation. The reproducibility dataset should be cited using DOI

    10.5281/zenodo.21794454

after the Zenodo record has been published.

## Licenses

- Source code: MIT License (`LICENSE`).
- Research data, manifests, numerical summaries, and derived reports:
  Creative Commons Attribution 4.0 International (`DATA_LICENSE.md`).
- Manuscript text: subject to the terms applicable to the journal article.
