# Public reproducibility dataset

## Dataset identity

Title:

**Backward Error of the Purcell–Egerváry–Castillo Matrix Inverse:
Reproducibility Dataset**

Version:

    1.0.0

Creator:

    Francisco R. Villatoro
    Universidad de Málaga

Reserved Zenodo DOI:

    10.5281/zenodo.21794454

The DOI becomes active when the Zenodo record is published.

## Public release asset

The public dataset is distributed as one compressed archive:

    castillo_stability_reproducibility_v1.0.0.tar.zst

Exact size:

    869,259,584 bytes

SHA-256:

    7e2865fa6897883db8b49a04e2dbe6be3d51b4394218b66de1a8cb150ad3eadf

The checksum is distributed as:

    castillo_stability_reproducibility_v1.0.0.tar.zst.sha256

## Included material

The public archive includes:

- the frozen scientific code and configurations;
- the Slurm scripts used for the computational record;
- the canonical campaign manifest;
- all 999 recovered-task status JSON files;
- the recovered-population prefix certificate;
- the bulk-file size and SHA-256 manifests;
- compact analysis selection and bootstrap data;
- the complete defect-replay record;
- final analysis, theory-audit, recovery, canonical, and defect-replay reports;
- software-environment information;
- a SHA-256 manifest of every file in the archive.

## Why the complete recovered CSV collection is not included

The complete recovered inverse and solution collection contains:

- 999 inverse gzip CSV shards;
- 999 solution gzip CSV shards;
- 1,998 files in total;
- 136,127,339,793 bytes.

Those files are retained by the author during peer review because they may be
needed for reviewer-requested recalculation, additional stratification, or
new diagnostic summaries.

They are not included in the public version 1.0.0 asset because the compact
release already contains the information and provenance needed to identify
exactly which matrices and complete record groups constitute the campaign
reported in the manuscript.

## Exact identification of the reported population

The manuscript population contains:

- 14,686,763 matrices;
- 146,867,630 inverse records;
- 1,321,808,670 solution records;
- 999 recovered task streams.

For each task, the recovery program reads the inverse and solution streams in
their deterministic generation order. It retains only complete paired matrix
groups and stops at the first incomplete or inconsistent group.

The exact recovered prefix for every task is recorded in the corresponding
status JSON. The independent certificate

    reports/recovered/recovered_prefix_certificate.json

reconstructs every prefix from the canonical manifest, deterministic task
partition, seed namespace, and matrix-identifier rule. It verifies all 999
tasks and all aggregate record counts without reading the large recovered CSV
files.

The original bulk collection is additionally documented by:

    reports/recovered/recovered_bulk_sha256.txt
    reports/recovered/recovered_bulk_sizes.txt

Thus, the public release defines the exact reported population even though it
does not distribute all 136,127,339,793 original recovered bytes.

## Verify

Verify the downloaded asset:

    sha256sum -c castillo_stability_reproducibility_v1.0.0.tar.zst.sha256

Test the compressed archive:

    zstd -t castillo_stability_reproducibility_v1.0.0.tar.zst

Inspect its contents without extracting:

    zstd -dc castillo_stability_reproducibility_v1.0.0.tar.zst | tar -tf -

Extract it:

    zstd -dc castillo_stability_reproducibility_v1.0.0.tar.zst | tar -xf -

## Licensing

Research data, manifests, numerical summaries, and derived reports are
licensed under Creative Commons Attribution 4.0 International.

Source code contained in the archive is licensed separately under the MIT
License.
