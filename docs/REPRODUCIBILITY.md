# Reproducibility notes

## Computational environment

The original campaign was executed on the Picasso HPC system at Universidad de
Málaga through Slurm.

The historical computational roots were:

    ~/castillo_stability_campaign
    ~/fscratch/castillo_stability_campaign

The archived Slurm scripts and reports intentionally preserve absolute Picasso
paths where those paths form part of the execution record. Users running the
software elsewhere must adapt filesystem locations, Slurm partitions,
constraints, module names, and resource requests.

## Frozen scientific implementation

The release preserves the scientific implementation used for the manuscript,
including:

- matrix families and deterministic seed generation;
- the Purcell–Egerváry–Castillo pivoting variants;
- the canonical campaign driver;
- solution and iterative-refinement experiments;
- recovery of timeout-truncated streams;
- canonical post-processing;
- theory and diagonal-dominance audits;
- defect-replay, compensation, and quadrature analyses;
- high-precision and matrix-family verifiers.

Release and packaging tools are kept separately under `release_tools/`.

## Canonical design

The canonical configuration fixes:

- the seed namespace;
- matrix dimensions and families;
- parameter cells;
- stochastic replica counts;
- pivoting variants;
- floating-point precisions;
- right-hand-side types;
- iterative-refinement levels;
- the number of Slurm tasks.

Matrix seeds and matrix identifiers are deterministic functions of the seed
namespace, cell identifier, matrix dimension, and replica number.

The task partition is deterministic:

    global_case_index mod num_tasks = task_id

Random right-hand sides have deterministic seeds derived from the matrix
identifier and right-hand-side type.

## Complete design and recovered population

The complete canonical design contains:

- 19,200,975 matrix cases;
- 192,009,750 expected inverse records;
- 1,728,087,750 expected solution records;
- 1,935 manifest blocks;
- 129 parameter cells.

The population analyzed in the manuscript is the successfully recovered
subset:

- 14,686,763 matrices;
- 146,867,630 inverse records;
- 1,321,808,670 solution records;
- 999 recovered task streams.

For every task, recovery processes the inverse and solution streams in their
original deterministic order. It retains a matrix only when both streams
contain a complete and consistent group:

- 10 inverse records;
- 90 solution records.

Recovery stops at the first incomplete, mismatched, or inconsistent group.
It does not skip that group and continue with later matrices. Consequently,
each recovered task output is the maximal valid prefix of its deterministic
task stream.

## Population certification

Every recovered-task status record stores:

- the task identifier;
- the number of recovered matrices;
- inverse and solution record counts;
- the first and last global case indices;
- the last matrix identifier;
- counts by manifest block;
- counts by matrix family;
- counts by dimension;
- information about where recovery stopped.

The independent program

    release_tools/verify_recovered_prefixes.py

reconstructs all 999 prefixes from the canonical manifest, deterministic task
partition, seed namespace, and matrix-identifier rule.

The resulting certificate is:

    reports/recovered/recovered_prefix_certificate.json

It verifies:

- 999 of 999 tasks;
- 14,686,763 matrices;
- 146,867,630 inverse records;
- 1,321,808,670 solution records;
- all first and last case indices;
- all last matrix identifiers;
- all block, family, and dimension counts;
- no certification failures.

## Public and retained data

The public version 1.0.0 reproducibility asset contains:

- frozen code and configurations;
- the canonical manifest;
- all recovered-task status records;
- the recovered-population certificate;
- compact analysis data;
- complete defect-replay data;
- final reports;
- software-environment information;
- SHA-256 provenance and content manifests.

The complete recovered inverse and solution CSV collection consists of 1,998
gzip shards occupying 136,127,339,793 bytes. It is retained by the author
during peer review and is not included in the public version 1.0.0 dataset.

The name, exact size, and SHA-256 digest of every retained bulk file are
documented in:

    reports/recovered/recovered_bulk_sizes.txt
    reports/recovered/recovered_bulk_sha256.txt

## Reproducing the complete campaign

A full rerun requires substantial computing time and storage. The principal
workflow is represented by the scripts under `slurm/`:

1. generate the canonical manifest;
2. execute the canonical task array;
3. recover valid prefixes from timeout-truncated streams where necessary;
4. finalize and summarize the recovered campaign;
5. select the analysis populations;
6. execute the analysis shards and bootstrap calculations;
7. run theory, diagonal-dominance, and defect-replay audits;
8. generate the final reports.

The scripts preserve the Picasso execution settings used for the manuscript.
They should be reviewed and adapted before use on another HPC system.

## Numerical reproducibility

The campaign design, seeds, matrix identifiers, task assignment, right-hand-side
generation, and reported recovered population are deterministic.

Exact byte-for-byte equality of every floating-point output is not required
across different processors, BLAS or LAPACK implementations, compiler stacks,
NumPy or SciPy builds, long-double implementations, or gzip versions.

Small platform-dependent floating-point differences may occur. Independent
reproductions should compare:

- documented diagnostics and tolerances;
- aggregate distributions and quantiles;
- fitted scaling behavior;
- failure classifications;
- theoretical inequalities;
- qualitative conclusions.

The compressed CSV files themselves may also differ at the byte level because
gzip metadata and software versions can differ even when the uncompressed
records are numerically equivalent.

## Integrity

The repository includes:

    REPOSITORY_CONTENT_SHA256.txt
    RELEASE_PROVENANCE_SHA256.txt
    RELEASE_ASSET_SHA256.txt

The public dataset includes its own internal content manifest and the hash and
size inventories of the retained bulk collection.

Verify all supplied checksums before analyzing or redistributing the release.


## Version 1.1.0 LAPACK baseline

Version 1.1.0 adds a SciPy/LAPACK `xGETRF+xGETRI` explicit-inverse baseline
on the exact deterministic balanced sample used by the large-scale Castillo
analysis.

The frozen execution programs are under `code/lapack_baseline/`, the Picasso
launchers under `slurm/lapack_baseline/`, and compact outputs under
`reports/lapack_baseline/`.

The final comparative statistics use a strict common-success and common
finite-metric sample across all six explicit-inverse methods. See
`docs/LAPACK_BASELINE.md` for sample sizes, environment versions, failure
accounting, and final quantiles.
