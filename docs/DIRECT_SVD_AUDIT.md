# Direct-SVD follow-up audit

Computational reproducibility material for **Growth, Pivoting, and Backward Error in the Purcell–Egerváry–Castillo Matrix Inverse**, Francisco R. Villatoro.

The complete Zenodo release contains portable code, the 999 exact balanced-selection NPZ files, the frozen 832-block selection summary, the 832 preserved direct-SVD task CSV files, canonical block metadata, and the exact numerical implementation/matrix generators.

No Picasso-specific path is required.

## Reconstruct manifest

From the extracted Zenodo release root:

```bash
export CASTILLO_REPRO_ROOT="$PWD"
export CASTILLO_DIRECT_SVD_WORKDIR="$PWD/work/direct_svd_crosscheck"
python3 code/direct_svd/build_direct_svd_manifest.py
```

## Re-finalize preserved task CSVs

```bash
export CASTILLO_REPRO_ROOT="$PWD"
export CASTILLO_DIRECT_SVD_WORKDIR="$PWD/work/direct_svd_crosscheck"
mkdir -p "$CASTILLO_DIRECT_SVD_WORKDIR"
cp -a results/direct_svd_crosscheck/tasks "$CASTILLO_DIRECT_SVD_WORKDIR/"
python3 code/direct_svd/direct_svd_crosscheck_finalize.py
python3 code/direct_svd/direct_svd_methodwise_pair_bootstrap.py
```

The resulting final CSV files must be byte-identical to `reports/direct_svd/`.

## Full rerun

After manifest reconstruction:

```bash
mkdir -p work/direct_svd_crosscheck/tasks
export CASTILLO_REPRO_ROOT="$PWD"
export CASTILLO_DIRECT_SVD_WORKDIR="$PWD/work/direct_svd_crosscheck"
sbatch slurm/direct_svd/direct_svd_crosscheck.slurm
```

The design uses ranks 0, 1000 and 1999 from each of 832 stochastic blocks (2496 matrices), five Castillo variants, LAPACK xGETRF+xGETRI, binary32/binary64, direct binary64 SVD evaluation, and 2000 block-bootstrap replications. Majority wins do not imply uniform or tailwise dominance.
