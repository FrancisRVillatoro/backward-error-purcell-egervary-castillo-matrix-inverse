#!/bin/bash
#SBATCH --job-name=cst_pair_find
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/find_paired_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/find_paired_%j.err

module purge

ARCHIVE="$HOME/castillo_picasso_complete_20260804.tar.zst"
BASE="$HOME/castillo_lapack_baseline"

ALL="$BASE/paired_candidate_members.txt"

echo "================================================================"
echo "CASTILLO SIX-METHOD PAIRED ARTIFACT DISCOVERY"
echo "================================================================"
date
hostname

echo
echo "===== ARCHIVE ====="
ls -lh "$ARCHIVE"

echo
echo "===== SCAN ARCHIVE ====="

tar --zstd -tf "$ARCHIVE" 2>"$BASE/paired_scan_tar.err" \
| grep -E \
'selected_[0-9]{4}\.csv\.gz$|theory_audit.*partial|analysis.*partial|balanced.*\.npz$|shard.*\.npz$|partial.*\.npz$|paired_common_quantiles\.csv$|paired_common_method_ratios\.csv$|theory_balanced_block_metrics\.csv$|selection_summary\.json$|recovered.*/inverse/inverse_[0-9]{4}\.csv\.gz$' \
| LC_ALL=C sort \
> "$ALL"

SCAN_STATUS=$?

echo "scan_status=$SCAN_STATUS"

echo
echo "===== TOTAL MATCHES ====="
wc -l "$ALL"

echo
echo "===== SELECTED PER-MATRIX CSV ====="

grep -E \
'selected_[0-9]{4}\.csv\.gz$' \
"$ALL" \
> "$BASE/paired_selected_csv_members.txt"

wc -l "$BASE/paired_selected_csv_members.txt"

echo
echo "===== NPZ PARTIAL / SHARD CANDIDATES ====="

grep -E \
'(partial|shard|balanced).*\.npz$' \
"$ALL" \
> "$BASE/paired_npz_members.txt"

wc -l "$BASE/paired_npz_members.txt"

echo
echo "===== RECOVERED INVERSE SHARDS ====="

grep -E \
'recovered.*/inverse/inverse_[0-9]{4}\.csv\.gz$' \
"$ALL" \
> "$BASE/paired_recovered_inverse_members.txt"

wc -l "$BASE/paired_recovered_inverse_members.txt"

echo
echo "===== ROOTS OF SELECTED CSV FILES ====="

grep -E \
'selected_[0-9]{4}\.csv\.gz$' \
"$ALL" \
| sed -E 's#/selected_[0-9]{4}\.csv\.gz$##' \
| sort \
| uniq -c \
| sort -nr

echo
echo "===== ROOTS OF NPZ CANDIDATES ====="

grep -E \
'(partial|shard|balanced).*\.npz$' \
"$ALL" \
| sed -E 's#/[^/]+\.npz$##' \
| sort \
| uniq -c \
| sort -nr \
| head -40

echo
echo "===== FIRST 80 MATCHES ====="
head -80 "$ALL"

echo
echo "===== LAST 80 MATCHES ====="
tail -80 "$ALL"

echo
echo "===== TAR STDERR ====="
cat "$BASE/paired_scan_tar.err"

echo
echo "================================================================"
echo "PAIRED_ARTIFACT_DISCOVERY_FINISHED"
echo "================================================================"
date
