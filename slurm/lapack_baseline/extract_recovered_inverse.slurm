#!/bin/bash
#SBATCH --job-name=cst_pair_ext
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/pair_extract_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/pair_extract_%j.err

module purge

ARCHIVE="$HOME/castillo_picasso_complete_20260804.tar.zst"
DEST="$HOME/fscratch/castillo_lapack_paired/recovered_inverse"
BASE="$HOME/castillo_lapack_baseline"

ROOT="castillo_picasso_complete_20260804/FSCRATCH/castillo_stability_campaign/results/recovered/inverse"

MEMBERS="$BASE/recovered_inverse_members_detailed.txt"

echo "================================================================"
echo "CASTILLO PAIRED AUDIT -- RECOVERED INVERSE EXTRACTION"
echo "================================================================"
date
hostname

echo
echo "===== DESTINATION ====="
echo "$DEST"

echo
echo "===== FSCRATCH BEFORE ====="
quota
du -sh "$HOME/fscratch/castillo_lapack_paired" 2>/dev/null

echo
echo "===== INVENTORY ARCHIVE MEMBERS ====="

tar --zstd -tvf "$ARCHIVE" 2>"$BASE/recovered_inverse_inventory.err" \
| grep -E '/results/recovered/inverse/inverse_[0-9]{4}\.csv\.gz$' \
> "$MEMBERS"

INVENTORY_STATUS=$?

COUNT=$(wc -l < "$MEMBERS")

BYTES=$(awk '{s += $3} END {printf "%.0f", s}' "$MEMBERS")

GIB=$(awk -v b="$BYTES" 'BEGIN {printf "%.6f", b/1073741824}')

echo "inventory_status=$INVENTORY_STATUS"
echo "member_count=$COUNT"
echo "member_bytes=$BYTES"
echo "member_GiB=$GIB"

echo
echo "===== FIRST MEMBERS ====="
head -5 "$MEMBERS"

echo
echo "===== LAST MEMBERS ====="
tail -5 "$MEMBERS"

echo
echo "===== EXTRACTION GATE ====="

EXTRACT_GATE=NO

if [ "$INVENTORY_STATUS" -eq 0 ] && [ "$COUNT" -eq 999 ] && [ "$BYTES" -gt 0 ]; then
    EXTRACT_GATE=YES
fi

echo "EXTRACT_GATE=$EXTRACT_GATE"

echo
echo "===== EXTRACT ====="

TAR_STATUS=999

if [ "$EXTRACT_GATE" = "YES" ]; then

    tar --zstd -xf "$ARCHIVE" \
        -C "$DEST" \
        --strip-components=6 \
        "$ROOT" \
        2>"$BASE/recovered_inverse_extract.err"

    TAR_STATUS=$?

else
    echo "EXTRACTION_NOT_STARTED"
fi

echo "tar_status=$TAR_STATUS"

echo
echo "===== TAR STDERR ====="
cat "$BASE/recovered_inverse_extract.err" 2>/dev/null

echo
echo "===== EXTRACTED FILE AUDIT ====="

ACTUAL_COUNT=$(find "$DEST" \
    -maxdepth 1 \
    -type f \
    -name 'inverse_????.csv.gz' \
    | wc -l)

ACTUAL_BYTES=$(find "$DEST" \
    -maxdepth 1 \
    -type f \
    -name 'inverse_????.csv.gz' \
    -printf '%s\n' \
    | awk '{s += $1} END {printf "%.0f", s}')

ACTUAL_GIB=$(awk -v b="$ACTUAL_BYTES" 'BEGIN {printf "%.6f", b/1073741824}')

echo "actual_count=$ACTUAL_COUNT"
echo "actual_bytes=$ACTUAL_BYTES"
echo "actual_GiB=$ACTUAL_GIB"

echo
echo "===== FILENAME COVERAGE ====="

python3 - "$DEST" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

actual = sorted(
    p.name
    for p in root.glob("inverse_????.csv.gz")
)

expected = [
    f"inverse_{i:04d}.csv.gz"
    for i in range(999)
]

missing = sorted(set(expected) - set(actual))
unexpected = sorted(set(actual) - set(expected))

print("filenames_exact =", actual == expected)
print("missing_files =", len(missing))
print("unexpected_files =", len(unexpected))

if missing:
    print("missing_first20 =", missing[:20])

if unexpected:
    print("unexpected_first20 =", unexpected[:20])
PY

echo
echo "===== DISK USAGE ====="

du -sh "$DEST"
du -sh "$HOME/fscratch/castillo_lapack_paired"

echo
echo "===== FSCRATCH AFTER ====="

quota

echo
echo "================================================================"

if [ "$TAR_STATUS" -eq 0 ] && \
   [ "$ACTUAL_COUNT" -eq 999 ] && \
   [ "$ACTUAL_BYTES" -eq "$BYTES" ]; then

    echo "CASTILLO_RECOVERED_INVERSE_EXTRACTION=PASS"

else

    echo "CASTILLO_RECOVERED_INVERSE_EXTRACTION=FAIL"

fi

echo "================================================================"
date
