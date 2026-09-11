#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import csv
import json
import os
import numpy as np

RELEASE_ROOT = Path(os.environ.get("CASTILLO_REPRO_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
SOURCE = RELEASE_ROOT
SELDIR = RELEASE_ROOT / "data" / "direct_svd_selection" / "selection"
ROOT = Path(os.environ.get("CASTILLO_DIRECT_SVD_WORKDIR", str(RELEASE_ROOT / "work" / "direct_svd_crosscheck"))).resolve()
OUT = ROOT / "direct_svd_validation_manifest.csv"
META = ROOT / "direct_svd_validation_manifest.json"
SUMMARY = RELEASE_ROOT / "data" / "direct_svd_selection" / "summaries" / "selection_summary_release.json"

RANKS = (0, 1000, 1999)
EXPECTED_SELECTION_FILES = 999
EXPECTED_BLOCKS = 832
EXPECTED_ROWS_PER_BLOCK = 2000
EXPECTED_SELECTION_ROWS = EXPECTED_BLOCKS * EXPECTED_ROWS_PER_BLOCK
EXPECTED_MATRICES = EXPECTED_BLOCKS * len(RANKS)


def as_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def task_id_from_path(path):
    stem = path.stem
    prefix = "task_"
    if not stem.startswith(prefix):
        raise RuntimeError(f"unexpected selection filename {path.name}")
    return int(stem[len(prefix):])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    # Canonical manifest is metadata only.  Membership in the 832-block
    # balanced population is derived from exact_balanced_selection, which is
    # the source used by the strict six-method LAPACK comparison.
    canonical = {}
    canonical_path = SOURCE / "reports/canonical/canonical_manifest.csv"
    with canonical_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            block = int(r["block_index"])
            if block in canonical:
                raise RuntimeError(f"duplicate canonical block_index {block}")
            canonical[block] = dict(r)

    stochastic = {
        block
        for block, r in canonical.items()
        if as_bool(r.get("stochastic", False))
    }

    # This is the exact 832-block population used by the frozen strict
    # six-method LAPACK finalizer.
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary_blocks = sorted(
        summary["selected_blocks"],
        key=lambda x: int(x["block_index"]),
    )
    summary_block_ids = [int(x["block_index"]) for x in summary_blocks]
    if len(summary_block_ids) != EXPECTED_BLOCKS:
        raise RuntimeError(
            f"expected {EXPECTED_BLOCKS} selected blocks in {SUMMARY}, "
            f"got {len(summary_block_ids)}"
        )
    if len(set(summary_block_ids)) != EXPECTED_BLOCKS:
        raise RuntimeError("duplicate block_index in selection_summary_release.json")
    if not set(summary_block_ids) <= stochastic:
        bad = sorted(set(summary_block_ids) - stochastic)
        raise RuntimeError(f"balanced summary contains non-stochastic blocks: {bad[:20]}")

    for x in summary_blocks:
        b = int(x["block_index"])
        r = canonical[b]
        for key in ("cell_id", "family", "n"):
            if key in x and str(x[key]) != str(r[key]):
                raise RuntimeError(
                    f"balanced summary/canonical mismatch block={b} field={key}: "
                    f"summary={x[key]!r} canonical={r[key]!r}"
                )

    files = sorted(SELDIR.glob("task_????.npz"))
    if len(files) != EXPECTED_SELECTION_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_SELECTION_FILES} selection files, got {len(files)}"
        )

    file_task_ids = [task_id_from_path(p) for p in files]
    if file_task_ids != list(range(EXPECTED_SELECTION_FILES)):
        raise RuntimeError(
            "selection task filenames are not exactly task_0000.npz ... task_0998.npz"
        )

    wanted = set(RANKS)
    target_found = {}
    selected_blocks = set()
    block_counts = Counter()
    rank_seen = {}
    total_selection_rows = 0

    for path in files:
        selection_task = task_id_from_path(path)

        with np.load(str(path), allow_pickle=False) as z:
            block = np.asarray(z["block"], dtype=np.int64)
            replica = np.asarray(z["replica"], dtype=np.int64)
            rank = np.asarray(z["rank"], dtype=np.int64)

        if not (
            block.ndim == 1
            and replica.ndim == 1
            and rank.ndim == 1
            and block.size == replica.size == rank.size
        ):
            raise RuntimeError(f"bad selection schema in {path}")

        total_selection_rows += int(block.size)

        for bb, rr, kk in zip(block, replica, rank):
            bb = int(bb)
            rr = int(rr)
            kk = int(kk)

            if bb not in canonical:
                raise RuntimeError(f"selection references non-canonical block {bb}")
            if bb not in stochastic:
                raise RuntimeError(f"selection references non-stochastic block {bb}")
            if kk < 0 or kk >= EXPECTED_ROWS_PER_BLOCK:
                raise RuntimeError(
                    f"selection rank outside 0..{EXPECTED_ROWS_PER_BLOCK-1}: "
                    f"task={selection_task} block={bb} rank={kk}"
                )

            selected_blocks.add(bb)
            block_counts[bb] += 1

            if bb not in rank_seen:
                rank_seen[bb] = np.zeros(EXPECTED_ROWS_PER_BLOCK, dtype=np.uint16)
            rank_seen[bb][kk] += 1

            if kk in wanted:
                key = (bb, kk)
                value = (selection_task, rr)
                if key in target_found:
                    raise RuntimeError(
                        f"duplicate target block/rank pair {key}: "
                        f"old={target_found[key]} new={value}"
                    )
                target_found[key] = value

    if total_selection_rows != EXPECTED_SELECTION_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_SELECTION_ROWS} balanced-selection rows, "
            f"got {total_selection_rows}"
        )

    if len(selected_blocks) != EXPECTED_BLOCKS:
        raise RuntimeError(
            f"expected {EXPECTED_BLOCKS} balanced blocks in NPZ selection, "
            f"got {len(selected_blocks)}"
        )

    if selected_blocks != set(summary_block_ids):
        missing = sorted(set(summary_block_ids) - selected_blocks)
        extra = sorted(selected_blocks - set(summary_block_ids))
        raise RuntimeError(
            f"balanced summary/NPZ block-set mismatch: "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    bad_block_counts = {
        b: n for b, n in block_counts.items()
        if n != EXPECTED_ROWS_PER_BLOCK
    }
    if bad_block_counts:
        raise RuntimeError(
            f"balanced blocks without exactly {EXPECTED_ROWS_PER_BLOCK} rows: "
            f"{list(sorted(bad_block_counts.items()))[:20]}"
        )

    bad_rank_coverage = []
    for block in sorted(selected_blocks):
        seen = rank_seen[block]
        if not np.all(seen == 1):
            missing = int(np.count_nonzero(seen == 0))
            duplicated = int(np.count_nonzero(seen > 1))
            bad_rank_coverage.append((block, missing, duplicated))
    if bad_rank_coverage:
        raise RuntimeError(
            "balanced-selection rank coverage is not exactly once per rank: "
            f"{bad_rank_coverage[:20]}"
        )

    expected_target_keys = {
        (block, rank)
        for block in selected_blocks
        for rank in RANKS
    }
    if set(target_found) != expected_target_keys:
        missing = sorted(expected_target_keys - set(target_found))
        extra = sorted(set(target_found) - expected_target_keys)
        raise RuntimeError(
            f"target-rank coverage mismatch: missing={missing[:20]} extra={extra[:20]}"
        )

    # Deterministic task numbering: sorted canonical block_index of the exact
    # balanced population, three preregistered ranks per task.
    selected = [canonical[b] for b in summary_block_ids]

    rows = []
    for task_index, r in enumerate(selected):
        block = int(r["block_index"])
        for rank in RANKS:
            selection_task, replica = target_found[(block, rank)]
            rows.append({
                "task_index": task_index,
                "block_index": block,
                "selection_rank": rank,
                "selection_task": selection_task,
                "replica": replica,
                "cell_id": r["cell_id"],
                "family": r["family"],
                "n": int(r["n"]),
                "parameters_json": r["parameters_json"],
            })

    if len(rows) != EXPECTED_MATRICES:
        raise RuntimeError(
            f"expected {EXPECTED_MATRICES} validation matrices, got {len(rows)}"
        )

    fields = [
        "task_index", "block_index", "selection_rank", "selection_task",
        "replica", "cell_id", "family", "n", "parameters_json",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    blocks_by_n = Counter(int(canonical[b]["n"]) for b in selected_blocks)
    matrices_by_n = Counter(int(r["n"]) for r in rows)
    blocks_by_family = Counter(canonical[b]["family"] for b in selected_blocks)
    excluded_stochastic = stochastic - selected_blocks

    meta = {
        "balanced_block_metadata_source": "exact_balanced_selection/summaries/selection_summary_release.json",
        "selection_source": "exact_balanced_selection/selection/task_????.npz",
        "selection_rule": "ranks 0, 1000, and 1999 in every block of the exact balanced 832-block population",
        "selection_rule_fixed_before_outcome_evaluation": True,
        "source_selection_files": len(files),
        "source_selection_rows": total_selection_rows,
        "canonical_blocks_total": len(canonical),
        "canonical_stochastic_blocks": len(stochastic),
        "balanced_blocks": len(selected_blocks),
        "balanced_summary_matches_npz_selection": True,
        "stochastic_blocks_excluded_from_balanced": len(excluded_stochastic),
        "matrices": len(rows),
        "tasks": EXPECTED_BLOCKS,
        "matrices_per_task": len(RANKS),
        "target_ranks": list(RANKS),
        "blocks_by_n": {str(k): v for k, v in sorted(blocks_by_n.items())},
        "matrices_by_n": {str(k): v for k, v in sorted(matrices_by_n.items())},
        "blocks_by_family": dict(sorted(blocks_by_family.items())),
        "gate": True,
    }
    META.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(meta, indent=2, sort_keys=True))
    print("DIRECT_SVD_MANIFEST_OK")


if __name__ == "__main__":
    main()
