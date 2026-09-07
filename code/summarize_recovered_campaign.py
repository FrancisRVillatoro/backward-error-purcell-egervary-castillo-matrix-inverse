#!/usr/bin/env python3
"""Summarize statistical completeness of recovered timeout shards."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def classify_block(expected, recovered, stochastic):
    if recovered > expected:
        return "invalid_overshoot"

    if recovered == expected:
        return "complete"

    if recovered == 0:
        return "absent"

    if not stochastic:
        return "partial_deterministic"

    if recovered >= 10000:
        return "partial_high"

    if recovered >= 5000:
        return "partial_5000"

    if recovered >= 2000:
        return "partial_2000"

    if recovered >= 500:
        return "partial_500"

    if recovered >= 200:
        return "partial_200"

    return "partial_sparse"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--recovered-root",
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    recovered_root = Path(args.recovered_root)
    report_dir = Path(args.report_dir)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest = list(csv.DictReader(handle))

    status_paths = sorted(
        (recovered_root / "status").glob(
            "task_*.json"
        )
    )

    statuses = []

    for path in status_paths:
        statuses.append(
            json.loads(
                path.read_text(encoding="utf-8")
            )
        )

    by_task = {
        int(status["task_id"]): status
        for status in statuses
    }

    missing_tasks = [
        task_id
        for task_id in range(999)
        if task_id not in by_task
    ]

    non_ok_tasks = [
        task_id
        for task_id, status in sorted(by_task.items())
        if status.get("status") != "RECOVERY_OK"
    ]

    recovered_by_block = Counter()

    for status in statuses:
        for key, value in status.get(
            "block_counts",
            {},
        ).items():
            recovered_by_block[int(key)] += int(value)

    block_rows = []
    class_counts = Counter()

    family_expected = Counter()
    family_recovered = Counter()

    n_expected = Counter()
    n_recovered = Counter()

    n_block_states = defaultdict(list)

    total_expected = 0
    total_recovered = 0

    for row in manifest:
        block_index = int(row["block_index"])
        expected = int(row["replicas"])
        recovered = int(
            recovered_by_block.get(block_index, 0)
        )

        stochastic = (
            row["stochastic"].strip().lower()
            == "true"
        )

        state = classify_block(
            expected,
            recovered,
            stochastic,
        )

        class_counts[state] += 1

        family = row["family"]
        n = int(row["n"])

        total_expected += expected
        total_recovered += recovered

        family_expected[family] += expected
        family_recovered[family] += recovered

        n_expected[n] += expected
        n_recovered[n] += recovered

        n_block_states[n].append(state)

        block_rows.append(
            {
                "block_index": block_index,
                "cell_id": row["cell_id"],
                "family": family,
                "stochastic": stochastic,
                "n": n,
                "parameters_json":
                    row["parameters_json"],
                "expected_matrices": expected,
                "recovered_matrices": recovered,
                "completion_fraction": (
                    recovered / expected
                    if expected
                    else 0.0
                ),
                "state": state,
            }
        )

    overshoots = [
        row
        for row in block_rows
        if row["state"] == "invalid_overshoot"
    ]

    block_csv = (
        report_dir
        / "recovered_completion_by_block.csv"
    )

    with block_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(block_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(block_rows)

    family_csv = (
        report_dir
        / "recovered_completion_by_family.csv"
    )

    with family_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [
            "family",
            "expected_matrices",
            "recovered_matrices",
            "completion_fraction",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for family in sorted(family_expected):
            expected = family_expected[family]
            recovered = family_recovered[family]

            writer.writerow(
                {
                    "family": family,
                    "expected_matrices": expected,
                    "recovered_matrices": recovered,
                    "completion_fraction":
                        recovered / expected,
                }
            )

    n_csv = (
        report_dir
        / "recovered_completion_by_n.csv"
    )

    with n_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [
            "n",
            "expected_matrices",
            "recovered_matrices",
            "completion_fraction",
            "all_blocks_complete",
            "minimum_block_recovered",
            "maximum_block_recovered",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for n in sorted(n_expected):
            rows_n = [
                row
                for row in block_rows
                if row["n"] == n
            ]

            expected = n_expected[n]
            recovered = n_recovered[n]

            writer.writerow(
                {
                    "n": n,
                    "expected_matrices": expected,
                    "recovered_matrices": recovered,
                    "completion_fraction":
                        recovered / expected,
                    "all_blocks_complete":
                        all(
                            row["state"] == "complete"
                            for row in rows_n
                        ),
                    "minimum_block_recovered":
                        min(
                            row["recovered_matrices"]
                            for row in rows_n
                        ),
                    "maximum_block_recovered":
                        max(
                            row["recovered_matrices"]
                            for row in rows_n
                        ),
                }
            )

    total_inverse_rows = sum(
        int(status["recovered_inverse_rows"])
        for status in statuses
    )

    total_solution_rows = sum(
        int(status["recovered_solution_rows"])
        for status in statuses
    )

    errors = []

    if missing_tasks:
        errors.append("missing recovery tasks")

    if non_ok_tasks:
        errors.append("non-OK recovery tasks")

    if overshoots:
        errors.append("block overshoots")

    if total_inverse_rows != 10 * total_recovered:
        errors.append("inverse row count mismatch")

    if total_solution_rows != 90 * total_recovered:
        errors.append("solution row count mismatch")

    complete_dimensions = sorted(
        n
        for n, states in n_block_states.items()
        if all(state == "complete" for state in states)
    )

    dimensions_with_2000 = sorted(
        n
        for n in n_block_states
        if all(
            row["recovered_matrices"] >= 2000
            or not row["stochastic"]
            for row in block_rows
            if row["n"] == n
        )
    )

    last_blocks = [
        int(status["last_case_index"])
        if status.get("last_case_index") is not None
        else -1
        for status in statuses
    ]

    report = {
        "status": (
            "RECOVERED_SUMMARY_OK"
            if not errors
            else "RECOVERED_SUMMARY_INCOMPLETE"
        ),
        "errors": errors,
        "tasks_expected": 999,
        "tasks_recovered": len(statuses),
        "missing_tasks": missing_tasks,
        "non_ok_tasks": non_ok_tasks,
        "expected_matrix_cases": total_expected,
        "recovered_matrix_cases": total_recovered,
        "overall_completion_fraction": (
            total_recovered / total_expected
        ),
        "recovered_inverse_rows": total_inverse_rows,
        "recovered_solution_rows": total_solution_rows,
        "block_state_counts":
            dict(sorted(class_counts.items())),
        "complete_dimensions": complete_dimensions,
        "dimensions_with_at_least_2000_per_stochastic_block":
            dimensions_with_2000,
        "minimum_last_case_index": (
            min(last_blocks) if last_blocks else None
        ),
        "maximum_last_case_index": (
            max(last_blocks) if last_blocks else None
        ),
        "completion_by_block_csv": str(block_csv),
        "completion_by_family_csv": str(family_csv),
        "completion_by_n_csv": str(n_csv),
    }

    json_path = (
        report_dir
        / "recovered_completion_summary.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path = (
        report_dir
        / "recovered_completion_summary.md"
    )

    with md_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("# Recovered Castillo campaign\n\n")
        handle.write(
            "- Status: `{}`\n".format(report["status"])
        )
        handle.write(
            "- Tasks recovered: `{}/999`\n".format(
                len(statuses)
            )
        )
        handle.write(
            "- Matrix cases: `{}/{}`\n".format(
                total_recovered,
                total_expected,
            )
        )
        handle.write(
            "- Overall fraction: `{:.6f}`\n".format(
                report["overall_completion_fraction"]
            )
        )
        handle.write(
            "- Inverse rows: `{}`\n".format(
                total_inverse_rows
            )
        )
        handle.write(
            "- Solution rows: `{}`\n".format(
                total_solution_rows
            )
        )
        handle.write(
            "- Complete dimensions: `{}`\n".format(
                complete_dimensions
            )
        )
        handle.write(
            "- Dimensions with at least 2000 "
            "matrices per stochastic block: `{}`\n".format(
                dimensions_with_2000
            )
        )

        handle.write("\n## Block states\n\n")

        for state, count in sorted(class_counts.items()):
            handle.write(
                "- `{}`: `{}`\n".format(
                    state,
                    count,
                )
            )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    if errors:
        sys.exit(1)

    print("RECOVERED_SUMMARY_OK")


if __name__ == "__main__":
    main()
