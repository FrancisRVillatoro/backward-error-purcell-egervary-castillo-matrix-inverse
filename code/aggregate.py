#!/usr/bin/env python3
"""Validate and summarize the sharded canonical campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def human_bytes(
    value: int,
) -> str:
    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "PiB",
    ]

    x = float(value)
    index = 0

    while (
        x >= 1024.0
        and index + 1 < len(units)
    ):
        x /= 1024.0
        index += 1

    return "{:.3f} {}".format(
        x,
        units[index],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-summary",
        required=True,
    )

    parser.add_argument(
        "--execution",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        required=True,
    )

    args = parser.parse_args()

    manifest_summary = json.loads(
        Path(
            args.manifest_summary
        ).read_text(
            encoding="utf-8"
        )
    )

    execution = json.loads(
        Path(args.execution).read_text(
            encoding="utf-8"
        )
    )

    outdir = Path(args.outdir)
    report_dir = Path(args.report_dir)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_tasks = int(
        execution["array_tasks"]
    )

    status_dir = (
        outdir / "status"
    )

    status_paths = sorted(
        status_dir.glob(
            "task_*.json"
        )
    )

    statuses: List[
        Dict[str, Any]
    ] = []

    invalid_status_files: List[
        str
    ] = []

    for path in status_paths:
        try:
            statuses.append(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception as exc:
            invalid_status_files.append(
                "{}: {}".format(
                    path,
                    repr(exc),
                )
            )

    by_task: Dict[
        int,
        Dict[str, Any],
    ] = {}

    duplicate_tasks: List[int] = []

    for status in statuses:
        task_id = int(
            status.get(
                "task_id",
                -1,
            )
        )

        if task_id in by_task:
            duplicate_tasks.append(
                task_id
            )

        by_task[task_id] = status

    missing_tasks = [
        task_id
        for task_id in range(num_tasks)
        if task_id not in by_task
    ]

    unexpected_tasks = sorted(
        task_id
        for task_id in by_task
        if (
            task_id < 0
            or task_id >= num_tasks
        )
    )

    non_ok_tasks = sorted(
        task_id
        for task_id, status
        in by_task.items()
        if status.get("status")
        != "TASK_OK"
    )

    total_matrix_cases = sum(
        int(
            status.get(
                "matrix_cases",
                0,
            )
        )
        for status in by_task.values()
    )

    total_inverse_records = sum(
        int(
            status.get(
                "inverse_records",
                0,
            )
        )
        for status in by_task.values()
    )

    total_solution_records = sum(
        int(
            status.get(
                "solution_records",
                0,
            )
        )
        for status in by_task.values()
    )

    total_inverse_failures = sum(
        int(
            status.get(
                "inverse_failures",
                0,
            )
        )
        for status in by_task.values()
    )

    total_solution_failures = sum(
        int(
            status.get(
                "solution_failures",
                0,
            )
        )
        for status in by_task.values()
    )

    total_inverse_bytes = sum(
        int(
            status.get(
                "inverse_size_bytes",
                0,
            )
        )
        for status in by_task.values()
    )

    total_solution_bytes = sum(
        int(
            status.get(
                "solution_size_bytes",
                0,
            )
        )
        for status in by_task.values()
    )

    expected_matrix_cases = int(
        manifest_summary[
            "num_matrix_cases"
        ]
    )

    expected_inverse_records = int(
        manifest_summary[
            "expected_inverse_records"
        ]
    )

    expected_solution_records = int(
        manifest_summary[
            "expected_solution_records"
        ]
    )

    fingerprints = sorted(
        set(
            str(
                status.get(
                    "campaign_fingerprint",
                    "",
                )
            )
            for status in by_task.values()
        )
    )

    missing_output_files: List[
        str
    ] = []

    size_mismatches: List[str] = []

    for task_id, status in sorted(
        by_task.items()
    ):
        for kind in (
            "inverse",
            "solution",
        ):
            path = Path(
                status.get(
                    kind + "_path",
                    "",
                )
            )

            expected_size = int(
                status.get(
                    kind + "_size_bytes",
                    -1,
                )
            )

            if not path.is_file():
                missing_output_files.append(
                    str(path)
                )

            elif (
                path.stat().st_size
                != expected_size
            ):
                size_mismatches.append(
                    "{} expected={} actual={}".format(
                        path,
                        expected_size,
                        path.stat().st_size,
                    )
                )

    failure_counts: Counter = Counter()
    family_case_counts: Counter = Counter()
    dtype_method_counts: Counter = Counter()

    for status in by_task.values():
        for key, value in status.get(
            "failure_counts",
            {},
        ).items():
            failure_counts[key] += int(value)

        for key, value in status.get(
            "family_case_counts",
            {},
        ).items():
            family_case_counts[key] += int(value)

        for key, value in status.get(
            "dtype_method_counts",
            {},
        ).items():
            dtype_method_counts[key] += int(value)

    errors: List[str] = []

    if invalid_status_files:
        errors.append(
            "invalid status files"
        )

    if missing_tasks:
        errors.append(
            "missing tasks"
        )

    if unexpected_tasks:
        errors.append(
            "unexpected tasks"
        )

    if duplicate_tasks:
        errors.append(
            "duplicate tasks"
        )

    if non_ok_tasks:
        errors.append(
            "non-OK tasks"
        )

    if (
        len(fingerprints) != 1
        or fingerprints == [""]
    ):
        errors.append(
            "campaign fingerprint mismatch"
        )

    if (
        total_matrix_cases
        != expected_matrix_cases
    ):
        errors.append(
            "matrix case count mismatch"
        )

    if (
        total_inverse_records
        != expected_inverse_records
    ):
        errors.append(
            "inverse record count mismatch"
        )

    if (
        total_solution_records
        != expected_solution_records
    ):
        errors.append(
            "solution record count mismatch"
        )

    if missing_output_files:
        errors.append(
            "missing output files"
        )

    if size_mismatches:
        errors.append(
            "output size mismatches"
        )

    shard_manifest = (
        report_dir
        / "canonical_shards_manifest.csv"
    )

    with shard_manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "task_id",
            "matrix_cases",
            "inverse_records",
            "solution_records",
            "inverse_failures",
            "solution_failures",
            "elapsed_seconds",
            "inverse_path",
            "inverse_size_bytes",
            "inverse_sha256",
            "solution_path",
            "solution_size_bytes",
            "solution_sha256",
            "campaign_fingerprint",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for task_id, status in sorted(
            by_task.items()
        ):
            writer.writerow(
                {
                    field:
                        status.get(field, "")
                    for field in fieldnames
                }
            )

    report = {
        "status": (
            "CANONICAL_COMPLETE"
            if not errors
            else "CANONICAL_INCOMPLETE"
        ),

        "errors":
            errors,

        "num_tasks_expected":
            num_tasks,

        "num_status_files":
            len(status_paths),

        "num_tasks_found":
            len(by_task),

        "missing_tasks":
            missing_tasks,

        "unexpected_tasks":
            unexpected_tasks,

        "duplicate_tasks":
            duplicate_tasks,

        "non_ok_tasks":
            non_ok_tasks,

        "invalid_status_files":
            invalid_status_files,

        "campaign_fingerprints":
            fingerprints,

        "expected_matrix_cases":
            expected_matrix_cases,

        "matrix_cases":
            total_matrix_cases,

        "expected_inverse_records":
            expected_inverse_records,

        "inverse_records":
            total_inverse_records,

        "expected_solution_records":
            expected_solution_records,

        "solution_records":
            total_solution_records,

        "inverse_failures":
            total_inverse_failures,

        "solution_failures":
            total_solution_failures,

        "inverse_size_bytes":
            total_inverse_bytes,

        "solution_size_bytes":
            total_solution_bytes,

        "total_size_bytes":
            (
                total_inverse_bytes
                + total_solution_bytes
            ),

        "missing_output_files":
            missing_output_files,

        "size_mismatches":
            size_mismatches,

        "failure_counts":
            dict(
                sorted(
                    failure_counts.items()
                )
            ),

        "family_case_counts":
            dict(
                sorted(
                    family_case_counts.items()
                )
            ),

        "dtype_method_counts":
            dict(
                sorted(
                    dtype_method_counts.items()
                )
            ),

        "shards_manifest":
            str(shard_manifest),

        "storage_layout": {
            "inverse":
                "gzip CSV shards; "
                "one per array task",

            "solution":
                "gzip CSV shards; "
                "one per array task",

            "join_key_inverse": [
                "matrix_id",
                "method",
                "dtype_name",
            ],

            "join_key_solution": [
                "matrix_id",
                "method",
                "dtype_name",
                "rhs_type",
                "refinement_steps",
            ],

            "assignment":
                "global_case_index "
                "mod num_tasks",
        },
    }

    json_path = (
        report_dir
        / "canonical_completion.json"
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
        / "canonical_completion.md"
    )

    with md_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# Canonical Castillo "
            "campaign completion\n\n"
        )

        handle.write(
            "- Status: `{}`\n".format(
                report["status"]
            )
        )

        handle.write(
            "- Tasks: `{}/{}`\n".format(
                len(by_task),
                num_tasks,
            )
        )

        handle.write(
            "- Matrix cases: `{}/{}`\n".format(
                total_matrix_cases,
                expected_matrix_cases,
            )
        )

        handle.write(
            "- Inverse records: `{}/{}`\n".format(
                total_inverse_records,
                expected_inverse_records,
            )
        )

        handle.write(
            "- Solution records: `{}/{}`\n".format(
                total_solution_records,
                expected_solution_records,
            )
        )

        handle.write(
            "- Inverse failures: `{}`\n".format(
                total_inverse_failures
            )
        )

        handle.write(
            "- Solution failures: `{}`\n".format(
                total_solution_failures
            )
        )

        handle.write(
            "- Inverse compressed size: `{}`\n".format(
                human_bytes(
                    total_inverse_bytes
                )
            )
        )

        handle.write(
            "- Solution compressed size: `{}`\n".format(
                human_bytes(
                    total_solution_bytes
                )
            )
        )

        handle.write(
            "- Total compressed size: `{}`\n".format(
                human_bytes(
                    total_inverse_bytes
                    + total_solution_bytes
                )
            )
        )

        handle.write(
            "- Shards manifest: `{}`\n".format(
                shard_manifest
            )
        )

        if errors:
            handle.write(
                "\n## Validation errors\n\n"
            )

            for error in errors:
                handle.write(
                    "- {}\n".format(error)
                )

        handle.write(
            "\n## Failure accounting\n\n"
        )

        handle.write(
            "| family | dtype | method | "
            "class | reason | count |\n"
        )

        handle.write(
            "|---|---|---|---|---|---:|\n"
        )

        for key, count in sorted(
            failure_counts.items()
        ):
            parts = key.split("|", 4)

            while len(parts) < 5:
                parts.append("")

            handle.write(
                "| {} | {} | {} | {} | {} | {} |\n".format(
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4].replace(
                        "|",
                        "/",
                    ),
                    count,
                )
            )

    print(
        json.dumps(
            {
                "status":
                    report["status"],

                "report_json":
                    str(json_path),

                "report_md":
                    str(md_path),

                "shards_manifest":
                    str(shard_manifest),

                "errors":
                    errors,

                "matrix_cases":
                    total_matrix_cases,

                "inverse_records":
                    total_inverse_records,

                "solution_records":
                    total_solution_records,

                "total_size":
                    human_bytes(
                        total_inverse_bytes
                        + total_solution_bytes
                    ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if errors:
        sys.exit(1)

    print("CANONICAL_FINALIZE_OK")


if __name__ == "__main__":
    main()
