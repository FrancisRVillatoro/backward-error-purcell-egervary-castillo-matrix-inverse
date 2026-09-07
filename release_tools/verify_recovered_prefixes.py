#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path

from families import stable_matrix_id


HOME = Path.home()
PROJECT = HOME / "castillo_stability_campaign"

MANIFEST = (
    PROJECT
    / "reports"
    / "canonical"
    / "canonical_manifest.csv"
)

CONFIG = PROJECT / "config" / "canonical.json"

STATUS_DIR = (
    HOME
    / "fscratch"
    / "castillo_stability_campaign"
    / "results"
    / "recovered"
    / "status"
)

SUMMARY = (
    PROJECT
    / "reports"
    / "recovered"
    / "recovered_completion_summary.json"
)

OUT_JSON = (
    PROJECT
    / "reports"
    / "recovered"
    / "recovered_prefix_certificate.json"
)

OUT_CSV = (
    PROJECT
    / "reports"
    / "recovered"
    / "recovered_prefix_certificate.csv"
)

NUM_TASKS = 999


def assigned_count(case_start, replicas, task_id):
    first = (task_id - case_start) % NUM_TASKS

    if first >= replicas:
        return 0, first

    count = 1 + (replicas - 1 - first) // NUM_TASKS

    return count, first


def normalize_counter(counter):
    return dict(
        sorted(
            (
                (str(k), int(v))
                for k, v in counter.items()
                if int(v) != 0
            ),
            key=lambda item: item[0],
        )
    )


def normalize_dimension_counter(counter):
    return dict(
        sorted(
            (
                (str(k), int(v))
                for k, v in counter.items()
                if int(v) != 0
            ),
            key=lambda item: int(item[0]),
        )
    )


def main():
    config = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )

    seed_namespace = str(
        config["seed_namespace"]
    )

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        blocks = list(reader)

    if not blocks:
        raise RuntimeError("Empty canonical manifest")

    required = {
        "block_index",
        "case_start",
        "replicas",
        "cell_id",
        "family",
        "n",
    }

    missing = required - set(blocks[0])

    if missing:
        raise RuntimeError(
            "Missing manifest columns: "
            + repr(sorted(missing))
            + "\nAvailable columns: "
            + repr(sorted(blocks[0].keys()))
        )

    summary = json.loads(
        SUMMARY.read_text(encoding="utf-8")
    )

    expected_total = int(
        summary["recovered_matrix_cases"]
    )

    expected_inverse = int(
        summary["recovered_inverse_rows"]
    )

    expected_solution = int(
        summary["recovered_solution_rows"]
    )

    certificate_rows = []

    total_matrices = 0
    failures = []

    for task_id in range(NUM_TASKS):
        tag = f"{task_id:04d}"

        status_path = (
            STATUS_DIR / f"task_{tag}.json"
        )

        if not status_path.is_file():
            failures.append(
                f"{tag}: missing status"
            )
            continue

        status = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )

        if status.get("status") != "RECOVERY_OK":
            failures.append(
                f"{tag}: status="
                f"{status.get('status')!r}"
            )
            continue

        target = int(
            status["recovered_matrices"]
        )

        remaining = target

        block_counts = Counter()
        family_counts = Counter()
        dimension_counts = Counter()

        first_case_index = None
        last_case_index = None
        last_matrix_id = None

        for block in blocks:
            if remaining <= 0:
                break

            block_index = int(
                block["block_index"]
            )

            case_start = int(
                block["case_start"]
            )

            replicas = int(
                block["replicas"]
            )

            n = int(block["n"])

            count, first_replica = assigned_count(
                case_start,
                replicas,
                task_id,
            )

            if count == 0:
                continue

            take = min(
                remaining,
                count,
            )

            first_taken_replica = first_replica

            last_taken_replica = (
                first_replica
                + (take - 1) * NUM_TASKS
            )

            first_taken_case = (
                case_start
                + first_taken_replica
            )

            last_taken_case = (
                case_start
                + last_taken_replica
            )

            if first_case_index is None:
                first_case_index = (
                    first_taken_case
                )

            last_case_index = last_taken_case

            last_matrix_id = stable_matrix_id(
                seed_namespace,
                block["cell_id"],
                n,
                last_taken_replica,
            )

            block_counts[
                str(block_index)
            ] += take

            family_counts[
                block["family"]
            ] += take

            dimension_counts[
                str(n)
            ] += take

            remaining -= take

        if remaining != 0:
            failures.append(
                f"{tag}: manifest exhausted "
                f"with remaining={remaining}"
            )
            continue

        checks = {
            "recovered_matrices":
                target
                == int(
                    status[
                        "recovered_matrices"
                    ]
                ),

            "inverse_rows":
                10 * target
                == int(
                    status[
                        "recovered_inverse_rows"
                    ]
                ),

            "solution_rows":
                90 * target
                == int(
                    status[
                        "recovered_solution_rows"
                    ]
                ),

            "first_case_index":
                first_case_index
                == status.get(
                    "first_case_index"
                ),

            "last_case_index":
                last_case_index
                == status.get(
                    "last_case_index"
                ),

            "last_matrix_id":
                last_matrix_id
                == status.get(
                    "last_matrix_id"
                ),

            "block_counts":
                normalize_counter(
                    block_counts
                )
                == normalize_counter(
                    status.get(
                        "block_counts",
                        {},
                    )
                ),

            "family_counts":
                normalize_counter(
                    family_counts
                )
                == normalize_counter(
                    status.get(
                        "family_counts",
                        {},
                    )
                ),

            "dimension_counts":
                normalize_dimension_counter(
                    dimension_counts
                )
                == normalize_dimension_counter(
                    status.get(
                        "dimension_counts",
                        {},
                    )
                ),
        }

        bad = [
            name
            for name, ok in checks.items()
            if not ok
        ]

        if bad:
            failures.append(
                f"{tag}: mismatches="
                + ",".join(bad)
            )

        certificate_rows.append(
            {
                "task_id": task_id,
                "recovered_matrices":
                    target,
                "first_case_index":
                    first_case_index,
                "last_case_index":
                    last_case_index,
                "last_matrix_id":
                    last_matrix_id,
                "all_checks_pass":
                    not bad,
            }
        )

        total_matrices += target

    aggregate_checks = {
        "tasks":
            len(certificate_rows)
            == NUM_TASKS,

        "matrix_total":
            total_matrices
            == expected_total,

        "inverse_total":
            10 * total_matrices
            == expected_inverse,

        "solution_total":
            90 * total_matrices
            == expected_solution,
    }

    for name, ok in aggregate_checks.items():
        if not ok:
            failures.append(
                "aggregate:" + name
            )

    with OUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "task_id",
            "recovered_matrices",
            "first_case_index",
            "last_case_index",
            "last_matrix_id",
            "all_checks_pass",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            certificate_rows
        )

    certificate = {
        "status":
            (
                "RECOVERED_PREFIX_CERTIFIED"
                if not failures
                else
                "RECOVERED_PREFIX_CERTIFICATION_FAILED"
            ),

        "num_tasks_expected":
            NUM_TASKS,

        "num_tasks_verified":
            len(certificate_rows),

        "manifest_blocks":
            len(blocks),

        "recovered_matrices":
            total_matrices,

        "recovered_inverse_rows":
            10 * total_matrices,

        "recovered_solution_rows":
            90 * total_matrices,

        "aggregate_checks":
            aggregate_checks,

        "failures":
            failures,

        "interpretation":
            (
                "For every task, the recovered campaign is "
                "the deterministic prefix of complete paired "
                "matrix groups generated by the frozen manifest, "
                "task partition, seed namespace, and matrix-ID rule."
            ),
    }

    OUT_JSON.write_text(
        json.dumps(
            certificate,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            certificate,
            indent=2,
            sort_keys=True,
        )
    )

    if failures:
        raise RuntimeError(
            "Recovered prefix certification failed"
        )

    print()
    print(
        "RECOVERED_PREFIX_CERTIFICATION_OK"
    )


if __name__ == "__main__":
    main()
