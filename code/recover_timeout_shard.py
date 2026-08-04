#!/usr/bin/env python3
"""
Recover one pair of truncated canonical gzip streams.

Only complete, paired matrices are retained:
  10 inverse rows per matrix
  90 solution rows per matrix

Original timeout files are never modified or deleted.
"""

import argparse
import csv
import gzip
import io
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


def load_config(path):
    raw = json.loads(path.read_text(encoding="utf-8"))

    base_name = raw.get("base_config")

    if base_name is None:
        return raw

    base = load_config(path.parent / str(base_name))

    for key, value in raw.items():
        if key != "base_config":
            base[key] = value

    return base


class TruncatedGzipCSV:
    def __init__(self, path):
        self.path = Path(path)
        self.source = self.path.open("rb")

        # Reading through stdin avoids gzip filename-suffix checks.
        self.process = subprocess.Popen(
            ["gzip", "-dc"],
            stdin=self.source,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if self.process.stdout is None:
            raise RuntimeError("Could not open gzip stdout")

        self.text = io.TextIOWrapper(
            self.process.stdout,
            encoding="utf-8",
            errors="strict",
            newline="",
        )

        self.reader = csv.DictReader(self.text)
        self.fieldnames = list(self.reader.fieldnames or [])

        if not self.fieldnames:
            raise RuntimeError(
                "Missing CSV header in {}".format(self.path)
            )

    def close(self):
        try:
            self.text.close()
        except Exception:
            pass

        try:
            if self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass

        try:
            returncode = self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
            returncode = self.process.wait()

        try:
            self.source.close()
        except Exception:
            pass

        return returncode


def grouped_rows(reader):
    current_id = None
    current_rows = []

    try:
        for row in reader:
            matrix_id = row.get("matrix_id")

            if not matrix_id:
                break

            if current_id is None:
                current_id = matrix_id
                current_rows = [row]

            elif matrix_id == current_id:
                current_rows.append(row)

            else:
                yield current_id, current_rows
                current_id = matrix_id
                current_rows = [row]

    except (
        csv.Error,
        UnicodeDecodeError,
        OSError,
    ):
        pass

    if current_id is not None and current_rows:
        yield current_id, current_rows


def valid_inverse_group(rows, expected_keys):
    if len(rows) != len(expected_keys):
        return False, "inverse_row_count={}".format(len(rows))

    matrix_ids = {
        row.get("matrix_id", "")
        for row in rows
    }

    case_indices = {
        row.get("case_index", "")
        for row in rows
    }

    keys = {
        (
            row.get("method", ""),
            row.get("dtype_name", ""),
        )
        for row in rows
    }

    if len(matrix_ids) != 1:
        return False, "inverse_matrix_id_mismatch"

    if len(case_indices) != 1 or "" in case_indices:
        return False, "inverse_case_index_mismatch"

    if keys != expected_keys:
        return False, "inverse_method_dtype_grid_mismatch"

    return True, ""


def valid_solution_group(rows, expected_keys):
    if len(rows) != len(expected_keys):
        return False, "solution_row_count={}".format(len(rows))

    matrix_ids = {
        row.get("matrix_id", "")
        for row in rows
    }

    keys = {
        (
            row.get("method", ""),
            row.get("dtype_name", ""),
            row.get("rhs_type", ""),
            row.get("refinement_steps", ""),
        )
        for row in rows
    }

    if len(matrix_ids) != 1:
        return False, "solution_matrix_id_mismatch"

    if keys != expected_keys:
        return False, "solution_grid_mismatch"

    return True, ""


def unique_source(directory, pattern):
    paths = sorted(directory.glob(pattern))

    if len(paths) != 1:
        raise RuntimeError(
            "Expected one source for {}, found {}: {}".format(
                pattern,
                len(paths),
                [str(path) for path in paths],
            )
        )

    return paths[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--source-root",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.task_id < 0 or args.task_id >= 999:
        raise ValueError("task-id must be in 0,...,998")

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    for name in ("inverse", "solution", "status", "tmp"):
        (output_root / name).mkdir(
            parents=True,
            exist_ok=True,
        )

    tag = "{:04d}".format(args.task_id)

    source_inverse = unique_source(
        source_root / "tmp",
        "inverse_{}.csv.gz.tmp.*".format(tag),
    )

    source_solution = unique_source(
        source_root / "tmp",
        "solution_{}.csv.gz.tmp.*".format(tag),
    )

    output_inverse = (
        output_root
        / "inverse"
        / "inverse_{}.csv.gz".format(tag)
    )

    output_solution = (
        output_root
        / "solution"
        / "solution_{}.csv.gz".format(tag)
    )

    status_path = (
        output_root
        / "status"
        / "task_{}.json".format(tag)
    )

    source_signature = {
        "inverse_path": str(source_inverse),
        "inverse_size": source_inverse.stat().st_size,
        "inverse_mtime_ns": source_inverse.stat().st_mtime_ns,
        "solution_path": str(source_solution),
        "solution_size": source_solution.stat().st_size,
        "solution_mtime_ns": source_solution.stat().st_mtime_ns,
    }

    if (
        status_path.exists()
        and output_inverse.exists()
        and output_solution.exists()
        and not args.force
    ):
        old = json.loads(
            status_path.read_text(encoding="utf-8")
        )

        if (
            old.get("status") == "RECOVERY_OK"
            and old.get("source_signature")
            == source_signature
        ):
            print("RECOVERY_ALREADY_COMPLETE={}".format(tag))
            return

    config = load_config(Path(args.config))

    methods = [
        str(value)
        for value in config["methods"]
    ]

    dtypes = [
        str(value)
        for value in config["dtypes"]
    ]

    rhs_types = [
        str(value)
        for value in config["solution"]["rhs_types"]
    ]

    refinement_steps = [
        str(int(value))
        for value
        in config["solution"]["refinement_steps"]
    ]

    expected_inverse_keys = {
        (method, dtype_name)
        for dtype_name in dtypes
        for method in methods
    }

    expected_solution_keys = {
        (
            method,
            dtype_name,
            rhs_type,
            refinement,
        )
        for dtype_name in dtypes
        for method in methods
        for rhs_type in rhs_types
        for refinement in refinement_steps
    }

    inverse_tmp = (
        output_root
        / "tmp"
        / "inverse_{}.csv.gz.part".format(tag)
    )

    solution_tmp = (
        output_root
        / "tmp"
        / "solution_{}.csv.gz.part".format(tag)
    )

    for path in (inverse_tmp, solution_tmp):
        if path.exists():
            path.unlink()

    start = time.perf_counter()

    inverse_stream = TruncatedGzipCSV(source_inverse)
    solution_stream = TruncatedGzipCSV(source_solution)

    inverse_groups = iter(
        grouped_rows(inverse_stream.reader)
    )

    solution_groups = iter(
        grouped_rows(solution_stream.reader)
    )

    recovered_matrices = 0
    recovered_inverse_rows = 0
    recovered_solution_rows = 0

    first_case_index = None
    last_case_index = None
    last_matrix_id = None

    block_counts = Counter()
    family_counts = Counter()
    dimension_counts = Counter()

    stopped_reason = "unknown"

    try:
        with gzip.open(
            str(inverse_tmp),
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=1,
        ) as inverse_handle, gzip.open(
            str(solution_tmp),
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=1,
        ) as solution_handle:

            inverse_writer = csv.DictWriter(
                inverse_handle,
                fieldnames=inverse_stream.fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )

            solution_writer = csv.DictWriter(
                solution_handle,
                fieldnames=solution_stream.fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )

            inverse_writer.writeheader()
            solution_writer.writeheader()

            while True:
                try:
                    inverse_id, inverse_rows = next(
                        inverse_groups
                    )
                except StopIteration:
                    stopped_reason = "inverse_stream_ended"
                    break

                try:
                    solution_id, solution_rows = next(
                        solution_groups
                    )
                except StopIteration:
                    stopped_reason = "solution_stream_ended"
                    break

                if inverse_id != solution_id:
                    stopped_reason = (
                        "matrix_id_mismatch:{}:{}".format(
                            inverse_id,
                            solution_id,
                        )
                    )
                    break

                inverse_ok, inverse_reason = valid_inverse_group(
                    inverse_rows,
                    expected_inverse_keys,
                )

                if not inverse_ok:
                    stopped_reason = inverse_reason
                    break

                solution_ok, solution_reason = valid_solution_group(
                    solution_rows,
                    expected_solution_keys,
                )

                if not solution_ok:
                    stopped_reason = solution_reason
                    break

                for row in inverse_rows:
                    inverse_writer.writerow(row)

                for row in solution_rows:
                    solution_writer.writerow(row)

                recovered_matrices += 1
                recovered_inverse_rows += len(inverse_rows)
                recovered_solution_rows += len(solution_rows)

                first = inverse_rows[0]

                case_index = int(first["case_index"])
                block_index = int(first["block_index"])
                family = first["family"]
                n = int(first["n"])

                if first_case_index is None:
                    first_case_index = case_index

                last_case_index = case_index
                last_matrix_id = inverse_id

                block_counts[str(block_index)] += 1
                family_counts[family] += 1
                dimension_counts[str(n)] += 1

    finally:
        inverse_returncode = inverse_stream.close()
        solution_returncode = solution_stream.close()

    os.replace(
        str(inverse_tmp),
        str(output_inverse),
    )

    os.replace(
        str(solution_tmp),
        str(output_solution),
    )

    # Recovered outputs must be complete gzip streams.
    subprocess.check_call(
        ["gzip", "-t", str(output_inverse)]
    )

    subprocess.check_call(
        ["gzip", "-t", str(output_solution)]
    )

    if recovered_inverse_rows != 10 * recovered_matrices:
        raise RuntimeError("Invalid recovered inverse count")

    if recovered_solution_rows != 90 * recovered_matrices:
        raise RuntimeError("Invalid recovered solution count")

    status = {
        "status": "RECOVERY_OK",
        "task_id": args.task_id,
        "source_signature": source_signature,
        "source_gzip_returncode_inverse":
            inverse_returncode,
        "source_gzip_returncode_solution":
            solution_returncode,
        "stopped_reason": stopped_reason,
        "recovered_matrices": recovered_matrices,
        "recovered_inverse_rows": recovered_inverse_rows,
        "recovered_solution_rows": recovered_solution_rows,
        "first_case_index": first_case_index,
        "last_case_index": last_case_index,
        "last_matrix_id": last_matrix_id,
        "block_counts": dict(sorted(block_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "dimension_counts": dict(
            sorted(
                dimension_counts.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "output_inverse": str(output_inverse),
        "output_solution": str(output_solution),
        "output_inverse_size": output_inverse.stat().st_size,
        "output_solution_size": output_solution.stat().st_size,
        "elapsed_seconds": time.perf_counter() - start,
    }

    status_tmp = status_path.with_suffix(".json.tmp")

    status_tmp.write_text(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        str(status_tmp),
        str(status_path),
    )

    print(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
    )

    print("RECOVERY_OK={}".format(tag))


if __name__ == "__main__":
    main()
