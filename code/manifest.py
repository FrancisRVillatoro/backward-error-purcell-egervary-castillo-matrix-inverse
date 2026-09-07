#!/usr/bin/env python3

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


PARAMETER_COLUMNS = [
    "condition_number",
    "profile",
    "node_type",
    "cluster_parameter",
    "column_scaling",
    "theta",
    "epsilon",
    "decay",
    "scale_range",
    "alpha",
    "kappa",
]


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    base_name = raw.get("base_config")
    if base_name is None:
        return raw

    base_path = path.parent / base_name
    base = load_config(base_path)

    merged = dict(base)
    for key, value in raw.items():
        if key != "base_config":
            merged[key] = value

    return merged


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def cell_identifier(family, parameters):
    payload = canonical_json(
        {
            "family": family,
            "parameters": parameters,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{family}:{digest}"


def expand_grid(grid):
    if not grid:
        return []

    keys = list(grid.keys())
    values = [grid[key] for key in keys]

    cells = []
    for combination in itertools.product(*values):
        cells.append(dict(zip(keys, combination)))

    return cells


def stochastic_for_cell(family_spec, parameters):
    conditions = family_spec.get("stochastic_when")

    if conditions is not None:
        for key, allowed_values in conditions.items():
            if parameters.get(key) not in allowed_values:
                return False
        return True

    return bool(family_spec.get("stochastic", False))


def expand_family_cells(family_spec):
    family = family_spec["name"]

    parameter_cells = []

    for parameters in expand_grid(family_spec.get("grid", {})):
        parameter_cells.append(parameters)

    for parameters in family_spec.get("cells", []):
        parameter_cells.append(dict(parameters))

    if not parameter_cells:
        parameter_cells.append({})

    result = []
    seen = set()

    for parameters in parameter_cells:
        parameter_key = canonical_json(parameters)

        if parameter_key in seen:
            raise RuntimeError(
                f"duplicate parameter cell in family {family}: "
                f"{parameters}"
            )

        seen.add(parameter_key)

        result.append(
            {
                "family": family,
                "parameters": parameters,
                "cell_id": cell_identifier(family, parameters),
                "stochastic": stochastic_for_cell(
                    family_spec,
                    parameters,
                ),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    campaign = str(config["campaign"])
    dimensions = [int(n) for n in config["dimensions"]]
    methods = list(config["methods"])
    dtypes = list(config["dtypes"])
    stochastic_replicates = int(config["stochastic_replicates"])

    rhs_types = list(config["solution"]["rhs_types"])
    refinement_steps = [
        int(value)
        for value in config["solution"]["refinement_steps"]
    ]

    family_cells = []

    for family_spec in config["families"]:
        family_cells.extend(expand_family_cells(family_spec))

    family_names = sorted(
        {cell["family"] for cell in family_cells}
    )

    stochastic_cells = [
        cell for cell in family_cells
        if cell["stochastic"]
    ]
    deterministic_cells = [
        cell for cell in family_cells
        if not cell["stochastic"]
    ]

    expected_family_count = int(config["expected_family_count"])
    expected_stochastic_cells = int(
        config["expected_stochastic_cells"]
    )
    expected_deterministic_cells = int(
        config["expected_deterministic_cells"]
    )

    assert len(family_names) == expected_family_count, (
        len(family_names),
        expected_family_count,
        family_names,
    )
    assert len(stochastic_cells) == expected_stochastic_cells, (
        len(stochastic_cells),
        expected_stochastic_cells,
    )
    assert len(deterministic_cells) == expected_deterministic_cells, (
        len(deterministic_cells),
        expected_deterministic_cells,
    )

    cell_ids = [cell["cell_id"] for cell in family_cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("cell_id collision")

    manifest_csv = outdir / f"{campaign}_manifest.csv"
    summary_json = outdir / f"{campaign}_manifest_summary.json"
    summary_md = outdir / f"{campaign}_manifest_summary.md"

    fieldnames = [
        "block_index",
        "campaign",
        "n",
        "family",
        "cell_id",
        "stochastic",
        "replicas",
        "case_start",
        "case_stop",
        "parameters_json",
    ] + PARAMETER_COLUMNS

    by_family_cells = Counter()
    by_family_stochastic_cells = Counter()
    by_family_deterministic_cells = Counter()
    by_family_matrix_cases = Counter()
    by_dimension_matrix_cases = Counter()

    composition_by_dimension = defaultdict(list)

    block_keys = set()
    case_cursor = 0
    block_index = 0

    with manifest_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for n in dimensions:
            for cell in family_cells:
                family = cell["family"]
                parameters = cell["parameters"]
                stochastic = bool(cell["stochastic"])

                replicas = (
                    stochastic_replicates
                    if stochastic
                    else 1
                )

                case_start = case_cursor
                case_stop = case_start + replicas

                block_key = (n, cell["cell_id"])

                if block_key in block_keys:
                    raise RuntimeError(
                        f"duplicate block key: {block_key}"
                    )
                block_keys.add(block_key)

                row = {
                    "block_index": block_index,
                    "campaign": campaign,
                    "n": n,
                    "family": family,
                    "cell_id": cell["cell_id"],
                    "stochastic": stochastic,
                    "replicas": replicas,
                    "case_start": case_start,
                    "case_stop": case_stop,
                    "parameters_json": canonical_json(parameters),
                }

                for name in PARAMETER_COLUMNS:
                    row[name] = parameters.get(name, "")

                writer.writerow(row)

                composition_by_dimension[n].append(
                    (
                        cell["cell_id"],
                        stochastic,
                        replicas,
                        canonical_json(parameters),
                    )
                )

                by_family_cells[family] += 1
                if stochastic:
                    by_family_stochastic_cells[family] += 1
                else:
                    by_family_deterministic_cells[family] += 1

                by_family_matrix_cases[family] += replicas
                by_dimension_matrix_cases[n] += replicas

                case_cursor = case_stop
                block_index += 1

    reference_dimension = dimensions[0]
    reference_composition = composition_by_dimension[
        reference_dimension
    ]

    for n in dimensions[1:]:
        if composition_by_dimension[n] != reference_composition:
            raise RuntimeError(
                "parameter composition differs between "
                f"n={reference_dimension} and n={n}"
            )

    matrix_cases = case_cursor
    inverse_records_per_matrix = len(methods) * len(dtypes)
    solution_records_per_matrix = (
        len(methods)
        * len(dtypes)
        * len(rhs_types)
        * len(refinement_steps)
    )

    expected_inverse_records = (
        matrix_cases * inverse_records_per_matrix
    )
    expected_solution_records = (
        matrix_cases * solution_records_per_matrix
    )

    payload = {
        "status": "MANIFEST_OK",
        "campaign": campaign,
        "config": str(config_path),
        "manifest_csv": str(manifest_csv),
        "seed_namespace": config["seed_namespace"],
        "dimensions": dimensions,
        "num_dimensions": len(dimensions),
        "families": family_names,
        "num_families": len(family_names),
        "num_parameter_cells": len(family_cells),
        "num_stochastic_cells": len(stochastic_cells),
        "num_deterministic_cells": len(deterministic_cells),
        "stochastic_replicates": stochastic_replicates,
        "num_blocks": block_index,
        "num_matrix_cases": matrix_cases,
        "methods": methods,
        "dtypes": dtypes,
        "rhs_types": rhs_types,
        "refinement_steps": refinement_steps,
        "inverse_records_per_matrix": inverse_records_per_matrix,
        "solution_records_per_matrix": solution_records_per_matrix,
        "expected_inverse_records": expected_inverse_records,
        "expected_solution_records": expected_solution_records,
        "composition_identical_across_dimensions": True,
        "by_family_cells": dict(by_family_cells),
        "by_family_stochastic_cells": dict(
            by_family_stochastic_cells
        ),
        "by_family_deterministic_cells": dict(
            by_family_deterministic_cells
        ),
        "by_family_matrix_cases": dict(
            by_family_matrix_cases
        ),
        "by_dimension_matrix_cases": {
            str(key): value
            for key, value in by_dimension_matrix_cases.items()
        },
    }

    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write(f"# {campaign} manifest summary\n\n")
        handle.write(f"- Status: `{payload['status']}`\n")
        handle.write(f"- Config: `{config_path}`\n")
        handle.write(f"- Manifest: `{manifest_csv}`\n")
        handle.write(f"- Dimensions: `{dimensions}`\n")
        handle.write(f"- Families: `{len(family_names)}`\n")
        handle.write(
            f"- Parameter cells: `{len(family_cells)}`\n"
        )
        handle.write(
            f"- Stochastic cells: `{len(stochastic_cells)}`\n"
        )
        handle.write(
            f"- Deterministic cells: "
            f"`{len(deterministic_cells)}`\n"
        )
        handle.write(
            f"- Replicates per stochastic cell: "
            f"`{stochastic_replicates}`\n"
        )
        handle.write(f"- Blocks: `{block_index}`\n")
        handle.write(f"- Matrix cases: `{matrix_cases}`\n")
        handle.write(
            f"- Inverse records per matrix: "
            f"`{inverse_records_per_matrix}`\n"
        )
        handle.write(
            f"- Expected inverse records: "
            f"`{expected_inverse_records}`\n"
        )
        handle.write(
            f"- Solution records per matrix: "
            f"`{solution_records_per_matrix}`\n"
        )
        handle.write(
            f"- Expected solution records: "
            f"`{expected_solution_records}`\n"
        )
        handle.write(
            "- Composition identical across dimensions: `True`\n"
        )

        handle.write("\n## Cells and matrix cases by family\n\n")
        handle.write(
            "| family | stochastic cells | deterministic cells "
            "| matrix cases |\n"
        )
        handle.write("|---|---:|---:|---:|\n")

        for family in family_names:
            handle.write(
                f"| {family} "
                f"| {by_family_stochastic_cells[family]} "
                f"| {by_family_deterministic_cells[family]} "
                f"| {by_family_matrix_cases[family]} |\n"
            )

        handle.write("\n## Matrix cases by dimension\n\n")
        handle.write("| n | matrix cases |\n")
        handle.write("|---:|---:|\n")

        for n in dimensions:
            handle.write(
                f"| {n} | {by_dimension_matrix_cases[n]} |\n"
            )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
