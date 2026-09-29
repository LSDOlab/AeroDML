from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[int, dict[str, Any]]:
    """Load parser JSON and return {case_id: summary_dict}."""
    with path.open() as f:
        rows = json.load(f)
    return {int(row["case"]): row for row in rows}


def _residual_too_high(row: dict[str, Any], max_residual: float | None) -> bool:
    """Return True if any stored final residual exceeds the tolerance."""
    if max_residual is None:
        return False

    values = []

    if row.get("p_final_residual") is not None:
        values.append(row["p_final_residual"])

    for val in row.get("final_solver_residuals", {}).values():
        if val is not None:
            values.append(val)

    return any(abs(float(v)) > max_residual for v in values)


def _invalid_reasons(
    case: int,
    euler: dict[int, dict[str, Any]] | None,
    rans: dict[int, dict[str, Any]],
    max_residual: float | None,
) -> list[str]:
    reasons = []
    summaries = [("rans", rans)]

    if euler is not None:
        summaries.insert(0, ("euler", euler))

    for solver_name, data in summaries:
        row = data.get(case)

        if row is None:
            reasons.append(f"missing_{solver_name}_summary")
            continue
        if not row.get("log_found", False):
            reasons.append(f"{solver_name}_log_missing")
        if not row.get("npz_exists", False):
            reasons.append(f"{solver_name}_npz_missing")
        if row.get("nans_exist", False):
            reasons.append(f"{solver_name}_has_nans")
        if row.get("converged") == 0:
            reasons.append(f"{solver_name}_not_converged")
        if _residual_too_high(row, max_residual):
            reasons.append(f"{solver_name}_residual_too_high")

    return reasons


def create_dataset_split(
    split_name: str = "split_default",
    test_ratio: float = 0.2,
    seed: int = 0,
    max_residual: float | None = None,
    manual_invalid_cases: list[int] | None = None,
    manual_valid_cases: list[int] | None = None,
    manual_test_cases: list[int] | None = None,
    manual_train_cases: list[int] | None = None,
    output_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """
    Create a named dataset split from paired euler/rans parser summaries.

    manual_invalid_cases: force these cases invalid.
    manual_valid_cases: force these cases valid, overriding automatic reasons.
    manual_test_cases/manual_train_cases: force valid cases into that split.
    """
    root = Path(output_dir)
    euler_path = root / "euler" / "parsed_case_summary.json"
    rans_path = root / "rans" / "parsed_case_summary.json"

    euler = _load_summary(euler_path) if euler_path.exists() else None
    rans = _load_summary(rans_path)

    manual_invalid = set(manual_invalid_cases or [])
    manual_valid = set(manual_valid_cases or [])
    manual_test = set(manual_test_cases or [])
    manual_train = set(manual_train_cases or [])

    overlap = manual_test & manual_train
    if overlap:
        raise ValueError(f"Cases cannot be both manual_test and manual_train: {sorted(overlap)}")

    all_cases = sorted(set(rans) | (set(euler) if euler else set()))
    rows = []

    for case in all_cases:
        reasons = _invalid_reasons(case, euler, rans, max_residual)

        if case in manual_invalid:
            reasons.append("manual_invalid")

        if case in manual_valid:
            reasons = []

        rows.append({
            "case": case,
            "valid": len(reasons) == 0,
            "split": None,
            "invalid_reasons": reasons,
        })

    valid_cases = [row["case"] for row in rows if row["valid"]]
    forced_test = sorted(set(valid_cases) & manual_test)
    forced_train = sorted(set(valid_cases) & manual_train)

    remaining = sorted(set(valid_cases) - set(forced_test) - set(forced_train))
    rng = random.Random(seed)
    rng.shuffle(remaining)

    n_test = round(test_ratio * len(valid_cases))
    n_random_test = max(0, n_test - len(forced_test))
    random_test = set(remaining[:n_random_test])

    test_cases = set(forced_test) | random_test
    train_cases = set(forced_train) | (set(remaining) - random_test)

    for row in rows:
        case = row["case"]
        if case in test_cases:
            row["split"] = "test"
        elif case in train_cases:
            row["split"] = "train"

    output = {
        "schema_version": 1,
        "split_name": split_name,
        "test_ratio": test_ratio,
        "seed": seed,
        "max_residual": max_residual,
        "source_files": {
            "rans": str(rans_path),
            **({"euler": str(euler_path)} if euler else {}),
        },
        "n_cases": len(rows),
        "n_valid": sum(row["valid"] for row in rows),
        "n_train": sum(row["split"] == "train" for row in rows),
        "n_test": sum(row["split"] == "test" for row in rows),
        "cases": rows,
    }

    output_path = root / f"{split_name}.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote split: {output_path}")
    print(
        f"cases: {output['n_cases']}, valid: {output['n_valid']}, "
        f"train: {output['n_train']}, test: {output['n_test']}"
    )

    return rows

if __name__ == "__main__":
    create_dataset_split(
        split_name="split_0",
        test_ratio=0.20,
        seed=133313,
        max_residual=3e-6,
    )

    # add optional manual invalid cases to the split
    # manual_invalid = [1]

    # create_dataset_split(
    #     split_name="split_1",
    #     test_ratio=0.20,
    #     seed=999,
    #     max_residual=3e-6,
    #     manual_invalid_cases = manual_invalid,
    # )
