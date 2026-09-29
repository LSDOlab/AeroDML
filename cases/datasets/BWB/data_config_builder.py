import json
from pathlib import Path

import numpy as np


def _jsonable(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def _as_runtime_value(value):
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _is_ranged(spec):
    return "ref_value" in spec and "delta" in spec


def _reference_value(spec):
    if "ref_value" in spec:
        return spec["ref_value"]
    if "value" in spec:
        return spec["value"]
    raise KeyError("Each config entry must have either 'value' or 'ref_value' + 'delta'.")


def _lhs_unit(n_samples, n_dim, seed):
    if n_samples == 0 or n_dim == 0:
        return np.zeros((n_samples, n_dim))

    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_dim))

    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        X[:, j] = (perm + rng.random(n_samples)) / n_samples

    return X


def _build_samples(config_ranges, n_lhs_samples, seed):
    """
    Supports two entry styles:

    Fixed value:
        "p_inf": {"value": 30089.6}

    LHS range:
        "p_inf": {"ref_value": 30089.6, "delta": 1000.0}

    The returned list always has n_lhs_samples + 1 entries.
    Entry 0 is always the reference/fixed value.
    Entries 1...n are LHS samples for ranged entries and fixed copies for fixed entries.
    """
    config_ranges = config_ranges or {}

    names = list(config_ranges.keys())
    ranged_meta = []
    lower = []
    upper = []

    # Reference case: exact ref_value/value, no delta.
    reference_sample = {}
    for name in names:
        spec = config_ranges[name]
        reference_sample[name] = {
            "value": _jsonable(_reference_value(spec)),
        }

        if _is_ranged(spec):
            ref = np.asarray(spec["ref_value"], dtype=float)
            delta = np.asarray(spec["delta"], dtype=float)
            delta = np.broadcast_to(delta, ref.shape)

            flat_ref = ref.reshape(-1)
            flat_delta = delta.reshape(-1)

            start = len(lower)
            lower.extend((flat_ref - flat_delta).tolist())
            upper.extend((flat_ref + flat_delta).tolist())
            stop = len(lower)

            ranged_meta.append((name, ref.shape, start, stop))

    lhs = _lhs_unit(n_lhs_samples, len(lower), seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    samples = [reference_sample]

    for i in range(n_lhs_samples):
        sample = {}

        # Start with fixed/reference copies for every variable.
        for name in names:
            sample[name] = {
                "value": _jsonable(_reference_value(config_ranges[name])),
            }

        # Replace ranged entries with their LHS values.
        if len(lower) > 0:
            row = lower + lhs[i, :] * (upper - lower)
            for name, shape, start, stop in ranged_meta:
                value = row[start:stop]
                if shape == ():
                    value = float(value[0])
                else:
                    value = value.reshape(shape)
                sample[name] = {
                    "value": _jsonable(value),
                }

        samples.append(sample)

    return samples


def _convert_values_for_runtime(cases):
    for case in cases:
        for group_key in ("geometry_configuration", "flow_conditions"):
            group = case.get(group_key, {})
            for _, item in group.items():
                item["value"] = _as_runtime_value(item["value"])
    return cases


def write_dataset_config(data_dir, n_samples, geometry_ranges, flow_ranges=None, seed=99):
    """
    Write cases.json.

    n_samples is the TOTAL number of cases.
    Case key "0" is always the reference case.
    Cases "1"... are LHS samples.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1 because case '0' is the reference case.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    n_lhs_samples = n_samples - 1

    geometry_samples = _build_samples(
        geometry_ranges,
        n_lhs_samples=n_lhs_samples,
        seed=seed,
    )
    flow_samples = _build_samples(
        flow_ranges or {},
        n_lhs_samples=n_lhs_samples,
        seed=seed + 1,
    )

    cases = []
    for i in range(n_samples):
        cases.append({
            "key": str(i),
            "geometry_configuration": geometry_samples[i],
            "flow_conditions": flow_samples[i],
        })

    config = {
        "cases": cases,
    }

    config_path = data_dir / "cases.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return _convert_values_for_runtime(cases)


def get_dataset_configurations(data_dir):
    data_dir = Path(data_dir)
    config_path = data_dir / "cases.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return _convert_values_for_runtime(config["cases"])