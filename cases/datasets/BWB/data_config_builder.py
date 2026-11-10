import json
from pathlib import Path

import numpy as np

def switch_func(x, funcs_list, bounds_list, scale=10.):
    f_0, f_end = funcs_list[0], funcs_list[-1]
    x_0, x_end = bounds_list[0], bounds_list[-1]

    y = f_0*(0.5*np.tanh(scale*(x_0-x)) + 0.5)
    
    for i in range(len(bounds_list) - 1):
        f_i = funcs_list[i+1]

        x_l, x_h = bounds_list[i], bounds_list[i+1]

        y = y + f_i*(0.5*(np.tanh(scale*(x-x_l)) - np.tanh(scale*(x-x_h))))

    y = y + f_end*(0.5*np.tanh(scale*(x-x_end)) + 0.5)

    return y

def atmos_model(h):
    '''
    outputs density and speed of sound based on altitude
    input: h (in km)
    '''
    gamma = 1.4
    R = 287 # m^2/(s^2K)

    h_t = 11 # km
    T0 = 288.16 # K
    T1 = 216.65 # K
    L = 6.5 # K/km

    p0 = 101325 # Pa
    p1 = 22632 # Pa
    g = 9.81 # m/s^2

    T_funcs = [
        T0 - L*h,
        T1
    ]
    T_bounds = [h_t]
    T = switch_func(h, T_funcs, T_bounds, scale=100)
    
    p_funcs = [
        p0*(T/T0)**(g/(1e-3*L*R)),
        p1*np.exp(g*(h_t-h)*1000/(R*T1))
    ]
    p_bounds = [h_t]
    p = switch_func(h, p_funcs, p_bounds, scale=100)
    
    rho = p/(R*T)
    a = (gamma*R*T)**0.5

    mu_0 = 1.716e-5 # Pa*s
    T_s = 273.15 # K
    C = 110.4 # K

    mu = mu_0*(T/T_s)**(3/2)*(T_s+C)/(T+C)

    print(f"Conditions @ {h:.2f}km:")
    print(f"-Temperature: {T:.2f} K")
    print(f"-Pressure: {p:.2f} Pa")
    print(f"-Density: {rho:.4f} kg/m³")
    print(f"-Speed of sound: {a:.2f} m/s")
    print(f"-Dynamic viscosity: {mu:.3e} Pa·s")
    return rho, a, mu, T, p

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
    return "delta" in spec or ("lower" in spec and "upper" in spec)


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
            if "delta" in spec:
                delta = np.broadcast_to(
                    np.asarray(spec["delta"], dtype=float),
                    ref.shape,
                )
                lower_bound = ref - delta
                upper_bound = ref + delta
            else:
                lower_bound = np.broadcast_to(
                    np.asarray(spec["lower"], dtype=float),
                    ref.shape,
                )
                upper_bound = np.broadcast_to(
                    np.asarray(spec["upper"], dtype=float),
                    ref.shape,
                )

            start = len(lower)
            lower.extend(lower_bound.reshape(-1).tolist())
            upper.extend(upper_bound.reshape(-1).tolist())
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
    Write a new cases.json; refuse to overwrite existing configurations.

    n_samples is the TOTAL number of cases.
    Case key "0" is always the reference case.
    Cases "1"... are LHS samples.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1 because case '0' is the reference case.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "cases.json"
    if config_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing configuration: {config_path}. "
            "To regenerate cases, delete cases.json first. Regenerating cases "
            "will break the correspondence with any previously generated data; "
            "generate that data again or use a new dataset directory."
        )

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

    with open(config_path, "x") as f:
        json.dump(config, f, indent=2)

    return _convert_values_for_runtime(cases)


def get_dataset_configurations(data_dir):
    data_dir = Path(data_dir)
    config_path = data_dir / "cases.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return _convert_values_for_runtime(config["cases"])
