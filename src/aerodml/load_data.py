import numpy as np
import h5py
from .utils import print_content
from .visualize_mesh import visualize_scalars
import matplotlib.pyplot as plt
import os
import time
import json

from typing import Callable

def load_cfd(dir: str, split_dir: str, analyze: bool = False, filter: Callable = None, num_cases: int = None) -> dict:
    """
    Load CFD .npz files and attach usable/train flags from a dataset split JSON.

    Args:
        dir:
            Directory containing CFD .npz files, e.g. "rans" or "euler".

        split_dir:
            Path to the selected split JSON, e.g. "split_default.json".
            This is the file created by create_dataset_split().

        analyze:
            If True, run your existing visualization/debug summary code.

        filter:
            Optional function that receives case_data and returns True/False.

        num_cases:
            Optional maximum number of usable cases to load.

    Returns:
        List of case dictionaries, same as before, but now with:
            case_data["usable"] = np.array(True/False)
            case_data["train"] = np.array(True/False)
    """

    if not (split_dir is None):
        # add json suffux if not present
        if not split_dir.endswith(".json"):
            split_dir += ".json"

        with open(split_dir, "r") as f:
            split_json = json.load(f)

        split_lookup = {
            str(row["case"]): row
            for row in split_json["cases"]
        }

    files = [f for f in os.listdir(dir) if f.endswith(".npz")]

    files = sorted(
        files,
        key=lambda name: int(name.replace(".npz", "")) if name.replace(".npz", "").isdigit() else name
    )

    data = []
    start = time.time()
    num_total_cases = len(files)
    num_failed_cases = 0

    print(
        f"* Loading CFD data from {dir} ({num_total_cases} cases found) "
        f"using split={split_dir}, filter={filter}, num_cases={num_cases}..."
    )

    for file in files:
        case_data = {}
        file_path = os.path.join(dir, file)

        with np.load(file_path, allow_pickle=True) as f:
            for key in f.files:
                case_data[key] = f[key]

            # Store the filename/case ID as an identifier.
            case_id = file.replace(".npz", "")
            case_data["case_id"] = case_id

        if not (split_dir is None):
            split_row = split_lookup.get(case_data["case_id"], None)

            if split_row is None:
                usable = False
                train = False
                invalid_reasons = ["missing_from_split_json"]
            else:
                usable = bool(split_row["valid"])
                train = split_row["split"] == "train"
                invalid_reasons = split_row.get("invalid_reasons", [])
        else:
            usable = True
            train = False
            invalid_reasons = []

        case_data["usable"] = np.array(usable)
        case_data["train"] = np.array(train)

        if not bool(case_data["usable"]):
            print(
                f"  * Warning: Case {case_data['case_id']} is unusable according to split JSON. "
                f"Reasons: {invalid_reasons}. Skipping this case."
            )
            num_failed_cases += 1
            continue

        if filter is not None and not filter(case_data):
            print(
                f"  * Info: Case {case_data['case_id']} did not pass the filter criteria. "
                f"Skipping this case."
            )
            num_failed_cases += 1
            continue

        case_data["info"] = {
            "converged": case_data["converged"],
            "file_name": file,
            "usable": case_data["usable"],
            "train": case_data["train"],
        }

        case_data["flow_condition"] = case_data["flow_condition"].item()
        case_data["geometry"] = case_data["geometry"].item()

        case_data["aircraft_geometry"] = case_data["geometry"]

        case_data["flow_condition"] = {
            "v_inf": np.linalg.norm(case_data["flow_condition"]["v_inf"]),
            "alpha": 0.0,
            "T_inf": case_data["flow_condition"]["T_inf"],
            "p_inf": case_data["flow_condition"]["p_inf"],
            "nuTilda_inf": case_data["flow_condition"]["nuTilda"],
        }

        if "nuTilda" not in case_data:
            case_data["nuTilda"] = (
                np.ones_like(case_data["p"]) * case_data["flow_condition"]["nuTilda_inf"]
            )

        data.append(case_data)

        if num_cases is not None and len(data) >= num_cases:
            break

    elapsed = time.time() - start

    if len(data) == 0:
        print(
            f"* Loaded 0/{num_total_cases} CFD cases ({num_failed_cases} failed) "
            f"from {dir} in {elapsed:.2f}s"
        )
        return data

    print(
        f"* Loaded {len(data)}/{num_total_cases} CFD cases ({num_failed_cases} failed) "
        f"from {dir} in {elapsed:.2f}s (~{elapsed / len(data):.2f}s per case)"
    )

    # Shape consistency checks.
    num_vertices, num_states = None, None

    for case in data:
        if num_vertices is None:
            num_vertices = case["vertex_coordinates"].shape[0]
            num_states = case["p"].shape[0]
        else:
            assert num_vertices == case["vertex_coordinates"].shape[0], (
                f"Case {case['case_id']} has different number of vertices than expected"
            )
            assert num_states == case["p"].shape[0], (
                f"Case {case['case_id']} has different number of states than expected"
            )

    if analyze:
        print(f"\n=== CFD Data Summary (case 0/{len(data)}) ===")
        print_content(data[0], 0)
        print("=========================\n")

        for case in data:
            scalars = case["vertex_coordinates"][:, 1]

            p = visualize_scalars(
                case["vertex_coordinates"],
                scalars,
                title=f"{case['span']} scalar field",
                cmap="viridis",
                clim=(0, 25),
            )
            p.show()

    return data

def load_lowfi_cfd_data(cfd_data:list[dict], lf_dir:str, split_dir:str):

    # Only extract the cases that are present in the high-fidelity data
    cfd_case_ids = set(case['case_id'] for case in cfd_data)
    case_filter = lambda case: case['case_id'] in cfd_case_ids
    euler_data = load_cfd(lf_dir, split_dir, filter=case_filter)
    assert len(euler_data) == len(cfd_data), f"Low-fidelity data length {len(euler_data)} does not match high-fidelity data length {len(cfd_data)}"

    # creating a mapping from case_id to low-fidelity data
    case_id_to_lf_data = {case['case_id']: case for case in euler_data}

    # add the data from the low-fidelity cases to the high-fidelity cases
    for case in cfd_data:
        lf_case = case_id_to_lf_data[case['case_id']]
        case['low_fidelity_data'] = {
            'p': lf_case['p'],
            'U': lf_case['U'],
            'T': lf_case['T'],
            'vertex_coordinates': lf_case['vertex_coordinates'],
            'lift': lf_case['lift'],
            'drag': lf_case['drag'],
        }
