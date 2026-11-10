import json
from typing import Callable
import time
import os
import numpy as np

def load_graph_vertices(dir:str, split_dir:str, file_filter:Callable=None):
    # ------------------------------------------------------------
    # Load the selected split JSON.
    # This file tells us which case IDs are valid and train/test.
    # ------------------------------------------------------------
    # add json suffux if not present
    if not split_dir.endswith(".json"):
        split_dir += ".json"

    with open(split_dir, "r") as f:
        split_json = json.load(f)

    split_lookup = {
        str(row["case"]): row
        for row in split_json["cases"]
    }

    # ------------------------------------------------------------
    # Find all CFD case files.
    # Sorting by case number keeps loading deterministic.
    # ------------------------------------------------------------
    files = [f for f in os.listdir(dir) if f.endswith(".npy")]

    files = sorted(
        files,
        key=lambda name: int(name.replace(".npy", "")) if name.replace(".npy", "").isdigit() else name
    )

    # Filter using only the filename, before loading any .npz data.
    if file_filter is not None:
        original_num_files = len(files)
        files = [file for file in files if file_filter(file)]
        print(f"Filtered files from {original_num_files} to {len(files)} using file_filter={file_filter}")  

    data = []
    start = time.time()
    num_total_cases = len(files)
    num_failed_cases = 0

    print(
        f"* Loading vertex data from {dir} ({num_total_cases} cases found) "
        f"using split={split_dir}, file_filter={file_filter}, ..."
    )

    vertex_cases = []
    for file in files:
        file_path = os.path.join(dir, file)

        # --------------------------------------------------------
        # Load the .npz exactly like before.
        # --------------------------------------------------------
        vertex_data = np.load(file_path)

        case_id = file.replace(".npy", "")
        # print_content(panel_data)

        split_row = split_lookup[case_id]
        usable = bool(split_row["valid"])
        train = split_row["split"] == "train"
        # print(f"Processing case_id={case_id}, usable={usable}, train={train}")

        if not usable:
            continue
        
        case = {
            'case_id': case_id,
            'vertices': vertex_data,
        }
        # print(case["vertices"].shape)
        vertex_cases.append(case)
    return vertex_cases

def load_graph_vertex_data(cfd_data:dict, dir:str, split_dir:str):
# def load_lowfi_cfd_data(cfd_data:list[dict], lf_dir:str, split_dir:str):

    # Only extract the cases that are present in the high-fidelity data
    cfd_case_ids = set(str(case['case_id']) for case in cfd_data)
    case_filter = lambda filename: str(filename).replace(".npy", "") in cfd_case_ids
    vertex_data = load_graph_vertices(dir, split_dir, file_filter=case_filter)
    assert len(vertex_data) == len(cfd_data), f"Preloaded vertex data length {len(vertex_data)} does not match high-fidelity data length {len(cfd_data)}"

    # creating a mapping from case_id to low-fidelity data
    case_id_to_vertex_data = {case['case_id']: case for case in vertex_data}

    # add the data from the low-fidelity cases to the high-fidelity cases
    for case in cfd_data:
        vertex_case = case_id_to_vertex_data[case['case_id']]
        case['preloaded_dual_vertices'] = {
            'dual_cells': vertex_case['vertices'],
        }