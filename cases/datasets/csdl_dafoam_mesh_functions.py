import gzip
import numpy as np
from pathlib import Path

def read_proc_addressing(dafoam_instance, comm, key="cell"):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    key = key.lower()
    if key not in ["boundary", "cell", "face", "point"]:
        raise ValueError( f'{key} does not have an associated ProcAddressing. Please specify "boundary", "cell", "face", or "point".')

    run_directory = Path(dafoam_instance.run_directory)
    if comm_size > 1:
        filename      = run_directory/f'processor{rank}'/'constant'/'polyMesh'/f'{key}ProcAddressing.gz'
    
    # Serial case: just return the range of number of faces or points
    else:
        if key == "cell":
            data = range(0, dafoam_instance.solver.getNLocalCells())
        elif key == "face":
            data = range(1, dafoam_instance.solver.getNLocalFaces() + 1)
        else:
            raise NotImplementedError("boundary and point proc_addressing haven't been implemented for serial case yet.")
        return data

    openf = gzip.open if filename.suffix == ".gz" else open

    with openf(filename, "rt") as f:
        # Skip header until we hit the size
        for line in f:
            line = line.strip()
            if line.isdigit():
                n = int(line)
                break
        else:
            raise RuntimeError("No labelList size found")

        # Expect opening parenthesis
        if f.readline().strip() != "(":
            raise RuntimeError("Malformed labelList: missing '('")

        data = np.fromiter(
                    (int(f.readline()) for _ in range(n)),
                    dtype=np.int64,
                    count=n
                )
    
    return data


from mpi4py import MPI

def assemble_cell_states(v_local, gid_local, comm):
    all_v   = comm.gather(v_local, root=0)
    all_gid = comm.gather(gid_local, root=0)
    if comm.rank != 0:
        return None
    n = max(g.max() for g in all_gid if g.size) + 1
    V = np.full((n,) + v_local.shape[1:], np.nan)
    for v, g in zip(all_v, all_gid):
        V[g] = v

    # print shape, mean, std, min, max, and count of NaNs for debugging
    print(f"Global state shape: {V.shape}, mean: {np.nanmean(V)}, std: {np.nanstd(V)}, min: {np.nanmin(V)}, max: {np.nanmax(V)}, NaN count: {np.isnan(V).sum()}")
    return V

def assemble_global_points_rank0_average(pts_local, gid_local, comm, root=0):
    """
    pts_local: (N,3) float
    gid_local: (N,) int, mapping local point -> original global point index
    returns X_global (Nglobal,3) on root, else None
    """
    rank = comm.Get_rank()

    pts_local = np.asarray(pts_local)
    gid_local = np.asarray(gid_local, dtype=np.int64)

    if pts_local.ndim == 1:
        pts_local = pts_local.reshape((-1, 3))
    assert pts_local.shape[0] == gid_local.shape[0]

    # Compute global size (handle empty ranks)
    local_max = gid_local.max() if gid_local.size else -1
    n_global = comm.allreduce(local_max, op=MPI.MAX) + 1

    all_pts = comm.gather(pts_local, root=root)
    all_gid = comm.gather(gid_local, root=root)

    if rank != root:
        return None, None
    # all_gid = np.concatenate(all_gid, axis=0)
    # print("unique gids:", np.unique(all_gid).size)
    # print("min,max:", all_gid.min(), all_gid.max())
    # print("has 0?", np.any(all_gid == 0))

    sums = np.zeros((n_global, 3), dtype=pts_local.dtype)
    cnts = np.zeros(n_global, dtype=np.int64)

    for pts, gids in zip(all_pts, all_gid):
        # add-at handles repeated gids correctly
        np.add.at(sums, gids, pts)
        np.add.at(cnts, gids, 1)

    if np.any(cnts == 0):
        missing = np.where(cnts == 0)[0]
        print(f"Warning: {missing.size} global points have no local contributions; first few: {missing[:10]}, last few: {missing[-10:]}")
        # raise RuntimeError(f"Missing {missing.size} global points; first few: {missing[:10]}, last few: {missing[-10:]}")

    X_global = sums / cnts[:, None]
    return X_global, missing

import os

def _gather_petsc_vec_to_root(vec, comm, root=0):
    """Return global 1D numpy array on root, else None."""
    rank = comm.Get_rank()
    start, end = vec.getOwnershipRange()
    local = np.asarray(vec.getArray())  # 1D view
    n_local = end - start

    counts = comm.gather(n_local, root=root)

    if rank == root:
        displs = np.zeros(len(counts), dtype=np.int64)
        displs[1:] = np.cumsum(counts[:-1])
        global_arr = np.empty(np.sum(counts), dtype=local.dtype)
    else:
        displs = None
        global_arr = None

    mpi_t = MPI._typedict[np.dtype(local.dtype).char]
    comm.Gatherv(local, [global_arr, counts, displs, mpi_t], root=root)
    return global_arr if rank == root else None

def write_data(
        dir_name,
        file_name,
        states,
        vertex_coordinates,
        forces,
        configuration,
        flow_condition,
        eval_time,
        converged,
        comm
    ):
    rank = comm.Get_rank()
    root = 0

    # 2) save everything on rank 0
    if rank == root:
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, file_name if file_name.endswith(".npz") else file_name + ".npz")
        if os.path.exists(path):
            # Quit MPI processes before raising error to avoid deadlock
            print(f"WARNING: file {path} already exists!")
            # comm.Abort(1)

        # vertex_coordinates can be a tuple; store each part
        save_dict = {
            "converged": np.array([converged], dtype=np.uint8),
            "lift": np.asarray(forces['lift'], dtype=np.float64),
            "drag": np.asarray(forces['drag'], dtype=np.float64),
            "eval_time": np.asarray(eval_time, dtype=np.float64)
        }

        n_cells = None
        for k, v in states.items():
            if not isinstance(k, str):
                k = str(k)
            if n_cells is None:
                n_cells = v.shape[0]
            assert v.shape[0] == n_cells, f"All state arrays must have the same number of cells. Found {v.shape[0]} for key '{k}' but expected {n_cells}."
            save_dict[k] = v

        vertex_coordinates_values, missing_vertices = vertex_coordinates
        save_dict["vertex_coordinates"] = vertex_coordinates_values
        save_dict["missing_vertices"] = missing_vertices
        save_dict["geometry"] = {}
        save_dict["flow_condition"] = {}

        # configuration: dict where values include 'name' and 'value' arrays
        print_str = ""
        for k, item in configuration.items():
            k = str(k)
            save_dict["geometry"][k] = np.asarray(item["value"])

        for k, item in flow_condition.items():
            k = str(k)
            print_str += f"{k}: {item['value']}  \n"
            save_dict["flow_condition"][k] = np.asarray(item["value"])

        # np.savez_compressed(path, **save_dict)
        np.savez(path, **save_dict)
        print("\n==============================================================")
        print(f"Saved data to {path}, lift: {save_dict['lift']}, drag: {save_dict['drag']}, converged: {save_dict['converged'][0]}")
        print(print_str)
        print("==============================================================\n")

    comm.Barrier()

from scipy.stats import qmc
import csv
def build_sample(config_ranges, N, rank, seed=2, csv_path="samples.csv"):
    """
    Build N Latin-hypercube samples from ref_value +/- delta
    and write them to a CSV.

    Returns:
        samples: list of configuration dicts
    """
    if seed is None:
        raise ValueError("Seed must be provided for MPI.")
    keys = list(config_ranges.keys())
    meta = []
    lower = []
    upper = []
    columns = []

    # Flatten all parameters into one vector of bounds
    for key in keys:
        name = config_ranges[key]["name"]
        ref = np.asarray(config_ranges[key]["ref_value"], dtype=float)
        delta = config_ranges[key]["delta"]

        if ref.ndim == 0:  # scalar
            shape = None
            flat = np.array([ref.item()])
            columns.append(name)
        else:              # array
            shape = ref.shape
            flat = ref.ravel()
            columns.extend([f"{name}_{i}" for i in range(flat.size)])

        lower.extend(flat - delta)
        upper.extend(flat + delta)
        meta.append((key, shape, flat.size))

    # Latin hypercube sampling
    sampler = qmc.LatinHypercube(d=len(lower), seed=seed)
    X = sampler.random(n=N)
    X = qmc.scale(X, lower, upper)

    # Write CSV
    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(X)

    samples = []
    for row in X:
        config = {}
        idx = 0
        for key, shape, size in meta:
            vals = row[idx:idx + size]
            idx += size

            if shape is None:
                value = float(vals[0])
            else:
                value = vals.reshape(shape)

            config[key] = {
                "value": value,
                "name": config_ranges[key]["name"]
            }

        samples.append(config)

    return samples