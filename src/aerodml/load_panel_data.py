import numpy as np
import h5py
from .utils import print_content, stacked_normalize
from .visualize_mesh import visualize_scalars
import matplotlib.pyplot as plt
import os
import time
import pickle
import json

from typing import Callable
from .utils import print_content
from scipy.spatial import cKDTree

import pyvista as pv

def generate_panel_sampling_points(config:dict, N:int):
    dual_graph = config['topology']['dual_graph']

    points = dual_graph['dual_cells']
    edges = dual_graph['dual_edges']
    known_data_mask = (dual_graph['dual_cell_types'] == 1) | (dual_graph['dual_cell_types'] == 2)

    # print(points.shape, points.dtype)
    # print(edges.shape, edges.dtype)
    # print(known_data_mask.shape, known_data_mask.dtype, known_data_mask.sum())

    # filter out points that are too close to the wall
    minimum_wall_distance = 0.12 # in whatever units the geometry is defined (e.g. meters)
    wall_distance = np.linalg.norm(points - points[dual_graph['closest_wall_id']], axis=1)
    known_data_mask |= (wall_distance < minimum_wall_distance)

    # filter out points with y<0.01 (too close to symmetry plane)
    known_data_mask |= (points[:, 1] < 0.01)

    candidate_indices = np.flatnonzero(~known_data_mask)
    if len(candidate_indices) < N:
        raise ValueError(f"Not enough candidate points to sample from. Found {len(candidate_indices)}, but need {N}. Consider reducing N or adjusting the known_data_mask criteria.")
    
    rng = np.random.default_rng(30)
    sample_indices = rng.choice(candidate_indices, size=N, replace=False)
    assert len(sample_indices) == N
    assert len(sample_indices) == len(set(sample_indices)) # no duplicates
    return sample_indices

def write_panel_generation_configs(configs):
    """
    This file generates panel method data for each CFD case in 'dir'
    It creates a new PANEL_CONFIG config file.

    This file contains each CFD case's:
    - case_id (filename without .npz)
    - all geometry configuration parameters needed to define the model
    - flow conditions

    This file will be used to generate the panel method data for each CFD case
    """    

    # find baseline config ('case_id' == 0)
    baseline_config = next(config for config in configs if config['case_id'] == '0')
    N = 10000
    sample_point_indices = generate_panel_sampling_points(baseline_config, N)

    panel_config = []
    for config in configs:
        # print_content(config, 0)
        sample_points = config['warped_geo']['dual_graph']['dual_cells'][sample_point_indices]
        assert sample_points.shape == (N, 3)
        assert sample_point_indices.shape == (N,)
        case_config = {
            'case_id': config['case_id'],
            'aircraft_geometry': config['aircraft_geometry'],
            'sample_points': sample_points,
            'sample_indices': sample_point_indices,
            'flow_condition': config['flow_condition'],
        }
        panel_config.append(case_config)

    # Save the panel config as a pickle file
    import pickle
    panel_config_path = os.path.join('PANEL_CONFIG.pkl')
    with open(panel_config_path, 'wb') as f:
        pickle.dump(panel_config, f)
    print(f"Saved panel generation config for {len(panel_config)} cases to {panel_config_path}")

def build_4way_cell_interp(points, tris, adjacency, owner_tris, projected_points):
    # triangle centroids: (num_cells, 3)
    tri_centroids = points[tris].mean(axis=1)

    # owner + 3 neighbors -> (N, 4)
    assert adjacency.shape == (tris.shape[0], 3)
    neighbor_tris = adjacency[owner_tris]   # (N, 3)

    if np.any(neighbor_tris < 0):
        bad = np.where(np.any(neighbor_tris < 0, axis=1))[0][:10]
        raise ValueError(
            f"Some owner triangles do not have 3 neighbors. Example target rows: {bad.tolist()}"
        )

    interp_cell_ids = np.concatenate(
        [owner_tris[:, None], neighbor_tris],
        axis=1,
    ).astype(np.int64)

    # centroid locations of the 4 stencil cells: (N, 4, 3)
    stencil_centroids = tri_centroids[interp_cell_ids]

    # 3D distances from projected point to each stencil centroid: (N, 4)
    dists = np.linalg.norm(
        stencil_centroids - projected_points[:, None, :],
        axis=2,
    )

    # inverse-distance weights
    interp_cell_weights = 1.0 / np.maximum(dists, 1e-12)
    interp_cell_weights /= interp_cell_weights.sum(axis=1, keepdims=True)
    return interp_cell_ids, interp_cell_weights

def build_cell_interp(points, cells_dict, cell_adjacency, owner_cells, projected_points):
    tris = np.asarray(cells_dict["triangle"], dtype=np.int64)
    quads = np.asarray(cells_dict["quad"], dtype=np.int64)

    num_tris = tris.shape[0]
    num_quads = quads.shape[0]
    num_cells = num_tris + num_quads

    owner_cells = np.asarray(owner_cells, dtype=np.int64)
    projected_points = np.asarray(projected_points)

    if owner_cells.ndim != 1:
        raise ValueError(f"owner_cells must be one-dimensional, got {owner_cells.shape}")

    if projected_points.shape != (owner_cells.shape[0], 3):
        raise ValueError(
            f"projected_points has shape {projected_points.shape}, "
            f"but expected ({owner_cells.shape[0]}, 3)"
        )

    if np.any(owner_cells < 0) or np.any(owner_cells >= num_cells):
        raise ValueError("owner_cells contains invalid global panel-cell IDs")

    # Centroids use the same global ordering as pressure and velocity:
    # [all triangle centroids, all quad centroids]
    tri_centroids = points[tris].mean(axis=1)
    quad_centroids = points[quads].mean(axis=1)
    cell_centroids = np.concatenate((tri_centroids, quad_centroids), axis=0)

    assert cell_centroids.shape == (num_cells, 3)

    # Adjacency rows must use the same global cell ordering.
    adjacency_rows = list(cell_adjacency["triangle"]) + list(cell_adjacency["quad"])

    if len(adjacency_rows) != num_cells:
        raise ValueError(
            f"Found {len(adjacency_rows)} adjacency rows, but expected {num_cells}"
        )

    # For triangle/quad data this will normally be 4, giving a stencil size of 5:
    # owner + up to 4 neighbors.
    max_neighbors = max(len(neighbors) for neighbors in adjacency_rows)
    stencil_size = 1 + max_neighbors
    num_targets = owner_cells.shape[0]

    # Initialize every slot with the owner cell.
    #
    # Padding must contain a valid cell ID because these IDs will later be used
    # for array indexing. Padding entries receive weight zero.
    interp_cell_ids = np.repeat(owner_cells[:, None], stencil_size, axis=1)

    # First entry is always the owner and is therefore valid.
    valid_mask = np.zeros((num_targets, stencil_size), dtype=bool)
    valid_mask[:, 0] = True

    for target_index, owner_cell in enumerate(owner_cells):
        neighbors = np.asarray(adjacency_rows[owner_cell], dtype=np.int64).reshape(-1)

        # Remove boundary markers, invalid IDs, the owner itself, and duplicates.
        neighbors = neighbors[
            (neighbors >= 0)
            & (neighbors < num_cells)
            & (neighbors != owner_cell)
        ]

        # Preserve the original adjacency order while removing duplicates.
        if neighbors.size:
            _, unique_indices = np.unique(neighbors, return_index=True)
            neighbors = neighbors[np.sort(unique_indices)]

        if neighbors.size > max_neighbors:
            raise ValueError(
                f"Cell {owner_cell} has {neighbors.size} valid neighbors, "
                f"but max_neighbors is {max_neighbors}"
            )

        num_neighbors = neighbors.size

        interp_cell_ids[target_index, 1:1 + num_neighbors] = neighbors
        valid_mask[target_index, 1:1 + num_neighbors] = True

    stencil_centroids = cell_centroids[interp_cell_ids]

    dists = np.linalg.norm(
        stencil_centroids - projected_points[:, None, :],
        axis=2,
    )

    # Inverse-distance weights for the owner and real neighbors.
    # Padded entries receive exactly zero weight.
    inverse_dists = np.where(
        valid_mask,
        1.0 / np.maximum(dists, 1e-12),
        0.0,
    )

    weight_sums = inverse_dists.sum(axis=1, keepdims=True)

    if np.any(weight_sums <= 0.0):
        bad_rows = np.flatnonzero(weight_sums[:, 0] <= 0.0)[:10]
        raise ValueError(
            f"Could not construct interpolation weights for rows {bad_rows.tolist()}"
        )

    interp_cell_weights = inverse_dists / weight_sums
    return interp_cell_ids, interp_cell_weights

def get_panel_mesh(path: str, cfd_mesh_topology: dict, panel_load_settings: dict) -> dict:
    with open(path, "rb") as f:
        connectivity = pickle.load(f)

    # Extract info
    cell_info = connectivity["cell_adjacency_data"]
    points = np.asarray(cell_info[0])
    cells_dict = cell_info[1]
    cell_adjacency = cell_info[2]
    tris = np.asarray(cells_dict["triangle"], dtype=np.int64)
    quads = np.asarray(cells_dict["quad"], dtype=np.int64)

    num_tris = tris.shape[0]
    num_quads = quads.shape[0]
    num_cells = num_tris + num_quads
    num_points = points.shape[0]

    # Global panel-cell ordering:
    # [all triangles, all quads]

    # This is the same ordering as pressure and velocity.
    graph = cfd_mesh_topology["dual_graph"]
    surface_mask = graph["dual_cell_types"] == 1
    target_projection_points = graph["dual_cells"][surface_mask]
    num_projection_points = target_projection_points.shape[0]

    # Build mixed triangle/quad PyVista surface.
    tri_faces = np.hstack((np.full((num_tris, 1), 3, dtype=np.int64), tris)).ravel()
    quad_faces = np.hstack((np.full((num_quads, 1), 4, dtype=np.int64), quads)).ravel()
    faces = np.concatenate((tri_faces, quad_faces))

    surf = pv.PolyData(points, faces)

    assert surf.n_cells == num_cells
    assert surf.n_points == num_points

    # Find the closest panel cell and projected surface location for each CFD surface node.
    start = time.time()

    surface_cells_to_panels, projected_points = surf.find_closest_cell(
        target_projection_points, return_closest_point=True
    )

    end = time.time()

    print(
        f"* Projected {num_projection_points} surface nodes onto panel mesh "
        f"in {end - start:.2f} seconds"
    )

    surface_cells_to_panels = np.asarray(surface_cells_to_panels, dtype=np.int64)
    projected_points = np.asarray(projected_points)

    assert surface_cells_to_panels.shape == (num_projection_points,)
    assert projected_points.shape == (num_projection_points, 3)

    # Owner cell + all edge-neighboring cells.
    # Triangle:
    #   owner + 3 neighbors + zero weighted cell owner that doesn't contribute
    # Quad:
    #   owner + 4 neighbors
    interp_cell_ids, interp_cell_weights = build_cell_interp(
        points=points,
        cells_dict=cells_dict,
        cell_adjacency=cell_adjacency,
        owner_cells=surface_cells_to_panels,
        projected_points=projected_points,
    )

    if panel_load_settings["visualize"]:
        fake_surface_field = np.asarray(connectivity["pressure_baseline"])
        clim = (fake_surface_field.min(), fake_surface_field.max())

        surf.cell_data["pressure"] = fake_surface_field

        interpolated_pressure = np.sum(
            fake_surface_field[interp_cell_ids] * interp_cell_weights,
            axis=1,
        )

        proj = pv.PolyData(target_projection_points)
        proj["interpolated_pressure"] = interpolated_pressure

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(surf, scalars="pressure", show_edges=False, clim=clim)
        plotter.add_mesh(proj, scalars="interpolated_pressure", point_size=8, render_points_as_spheres=True, clim=clim)

        from aerodml.visualize_mesh import available_cameras
        plotter.camera_position = available_cameras['pos_2']['position']
        plotter.camera.view_angle = available_cameras['pos_2']['view_angle']
        plotter.camera.clipping_range = available_cameras['pos_2']['clipping_range']
        plotter.show(screenshot='PANEL_INTERPOLATION.png')

    panel_topology = {
        "triangles": tris,
        "quads": quads,
        "adjacency": cell_adjacency,
        "surface_points": points,
        "projected_surface_map": surface_cells_to_panels,
        "interp_cell_ids": interp_cell_ids,
        "interp_cell_weights": interp_cell_weights,
        "num_triangles": num_tris,
        "num_quads": num_quads,
        # "edges2cells": edges2cells,
        # "points2cells": points2cells,
    }

    return panel_topology

import jax.numpy as jnp
def get_lowfi_panel_mesh(path:str, cfd_mesh_topology:str, panel_load_settings:dict[str])->dict:
    panel_topology = get_panel_mesh(path, cfd_mesh_topology, panel_load_settings)

    def data_transfer_function(graph_topology, euler_topology, graph_x, panel_data, farfield):
        graph = graph_topology['dual_graph']
        edges = graph['dual_edges']
        points = graph_x['dual_graph']['dual_cells']

        cell_ids = panel_topology["interp_cell_ids"]          # (N, 4)
        cell_weights = panel_topology["interp_cell_weights"]  # (N, 4)

        # ======= Known nodes =======
        # 1: surface
        surface_nodes = graph['dual_cell_types'] == 1
        # surface_pressures = panel_data['pressure'][panel_topology['projected_surface_map']]
        # surface_vels = panel_data['velocity'][panel_topology['projected_surface_map']]
        surface_pressures = jnp.sum(
            panel_data['pressure'][cell_ids] * cell_weights,
            axis=1,
        )
        surface_vels = jnp.sum(
            panel_data['velocity'][cell_ids] * cell_weights[:, :, None],
            axis=1,
        )

        # Everywhere else is far-field values
        points = graph_x['dual_graph']['dual_cells']   # shape (N, 3)
        num_points = points.shape[0]
        ff_pressures = farfield['p_inf'] * jnp.ones(num_points)
        ff_vels = jnp.zeros((num_points, 3))
        ff_vels = ff_vels.at[:, 0].set(farfield["v_inf"])   # x-component only
        ff_nuTilda = farfield['nuTilda_inf'] * np.ones(num_points)
        ff_T = farfield['T_inf'] * np.ones(num_points)

        panel_p = ff_pressures.at[surface_nodes].set(surface_pressures)
        panel_U = ff_vels.at[surface_nodes].set(surface_vels)
        panel_T = ff_T
        panel_nulTilda = ff_nuTilda
        return {
            "p": panel_p,
            "U": panel_U,
            "T": panel_T,
            "nuTilda": panel_nulTilda,
        }

    return {'mapping_function': data_transfer_function, 'topology': panel_topology}

def load_panel_data(dir:str, split_dir:str, file_filter: Callable=None):
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
    files = [f for f in os.listdir(dir) if f.endswith(".npz")]

    files = sorted(
        files,
        key=lambda name: int(name.replace(".npz", "")) if name.replace(".npz", "").isdigit() else name
    )

    # Filter using only the filename, before loading any .npz data.
    if file_filter is not None:
        files = [file for file in files if file_filter(file)]

    data = []
    start = time.time()
    num_total_cases = len(files)
    num_failed_cases = 0

    print(
        f"* Loading panel data from {dir} ({num_total_cases} cases found) "
        f"using split={split_dir}, filter={filter}, ..."
    )

    panel_cases = []
    for file in files:
        file_path = os.path.join(dir, file)

        # --------------------------------------------------------
        # Load the .npz exactly like before.
        # --------------------------------------------------------
        panel_data = np.load(file_path)

        case_id = file.replace(".npz", "")
        # print_content(panel_data)

        split_row = split_lookup[case_id]
        usable = bool(split_row["valid"])
        train = split_row["split"] == "train"

        if not usable:
            continue
        
        case = {
            'case_id': case_id,
            'panel_mesh': panel_data['panel_mesh'],
            'pressure': panel_data['pressure'],
            'velocity': panel_data['velocity'],
            'lift': panel_data['L']/2.0, # divide by 2 because symmetry
            'drag': panel_data['Di']/2.0, # divide by 2 because symmetry
            'moment': panel_data['M']/2.0, # divide by 2 because symmetry
        }

        # print('pressure', case['pressure'].shape)
        # print('velocity', case['velocity'].shape)
        # # print('velocity', case['Cp'].shape)
        # exit()
        if 'sample_points' in panel_data:
            sample_points = panel_data['sample_points'][:1000]
            sample_velocity = panel_data['sample_velocity'][:1000]
            sample_pressure = panel_data['sample_pressure'][:1000]

            case['sample_points'] = sample_points
            case['sample_velocity'] = sample_velocity
            case['sample_pressure'] = sample_pressure

        panel_cases.append(case)
    return panel_cases

def load_lowfi_panel_data(cfd_data:list[dict], dir:str, split_dir:str):
    panel_cases = load_panel_data(dir, split_dir)

    # map case_id to panel_case for quick lookup
    panel_case_lookup = {case['case_id']: case for case in panel_cases}

    # for each case in cfd_data, load the corresponding panel method data from "dir"
    # in-place add the panel method data to the case dict under the key 'panel'
    for case in cfd_data:
        case_id = case['case_id']
        case['low_fidelity_data'] = panel_case_lookup[case_id]

import jax
