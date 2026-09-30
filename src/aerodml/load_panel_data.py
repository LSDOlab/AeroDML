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

def get_panel_mesh(path:str, cfd_mesh_topology:str, panel_load_settings:dict[str])->dict:
    # load the pickle file called "CONNECTIVITY.pkl" in the directory "dir"
    with open(path, "rb") as f:
        connectivity = pickle.load(f)

    tris = connectivity['triangles']
    adjacency = connectivity['cell_adjacency']
    points = connectivity['panel_mesh']
    num_cells = tris.shape[0]
    num_points = points.shape[0]

    # project these points onto triangular surface mesh
    volume_points = cfd_mesh_topology['dual_graph']['dual_cells']
    surface_mask = cfd_mesh_topology['dual_graph']['dual_cell_types'] == 1
    target_projection_points = volume_points[surface_mask]
    num_projection_points = target_projection_points.shape[0]

    # print(tris.shape, adjacency.shape, points.shape, target_projection_points.shape)

    # ================== PROJECT FOR INTERPOLATION ==================
    # build triangular surface mesh
    faces = np.hstack([
        np.full((num_cells, 1), 3, dtype=np.int64),
        tris.astype(np.int64),
    ]).ravel()

    surf = pv.PolyData(points, faces)

    # What triangle is each target_projection_point closest to?
    # Also returns the actual projected point on the triangle surface.
    start = time.time()
    surface_cells_to_tris, projected_points = surf.find_closest_cell(
        target_projection_points,
        return_closest_point=True,
    )
    end = time.time()
    print(f"* Projected {num_projection_points} surface nodes onto panel mesh in {end - start:.2f} seconds")
    surface_cells_to_tris = np.asarray(surface_cells_to_tris, dtype=np.int64)
    projected_points = np.asarray(projected_points)

    assert surface_cells_to_tris.shape == (num_projection_points,)
    assert projected_points.shape == (num_projection_points, 3)

    # Also find 4 nearest triangle centers for each target_projection_point, and their distances. This will be used for interpolation.
    interp_cell_ids, interp_cell_weights = build_4way_cell_interp(
        points=points,
        tris=tris,
        adjacency=adjacency,
        owner_tris=surface_cells_to_tris,
        projected_points=projected_points,
    )

    # project the target_projection_points onto the source mesh triangles
    if panel_load_settings['visualize']:
        # simple scalar per triangle center
        fake_surface_field = connectivity['pressure_baseline']
        clim = (fake_surface_field.min(), fake_surface_field.max())

        # build triangular surface mesh
        faces = np.hstack([
            np.full((num_cells, 1), 3, dtype=np.int64),
            tris.astype(np.int64),
        ]).ravel()

        surf = pv.PolyData(points, faces)

        # color each projection point by its matched triangle
        proj = pv.PolyData(target_projection_points)
        # proj['triangle_id'] = fake_surface_field[surface_cells_to_tris]
        proj['triangle_id']  = np.sum(
            connectivity['pressure_baseline'][interp_cell_ids] * interp_cell_weights,
            axis=1,
        )
        

        p = pv.Plotter()
        p.add_mesh(surf, show_edges=False, clim=clim)
        # p.add_mesh(surf, scalars=fake_surface_field, show_edges=True, clim=clim)
        p.add_mesh(proj, scalars='triangle_id', point_size=8, render_points_as_spheres=True, clim=clim)
        p.show()

    # =========================== 3D INTERPOLATION ===========================
    graph = cfd_mesh_topology['dual_graph']
    edges = graph['dual_edges']
    dual_points = cfd_mesh_topology['dual_graph']['dual_cells']

    # ======= Known nodes =======
    # 1: surface
    # 2: far-field
    # 3: N sampled points
    surface_nodes = graph['dual_cell_types'] == 1
    ff_nodes = graph['dual_cell_types'] == 2
    sampled_indices = connectivity['mesh_sample_indices'][:1000]

    # print all shapes and dtypes
    # print(f"surface_nodes: {surface_nodes.shape}, {surface_nodes.dtype}")
    # print(f"ff_nodes: {ff_nodes.shape}, {ff_nodes.dtype}")
    # print(f"sampled_indices: {sampled_indices.shape}, {sampled_indices.dtype}")
    # print(f"edges: {edges.shape}, {edges.dtype}")
    # print(f"points: {points.shape}")

    # =============== Interpolate ==============
    # sampled points should not already be known
    assert not np.any(surface_nodes[sampled_indices])
    assert not np.any(ff_nodes[sampled_indices])

    surface_indices = np.flatnonzero(surface_nodes)
    ff_indices = np.flatnonzero(ff_nodes)
    known_indices = np.concatenate([
        surface_indices,
        ff_indices,
        sampled_indices,
    ])

    # simple Euclidean IDW from k nearest known nodes
    k = 4
    known_points = dual_points[known_indices]
    tree = cKDTree(known_points)
    _, nbr_idx = tree.query(dual_points, k=min(k, known_points.shape[0]))
    
    panel_topology = {
        'triangles': tris,
        'adjacency': adjacency,
        'surface_points': points,
        'projected_surface_map': surface_cells_to_tris,
        'sample_indices': sampled_indices,
        'nbr_idx': nbr_idx,
        'known_indices': known_indices,
        "interp_cell_ids": interp_cell_ids,               # (N, 4)
        "interp_cell_weights": interp_cell_weights,       # (N, 4)
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

def load_panel_data(dir:str, split_dir:str):
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
        sample_points = panel_data['sample_points'][:1000]
        sample_velocity = panel_data['sample_velocity'][:1000]
        sample_pressure = panel_data['sample_pressure'][:1000]

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
            'sample_points': sample_points,
            'sample_velocity': sample_velocity,
            'sample_pressure': sample_pressure,
            'lift': panel_data['L']/2.0, # divide by 2 because symmetry
            'drag': panel_data['Di']/2.0, # divide by 2 because symmetry
        }
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

if __name__ == "__main__":
    from run import get_mesh, load_cfd_data, build_geometry_data
    # High-level options
    device = jax.devices("gpu")[0]
    # new_cfd_data = True
    new_cfd_data = False

    # ============= Pre-processing ==============:
    # device = jax.devices()[0]
    mesh_load_settings =   {'verbose': False, 'run_checks': True, 'visualize': False}
    dual_load_settings =   {'verbose': False, 'run_checks': True, 'visualize': False}
    coarse_load_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    panel_load_settings =  {'verbose': False, 'run_checks': True, 'visualize': False}
    panel_data_settings =  {'verbose': False, 'run_checks': True, 'visualize': False}

    # Flow conditions (SI units):
    from utils import rho_physical
    R = 287.05
    T_ref = 228.714
    nuTilda_ref = 4.5e-5
    p_ref = 30089.6
    v_ref = 227.38
    rho_ref = rho_physical(p_ref, T_ref, R)
    constants = {
        'T': {      'mean': np.float64(232),                'std': 3*np.float64(11.0),             'ref': T_ref,},
        'nuTilda': {'mean': np.float64(0.00559),            'std': 3*np.float64(0.0179),           'ref': nuTilda_ref,},
        'p': {      'mean': np.float64(3.05e+04),           'std': 3*np.float64(3.86e+03),         'ref': p_ref,},
        'v': {      'mean': np.array([202.8, 14.2, -8.01]), 'std': 3*np.array([48.15, 32.3, 44.6]),'ref': v_ref,},
        'rho': {                                                                                   'ref': rho_ref,},
        'R': R, 'gamma': 1.4,
        'device': device,
    }

    # ============= Main Setup and Preprocessing ==============:
    geometry_topology = get_mesh('mesh_data', mesh_load_settings=mesh_load_settings, dual_load_settings=dual_load_settings, coarse_load_settings=coarse_load_settings)
    if not new_cfd_data:
        panel_topology = get_panel_mesh('panel_data', geometry_topology, panel_load_settings)
    else:
        panel_topology = None

    # case_dir = '639k_adiabatic'
    case_dir = 'camber_140'
    cfd_data = load_cfd_data(dir = f'cfd_data/{case_dir}', num_cases = None)
    if not new_cfd_data:
        load_panel_data(cfd_data, dir = f'panel_data/{case_dir}')
    
    from visualize_mesh import visualize_raw_panel_data
    for cfd_case in cfd_data:
        if cfd_case['case_id'] == '0':
            print(cfd_case['panel']['pressure'].shape)
            visualize_raw_panel_data(panel_topology, cfd_case['panel'], 'p', screenshot_path = f"plots/panel_projection_debug_{case_dir}.png")
            # visualize_raw_panel_data(panel_topology, cfd_case['panel'], 'p', screenshot_path = None)
    exit()
    train_data, test_data = build_geometry_data(
        geometry_topology, cfd_data, panel_topology,
        faces = False, verbose=0, visualize=False, device=device
    )
    
    # =============================== Configurations ===============================
    # Uncomment this line to generate the PANEL_CONFIG.pkl file, which contains the sampled panel points for each case
    # write_panel_generation_configs(train_data + test_data)
    # exit()

    # =============================== Compare panel method results to CFD results ===============================
    topology = train_data[0]['topology']
    # Fake eval which just reads the panel method data for the sampled points and returns it as the "predicted" field values for those points. This is just to test the visualization pipeline.
    def panel_eval(params, freestream_config, pts):
        is_interior = topology['dual_graph']['dual_cell_types'] == 0
        v = pts['panel']['v'][is_interior]
        T = pts['panel']['T'][is_interior]
        p = pts['panel']['p'][is_interior]
        nuTilda = pts['panel']['nuTilda'][is_interior]

        v_panel = stacked_normalize(v, constants['v']['mean'], constants['v']['std'])
        T_panel = stacked_normalize(T, constants['T']['mean'], constants['T']['std'])
        p_panel = stacked_normalize(p, constants['p']['mean'], constants['p']['std'])
        nuTilda_panel = stacked_normalize(nuTilda, constants['nuTilda']['mean'], constants['nuTilda']['std'])
        return v_panel, T_panel, p_panel, nuTilda_panel
    
    vid_data = train_data + test_data
    # vid_data = sorted(vid_data, key=lambda x: int(x['case_id']))

    images = []
    from visualize_mesh import build_pyvista_mesh_topology, visualize_raw_panel_data
    from postprocessing_viz import visualize_surface_post
    pv_mesh = build_pyvista_mesh_topology(topology['foam_mesh'])
    # state, camera = 'v', 'TE'
    # state, camera, plane = 'T', 'TE_underside', 0.5
    # state, camera, plane = 'T', 'TE', 0.5
    state, camera, plane = 'p', 'pos_2', 0.5
    # state, camera, plane = 'p', 'pos_2', 10.5
    # state, camera, plane = 'v', 'pos_2', 10.5
    
    # If interactive or not
    video = False
    for config in vid_data:
        print(f"Visualizing case {config['case_id']}...")
        title = f"Case {config['case_id']} (train={config['train']})"
        image = visualize_surface_post(panel_eval, None, config, constants, title = title, state=state, window_size=(993, 960), return_image=video, enable_picking=False, pyvista_mesh_topology=pv_mesh, camera = camera, y_plane = plane, print_l2=True)
        images.append(image)
