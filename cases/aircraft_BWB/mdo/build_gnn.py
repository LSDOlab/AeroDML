# import os
# import os
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

# flags = os.environ.get("XLA_FLAGS", "")
# os.environ["XLA_FLAGS"] = (
#     flags + " --xla_gpu_deterministic_ops=true"
# ).strip()

import jax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")
import jax

print("JAX version:", jax.__version__)
print("X64 enabled:", jax.config.read("jax_enable_x64"))
print("Default backend:", jax.default_backend())
print("Available devices:", jax.devices())

for device in jax.devices():
    print( "Platform:", device.platform, "| Device:", device.device_kind)
# exit()
# For optimization:
from mdo_utils import find_projected_points, read_and_split, read_mesh

import aerodml as adn
from aerodml.postprocessing import compute_forces, integrate_forces, integrate_moments

import os
import jax.numpy as jnp
import numpy as np
import time

from aerodml.utils import rho_physical, print_content
from typing import Callable

# model 
from aerodml.networks import build_network, ParameterInfo, Param
from aerodml.checks import check_model, check_res_func
from aerodml.model import get_model

# loaders
from aerodml.load_mesh import (
    load_foam_mesh,
    load_dual_graph,
    check_mesh_settings,
    load_coarse_dual_graph,
    build_geometry_data,
    get_lowfi_CFD_mesh,
)
from aerodml.load_panel_data import get_lowfi_panel_mesh, load_lowfi_panel_data
from aerodml.load_data import load_cfd, load_lowfi_cfd_data

# training
from aerodml.train import build_residual_function, train_model

# paths
from pathlib import Path
HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parents[0]          # aircraft_BWB
CASES_DIR = HERE.parents[1]         # cases

BWB_DIR = CASES_DIR / "datasets" / "BWB"
MODEL_DIR = CASE_DIR / "model" 
PARAM_DIR = MODEL_DIR / "saved_parameters"

print(f"CASE_DIR: {CASE_DIR}")
print(f"PARAM_DIR: {PARAM_DIR}")

import sys
sys.path.insert(0, str(BWB_DIR))
from bwb_geometry import compute_geometry_csdl
from bwb_helper_functions import rotate_pitch

from idwarp_jax.driver import deform_mesh
from geometry import build_surface_csdl

# ......................................................................
# ==============================SETTINGS================================
# deform_full_volume = False
deform_full_volume = True

mesh_dir = 'mesh_639k'
x_surf_hash_gid = 'a63d3d5d35e5809c' # read and choose hash from mesh_dir

# x_surf_hash = 'a63d3d5d35e5809c' # old geo
x_surf_hash = 'NEWHASH' # new geo

geometry_dir ='bwb_geometry/'
stp_file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_new_stored_refit.pickle'

print_timings = True # False
# .......................................................................

geometry_directory        =  os.path.join(BWB_DIR, geometry_dir)
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name
dafoam_directory    = os.path.join(MODEL_DIR, f'{mesh_dir}/')
surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'
# ==============================SETTINGS================================
# ......................................................................

# ======================== CSDL surf --> OPENFOAM surf =======================
surf_indices = np.load(f"{dafoam_directory}/surf_gid_full_{x_surf_hash_gid}.npy")
unique_gids, surf_indices_list = np.unique(surf_indices, return_index=True)
surf_25r_to_1r_map = surf_indices_list.tolist()
num_points_25r = surf_indices.shape[0]

# ============================== volume points ================================
jax.config.update("jax_enable_x64", False)
# from vol_warper import build_volume_pts_func
from idwarp_jax.vol_warper import build_volume_pts_func
mesh_info = read_mesh(dafoam_directory, {'run_checks':True, 'verbose':False, 'visualize':False})
foam_mesh = mesh_info['geometry_topology']['foam_mesh']
topology = mesh_info['geometry_topology']
R = 287.05
T_ref = 228.714
nuTilda_ref = 4.5e-5
p_ref = 30089.6
v_ref = 227.38
rho_ref = rho_physical(p_ref, T_ref, R)
flow_condition = {
    'v_inf': np.array([227.3805]),
    'alpha': np.array([0]),
    'T_inf': np.array([228.714]),
    'p_inf': np.array([30089.6]),
    'nuTilda_inf': np.array([4.5e-05]),
}
constants = {
    'T': {      'mean': np.float64(232),                'std': 3*np.float64(11.0),             'ref': T_ref,},
    'nuTilda': {'mean': np.float64(0.00559),            'std': 3*np.float64(0.0179),           'ref': nuTilda_ref,},
    'p': {      'mean': np.float64(3.05e+04),           'std': 3*np.float64(3.86e+03),         'ref': p_ref,},
    'v': {      'mean': np.array([202.8, 14.2, -8.01]), 'std': 3*np.array([48.15, 32.3, 44.6]),'ref': v_ref,},
    'rho': {                                                                                   'ref': rho_ref,},
    'R': R, 'gamma': 1.4,
    # 'gnn_device': jax.devices("cpu")[0],
    # 'vol_device': jax.devices("cpu")[0],
    # 'panel_device': jax.devices("cpu")[0],
    'gnn_device': jax.devices("gpu")[0],
    'vol_device': jax.devices("gpu")[0],
    'panel_device': jax.devices("gpu")[0],
}

# Create deformation function
copied_points = np.array(foam_mesh["points"],copy=True,)
if deform_full_volume:
    base_volume_points = copied_points
else:
    # base_volume_points = np.array(topology["dual_graph"]["dual_cells"], copy=True)
    points = np.asarray(foam_mesh["points"])
    wall_faces = [
        np.asarray(face)
        for face, face_type in zip(foam_mesh["faces"]["points"], foam_mesh["faces"]["type"])
        if face_type == 1
    ]
    global_conn = np.concatenate(wall_faces)
    wall_ids = np.unique(global_conn)
    wall_points = points[wall_ids]
    base_volume_points = np.concatenate([topology["dual_graph"]["dual_cells"], wall_points], axis=0,)

jax.config.update("jax_enable_x64", False)
compute_volume_pts, d_compute_volume_pts, aux = build_volume_pts_func(
    foam_mesh, base_volume_points, constants['vol_device'], return_aux = True, print_timings = print_timings,
    cached_routes = os.path.join(dafoam_directory, f"cached_routes_{x_surf_hash}"),
    progress=False,
)
unique_wall_points = aux['wall_points']
wall_faces = aux['wall_faces']

num_wall_faces = len(wall_faces)
num_points_1r = unique_wall_points.shape[0]

# ============================== GNN MODEL ================================
from aerodml.load_mesh import build_geometry_case, build_panel_case
from aerodml.utils import stacked_unnormalize

def build_ml_nn_eval_func(
        topology,
        constants,
        flow_condition,
        load_model_path,
        deform_full_volume=True,
        lowfi:dict=None,
        model_config:dict=None,
    ):

    # With lowfi, pass lowfi_data to both evaluators; the VJP also returns d_lowfi_data.
    # Without lowfi, the existing call signatures and two VJP outputs are preserved.
    if deform_full_volume:
        update_geometry_points = lambda x: build_geometry_case(topology, x, False)
    else:
        update_geometry_points = lambda x: build_geometry_case(topology, x, False, precomputed_dual_cells_and_surf=x)

    gnn, param_info = get_model(constants, 'panel' if lowfi is not None else False, root=PARAM_DIR, model_config=model_config)
    params = param_info.load('', custom_path=load_model_path)

    def _compute_ml_nn_full(global_x_vol, pitch, lowfi_data = None):
        # Get new points to serve as input
        warped_geo = update_geometry_points(global_x_vol)

        if lowfi is not None:
            if lowfi_data is None:
                raise ValueError("lowfi_data is required when lowfi is provided")
            warped_geo["low_fidelity_inputs"] = lowfi["mapping_function"](
                topology,
                lowfi["topology"],
                warped_geo,
                lowfi_data,
                flow_condition,
            )

        # Predicted normalized flow quantities from GNN
        v_pred, T_pred, p_pred, nuTilda_pred = gnn(params, flow_condition, warped_geo, topology)
        topology['foam_mesh']['points'] = copied_points
        
        # Unnormalize predicted flow quantities
        p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
        T_pred = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
        nuTilda_pred = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])
        p_pred = jnp.clip(p_pred, a_min=1000, a_max=np.inf)
        v_pred = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])

        # Compute face forces
        force_info = compute_forces(p_pred, v_pred, nuTilda_pred, T_pred, warped_geo, topology)
        
        # Compute scalar values
        L, D = integrate_forces(force_info)
        M = integrate_moments(force_info, axis="y")

        # Postprocess face values
        forces_global = force_info['forces']
        centroids_global = force_info['surface_centroids']

        # Return aux surface forces and centroids
        # NOTE: these must be rotated BACK to the body-fixed points
        surf_aux = {
            'surface_forces': forces_global,
            'surface_points': centroids_global,
        }
        
        # auxiliary info to return for debugging/visualization
        aux = {
            'p':p_pred, 'T': T_pred, 'nuTilda': nuTilda_pred, 'v': v_pred,
            'warped_geo': warped_geo,
        }
        return L, D, M, surf_aux, aux
    print("X64 enabled:", jax.config.read("jax_enable_x64"))

    def _compute_ml_nn_outputs(global_x_vol, pitch, lowfi_data = None):
        L, D, M, surf_aux, _ = _compute_ml_nn_full(global_x_vol, pitch, lowfi_data = lowfi_data)
        return L, D, M, surf_aux

    _compute_ml_nn_jit = jax.jit(_compute_ml_nn_outputs, device=constants['gnn_device'])
    _compute_ml_nn_full_jit = jax.jit(_compute_ml_nn_full, device=constants['gnn_device'])

    def _d_compute_ml_nn(global_x_vol, pitch, d_L, d_D, d_M, d_surf_aux, lowfi_data = None):
        primal_outputs, pullback = jax.vjp(
            _compute_ml_nn_outputs,
            global_x_vol,
            pitch,
            lowfi_data,
        )

        # Computes the VJP with respect to global_x_vol, pitch, and lowfi_data.
        d_vol, d_pitch, d_lowfi_data = pullback((d_L, d_D, d_M, d_surf_aux))
        if lowfi is not None:
            return d_vol, d_pitch, d_lowfi_data
        return d_vol, d_pitch


    _d_compute_ml_nn_jit = jax.jit(
        _d_compute_ml_nn,
        device=constants['gnn_device'],
    )

    def compute_ml_nn(global_x_vol_in, pitch, lowfi_data = None):
        start = time.time()
        # print("X64 enabled:", jax.config.read("jax_enable_x64"))
        new_vol_pts_gpu = jax.device_put(jnp.asarray(global_x_vol_in, dtype=jnp.float32),constants['gnn_device'])
        pitch_gpu = jax.device_put(jnp.asarray(pitch, dtype=jnp.float32),constants['gnn_device'])
        lowfi_data_gpu = jax.device_put(jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), lowfi_data), constants['gnn_device'])
        L, D, M, surf_aux = _compute_ml_nn_jit(new_vol_pts_gpu, pitch_gpu, lowfi_data_gpu)
        L.block_until_ready()
        D.block_until_ready()
        M.block_until_ready()
        end = time.time()
        if print_timings:
            print(f'* evaluated GNN (pitch: {pitch}, L: {(2*L/1000):.3f}kN) ({end - start:.3f} sec)')
        return L, D, M, surf_aux

    def d_compute_ml_nn(global_x_vol, pitch, d_L, d_D, d_M, d_surf_aux, lowfi_data = None):
        start = time.time()
        # d_outputs
        d_L_gpu = jax.device_put(jnp.asarray(d_L, dtype=jnp.float32),constants['gnn_device'])
        d_D_gpu = jax.device_put(jnp.asarray(d_D, dtype=jnp.float32),constants['gnn_device'])
        d_M_gpu = jax.device_put(jnp.asarray(d_M, dtype=jnp.float32),constants['gnn_device'])
        d_surf_aux_gpu = jax.device_put(jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), d_surf_aux,), constants['gnn_device'])
        
        # d_inputs
        global_x_vol_gpu = jax.device_put(jnp.asarray(global_x_vol, dtype=jnp.float32),constants['gnn_device'])
        pitch_gpu = jax.device_put(jnp.asarray(pitch, dtype=jnp.float32),constants['gnn_device'])

        lowfi_data_gpu = jax.device_put(jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), lowfi_data), constants['gnn_device'])

        # compute vjp
        derivatives = _d_compute_ml_nn_jit(global_x_vol_gpu, pitch_gpu, d_L_gpu, d_D_gpu, d_M_gpu, d_surf_aux_gpu, lowfi_data_gpu)
        jax.block_until_ready(derivatives)
        end = time.time()
        if print_timings:
            print(f"* evaluated D_GNN ({end - start:.3f} sec)")
        return derivatives
    return compute_ml_nn, d_compute_ml_nn, _compute_ml_nn_full_jit
