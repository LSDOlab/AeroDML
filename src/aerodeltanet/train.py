import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"   # preallocate 95% of total GPU memory
# optional, but this is the default behavior:
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

import jax
from .postprocessing import compute_forces, integrate_forces
jax.config.update("jax_enable_x64", False)

import os
import jax.numpy as jnp
import numpy as np
from .utils import (
    freestream_ic_from_physical,
    nondim_to_phys_state,
    plot_progress,
    plot_parameterization,
    save_aux_data,
    save_all_aux_data,
    reset_aux_data,
    mask_set,
    rho_physical,
)

from .networks import ParameterInfo

from .utils import plot_states, normalize, unnormalize, stack_func, stacked_normalize, stacked_unnormalize, debug_data
from typing import Callable

import matplotlib.pyplot as plt
import time

FARFIELD = 2
BOUNDARY = 1
SYMMETRY = 3

def get_states(rho, v, p, T, nuTilda, freestream, geometry, constants, direction):
    ftype = geometry['foam_mesh']['faces']['type']

    if direction == 'owner':
        cells_1 = geometry['foam_mesh']['faces']['owner']
        cells_2 = geometry['foam_mesh']['faces']['neighbour']
    elif direction == 'neighbor':
        cells_1 = geometry['foam_mesh']['faces']['neighbour']
        cells_2 = geometry['foam_mesh']['faces']['owner']
    else:
        raise ValueError("Direction must be 'left' or 'right'")

    # primitive variables at faces:
    is_boundary = cells_1 < 0
    # four cases:
    # 1) internal face
    rho_cell, v_cell, p_cell, T_cell, nuTilda_cell = rho[cells_1], v[cells_1], p[cells_1], T[cells_1], nuTilda[cells_1]
    rho_inf = rho_physical(freestream['p_inf'], freestream['T_inf'], constants['R'])

    # 2) freestream boundary face (ONE SIDE OF THE FACE IS REAL, THE OTHER DNE)
    # - 'Other side' should be freestream conditions
    is_freestream_face_ghost = is_boundary & (ftype == FARFIELD)
    rho_cell = rho_cell.at[is_freestream_face_ghost].set(rho_inf/constants['rho']['ref'])
    v_cell = v_cell.at[is_freestream_face_ghost].set(freestream['v_inf']/constants['v']['ref'])
    p_cell = p_cell.at[is_freestream_face_ghost].set(freestream['p_inf']/constants['p']['ref'])
    T_cell = T_cell.at[is_freestream_face_ghost].set(freestream['T_inf']/constants['T']['ref'])
    nuTilda_cell = nuTilda_cell.at[is_freestream_face_ghost].set(freestream['nuTilda_inf']/constants['nuTilda']['ref'])

    # 3) wall boundary face
    # - 'Other side' should be no slip conditions for velocity, and zero gradient for other variables (i.e. same value as real side)
    is_wall_face_ghost = is_boundary & (ftype == BOUNDARY)
    rho_cell = rho_cell.at[is_wall_face_ghost].set(rho[cells_2][is_wall_face_ghost]) # zero gradient
    v_cell = v_cell.at[is_wall_face_ghost].set(jnp.zeros_like(v[cells_2][is_wall_face_ghost])) # no slip
    p_cell = p_cell.at[is_wall_face_ghost].set(p[cells_2][is_wall_face_ghost]) # zero gradient
    T_cell = T_cell.at[is_wall_face_ghost].set(T[cells_2][is_wall_face_ghost]) # zero gradient
    nuTilda_cell = nuTilda_cell.at[is_wall_face_ghost].set(jnp.zeros_like(nuTilda[cells_2][is_wall_face_ghost])) # zero

    # 4) symmetry boundary face (IMPORTANT: HARDCODED FOR y=0 plane)
    is_symmetry_face_ghost = is_boundary & (ftype == SYMMETRY)
    rho_cell = rho_cell.at[is_symmetry_face_ghost].set(rho[cells_2][is_symmetry_face_ghost]) # zero gradient
    v_cell = v_cell.at[is_symmetry_face_ghost].set(v[cells_2][is_symmetry_face_ghost] * jnp.array([1, -1, 1])) # invert y velocity
    p_cell = p_cell.at[is_symmetry_face_ghost].set(p[cells_2][is_symmetry_face_ghost]) # zero gradient
    T_cell = T_cell.at[is_symmetry_face_ghost].set(T[cells_2][is_symmetry_face_ghost]) # zero gradient
    nuTilda_cell = nuTilda_cell.at[is_symmetry_face_ghost].set(nuTilda[cells_2][is_symmetry_face_ghost]) # zero gradient

    assert is_freestream_face_ghost.sum() + is_wall_face_ghost.sum() + is_symmetry_face_ghost.sum() == is_boundary.sum()
    # print(f"direction={direction}, is_freestream_face_ghost={is_freestream_face_ghost.sum()} faces")
    # print(f"direction={direction}, is_wall_face={is_wall_face_ghost.sum()} faces")
    # print(f"direction={direction}, is_symmetry_face={is_symmetry_face_ghost.sum()} faces")
    # print(f"direction={direction}, num_boundary_faces={is_boundary.sum()} faces, num_internal_faces={(~is_boundary).sum()} faces")
    return rho_cell, v_cell, p_cell, T_cell, nuTilda_cell

def compute_face_fluxes(rho, v, p, T, nuTilda, freestream, graph_points, geometry, constants):
    # Estimate states at each face using graph points and geometry
    rho_L, V_L, p_L, T_L, nuTilda_L = get_states(rho, v, p, T, nuTilda, freestream, geometry, constants, direction='owner')
    rho_R, V_R, p_R, T_R, nuTilda_R = get_states(rho, v, p, T, nuTilda, freestream, geometry, constants, direction='neighbor')

    n = graph_points['foam_mesh']['faces']['normals'] # unit normals

    # Compute conserved quantities per side
    vx_L = V_L[:, 0]; vx_R = V_R[:, 0]
    vy_L = V_L[:, 1]; vy_R = V_R[:, 1]
    vz_L = V_L[:, 2]; vz_R = V_R[:, 2]
    momx_L = rho_L * vx_L;  momx_R = rho_R * vx_R
    momy_L = rho_L * vy_L;  momy_R = rho_R * vy_R
    momz_L = rho_L * vz_L;  momz_R = rho_R * vz_R
    en_L   = p_L/(constants['gamma']-1.0) + 0.5*rho_L*(vx_L**2 + vy_L**2 + vz_L**2)
    en_R   = p_R/(constants['gamma']-1.0) + 0.5*rho_R*(vx_R**2 + vy_R**2 + vz_R**2)

    # Compute F(U) dot n on each side
    vn_L = jnp.sum(V_L * n, axis=1)
    vn_R = jnp.sum(V_R * n, axis=1)

    Fm_L = rho_L * vn_L
    Fx_L = momx_L * vn_L + p_L * n[:, 0]
    Fy_L = momy_L * vn_L + p_L * n[:, 1]
    Fz_L = momz_L * vn_L + p_L * n[:, 2]
    Fe_L = (en_L + p_L) * vn_L

    Fm_R = rho_R * vn_R
    Fx_R = momx_R * vn_R + p_R * n[:, 0]
    Fy_R = momy_R * vn_R + p_R * n[:, 1]
    Fz_R = momz_R * vn_R + p_R * n[:, 2]
    Fe_R = (en_R + p_R) * vn_R

    # Numerical flux = 0.5 * (F(U_L) + F(U_R)) - 0.5 * s_max * (U_R - U_L)
    # alpha:
    eps = 1e-10
    c_L = jnp.sqrt(jnp.maximum(constants['gamma'] * p_L / rho_L, eps))
    c_R = jnp.sqrt(jnp.maximum(constants['gamma'] * p_R / rho_R, eps))

    tuning_coeff = 0.1
    alpha = tuning_coeff*jnp.maximum(jnp.abs(vn_L) + c_L, jnp.abs(vn_R) + c_R)

    # flux:
    d_rho = rho_R - rho_L
    d_momx = momx_R - momx_L
    d_momy = momy_R - momy_L
    d_momz = momz_R - momz_L
    d_en = en_R - en_L

    # debug_data(0.5 * alpha * (d_rho), 'alpha', ordering='largest', num_values=5)
    flux_Mass = 0.5 * (Fm_L + Fm_R) - 0.5 * alpha * (d_rho)
    flux_mom_x = 0.5 * (Fx_L + Fx_R) - 0.5 * alpha * (d_momx)
    flux_mom_y = 0.5 * (Fy_L + Fy_R) - 0.5 * alpha * (d_momy)
    flux_mom_z = 0.5 * (Fz_L + Fz_R) - 0.5 * alpha * (d_momz)
    flux_energy = 0.5 * (Fe_L + Fe_R) - 0.5 * alpha * (d_en)

    # Set wall faces to have zero flux since vn=0 there
    is_wall_face = (geometry['foam_mesh']['faces']['type'] == BOUNDARY)
    flux_Mass = flux_Mass.at[is_wall_face].set(0.0)
    flux_mom_x = flux_mom_x.at[is_wall_face].set(0.0)
    flux_mom_y = flux_mom_y.at[is_wall_face].set(0.0)
    flux_mom_z = flux_mom_z.at[is_wall_face].set(0.0)
    flux_energy = flux_energy.at[is_wall_face].set(0.0)

    nfaces = geometry['foam_mesh']['faces']['nFaces']
    assert flux_Mass.shape == flux_mom_x.shape == flux_mom_y.shape == flux_mom_z.shape == flux_energy.shape == (nfaces,)
    return flux_Mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy, None

def flux_residual(face_flux, foam_mesh, graph_points):
    # gather face to cell mapping
    L_cell = foam_mesh['faces']['owner']
    R_cell = foam_mesh['faces']['neighbour']

    L_internal = L_cell >= 0
    R_internal = R_cell >= 0

    # integrated flux across each face
    flux_int = face_flux * graph_points['foam_mesh']['faces']['areas']
    assert flux_int.shape == face_flux.shape
    # area stats
    # print(f"Face area stats: min={graph_points['foam_mesh']['faces']['areas'].min()}, max={graph_points['foam_mesh']['faces']['areas'].max()}, mean={graph_points['foam_mesh']['faces']['areas'].mean()}")
    
    # Compute net flux
    cell_flux = jnp.zeros_like(graph_points['foam_mesh']['cells']['centroids'][:, 0])
    cell_flux = cell_flux.at[L_cell[L_internal]].add(flux_int[L_internal])
    cell_flux = cell_flux.at[R_cell[R_internal]].add(-flux_int[R_internal])

    # TODO: face-area weighted
    # cell_surface_area = jnp.zeros_like(cell_flux)
    # cell_surface_area = cell_surface_area.at[L_cell].add(graph_points['foam_mesh']['faces']['areas'])
    # cell_surface_area = cell_surface_area.at[R_cell].add(graph_points['foam_mesh']['faces']['areas'])
    # cell_flux = cell_flux / cell_surface_area+1.0
    return cell_flux

def compute_pinn_residuals(v, T, p, nuTilda, freestream, graph_points, geometry, constants):
    # Compute PINN residuals for momentum, energy, continuity, and turbulence equations

    # 1) unnormalize predictions to get physical units
    # 2) non-dimensionalize to get values in non-dimensional units
    # 3) compute conserved variables
    # 4) compute fluxes for subset of faces
    # 5) compute residuals as sum of fluxes for each cell

    # 1) unnormalize and compute density (already done)
    p_phys, T_phys, v_phys, nuTilda_phys = p, T, v, nuTilda
    rho_phys = rho_physical(p_phys, T_phys, constants['R'])

    # 2) non-dimensionalize
    p_nd = p_phys / constants['p']['ref']
    T_nd = T_phys / constants['T']['ref']
    v_nd = v_phys / constants['v']['ref']
    nuTilda_nd = nuTilda_phys / (constants['nuTilda']['ref'])
    rho_nd = rho_phys / constants['rho']['ref']

    # 4) Compute fluxes for subset of faces (all faces for now)
    mass_flux, mom_flux_x, mom_flux_y, mom_flux_z, energy_flux, _ = compute_face_fluxes(
        rho_nd, v_nd, p_nd, T_nd, nuTilda_nd,
        freestream, graph_points,
        geometry, constants,
    )

    # 3) compute conserved variables (flux is not normalized wrt variable)
    mass_flux_cell = flux_residual(mass_flux, geometry['foam_mesh'], graph_points)
    mom_flux_x_cell = flux_residual(mom_flux_x, geometry['foam_mesh'], graph_points)
    mom_flux_y_cell = flux_residual(mom_flux_y, geometry['foam_mesh'], graph_points)
    mom_flux_z_cell = flux_residual(mom_flux_z, geometry['foam_mesh'], graph_points)
    energy_flux_cell = flux_residual(energy_flux, geometry['foam_mesh'], graph_points)
    
    # Output
    res_continuity = mass_flux_cell
    # res_mom_x = mom_flux_x_cell
    # res_mom_y = mom_flux_y_cell
    # res_mom_z = mom_flux_z_cell
    # res_energy = energy_flux_cell
    res_mom_x = jnp.zeros_like(p)
    res_mom_y = jnp.zeros_like(p)
    res_mom_z = jnp.zeros_like(p)
    res_energy = jnp.zeros_like(p)
    res_turbulence = jnp.zeros_like(p)
    return res_mom_x, res_mom_y, res_mom_z, res_energy, res_continuity,  res_turbulence

def build_residual_function(model:Callable, return_full_residuals:bool=False)->Callable:
    def residual(
            params:dict, data:dict, freestream:dict, graph_points:dict, geometry:dict, constants:dict,
            compute_data_loss:bool = True, compute_pinn_loss:bool=False,
        )->dict:
        print(f'* COMPILING RESIDUALS... (w/ data loss = {compute_data_loss} | w/ PINN loss = {compute_pinn_loss})')
        v_pred, T_pred, p_pred, nuTilda_pred = model(params, freestream, graph_points, geometry)
        
        # outputs
        if return_full_residuals:
            def m(x):
                return x
        else:
            def m(x):
                return jnp.mean(x)

        v_data_raw, T_data_raw, p_data_raw, nuTilda_data_raw = data['U'], data['T'], data['p'], data['nuTilda']
        v_data = stacked_normalize(v_data_raw, constants['v']['mean'], constants['v']['std'])
        T_data = stacked_normalize(T_data_raw, constants['T']['mean'], constants['T']['std'])
        p_data = stacked_normalize(p_data_raw, constants['p']['mean'], constants['p']['std'])
        nuTilda_data = stacked_normalize(nuTilda_data_raw, constants['nuTilda']['mean'], constants['nuTilda']['std'])

        # checks
        n = data['U'].shape[0]
        assert v_pred.shape == v_data.shape == (n, 3), f"Expected velocity shape {(n, 3)}, got {v_pred.shape} and {v_data.shape}"
        assert T_pred.shape == T_data.shape == (n,), f"Expected temperature shape {(n,)}, got {T_pred.shape} and {T_data.shape}"
        assert p_pred.shape == p_data.shape == (n,), f"Expected pressure shape {(n,)}, got {p_pred.shape} and {p_data.shape}"
        assert nuTilda_pred.shape == nuTilda_data.shape == (n,), f"Expected nuTilda shape {(n,)}, got {nuTilda_pred.shape} and {nuTilda_data.shape}"

        # ============== DATA-BASED RESIDUALS ==============
        if compute_data_loss:
            res_vx = (v_pred[:, 0] - v_data[:, 0])
            res_vy = (v_pred[:, 1] - v_data[:, 1])
            res_vz = (v_pred[:, 2] - v_data[:, 2])
            res_T = (T_pred - T_data)
            res_p = (p_pred - p_data)
            res_nuTilda = (nuTilda_pred -nuTilda_data)

            weights = jnp.ones_like(res_T)
            # increase weights for surface neighbors since they are more important for accurate force prediction
            weights = weights.at[geometry['foam_mesh']['faces']['surface_neighbors']].set(4.0)  
            assert res_vx.shape == res_vy.shape == res_vz.shape == res_T.shape == res_p.shape == res_nuTilda.shape == weights.shape
            # data_loss = m(res_vx**2) + m(res_vy**2) + m(res_vz**2) + m(res_T**2) + m(res_p**2) + m(res_nuTilda**2)

            res_vx_w = (res_vx**2.0)
            res_vy_w = (res_vy**2.0)
            res_vz_w = (res_vz**2.0)
            res_T_w = (res_T**2.0)
            res_p_w = weights*(res_p**2.0)
            res_nuTilda_w = res_nuTilda**2.0
            data_loss = m(res_vx_w) + m(res_vy_w) + m(res_vz_w) + m(res_T_w) + m(res_p_w) + m(res_nuTilda_w)
        else:
            res_vx = res_vy = res_vz = res_T = res_p = res_nuTilda = jnp.zeros_like(T_pred)
            data_loss = 0.0
        
        # ============== PINN-BASED RESIDUALS ==============
        if compute_pinn_loss:
            assert graph_points['foam_mesh']['faces']['areas'] is not None, "Graph points must include face areas, centroids, normals to compute PINN residuals"
            v_phys = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])
            T_phys = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
            p_phys = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
            nuTilda_phys = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])

            res_mom_x, res_mom_y, res_mom_z, res_energy, res_continuity, res_turbulence = compute_pinn_residuals(v_phys, T_phys, p_phys, nuTilda_phys, freestream, graph_points, geometry, constants)
            assert res_mom_x.shape == res_mom_y.shape == res_mom_z.shape == res_energy.shape == res_continuity.shape == res_turbulence.shape
            pinn_loss = m(res_mom_x**2) + m(res_mom_y**2) + m(res_mom_z**2) + m(res_energy**2) + m(res_continuity**2) + m(res_turbulence**2)
            pinn_loss = 0.05*pinn_loss
        else:
            res_mom_x = res_mom_y = res_mom_z = res_energy = res_continuity = res_turbulence = jnp.zeros_like(T_pred)
            pinn_loss = 0.0

        loss = data_loss + pinn_loss

        # ============== Extra info ==============:
        p_pred_phys = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
        v_pred_phys = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])
        nuTilda_pred_phys = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])
        T_pred_phys = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
        forces_reconstructed = compute_forces(p_pred_phys, v_pred_phys, nuTilda_pred_phys, T_pred_phys, graph_points, geometry)
        L_r, D_r = integrate_forces(forces_reconstructed)

        forces_data = compute_forces(p_data_raw, v_data_raw, nuTilda_data_raw, T_data_raw, graph_points, geometry)
        L_d, D_d = integrate_forces(forces_data)
        
        residuals = {
            # Data-based residuals
            'res_vx': m(jnp.abs(res_vx)), 'res_vy': m(jnp.abs(res_vy)), 'res_vz': m(jnp.abs(res_vz)),
            'res_T': m(jnp.abs(res_T)), 'res_p': m(jnp.abs(res_p)), 'res_nuTilda': m(jnp.abs(res_nuTilda)),
            'L_pred': L_r, 'D_pred': D_r, 'L_data': L_d, 'D_data': D_d, 'data_loss': data_loss,

            # PINN-based residuals
            'res_mom_x': m(jnp.abs(res_mom_x)), 'res_mom_y': m(jnp.abs(res_mom_y)), 'res_mom_z': m(jnp.abs(res_mom_z)),
            'res_energy': m(jnp.abs(res_energy)), 'res_continuity': m(jnp.abs(res_continuity)), 'res_turbulence': m(jnp.abs(res_turbulence)),
            'pinn_loss': pinn_loss,
        }
        return loss, residuals 
    return residual

# def compute_latest_metric(training_stats, eps=1e-8):
#     keys = ("aux_history_tr", "aux_history_te", "aux_history_te_2")
#     per_case_metrics = []
#     for key in keys:
#         if key not in training_stats or len(training_stats[key]) == 0:
#             continue

#         s = training_stats[key][-1]

#         # Relative squared errors so lift/drag scales don't dominate unfairly
#         drag_err = ((s["D_pred"] - s["D_data"]) / (abs(s["D_data"]) + eps)) ** 2
#         lift_err = ((s["L_pred"] - s["L_data"]) / (abs(s["L_data"]) + eps)) ** 2

#         # loss is already an MSE-like quantity, so keep it as-is
#         loss_term = float(s.get("loss", 0.0))

#         case_metric = (drag_err + lift_err + loss_term) / 3.0
#         per_case_metrics.append(case_metric)
#     if not per_case_metrics:
#         return float("inf")
#     return float(sum(per_case_metrics) / len(per_case_metrics))


def compute_latest_metric(training_stats, eps=1e-8):
    keys = ("aux_history_tr", "aux_history_te", "aux_history_te_2")

    force_weight = 0.05
    worst_case_weight = 0.10

    per_case_metrics = []

    for key in keys:
        if key not in training_stats or len(training_stats[key]) == 0:
            continue

        s = training_stats[key][-1]

        loss_term = float(s.get("loss", 0.0))

        dL = float(s["L_pred"] - s["L_data"])
        dD = float(s["D_pred"] - s["D_data"])

        # One force-vector error, not separate relative L/D errors.
        # This avoids exploding when L_data or D_data is small.
        force_scale = max(
            float((s["L_data"]**2 + s["D_data"]**2) ** 0.5),
            eps,
        )

        force_err = ((dL**2 + dD**2) ** 0.5) / force_scale

        case_metric = loss_term + force_weight * force_err
        per_case_metrics.append(case_metric)

    if not per_case_metrics:
        return float("inf")

    # Average across all 3 cases, with a small penalty for the worst case
    # so one terrible validation case still matters.
    return float(
        sum(per_case_metrics) / len(per_case_metrics)
        + worst_case_weight * max(per_case_metrics)
    )


def train_model(
        model:Callable,
        res_func:Callable,
        params:dict,
        training_configs:list,
        test_configs:list,
        constants:dict,
        solver_settings:dict,
        param_info:ParameterInfo,
    )->dict:
    import optax

    # Main graph topology
    topology = training_configs[0]['topology']

    # build pinn loss residual and gradient functions if needed
    compute_pinn_loss = solver_settings['pinn_loss']
    if compute_pinn_loss:
        # res_jit = jax.jit(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants), device=constants['device'])
        grad_jit = jax.jit(jax.grad(lambda params, data_config, flow_config, graph_config,: res_func(params, data_config, flow_config, graph_config, topology, constants)[0]), device=constants['device'])
        
        # res_jit_pinn = jax.jit(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants, compute_data_loss=False, compute_pinn_loss=True), device=constants['device'])
        grad_jit_pinn = jax.jit(jax.grad(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants, compute_data_loss=False, compute_pinn_loss=True)[0]), device=constants['device'])

        res_jit = res_jit_both = jax.jit(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants, compute_data_loss=True, compute_pinn_loss=True), device=constants['device'])
        # grad_jit_both = jax.jit(jax.grad(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants, compute_data_loss=True, compute_pinn_loss=True)[0]), device=constants['device'])
    else:
        # build data loss residual and gradient functions
        res_jit = jax.jit(lambda params, data_config, flow_config, graph_config: res_func(params, data_config, flow_config, graph_config, topology, constants), device=constants['device'])
        grad_jit = jax.jit(jax.grad(lambda params, data_config, flow_config, graph_config,: res_func(params, data_config, flow_config, graph_config, topology, constants)[0]), device=constants['device'])

    # Prepare optimizer
    schedule = solver_settings['lr']
    solver = optax.adamw(schedule, b1=0.9, b2=0.99, eps=1e-8, weight_decay=1e-5)
    # solver = optax.adam(schedule)
    opt_state = solver.init(params)
    def update_fn(df_dx, opt_state, params):
        return solver.update(df_dx, opt_state, params)
    update_fn = jax.jit(update_fn, device=constants['device'])

    # Prepare batching
    rng = np.random.default_rng(seed=solver_settings['seed'])
    batch_size = solver_settings['batch_size']
    num_training_configs = len(training_configs)
    shuffled_training_box = rng.permutation(num_training_configs).tolist()

    if solver_settings['save'] is not None:
        save_name = solver_settings['save']
    else:
        save_name = f"training_stats_{int(time.time())}"

    # Start training loop
    assert solver_settings['print_stride'] > 0, "print_stride must be > 0 to save training stats and see progress"
    i = 0
    next_print = 0
    best_save_metric = np.inf

    training_stats = {
        'iter_history': [], 'aux_history_tr': [], 'aux_history_te':[], 'aux_history_te_2':[],
        'case_tr':[], 'case_te':[], 'case_te_2':[], 'save_metric':[]
    } # for saving
    reset_aux_data(solver_settings['save'])
    tic = current_time = time.time()
    while i < solver_settings['max_iter']:
        try:
            batch_indices = []
            for _ in range(batch_size):
                if len(shuffled_training_box) == 0:
                    shuffled_training_box = rng.permutation(num_training_configs).tolist()
                training_sample = shuffled_training_box.pop()
                batch_indices.append(training_sample)
                # There is a very small chance of duplicates within a batch

            df_dx_accum = None
            for config_idx in batch_indices:
                data_train = jax.device_put(training_configs[config_idx]['data'], device=constants['device'])
                graph_train = jax.device_put(training_configs[config_idx]['warped_geo'], device=constants['device'])
                flow_train = jax.device_put(training_configs[config_idx]['flow_condition'], device=constants['device'])
                df_dx = grad_jit(params, data_train, flow_train, graph_train)

                # add different design condition PINN loss
                if compute_pinn_loss:
                    flow_train_pinn = {**flow_train}
                    flow_train_pinn['v_inf'] = jnp.array([197.04, 0.0, 0.0]) # hardcoded as mach 0.65 for now, but can be randomized later
                    df_dx_pinn = grad_jit_pinn(params, data_train, flow_train_pinn, graph_train)
                    df_dx_total = jax.tree_util.tree_map(lambda a, b: a + b, df_dx, df_dx_pinn)
                else:
                    df_dx_total = df_dx

                if df_dx_accum is None:
                    df_dx_accum = df_dx_total
                else:
                    df_dx_accum = jax.tree_util.tree_map(lambda a, b: a + b, df_dx_accum, df_dx_total)

                i += 1 # Count iterations

            # average gradients over batch
            df_dx_accum = jax.tree_util.tree_map(lambda g: g / batch_size, df_dx_accum)
            updates, opt_state = update_fn(df_dx_accum, opt_state, params)
            params = optax.apply_updates(params, updates)

        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting.")
            break
        
        # if solver_settings['print_stride'] and (i % solver_settings['print_stride'] == 0):
        while i >= next_print:
            next_print += solver_settings['print_stride']

            iter_time = time.time() - current_time

            # save training stats
            config_idx = 2
            data_train = jax.device_put(training_configs[config_idx]['data'], device=constants['device'])
            graph_train = jax.device_put(training_configs[config_idx]['warped_geo'], device=constants['device'])
            flow_train = jax.device_put(training_configs[config_idx]['flow_condition'], device=constants['device'])
            training_loss, training_residuals = res_jit(params, data_train, flow_train, graph_train)
            aux_data = training_residuals
            aux_data = {k: float(jax.device_get(v)) for k, v in training_residuals.items()}
            aux_data['loss'] = float(jax.device_get(training_loss))
            training_stats['iter_history'].append(i)
            training_stats['aux_history_tr'].append(aux_data)
            training_stats['case_tr'].append(training_configs[config_idx]['case_id'])

            # save validation stats 1
            VALIDATION_CONFIG_IDX = 2
            test_config = test_configs[VALIDATION_CONFIG_IDX]
            data_test = jax.device_put(test_config['data'], device=constants['device'])
            graph_test = jax.device_put(test_config['warped_geo'], device=constants['device'])
            flow_test = jax.device_put(test_config['flow_condition'], device=constants['device'])
            # flow_test['v_inf'] = jnp.array([197.04, 0.0, 0.0]) # hardcoded as mach 0.65 for now, but can be randomized later
            test_loss, test_residuals = res_jit(params, data_test, flow_test, graph_test)
            aux_data = test_residuals
            aux_data = {k: float(jax.device_get(v)) for k, v in test_residuals.items()}
            aux_data['loss'] = float(jax.device_get(test_loss))
            training_stats['aux_history_te'].append(aux_data)
            training_stats['case_te'].append(test_config['case_id'])

            # save test stats
            VALIDATION_CONFIG_IDX = -2
            test_config = test_configs[VALIDATION_CONFIG_IDX]
            data_test = jax.device_put(test_config['data'], device=constants['device'])
            graph_test = jax.device_put(test_config['warped_geo'], device=constants['device'])
            flow_test = jax.device_put(test_config['flow_condition'], device=constants['device'])
            # flow_test['v_inf'] = jnp.array([197.04, 0.0, 0.0]) # hardcoded as mach 0.65 for now, but can be randomized later
            test_loss, test_residuals = res_jit(params, data_test, flow_test, graph_test)
            aux_data = test_residuals
            aux_data = {k: float(jax.device_get(v)) for k, v in test_residuals.items()}
            aux_data['loss'] = float(jax.device_get(test_loss))
            training_stats['aux_history_te_2'].append(aux_data)
            training_stats['case_te_2'].append(test_config['case_id'])

            # Just save every iteration
            param_info.save(params, solver_settings['save'] + '_TEMP')
            save_metric = compute_latest_metric(
                training_stats
            )
            training_stats['save_metric'].append(save_metric)
            if save_metric < best_save_metric:
                param_info.save(params, solver_settings['save'] + '_TEMP_BEST')
                best_save_metric = save_metric

            save_aux_data(training_stats, dir=f'training_stats', suffix=save_name)
            loss_str = f"Iter {i}:\tTrLoss={training_loss:.3e}, TeLoss={test_loss:.3e} , SaveMetric={save_metric:.3e} ,"
            res_str = f"res_vx={training_residuals['res_vx']:.3e}, res_T={training_residuals['res_T']:.3e}, res_p={training_residuals['res_p']:.3e}, res_nuTilda={training_residuals['res_nuTilda']:.3e}, Iter={iter_time:.2f} seconds ,"
            forces_str = f"L_pred={training_residuals['L_pred']:.2e}, L_data={training_residuals['L_data']:.2e}, D_pred={training_residuals['D_pred']:.2e},D_data={training_residuals['D_data']:.2e}"
            print(loss_str, res_str, forces_str)

            current_time = time.time()

    total_time = time.time() - tic
    print("Total training time: ", total_time)

    save_all_aux_data(training_stats, f'training_stats/{save_name}')
    save_all_aux_data(training_stats, f'BACKUP')
    return params
