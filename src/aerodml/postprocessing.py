import numpy as np
import jax
import jax.numpy as jnp
import os
import csv
import matplotlib.pyplot as plt
from .utils import print_content

def compute_surface_areas_and_normals(points: jax.Array, mesh_topology:dict):
    """Compatibility wrapper around the geometry-builder implementation."""
    from .load_mesh import compute_face_geometry

    surface_face_ids = np.flatnonzero(mesh_topology['faces']['type'] == 1).astype(np.int32)
    normals, areas, _ = compute_face_geometry(mesh_topology, points, surface_face_ids)
    return areas, normals

def wall_shear_force_wall_function(pressure, velocity, temperature, normals, areas,
                                   adjacent_centroids, surface_centroids):
    import jax.numpy as jnp
 
    # --- geometry / kinematics ---
    d_vec = adjacent_centroids - surface_centroids
    y = jnp.abs(jnp.sum(d_vec * normals, axis=1))
    y = jnp.maximum(y, 1e-8)
 
    v_normal = jnp.sum(velocity * normals, axis=1, keepdims=True)
    v_tangent = velocity - v_normal * normals
    # Keep the norm itself differentiable at stagnation. Applying maximum()
    # after jnp.linalg.norm() protects the forward division, but its VJP can
    # already contain 0/0 when v_tangent is zero (or underflows to zero in
    # float32). Putting the epsilon inside sqrt keeps every intermediate in
    # the reverse pass finite.
    speed_eps = jnp.asarray(1e-12, dtype=v_tangent.dtype)
    U = jnp.sqrt(jnp.sum(v_tangent * v_tangent, axis=1) + speed_eps**2)
    t_hat = v_tangent / U[:, None]
 
    # --- thermodynamics ---
    R = 287.0
    T_ref = 273.15
    S = 110.4
    mu_ref = 1.716e-5  # Pa*s
 
    T = jnp.maximum(temperature, 50.0)
    mu_lam = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
 
    rho = pressure / (R * T)
    rho = jnp.maximum(rho, 1e-8)
 
    # IMPORTANT: wall laws use MOLECULAR viscosity here, not mu_lam + mu_t
    nu = mu_lam / rho
    nu = jnp.maximum(nu, 1e-12)
 
    # --- solve wall law for u_tau via u_plus ---
    kappa = 0.41
    B = 5.2
    A = jnp.exp(-kappa * B)
 
    Re_y = U * y / nu
    Re_y = jnp.maximum(Re_y, 1e-12)
 
    def spalding(u_plus):
        ku = kappa * u_plus
        return u_plus + A * (
            jnp.exp(ku) - 1.0 - ku - 0.5 * ku**2 - (1.0/6.0) * ku**3 - (1.0/24.0) * ku**4
        )
 
    def dspalding(u_plus):
        ku = kappa * u_plus
        return 1.0 + A * kappa * (
            jnp.exp(ku) - 1.0 - ku - 0.5 * ku**2 - (1.0/6.0) * ku**3
        )
 
    # decent initial guess
    u_plus = jnp.where(
        Re_y < 100.0,
        jnp.sqrt(Re_y),
        (1.0 / kappa) * jnp.log(jnp.maximum(Re_y, 1.0)) + 2.0
    )
 
    # Newton iterations
    for _ in range(8):
        f = spalding(u_plus) - Re_y / u_plus
        df = dspalding(u_plus) + Re_y / (u_plus**2)
        step = f / jnp.maximum(df, 1e-12)
        u_plus = jnp.maximum(u_plus - step, 1e-6)
 
    u_tau = U / jnp.maximum(u_plus, 1e-6)
 
    # wall shear magnitude
    tau_w = rho * u_tau**2
 
    # viscous force on body opposes tangential flow
    Fv = -tau_w[:, None] * areas[:, None] * t_hat
 
    # zero out near-stagnation noise
    Fv = jnp.where((U > 1e-10)[:, None], Fv, 0.0)
 
    return Fv, tau_w, u_tau
 
def compute_forces(pressure, velocity, nu_tilda, temperature, warped_points, topology):
    areas = warped_points['foam_mesh']['faces']['surface_areas']
    normals = warped_points['foam_mesh']['faces']['surface_normals']
 
    # Compute pressure forces: Fp = p * A * n
    scalar_idx = topology['foam_mesh']['faces']['surface_neighbors']  # precomputed mapping from surface face to neighboring cell
    pressure = pressure[scalar_idx]  # (Fsurf,)
    Fp = pressure[:, None] * areas[:, None] * normals  # (Fsurf, 3)
    assert Fp.shape == (areas.shape[0], 3), f"Expected Fp shape {(areas.shape[0], 3)}, got {Fp.shape}"
 
    # Compute APPROXIMATE viscous forces:
    # Formulation helped by AI
    velocity = velocity[scalar_idx]  # (Fsurf, 3)
    nu_tilda = nu_tilda[scalar_idx]  # (Fsurf,)
    temperature = temperature[scalar_idx]  # (Fsurf,)
    adjacent_centroids = warped_points['dual_graph']['dual_cells'][scalar_idx]  # (Fsurf, 3)
    surface_centroids = warped_points['foam_mesh']['faces']['surface_centroids']  # (Fsurf, 3)
    normals = normals # (Fsurf, 3), pointing outward
    areas = areas # (Fsurf,)
    assert velocity.shape[0] == areas.shape[0] == nu_tilda.shape[0] == adjacent_centroids.shape[0] == surface_centroids.shape[0] == normals.shape[0], "Mismatched number of surface faces between different arrays"
    
    # Use wall function:
    Fv, tau_w, u_tau = wall_shear_force_wall_function(
        pressure=pressure,
        velocity=velocity,
        temperature=temperature,
        normals=normals,
        areas=areas,
        adjacent_centroids=adjacent_centroids,
        surface_centroids=surface_centroids,
    )
 
    F = Fp + Fv
    return {
        "forces": F,
        "surface_centroids": surface_centroids,
        # extra:
        "pressure_forces": Fp,
        "viscous_forces": Fv,
        "surface_areas": areas,
        "surface_normals": normals,
    }

def integrate_forces(surface_forces):
    # Integrate forces to get lift and drag
    L = -surface_forces["forces"][:, 2].sum()  # Lift is the z-component of force
    D = -surface_forces["forces"][:, 0].sum()  # Drag is the x-component of force
    D += 300
    return L, D

def integrate_moments(surface_loads, axis="y"):
    axis_indices = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_indices:
        raise ValueError(f"axis must be one of {tuple(axis_indices)}, got {axis!r}")

    surface_forces = -surface_loads["forces"]
    surface_centroids = surface_loads["surface_centroids"]

    if surface_forces.shape != surface_centroids.shape:
        raise ValueError(f"Forces and centroids must have matching shapes, got {surface_forces.shape} and {surface_centroids.shape}")

    surface_moments = jnp.cross(surface_centroids, surface_forces)
    return surface_moments[:, axis_indices[axis]].sum()


if __name__ == "__main__":
    # data = load_cfd(analyze=True)
    from run import build_geometry_data, load_cfd_data, get_mesh
    import jax
    device = jax.devices()[0]
    mesh_load_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    dual_load_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    coarse_load_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    constants = {
        'T': {'mean': np.float64(233), 'std':np.float64(8.97)},
        'nuTilda': {'mean': np.float64(0.00507), 'std': np.float64(0.0137)}, 
        'p': {'mean': np.float64(3.09e+04), 'std': np.float64(2.87e+03)},
        'v': {'mean': np.array([203, 14.6, -5.56]), 'std': np.array([41.7, 29.3, 36.6])},
        'device': device,
    }

    # ============= Main Loop ==============:
    geometry_topology = get_mesh('mesh_data', mesh_load_settings=mesh_load_settings, dual_load_settings=dual_load_settings, coarse_load_settings=coarse_load_settings)
    cfd_data = load_cfd_data()
    train_data, test_data = build_geometry_data(geometry_topology, cfd_data, verbose=0, visualize=False)
