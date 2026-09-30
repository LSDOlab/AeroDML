import numpy as np
import jax
import jax.numpy as jnp
import os
import csv
import matplotlib.pyplot as plt

def compute_surface_areas_and_normals(points: jax.Array, mesh_topology:dict):
    faces = mesh_topology["faces"]
    face_type = faces["type"]
    face_points = faces["points"]                    # ragged list[list[int]]
    owner = faces["owner"]

    # ===================== PREPARATION =====================
    # Global face ids of surface faces, in mesh order
    surface_face_ids = np.flatnonzero(face_type == 1).astype(np.int32)

    # Ragged list of vertex ids for just the surface faces
    surf_face_points = [face_points[fid] for fid in surface_face_ids]

    num_surface_faces = len(surf_face_points)
    counts = np.asarray([len(fp) for fp in surf_face_points], dtype=np.int32) # number of vertices per surface face

    # Flattened vertex ids for p_i
    flat_idx = np.concatenate(
        [np.asarray(fp, dtype=np.int32) for fp in surf_face_points], axis=0
    )

    # Flattened vertex ids for p_{i+1} offset by one for cross product later with flat_idx
    next_idx = np.concatenate(
        [np.roll(np.asarray(fp, dtype=np.int32), -1) for fp in surf_face_points], axis=0
    )

    # We need to keep track of which face each vertex belongs to for the cross product later, so we create an array mapping each vertex -> face id
    face_id = np.concatenate(
        [np.full(len(fp), i, dtype=np.int32) for i, fp in enumerate(surf_face_points)],
        axis=0,
    )

    # ===================== REAL CALCUALTION =====================
    num_surface_faces = counts.shape[0]

    # p_i and p_{i+1} for every polygon edge on every surface face
    pi = points[flat_idx]      # (K, 3)
    pj = points[next_idx]      # (K, 3)

    # Sum of all cross-products per face in order leads to the area vector, whose magnitude is the area and direction is the normal
    # 0.5 * sum_i cross(p_i, p_{i+1}) = oriented area vector
    area_vec = 0.5 * jnp.zeros((num_surface_faces, 3)).at[face_id].add(jnp.cross(pi, pj))  # (Fsurf, 3)

    # Face areas
    areas = jnp.linalg.norm(area_vec, axis=1)  # (Fsurf,)

    # Unit normals from area vectors
    eps = 1e-8
    normals = -area_vec / jnp.maximum(areas[:, None], eps)
    assert areas.shape == (num_surface_faces,)
    assert normals.shape == (num_surface_faces, 3)
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
    U = jnp.linalg.norm(v_tangent, axis=1)
    t_hat = v_tangent / jnp.maximum(U[:, None], 1e-12)
 
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
    areas, normals = compute_surface_areas_and_normals(warped_points['foam_mesh']['points'], topology['foam_mesh'])
 
    # Compute pressure forces: Fp = p * A * n
    scalar_idx = topology['foam_mesh']['faces']['surface_neighbors']  # precomputed mapping from surface face to neighboring cell
    pressure = pressure[scalar_idx]  # (Fsurf,)
    Fp = pressure[:, None] * areas[:, None] * normals  # (Fsurf, 3)
    assert Fp.shape == (areas.shape[0], 3), f"Expected Fp shape {(areas.shape[0], 3)}, got {Fp.shape}"
 
    # Compute approximate viscous forces: *This formulation has not been validated for production use.
    velocity = velocity[scalar_idx]  # (Fsurf, 3)
    nu_tilda = nu_tilda[scalar_idx]  # (Fsurf,)
    temperature = temperature[scalar_idx]  # (Fsurf,)
    adjacent_centroids = warped_points['dual_graph']['dual_cells'][scalar_idx]  # (Fsurf, 3)
    surface_centroids = warped_points['dual_graph']['dual_cells'][topology['dual_graph']['dual_cell_types'] == 1]  # (Fsurf, 3)
    normals = normals # (Fsurf, 3), pointing outward
    areas = areas # (Fsurf,)
    assert velocity.shape[0] == areas.shape[0] == nu_tilda.shape[0] == adjacent_centroids.shape[0] == surface_centroids.shape[0] == normals.shape[0], "Mismatched number of surface faces between different arrays"
    
    # Wall function:
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
    return F


def integrate_forces(surface_forces):
    # Integrate forces to get lift and drag
    L = -surface_forces[:, 2].sum() 
    D = -surface_forces[:, 0].sum() 
    D += 300
    return L, D

def predict_forces(model_predict, params, graph_points, geometry):
    v, T, p, nuTilda = model_predict(params, graph_points, geometry)
    L, D = compute_forces(p, v, nuTilda, geometry)
    return L, D

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
