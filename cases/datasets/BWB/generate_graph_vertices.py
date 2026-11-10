import jax
jax.config.update("jax_default_matmul_precision", "highest")

import os
# os.environ.setdefault("JAX_PLATFORMS", "cpu")
from data_config_builder import get_dataset_configurations
import pickle
from pathlib import Path
import numpy as np
import time
import jax

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle, rotate_pitch
from bwb_geometry import build_bwb_geometry, build_initial_bwb_geometry, compute_geometry_csdl

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
from idwarp_jax.vol_warper import build_volume_pts_func
from aerodml.load_mesh import load_foam_mesh, load_dual_graph

# other
import csdl_alpha as csdl

# ======================================================================
# ==============================SETTINGS================================
deform_full_volume = True

data_dir = 'data/sample_tcp1_a'
# data_dir = 'data/demo_tcp1_a'
mesh_dir = 'mesh_639k'
x_surf_hash = 'a63d3d5d35e5809c' # read and choose hash from mesh_dir
cached_routes_path = None

data_dir = 'data/sample_tcp1_b'
x_surf_hash = 'NEWHASH' # read and choose hash from mesh_dir
cached_routes_path = str(Path(__file__).resolve().parents[2] / 'aircraft_BWB' / 'model' / 'mesh_639k' / 'cached_routes_NEWHASH')

geometry_dir ='bwb_geometry/'
stp_file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_new_stored_refit.pickle'

# ======================================================================
# ======================================================================

geometry_directory        =  os.path.join(os.getcwd(), geometry_dir)
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name
dafoam_directory    = os.path.join(os.getcwd(), f'{mesh_dir}/')
surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'

surf_indices = np.load(f"{dafoam_directory}/surf_gid_full_{x_surf_hash}.npy")
unique_gids, surf_indices_list = np.unique(surf_indices, return_index=True)
surf_25r_to_1r_map = surf_indices_list.tolist()

if __name__ == '__main__':

    geo_recorder = csdl.Recorder(inline=True, debug=False)
    geo_recorder.start()

    _, geometry_config_map, x_surf_dafoam_full_geo  = compute_geometry_csdl(
        None,
        geometry_pickle_file_path,
        stp_file_path,
        surface_mesh_projection_file_path,
        None,
        None,
        geo_recorder,
        timing_enabled=True,
    )

    geo_recorder.stop()
    geo_sim = csdl.experimental.JaxSimulator(
        geo_recorder,
        additional_inputs = [var for var in geometry_config_map.values()],
        additional_outputs = [x_surf_dafoam_full_geo],
        gpu = False
    )

    def compute_CFD_surface_pts(configuration):
        # Set inputs and run
        geometry_configuration_dict = {}
        assert len(configuration) == len(geometry_config_map), f"Configuration length {len(configuration)} does not match expected {len(geometry_config_map)}"
        for geo_param_name, geo_param_value in configuration.items():
            geo_variable = geometry_config_map[geo_param_name]
            geometry_configuration_dict[geo_variable] = geo_param_value['value']

        for var, val in geometry_configuration_dict.items():
            geo_sim[var] = val

        # geo_sim[geometry_config_map['temp']] = 1.0
        geo_sim.run()
        surf_pts_val = geo_sim[x_surf_dafoam_full_geo]
        return surf_pts_val


    # base_volume_points = np.array(topology["dual_graph"]["dual_cells"], copy=True)
    foam_mesh = load_foam_mesh(dafoam_directory, {'run_checks':True, 'verbose':False, 'visualize':False})
    dual_graph = load_dual_graph(dafoam_directory, foam_mesh, {'run_checks':True, 'verbose':False, 'visualize':False})
    
    foam_mesh = foam_mesh
    topology = {'foam_mesh': foam_mesh, 'dual_graph': dual_graph}

    points = np.asarray(foam_mesh["points"])
    if deform_full_volume:
        # Warp and save every OpenFOAM mesh point.
        base_volume_points = np.array(points, copy=True)
    else:
        # Warp only the points needed to construct the GNN geometry: all dual
        # cells followed by the surface points in wall_pt_ids order.
        wall_ids = foam_mesh['wall_pt_ids']
        wall_points = points[wall_ids]
        base_volume_points = np.concatenate(
            [topology["dual_graph"]["dual_cells"], wall_points], axis=0,
        )

    deformation_mode = "full volume" if deform_full_volume else "partial volume"
    print(f"Generating {deformation_mode} vertices ({len(base_volume_points)} points per case)")

    jax.config.update("jax_enable_x64", False)
    device = jax.devices("gpu")[0]
    compute_volume_pts, _, _ = build_volume_pts_func(
        foam_mesh, base_volume_points, device, return_aux = True, print_timings = 1,
        cached_routes = cached_routes_path,
        progress=False,
    )

    # Create vertices directory
    vertex_dir = Path(data_dir) / "vertices"

    # Compute and save surfaces
    dataset_configurations = get_dataset_configurations(data_dir)
    for dataset_input in dataset_configurations:
        geo_config = dataset_input['geometry_configuration']
        flow_condition = {key: val['value'] for key,val in  dataset_input['flow_conditions'].items()}
        case_id = dataset_input["key"]
        
        # compute
        print(f'computing surface points (case {case_id})...')
        start = time.time()
        surface_pts = compute_CFD_surface_pts(geo_config)
        # surface_pts = np.load(f"{dafoam_directory}/surf_dafoam_initial_47466_25.npy")
        end = time.time()
        print(f'completed in {end - start} sec')

        # compute volume points
        jax.config.update("jax_enable_x64", False)
        new_surf_pts = surface_pts[surf_25r_to_1r_map]
        new_vol_pts = compute_volume_pts(new_surf_pts, flow_condition)
        jax.config.update("jax_enable_x64", True)
        new_vol_pts = np.asarray(new_vol_pts)

        # save
        file_path = vertex_dir / f"{case_id}.npy"
        np.save(file_path, new_vol_pts)
