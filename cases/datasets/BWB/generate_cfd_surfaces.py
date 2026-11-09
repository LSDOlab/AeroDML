import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from data_config_builder import get_dataset_configurations
import pickle
from pathlib import Path
import numpy as np
import time

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle, rotate_pitch
from bwb_geometry import build_bwb_geometry, build_initial_bwb_geometry, compute_geometry_csdl

# from mpi4py import MPI
# comm = MPI.COMM_WORLD

# other
import csdl_alpha as csdl

# ======================================================================
# ==============================SETTINGS================================

# ===============RANS
data_dir = 'data/sample_tcp1_a'
x_surf_hash = 'a63d3d5d35e5809c' # read and choose hash from mesh_dir

data_dir = 'data/sample_tcp1_b'
x_surf_hash = 'NEWHASH' # read and choose hash from mesh_dir
mesh_dir = 'mesh_639k'
surf_dir_mesh = "surf_pts"

# ===============EULER
data_dir = 'data/sample_tcp1_b'
x_surf_hash = 'NEWHASH_EULER' # read and choose hash from mesh_dir
mesh_dir = 'mesh_112k'
surf_dir_mesh = "surf_pts_euler"

# ===============GEOMETRY
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

    # Create surface_pts directory
    surf_dir = Path(data_dir) / surf_dir_mesh

    # Compute and save surfaces
    dataset_configurations = get_dataset_configurations(data_dir)
    for dataset_input in dataset_configurations:
        geo_config = dataset_input['geometry_configuration']
        case_id = dataset_input["key"]
        
        # compute
        print(f'computing surface points (case {case_id})...')
        start = time.time()
        surface_pts = compute_CFD_surface_pts(geo_config)
        end = time.time()
        print(f'completed in {end - start} sec')

        # save
        file_path = surf_dir / f"{case_id}.npy"
        np.save(file_path, surface_pts)

        # import pyvista as pv
        # plotter = pv.Plotter()
        # cloud = pv.PolyData(np.asarray(surface_pts))
        # cloud.plot(
        #     style="points",
        #     point_size=3,
        #     render_points_as_spheres=True,
        # )
        # plotter.show(interactive_update=True)
        # plotter.close()