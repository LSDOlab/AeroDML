# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import sys
import time
from pathlib import Path
import shutil

# CSDL packages
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

# project/ directory
ROOT = Path(__file__).resolve().parents[1]

# allow imports from project/A
sys.path.insert(0, str(ROOT))
print(ROOT)

# IDWarp and DAFoam
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, DAFoamMeshWarper

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle, rotate_pitch
from bwb_geometry import build_bwb_geometry, build_initial_bwb_geometry, compute_geometry_csdl

from csdl_dafoam_helper_functions import Timer, hash_array_tol, quiet_barrier, compute_vertex_normals, average_normals_at_duplicate_points

# Plotting
from vedo import Points, Arrows, Mesh, show
import matplotlib.pyplot as plt
from csdl_dafoam_utils import is_headless

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------



# ===============================
# region USER INPUT
# ===============================
# Keyword for optimization name (optimization results folder will be saved with this name)

# # Geometry
# geometry_directory        =  os.path.join(os.getcwd(), 'bwb_geometry/')
# stp_file_name             = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
# geometry_pickle_file_name = 'bwb_stored_refit.pickle'

def build_cfd_func(
        mesh_dir:str,
        da_options:dict,
        operating_reference:dict,
        geometry_dir:str='bwb_geometry/',
        stp_file_name:str='bwbv2_no_wingtip_coarse_refined_flat.stp',
        geometry_pickle_file_name:str='bwb_new_stored_refit.pickle',
        x_surf_hash_str:str=None,
        save_primal_time = None,
    ):
    geometry_directory        =  os.path.join(os.getcwd(), geometry_dir)

    # Mesh
    average_normals_at_edges  = False # if true, this will average the normals of the shared point between two surfaces (might be useful for some cases)

    # MPI and timing
    # MPI
    from mpi4py import MPI
    comm           = MPI.COMM_WORLD
    timing_enabled = True  # True if we want timing printed for the CSDL operations

    # Plotting
    show_plots        = False
    interactive_plots = False

    # DAFoam
    # dafoam_directory    = os.path.join(os.getcwd(), 'openfoam_739k_bwb_symmetry/')
    dafoam_directory    = os.path.join(os.getcwd(), f'{mesh_dir}/')
    dafoamPrintInterval = 1 # This doesn't actually seem to affect anything...

    # Initial/reference values for DAFoam (best to use base conditions)
    U0 = operating_reference['U0']
    p0 = operating_reference['p0']
    T0 = operating_reference['T0']
    nuTilda0 = operating_reference['nuTilda0']
    CL_target = operating_reference['CL_target']
    aoa0 = operating_reference['aoa0']
    pitch0 = operating_reference['pitch0']
    A0 = operating_reference['A0']
    rho0 = operating_reference['rho0'] 

    # ===============================
    # region SETUP
    # ===============================
    # MPI information
    rank      = comm.Get_rank()
    comm_size = comm.Get_size()
    rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)

    # region DAFoam instance
    # region Mesh options
    mesh_options = {
        "gridFile": dafoam_directory,
        "fileType": "OpenFOAM",
        "symmetryPlanes": [
            [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        ],
        "useRotations": True,
        # "useRotations": False,
        "LdefFact": 100.0
    }

    dafoam_instance = instantiateDAFoam(da_options, comm, str(dafoam_directory), mesh_options)
    dafoam_instance.printInterval = dafoamPrintInterval
    x_surf_dafoam_initial_local   = dafoam_instance.getSurfaceCoordinates()
    x_vol_dafoam_initial_local    = dafoam_instance.xv0
    local_n_surf  = x_surf_dafoam_initial_local.shape[0]
    local_n_vol   = x_vol_dafoam_initial_local.shape[0]

    # exit()
    print(f'Rank {rank_str} has {local_n_surf} surface points and {local_n_vol} volume points.')

    # Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
    # and geometry evaluation functions)
    (x_surf_dafoam_initial, 
    x_surf_dafoam_initial_size,
    x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_local, comm)

    x_surf_dafoam_initial = comm.bcast(x_surf_dafoam_initial, root=0)

    # Get hash for surface mesh projection file read/write (broadcast to other ranks)
    if rank == 0:
        if x_surf_hash_str is None:
            x_surf_hash = hash_array_tol(x_surf_dafoam_initial)
        else:
            x_surf_hash = x_surf_hash_str
    else:
        x_surf_hash = None

    x_surf_hash = comm.bcast(x_surf_hash, root=0)

    # region File paths
    geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
    stp_file_path                     = Path(geometry_directory)/stp_file_name
    surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'
    # surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_5ef4b5407145897b.pickle'

    if save_primal_time:
        from csdl_dafoam_mesh_functions import read_proc_addressing, assemble_global_points_rank0_average, write_data, assemble_cell_states, build_sample
        surf_local = np.concatenate([dafoam_instance.boundaries[dafoam_instance.basicFamilies[i]]["indicesRed"]
                                    for i in dafoam_instance.families[dafoam_instance.allWallsGroup]]).astype(np.int64)
        surf_gid_local = read_proc_addressing(dafoam_instance, comm, "point")[surf_local]
        surf_gid_full = comm.gather(surf_gid_local, root=0)
        surf_gid_full = np.concatenate(surf_gid_full) if comm.rank == 0 else None
        if comm.rank == 0:
            print(surf_gid_full.shape, x_surf_dafoam_initial.shape)
            np.save(Path(dafoam_directory)/f"surf_gid_full_{x_surf_hash}.npy", surf_gid_full)

    quiet_barrier(comm)

    # region Surface normal computation
    points  = x_surf_dafoam_initial

    # Will get populated if needed
    geo_sim = None
    geometry_config_map = None
    x_surf_dafoam_full_geo = None

    def lazy_geo_sim(jax_sim=False):
        nonlocal geo_sim
        nonlocal geometry_config_map
        nonlocal x_surf_dafoam_full_geo

        if geo_sim is not None:
            return
        normals_local, face_normals_local, face_centers_local = compute_vertex_normals(dafoam_instance, outward_ref=None)

        normals      = gather_array_to_rank0(-normals_local, comm)[0]
        face_normals = gather_array_to_rank0(-face_normals_local, comm)[0]
        face_centers = gather_array_to_rank0(face_centers_local, comm)[0]

        # Edge normal handling
        if comm.rank == 0:
            if average_normals_at_edges:
                normals = average_normals_at_duplicate_points(x_surf_dafoam_initial, normals)

        if comm.rank == 0:
            geo_recorder = csdl.Recorder(inline=True, debug=False)
            geo_recorder.start()

            _, geometry_config_map, x_surf_dafoam_full_geo  = compute_geometry_csdl(
                comm,
                geometry_pickle_file_path,
                stp_file_path,
                surface_mesh_projection_file_path,
                x_surf_dafoam_initial,
                normals,
                geo_recorder,
                timing_enabled=timing_enabled,
            )

            geo_recorder.stop()
            if jax_sim:
                geo_sim = csdl.experimental.JaxSimulator(
                    geo_recorder,
                    additional_inputs = [var for var in geometry_config_map.values()],
                    additional_outputs = [x_surf_dafoam_full_geo],
                    gpu = False
                )
            else:
                geo_sim = csdl.experimental.PySimulator(geo_recorder)
        else:
            geo_sim = None

    def compute_CFD_surface_pts(comm, configuration, jax_sim = False):
        # Only creates the geo_sim once
        lazy_geo_sim(jax_sim)

        if comm.rank == 0:
            # Set inputs and run
            geometry_configuration_dict = {}
            assert len(configuration) == len(geometry_config_map), f"Configuration length {len(configuration)} does not match expected {len(geometry_config_map)}"
            for geo_param_name, geo_param_value in configuration.items():
                geo_variable = geometry_config_map[geo_param_name]
                geometry_configuration_dict[geo_variable] = geo_param_value['value']

            for var, val in geometry_configuration_dict.items():
                geo_sim[var] = val

            print('evaluating surface points...')
            geo_sim.run()
            surf_pts_val = geo_sim[x_surf_dafoam_full_geo]
        else:
            surf_pts_val = None
        surf_pts_val = comm.bcast(surf_pts_val, root=0)

        return surf_pts_val

    # region Surface mesh distribution
    i0, i1          = x_surf_dafoam_initial_indices[rank]

    recorder = csdl.Recorder(inline=False, debug=False)
    recorder.start()
    x_surf_dafoam_full = csdl.Variable(name='x_surf_dafoam_full', shape=x_surf_dafoam_initial.shape)

    flow_conditions_group = csdl.VariableGroup()
    flow_conditions_group.v_inf = csdl.Variable(value=np.array([U0, 0.0, 0.0]), name="v_inf")
    flow_conditions_group.p_inf = csdl.Variable(value=p0, name="p_inf")
    flow_conditions_group.T_inf = csdl.Variable(value=T0, name="T_inf")
    flow_conditions_group.pitch_inf = csdl.Variable(value=0.0, name="pitch_inf")
    # comm.Barrier()

    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

        x_surf_dafoam   = x_surf_dafoam_full[i0:i1, :]
        x_surf_dafoam   = x_surf_dafoam.flatten()
        
        use_idwarp = True
        if use_idwarp:
            # region IDWarp and DAFoam
            idwarp_model    = DAFoamMeshWarper(dafoam_instance)
            x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)
        else:
            # region IDWarp and DAFoam
            from custom_mesh_warper import CustomMeshWarper
            custom_warp_model    = CustomMeshWarper(dafoam_instance)
            x_vol_dafoam    = custom_warp_model.evaluate(x_surf_dafoam)

        # print(f'Rank {rank_str}: {x_vol_dafoam.shape[0]} volume points after warping.')
        print(f'Rank {rank_str}: {x_vol_dafoam.shape[0]/3} volume points after warping.')

        flow_conditions_group.v_inf = mpi_region.split_custom(
            flow_conditions_group.v_inf,
            split_func=lambda x: x,
        )
        flow_conditions_group.p_inf = mpi_region.split_custom(
            flow_conditions_group.p_inf,
            split_func=lambda x: x,
        )
        flow_conditions_group.T_inf = mpi_region.split_custom(
            flow_conditions_group.T_inf,
            split_func=lambda x: x,
        )
        flow_conditions_group.pitch_inf = mpi_region.split_custom(
            flow_conditions_group.pitch_inf,
            split_func=lambda x: x,
        )
        
        # rotate the volume points (and surface pts just in case???)
        x_vol_dafoam = rotate_pitch(x_vol_dafoam, flow_conditions_group.pitch_inf)

        # DAFOAM requires angle of attack
        zero_aoa = csdl.Variable(value=0.0, name="zero_aoa_internal")
        dafoam_input_variables_group = csdl.VariableGroup()
        dafoam_input_variables_group.aero_vol_coords = x_vol_dafoam
        dafoam_input_variables_group.patch_velocity = csdl.concatenate((
            csdl.norm(flow_conditions_group.v_inf),
            zero_aoa,
        ))
        dafoam_input_variables_group.pressure = flow_conditions_group.p_inf
        dafoam_input_variables_group.temperature = flow_conditions_group.T_inf

        # DAFoamSolver Implicit component setup and evaluation
        dafoam_solver           = DAFoamSolver(dafoam_instance, warm_start_time=9999, save_primal_time = save_primal_time)
        # dafoam_solver           = DAFoamSolver(dafoam_instance, save_primal_time = save_primal_time)
        dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

        # DAFoamFunctions Explicit component setup and evaluation
        dafoam_functions = DAFoamFunctions(dafoam_instance)
        dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                            dafoam_input_variables_group)

        mpi_region.set_as_global_output(dafoam_function_outputs.lift)
        mpi_region.set_as_global_output(dafoam_function_outputs.drag)

    recorder.stop()

    # ===============================
    # region SIM SETUP
    # ===============================
    sim = csdl.experimental.PySimulator(recorder)
    from csdl_dafoam_mesh_functions import read_proc_addressing, assemble_global_points_rank0_average, write_data, assemble_cell_states, build_sample
    vert_indices = read_proc_addressing(dafoam_instance, comm, key="point")
    cell_indices = read_proc_addressing(dafoam_instance, comm, key="cell")
    face_indices = read_proc_addressing(dafoam_instance, comm, key="face")


    flow_conditions_map = {
        'v_inf': flow_conditions_group.v_inf,
        'p_inf': flow_conditions_group.p_inf,
        'T_inf': flow_conditions_group.T_inf,
        'pitch': flow_conditions_group.pitch_inf,
    }

    normalize_states = dafoam_instance.getOption("normalizeStates")
    use_rans = "nuTilda" in normalize_states
    field_specs = {
        "U": "vector",
        "p": "scalar",
        "T": "scalar",
    }

    if use_rans:
        field_specs["nuTilda"] = "scalar"

    def run_simulation(name:str, configuration:dict, flow_condition:dict, geo_only:bool=False, dir:str = None, surface_pts:np.ndarray = None, write:bool = True):
        if not isinstance(name, str):
            name = str(name)

        # for key in flow_condition.keys():
        assert len(flow_condition) == len(flow_conditions_map), f"Flow condition length {len(flow_condition)} does not match expected {len(flow_conditions_map)}"
        for flow_variable_name, flow_variable_value in flow_condition.items():
            flow_variable = flow_conditions_map[flow_variable_name]
            sim[flow_variable] = flow_variable_value['value']

        # UNCOMMENT TO RUN SIMULATION
        comm.Barrier()
        start = time.time()
        
        # get mesh surface
        if surface_pts is None:
            surface_pt_val = compute_CFD_surface_pts(comm, configuration)
        else:
            surface_pt_val = surface_pts
        assert surface_pt_val.shape == x_surf_dafoam_initial.shape, f"Surface points shape {surface_pt_val.shape} does not match expected {x_surf_dafoam_initial.shape}"
        comm.Barrier()

        # run CFD
        sim[x_surf_dafoam_full] = surface_pt_val
        sim.run()
        comm.Barrier()

        end = time.time()
        eval_time = end - start
        if rank == 0:
            print(f'Simulation {name} evaluation time: {eval_time:.2f} seconds')

        vertices = sim[x_vol_dafoam].reshape((-1,3))
        assembled_X, missing = assemble_global_points_rank0_average(vertices, vert_indices, comm, root=0)

        # COMMENT
        # assembled_X, missing = assemble_global_points_rank0_average(x_vol_dafoam_initial_local, vert_indices, comm, root=0)

        nC = dafoam_instance.solver.getNLocalCells()
        states = {}
        for field_name, spec in field_specs.items():
            field_type = spec

            if field_type == "vector":
                local = np.zeros(nC * 3)
                dafoam_instance.solver.getOFField(field_name, "vector", local)
                local = local.reshape(nC, 3)

            elif field_type == "scalar":
                local = np.zeros(nC)
                dafoam_instance.solver.getOFField(field_name, "scalar", local)

            else:
                raise ValueError(f"Unsupported field type: {field_type}")

            states[field_name] = assemble_cell_states(local, cell_indices, comm)

        forces = {'drag': sim[dafoam_function_outputs.drag], 'lift': sim[dafoam_function_outputs.lift]}
        flow_condition_dict = {
            k: {"value": sim[v]} for k, v in flow_conditions_map.items()
        }
        flow_condition_dict["nuTilda"] = {"value": nuTilda0}

        dir_name = 'DATA_GEO' if geo_only else 'DATA'
        if dir is not None:
            dir_name = dir
        
        if write:
            write_data(
                dir_name=dir_name,
                file_name = name,
                states = states,
                vertex_coordinates = (assembled_X, missing),
                forces = forces,
                configuration = configuration,
                flow_condition = flow_condition_dict,
                eval_time = eval_time,
                converged=dafoam_solver.last_time_converged,
                comm=dafoam_instance.comm
            )
        else:
            print(f'=======RUN {name} NOT SAVED=======')
            print(forces)
            print(flow_condition_dict)
            print(f'=======RUN {name} NOT SAVED=======')

        comm.Barrier()
        return forces
    # exit()

    return {
        'sim':sim,
        'compute_surface_pts':compute_CFD_surface_pts,
        'model':run_simulation,
        'dafoam_instance': dafoam_instance,
    }

if __name__ == '__main__':
    exit()