# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import sys
import time
from pathlib import Path
import shutil

# MPI
from mpi4py import MPI

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
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle
from bwb_geometry import build_bwb_geometry

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
        geometry_pickle_file_name:str='bwb_stored_refit.pickle',
        save_primal_time = None,
    ):
    geometry_directory        =  os.path.join(os.getcwd(), geometry_dir)

    # Mesh
    average_normals_at_edges  = False # if true, this will average the normals of the shared point between two surfaces (might be useful for some cases)

    # MPI and timing
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

    # debug_case_state("BEFORE instantiateDAFoam")

    dafoam_instance = instantiateDAFoam(da_options, comm, str(dafoam_directory), mesh_options)
    dafoam_instance.printInterval = dafoamPrintInterval
    x_surf_dafoam_initial_local   = dafoam_instance.getSurfaceCoordinates()
    x_vol_dafoam_initial_local    = dafoam_instance.xv0
    local_n_surf  = x_surf_dafoam_initial_local.shape[0]
    local_n_vol   = x_vol_dafoam_initial_local.shape[0]

    # debug_case_state("AFTER instantiateDAFoam")

    # print("PYTHON CWD =", Path.cwd().resolve(), flush=True)
    # print("DAFOAM CASE DIR =", Path(dafoam_directory).resolve(), flush=True)
    # print("CASE EXISTS =", Path(dafoam_directory).exists(), flush=True)
    # print("PROCESSORS VISIBLE =", sorted([p.name for p in Path(dafoam_directory).glob("processor*")])[:10], flush=True)
    # case_dir = Path(dafoam_directory).resolve()

    # exit()
    print(f'Rank {rank_str} has {local_n_surf} surface points and {local_n_vol} volume points.')
    # comm.Barrier()
    # exit()

    # Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
    # and geometry evaluation functions)
    (x_surf_dafoam_initial, 
    x_surf_dafoam_initial_size,
    x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_local, comm)

    # comm.Barrier()
    # if comm.rank == 0:
    #     print('x_surf_dafoam_initial (OF mesh): ', x_surf_dafoam_initial.shape)
    # comm.Barrier()
    # print(f'x_surf_dafoam_initial_local ({rank}):', x_surf_dafoam_initial_local.shape)
    # exit()

    # Get hash for surface mesh projection file read/write (broadcast to other ranks)
    if rank == 0:
        x_surf_hash = hash_array_tol(x_surf_dafoam_initial)
    else:
        x_surf_hash = None

    x_surf_hash = comm.bcast(x_surf_hash, root=0)

    # region File paths
    geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
    stp_file_path                     = Path(geometry_directory)/stp_file_name
    surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'
    # surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_5ef4b5407145897b.pickle'

    # ===============================
    # region CSDL RECORDER
    # ===============================
    # recorder 
    recorder = csdl.Recorder(inline=True, debug=False)
    recorder.start()
    geometry, geometry_config_map, other = build_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled=timing_enabled)

    # region Surface normal computation
    points  = x_surf_dafoam_initial
    normals_local, face_normals_local, face_centers_local = compute_vertex_normals(dafoam_instance, outward_ref=None)

    normals      = gather_array_to_rank0(-normals_local, comm)[0]
    face_normals = gather_array_to_rank0(-face_normals_local, comm)[0]
    face_centers = gather_array_to_rank0(face_centers_local, comm)[0]

    # Edge normal handling
    if rank == 0:
        if average_normals_at_edges:
            normals = average_normals_at_duplicate_points(x_surf_dafoam_initial, normals)

    # Plot the initial points and normals over the geometry for reference
    if rank == 0 and show_plots and not is_headless():
        geo_plot  = geometry.plot(show=False)
        scatter   = Points(points, r=2, c='green')
        arrows    = Arrows(points, points + 0.2*normals, c='red', s=0.5)

        # Plot duplicate points
        uniq, idx, counts  = np.unique(x_surf_dafoam_initial, axis=0, return_index=True, return_counts=True)
        duplicate_points   = uniq[counts > 1]
        scatter_duplicates = Points(duplicate_points, r=2, c='yellow')

        scatter_face   = Points(face_centers, r=2, c='blue')
        arrows_face    = Arrows(face_centers, face_centers + 0.2*face_normals, c='orange', s=0.5)
        show(geo_plot, scatter, arrows, scatter_duplicates, scatter_face, arrows_face, axes=1, interactive=interactive_plots)

    quiet_barrier(comm)

    # region Surface mesh projection
    # Now do we do the same check for the surface mesh projection
    if surface_mesh_projection_file_path.is_file():
        if rank == 0:
            print('Found surface mesh projection pickle!')
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

    else:
        if rank == 0:
            print('No projected surface mesh file found.')
            try:
                # # ORIGINAL CODE
                with Timer('projecting on surface mesh', rank, timing_enabled):

                    projected_surf_mesh_dafoam = geometry.project(
                        x_surf_dafoam_initial, 
                        grid_search_density_parameter = 1,      # 1 
                        projection_tolerance          = 1e-3,   # 1.e-3m 
                        grid_search_density_cutoff    = 12,    # 150
                        force_reprojection            = False,
                        plot                          = show_plots and not is_headless(),
                        interactive                   = interactive_plots,                     
                        direction                     = normals,
                        num_workers                   = comm_size
                    )

                print('Writing surface mesh projection pickle...')
                write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
                print('Done!')

            # Added this exception because I was getting an ungraceful MPI termination
            except Exception as e:
                import traceback
                print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
                comm.Abort(1) # Abort MPI processes instead of letting them hang

        quiet_barrier(comm)

        if rank != 0:
            projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

    print(f'Rank {rank_str} done reading projected surface mesh!')
    quiet_barrier(comm)

    recorder.inline = False
    with Timer(f'evaluating geometry component', rank, timing_enabled):
        x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

    # region Surface mesh distribution
    i0, i1          = x_surf_dafoam_initial_indices[rank]

    flow_conditions_group = csdl.VariableGroup()
    flow_conditions_group.v_inf = csdl.Variable(value=np.array([U0, 0.0, 0.0]), name="v_inf")
    flow_conditions_group.p_inf = csdl.Variable(value=p0, name="p_inf")
    flow_conditions_group.T_inf = csdl.Variable(value=T0, name="T_inf")
    # comm.Barrier()

    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

        x_surf_dafoam   = x_surf_dafoam_full[i0:i1, :]
        x_surf_dafoam   = x_surf_dafoam.flatten()
        
        # region IDWarp and DAFoam
        idwarp_model    = DAFoamMeshWarper(dafoam_instance)
        x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)

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

    def run_simulation(name:str, configuration:dict, flow_condition:dict, geo_only:bool=False, dir:str = None):
        if not isinstance(name, str):
            name = str(name)
        
        # Set inputs and run
        assert len(configuration) == len(geometry_config_map), f"Configuration length {len(configuration)} does not match expected {len(geometry_config_map)}"
        for geo_param_name, geo_param_value in configuration.items():
            geo_variable = geometry_config_map[geo_param_name]
            sim[geo_variable] = geo_param_value['value']

        # for key in flow_condition.keys():
        assert len(flow_condition) == len(flow_conditions_map), f"Flow condition length {len(flow_condition)} does not match expected {len(flow_conditions_map)}"
        for flow_variable_name, flow_variable_value in flow_condition.items():
            flow_variable = flow_conditions_map[flow_variable_name]
            sim[flow_variable] = flow_variable_value['value']

        # UNCOMMENT TO RUN SIMULATION
        comm.Barrier()
        start = time.time()
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
        comm.Barrier()

    return {
        'geometry_config_map': geometry_config_map,
        'sim':sim,
        'model':run_simulation,
        'dafoam_instance': dafoam_instance,
    }

if __name__ == '__main__':
    exit()