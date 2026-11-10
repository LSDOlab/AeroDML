# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import time
from pathlib import Path
import shutil
import sys
import pickle

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
# from csdl_dafoam_helper_functions import Timer
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle, rotate_pitch
from bwb_geometry import build_bwb_geometry, build_initial_bwb_geometry

# VortexAD analysis
from VortexAD import PanelMethod
from VortexAD import find_cell_adjacency, TE_detection
from VortexAD import plot_pressure_distribution
import meshio

def build_panel_func(
        operating_reference:dict,
        geometry_dir:str='bwb_geometry/',
        stp_file_name:str='bwbv2_no_wingtip_coarse_refined_flat.stp',
        geometry_pickle_file_name:str='bwb_stored_refit.pickle',
        panel_surface_mesh_projection_file_name:str='panel_projections_new.pickle'
    ):
    comm = None
    rank = 0
    shutdown_inline = True

    timing_enabled = True  # True if we want timing printed for the CSDL operations

    geometry_directory        =  os.path.join(os.getcwd(), geometry_dir)
    geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
    stp_file_path                     = Path(geometry_directory)/stp_file_name

    # LOAD PANEL:
    file_path = 'panel/' 
    grid_type = 'mixed'
    if grid_type == 'mixed':
        file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh'
    elif grid_type == 'tri':
        file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch.stl'

    mesh = meshio.read(file_path + file_name)

    points_orig = mesh.points
    cells = mesh.cells
    cells_dict = mesh.cells_dict

    cell_adjacency_data = find_cell_adjacency(points=points_orig, cells=cells_dict)

    points_orig = cell_adjacency_data[0] 
    cells_dict = cell_adjacency_data[1] 
    cell_adjacency = cell_adjacency_data[2] 
    edges2cells = cell_adjacency_data[3]
    points2cells = cell_adjacency_data[4]

    TE_properties = TE_detection(
        points=points_orig,
        cells=cells_dict,
        edges2cells=edges2cells,
        points2cells=points2cells,
        threshold_theta=125.
    )

    cell_types = cells_dict.keys()
    combined_cells = []
    for cell_type in cell_types:
        combined_cells += cells_dict[cell_type].tolist()

    # ====================================== PANEL RECORDER START ============================================
    recorder = csdl.Recorder(inline=True, debug=False)
    recorder.start()

    # Initial geometry:
    initial_refit_outputs, _ = build_initial_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled=timing_enabled)
    geometry_project = initial_refit_outputs[1]

    #     if surface_mesh_projection_file_path.is_file():
    panel_surface_mesh_projection_file_path = Path(f'panel/{panel_surface_mesh_projection_file_name}')
    if panel_surface_mesh_projection_file_path.is_file():
        print('Found surface mesh projection pickle!')
        projected_panel_mesh = read_simple_pickle(panel_surface_mesh_projection_file_path)
    else:
        projected_panel_mesh = geometry_project.project(
            points_orig,
            grid_search_density_parameter=1,
            newton_tolerance=1.e-10,
            grid_search_density_cutoff=30,
            projection_tolerance=1.e-2,
            force_reprojection=False,
            plot=False,
        )
        print('Writing surface mesh projection pickle...')
        write_simple_pickle(projected_panel_mesh, panel_surface_mesh_projection_file_path)
        print('Done!')
        exit('rerun')

    # Parameterize
    geometry, geometry_config_map, other = build_bwb_geometry(comm, geometry_project, geometry_pickle_file_path, timing_enabled=timing_enabled)

    # Finally start panel method stuff
    recorder.inline = not shutdown_inline
    panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
    panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')
    panel_mesh.add_name('panel_mesh')
    # panel_center = geometry.evaluate(projected_panel_centers, plot=False)

    # SET UP VELOCITY, DENSITY, AND PITCH ARRAYS
    U0 = operating_reference['U0']
    p0 = operating_reference['p0']
    T0 = operating_reference['T0']
    A0        = 527           # Projected area of entire BWB. Used for normalizing CD and CL
    sos = np.sqrt(1.4 * 287.0 * T0)
    # print(sos)

    print("All operating conditions:")
    print(f"U0: {U0}, p0: {p0}, T0: {T0}, A0: {A0}, sos: {sos}")
    # exit()
    # exit()
    sos_array = csdl.Variable(value=sos)
    # sos_array = sos_var.expand(panel_mesh.shape, 'i->aji')

    #################################### ONLY X IS NEGATIVE, BE CAREFUL WHEN IMPLMENTING AOA #################################
    p_inf = csdl.Variable(value=p0)
    t_inf = csdl.Variable(value=T0)
    V_inf_dv = csdl.Variable(value = np.array([U0, 0.0, 0.0]))
    pitch_dv = csdl.Variable(value = np.array([-6.0])) # DEGREES

    V_inf = -V_inf_dv
    rho_inf = p_inf / (t_inf * 287.0)

    # For off-body
    rho_array = rho_inf
    V_inf_array = csdl.norm(V_inf_dv)

    # point_velocities = csdl.Variable(value=0.0, shape=panel_mesh.shape)
    # point_velocities = point_velocities.set()
    point_velocities = csdl.expand(V_inf, panel_mesh.shape, 'i->aji')

    pm_solver_inputs = {
        'V_inf': point_velocities,
        'rho': rho_array,
        'sos': sos_array,
        # 'compressibility': True,
        'compressibility': False,
        'Cp cutoff': -5.,
        'partition_size': 1,
        'reuse_AIC': True,
        # 'ref_area': planform_area, # does not matter bc we don't use the coefficients,
        # 'ref_chord': MAC,
    }
    panel_method = PanelMethod(
        solver_input_dict=pm_solver_inputs,
        skip_geometry=True # not running geometry
    )
    # inserting grid data from above
    pmesh_single = panel_mesh[0,:]

    pmesh_single = rotate_pitch(pmesh_single.flatten(), pitch_dv, use_csdl=True).reshape(-1,3)
    panel_method.insert_grid_data(
        mesh=pmesh_single,
        cell_adjacency_data=cell_adjacency_data,
        TE_properties=TE_properties
    )

    panel_method.declare_outputs([
        'Cp',
        'L',
        'Di',
        'M',
        'panel_forces',
        'Di_Trefftz',
        'Vx', 
        'Vy',
        'Vz',
        'V_mag',
    ])
    output_dict = panel_method.evaluate()

    mesh_velocity_x = output_dict['Vx']
    mesh_velocity_y = output_dict['Vy']
    mesh_velocity_z = output_dict['Vz']
    mesh_velocity_magnitudes = output_dict['V_mag']
    Cp = output_dict['Cp']

    # print(outputs['L'].value)
    # print(rho_inf.value)

    # # print min, max, avg of panel mesh[0,:]:
    # print(f"min: {np.min(panel_mesh[0,:].value)}, max: {np.max(panel_mesh[0,:].value)}, avg: {np.mean(panel_mesh[0,:].value)}")

    # [992222.62631303]
    # [656416.33610034]
    # U0: 227.35923760648035, p0: 30089.6, T0: 228.714, A0: 527, sos: 303.1456501419738

    outputs = [mesh_velocity_x, mesh_velocity_y, mesh_velocity_z, mesh_velocity_magnitudes, output_dict['L'], output_dict['Di'], output_dict['M'], output_dict['panel_forces'], output_dict['Di_Trefftz']]
    outputs = outputs + [V_inf_dv, p_inf, t_inf, pitch_dv, rho_inf, Cp, panel_mesh]


    inputs = [geo_param for geo_param in geometry_config_map.values()]
    flow_conditions_map = {
        'v_inf': V_inf_dv,
        'p_inf': p_inf,
        'T_inf': t_inf,
        'pitch': pitch_dv
    }
    inputs = inputs + [flow_conditions_map[key] for key in flow_conditions_map.keys()]

    # outputs = csdl.average(total_velocity)
    jax_sim = csdl.experimental.JaxSimulator(
        recorder,
        gpu=True,
        save_on_update=True,
        output_saved=True,
        additional_outputs=outputs,
        additional_inputs=inputs
    )

    # start = time.time()
    # # jax_sim.compute_totals()
    # jax_sim.run()
    # end = time.time()
    # print(end - start)
    # start = time.time()
    # # jax_sim.compute_totals()
    # jax_sim.run()
    # end = time.time()
    # print(end - start)
    # exit()

    # def run_simulation(name:str, configuration:dict,sample_points:dict['str', np.ndarray], flow_condition:dict, dir:str):
    def run_simulation(name:str, configuration:dict, flow_condition:dict, sample_points:np.ndarray = None, geo_only:bool=False, dir:str = None):
        if not isinstance(name, str):
            name = str(name)
        
        if sample_points is not None:
            raise NotImplementedError('Sample points are not supported in this simulation.')

        # =====================SET INPUTS========================
        # Set inputs and run
        assert len(configuration) == len(geometry_config_map), f"Configuration length {len(configuration)} does not match expected {len(geometry_config_map)}: {configuration.keys()} vs {geometry_config_map.keys()}"
        for geo_param_name, geo_param_value in configuration.items():
            geo_variable = geometry_config_map[geo_param_name]
            jax_sim[geo_variable] = geo_param_value['value']

        # for key in flow_condition.keys():
        assert len(flow_condition) == len(flow_conditions_map), f"Flow condition length {len(flow_condition)} does not match expected {len(flow_conditions_map)}"
        for flow_variable_name, flow_variable_value in flow_condition.items():
            flow_variable = flow_conditions_map[flow_variable_name]
            jax_sim[flow_variable] = flow_variable_value['value']
            print(f'Setting flow condition {flow_variable_name} to {flow_variable_value["value"]}')

        # ===================== RUN ========================
        start = time.time()
        jax_sim.run()
        end = time.time()

        L, Di = jax_sim[output_dict['L']][0], jax_sim[output_dict['Di']][0]
        M = jax_sim[output_dict['M']][0]
        v_inf = np.linalg.norm(flow_condition['v_inf']['value'])
        Cl = L/(0.5*rho_inf.value*A0*v_inf**2).item()
        print(f'   Finished case {name} ({(end-start):2f} sec): L={L:3f} ({Cl:3f}), Di={Di:3f}')

        rhoinf = jax_sim[rho_inf][0]
        Vinf = jax_sim[V_inf_dv][0]

        # surface pressure and velocity
        Cp_val = jax_sim[Cp][0]
        panel_mesh_val = jax_sim[panel_mesh][0]
        pressure_phys = 1/2*jax_sim[rho_inf]*jax_sim[V_inf_dv][0]**2*Cp_val+p0
        vx_phys = jax_sim[mesh_velocity_x][0]
        vy_phys = jax_sim[mesh_velocity_y][0]
        vz_phys = jax_sim[mesh_velocity_z][0]
        vx_phys = np.clip(vx_phys, 0, 350)
        vz_phys = np.clip(vz_phys, -300, 300)
        pressure_phys = np.clip(pressure_phys, 100, 50000)
        velocity_phys = np.stack((vx_phys, vy_phys, vz_phys), axis=-1)

        # evaluated volume velocity and pressure
        # sampled_velocity = jax_sim[total_velocity][0]
        # assert sampled_velocity.shape == (N_EVAL, 3), f'Evaluated velocity shape {sampled_velocity.shape} does not match expected shape {(N_EVAL, 3)}'
        # sampled_velocity = np.clip(sampled_velocity, [0, -300, -300], [350, 300, 300])
        # sampled_velocity_magnitudes = np.linalg.norm(sampled_velocity, axis=1)
        # sampled_pressure = p0 + 0.5 * rhoinf * (Vinf**2 - sampled_velocity_magnitudes**2)
        # sampled_pressure = np.clip(sampled_pressure, 100, 50000)
        # assert sampled_pressure.shape == (N_EVAL,), f'Sampled pressure shape {sampled_pressure.shape} does not match expected shape {(N_EVAL,)}'
        if 0:
            from VortexAD.utils.plot_unstructured import plot_pressure_distribution

            # uncomment to plot pressure
            vals = pressure_phys
            bounds = [1e4,4e4]

            # uncomment to plot velocity magnitude
            # vals = velocity_magnitude
            # vals = vx_phys
            # vals = vy_phys
            # vals = vz_phys
            # bounds = None
            
            plotter = plot_pressure_distribution(panel_mesh_val, vals.reshape((1,) + vals.shape), triangles, interactive=True, bounds = bounds, cmap='viridis')

            # plot points in space colored by velocity magnitude
            points = jax_sim[eval_points][0]
            velocities = jax_sim[total_velocity][0]
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            v_inf_magnitude = np.linalg.norm(jax_sim[V_inf_dv][0])
            pressures = 1/2*jax_sim[rho_inf]*jax_sim[V_inf_dv][0]**2*Cp_val+p0
            print(f'pressure range at eval points: {pressures.min()} to {pressures.max()}')

            # bounds = None
            plotter.add_points(points, scalars=sampled_pressure, cmap='viridis', point_size=15, clim=bounds)

            plotter.show()

        # Save pickle file to data storage directory with panel mesh, velocity and pressure
        # num_panels = triangles.shape[0]
        num_panels = len(combined_cells)
        num_points = panel_mesh_val.shape[0]
        assert pressure_phys.shape == (num_panels,), f"Pressure shape {pressure_phys.shape} does not match expected shape {(num_panels,)}"
        assert velocity_phys.shape == (num_panels, 3), f"Velocity shape {velocity_phys.shape} does not match expected shape {(num_panels, 3)}"

        # save_name = f'{name}.pkl'
        # save_path = os.path.join(dir, save_name)
        # with open(save_path, 'wb') as f:
        #     pickle.dump(output_data, f)

        # save as npz
        save_name = f'{name}.npz'
        save_path = os.path.join(dir, save_name)
        np.savez(save_path,
            panel_mesh=panel_mesh_val, velocity=velocity_phys, pressure=pressure_phys,
            # sample_velocity=sampled_velocity, sample_points=points, sample_pressure=sampled_pressure,
            L=L, Di=Di, M=M,
        )

        # Check if connectivity data is already saved as CONNECTIVITY.pkl in the storage directory, if not save it
        if name == '0':
            print('SAVED MESH TOPOLOGY AND CONNECTIVITY DATA')
            connectivity_save_path = os.path.join(dir, 'CONNECTIVITY_NEW.pkl')
            connectivity_data = {
                # 'triangles': triangles,
                # 'cell_adjacency': cell_adjacency,
                'cell_adjacency_data': cell_adjacency_data,
                'panel_mesh': panel_mesh_val,
                # 'mesh_sample_indices': sample_points['sample_indices'],
                'pressure_baseline': pressure_phys,
            }
            with open(connectivity_save_path, 'wb') as f:
                pickle.dump(connectivity_data, f)


    return {
        'geometry_config_map': geometry_config_map,
        'model':run_simulation,
    }
