# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import time
from pathlib import Path
import shutil
import sys

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
from csdl_dafoam_helper_functions import Timer
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle
from bwb_geometry import build_bwb_geometry

# For off-body analysis
from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel
def off_body_analysis(mesh_dict, wake_mesh_dict, eval_points, mu, sigma, velocity, upper_TE_cells, lower_TE_cells):

    num_panels = mesh_dict['panel_corners'].shape[1]
    num_eval_pts = eval_points.shape[1]
    num_wake_panels = wake_mesh_dict['panel_corners'].shape[1]
    num_nodes = 1

    AIC_mu = csdl.Variable(value=np.zeros((1, num_eval_pts, num_panels, 3)))
    AIC_mu_wake = csdl.Variable(value=np.zeros((1, num_eval_pts, num_wake_panels, 3)))

    # generating AIC matrix for surface
    num_interactions = num_eval_pts * num_panels
    expanded_shape = (num_nodes, num_eval_pts, num_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    csdl_mesh_dict = isinstance(mesh_dict['panel_corners'], csdl.Variable)
    csdl_wake_mesh_dict = isinstance(wake_mesh_dict['panel_corners'], csdl.Variable)
    if not csdl_mesh_dict:
        panel_corners = csdl.Variable(value=mesh_dict['panel_corners'])
        coll_point = csdl.Variable(value=mesh_dict['panel_center'])
        panel_x_dir = csdl.Variable(value=mesh_dict['panel_x_dir'])
        panel_y_dir = csdl.Variable(value=mesh_dict['panel_y_dir'])
        panel_normal = csdl.Variable(value=mesh_dict['panel_normal'])
        S = csdl.Variable(value=mesh_dict['S'])
        SL = csdl.Variable(value=mesh_dict['SL'])
        SM = csdl.Variable(value=mesh_dict['SM'])
    else:
        panel_corners = mesh_dict['panel_corners']
        coll_point = mesh_dict['panel_center']
        panel_x_dir = mesh_dict['panel_x_dir']
        panel_y_dir = mesh_dict['panel_y_dir']
        panel_normal = mesh_dict['panel_normal']
        S = mesh_dict['S']
        SL = mesh_dict['SL']
        SM = mesh_dict['SM']

    if not csdl_wake_mesh_dict:
        panel_corners_w = csdl.Variable(value=wake_mesh_dict['panel_corners'])
    else:
        panel_corners_w = wake_mesh_dict['panel_corners']
        
    panel_corners_expanded = panel_corners.expand(expanded_shape, 'ijkl->iajkl')
    panel_corners_vec = panel_corners_expanded.reshape(vectorized_shape)

    eval_points_expanded = eval_points.expand(expanded_shape, 'ijk->ijabk')
    eval_points_vec = eval_points_expanded.reshape(vectorized_shape)

    ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_vec[:,:,0,:],panel_corners_vec[:,:,1,:],p_eval=eval_points_vec[:,:,0,:], mode='wake', vc=1.e-4)
    ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_vec[:,:,1,:],panel_corners_vec[:,:,2,:],p_eval=eval_points_vec[:,:,0,:], mode='wake', vc=1.e-4)
    ind_vel_s_31 = compute_vortex_line_ind_vel(panel_corners_vec[:,:,2,:],panel_corners_vec[:,:,0,:],p_eval=eval_points_vec[:,:,0,:], mode='wake', vc=1.e-4)

    ind_vel_s = ind_vel_s_12+ind_vel_s_23+ind_vel_s_31 # (nn, num_interactions, 3)
    ind_vel_s_mat = ind_vel_s.reshape((num_nodes, num_eval_pts, num_panels, 3))
    AIC_mu = ind_vel_s_mat

    coll_point_exp = coll_point.expand(expanded_shape, 'ijk->iajbk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    panel_x_dir_exp = panel_x_dir.expand(expanded_shape, 'ijk->iajbk')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)

    panel_y_dir_exp = panel_y_dir.expand(expanded_shape, 'ijk->iajbk')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)

    panel_normal_exp = panel_normal.expand(expanded_shape, 'ijk->iajbk')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    S_exp = S.expand(expanded_shape[:-1], 'ijk->iajk')
    S_exp_vec = S_exp.reshape(vectorized_shape[:-1])

    SL_exp = SL.expand(expanded_shape[:-1], 'ijk->iajk')
    SL_exp_vec = SL_exp.reshape(vectorized_shape[:-1])

    SM_exp = SM.expand(expanded_shape[:-1], 'ijk->iajk')
    SM_exp_vec = SM_exp.reshape(vectorized_shape[:-1])

    a = eval_points_vec - panel_corners_vec # Rc - Ri
    P_JK = eval_points_vec - coll_point_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP
    # print(A.shape)
    # exit()
    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_exp_vec - AL*SM_exp_vec

    A = A.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    AM = AM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    B = B.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    BM = BM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    SL_i_exp_vec = SL_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    SM_i_exp_vec = SM_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    A1 = A1.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    PN = PN.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    S_i_exp_vec = S_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')

    A_list = [A[:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,ind] for ind in range(3)]
    B_list = [B[:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,ind] for ind in range(3)]
    SL_list = [SL_i_exp_vec[:,:,ind] for ind in range(3)]
    SM_list = [SM_i_exp_vec[:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,ind] for ind in range(3)]
    S_list = [S_i_exp_vec[:,:,ind] for ind in range(3)]

    ind_vel_source_local = compute_source_influence_new(
        A_list, 
        AM_list, 
        B_list, 
        BM_list, 
        SL_list, 
        SM_list, 
        A1_list, 
        PN_list, 
        S_list, 
        panel_x_dir_exp_vec[:,:,0,:], # these don't change along dimension 2
        panel_y_dir_exp_vec[:,:,0,:],
        panel_normal_exp_vec[:,:,0,:],
        mode='velocity'
    )

    ind_vel_source_global = ind_vel_source_local
    ind_vel_source_global_mat = ind_vel_source_global.reshape((num_nodes, num_eval_pts, num_panels, 3))
    AIC_sigma = ind_vel_source_global_mat

    # generating AIC matrix for wake
    num_interactions_w = num_eval_pts * num_wake_panels
    expanded_shape_w = (num_nodes, num_eval_pts, num_wake_panels, 4, 3)
    vectorized_shape_w = (num_nodes, num_interactions_w, 4, 3)

    # print(panel_corners_w.shape)
    panel_corners_expanded_w = panel_corners_w.expand(expanded_shape_w, 'ijkl->iajkl')
    panel_corners_vec_w = panel_corners_expanded_w.reshape(vectorized_shape_w)

    eval_points_expanded_w = eval_points.expand((num_nodes, num_eval_pts, num_wake_panels, 3), 'ijk->ijak')
    eval_points_vec_w = eval_points_expanded_w.reshape((num_nodes, num_interactions_w, 3))

    ind_vel_s_12_w = compute_vortex_line_ind_vel(panel_corners_vec_w[:,:,0,:],panel_corners_vec_w[:,:,1,:],p_eval=eval_points_vec_w, mode='wake', vc=1.e-4)
    ind_vel_s_23_w = compute_vortex_line_ind_vel(panel_corners_vec_w[:,:,1,:],panel_corners_vec_w[:,:,2,:],p_eval=eval_points_vec_w, mode='wake', vc=1.e-4)
    ind_vel_s_34_w = compute_vortex_line_ind_vel(panel_corners_vec_w[:,:,2,:],panel_corners_vec_w[:,:,3,:],p_eval=eval_points_vec_w, mode='wake', vc=1.e-4)
    ind_vel_s_41_w = compute_vortex_line_ind_vel(panel_corners_vec_w[:,:,3,:],panel_corners_vec_w[:,:,0,:],p_eval=eval_points_vec_w, mode='wake', vc=1.e-4)

    ind_vel_s_w = ind_vel_s_12_w+ind_vel_s_23_w+ind_vel_s_34_w+ind_vel_s_41_w # (nn, num_interactions, 3)
    ind_vel_s_mat_w = ind_vel_s_w.reshape((num_nodes, num_eval_pts, num_wake_panels, 3))
    AIC_mu_wake = ind_vel_s_mat_w

    mu_wake = mu[:,upper_TE_cells] - mu[:,lower_TE_cells]

    induced_vel = csdl.Variable(shape=(num_nodes, num_eval_pts, 3), value=0.)
    for direction in range(3):
        sigma_induced_vel = csdl.matvec(AIC_sigma[0,:,:,direction], sigma[0,:])
        mu_surf_induced_vel = csdl.matvec(AIC_mu[0,:,:,direction], mu[0,:])
        mu_wake_induced_vel = csdl.matvec(AIC_mu_wake[0,:,:,direction], mu_wake[0,:])
        induced_vel = induced_vel.set(
            csdl.slice[0,:,direction],
            # value=mu_surf_induced_vel+mu_wake_induced_vel
            value=sigma_induced_vel+mu_surf_induced_vel+mu_wake_induced_vel
        )
    velocity_expanded = csdl.expand(velocity, induced_vel.shape, 'ij->iaj')
    total_velocity = velocity_expanded + induced_vel
    return total_velocity

def build_panel_func(
        operating_reference:dict,
        geometry_dir:str='bwb_geometry/',
        stp_file_name:str='bwbv2_no_wingtip_coarse_refined_flat.stp',
        geometry_pickle_file_name:str='bwb_stored_refit.pickle',
    ):
    comm = None
    rank = 0

    timing_enabled = True  # True if we want timing printed for the CSDL operations

    geometry_directory        =  os.path.join(os.getcwd(), geometry_dir)
    geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
    stp_file_path                     = Path(geometry_directory)/stp_file_name

    # surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'
    recorder = csdl.Recorder(inline=True, debug=False)
    recorder.start()
    geometry, geometry_config_map, other = build_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled=timing_enabled)

    # ===============================================================================================================
    # =============================================== READ GEOMETRY MESH ============================================
    oml_indices                 = [key for key in geometry.functions.keys()]
    wing_c_indices              = [0,1,8,9]
    wing_r_transition_indices   = [2,3]
    wing_r_indices              = [4,5,6,7]
    wing_l_transition_indices   = [10,11]
    wing_l_indices              = [12,13,14,15] 

    with Timer('declaring geometry components', rank, timing_enabled):
        left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
        left_wing               = geometry.declare_component(wing_l_indices)
        right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
        right_wing              = geometry.declare_component(wing_r_indices)
        center_wing             = geometry.declare_component(wing_c_indices)
        oml = geometry.declare_component(oml_indices)

    # ===========================================================================================================================
    # =============================================== READ VORTEXAD MESH AND PROJECT ============================================
    from VortexAD import steady_panel_solver
    from VortexAD.utils.cell_adjacency import find_cell_adjacency
    from VortexAD.utils.TE_detection import TE_detection
    from VortexAD.utils.plot_unstructured import plot_pressure_distribution

    file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch.stl'
    file_path = os.getcwd() + '/panel_geometry/' 

    import meshio
    mesh = meshio.read(
        file_path + file_name,  # string, os.PathLike, or a buffer/open file
        # file_format="stl",  # optional if filename is a path; inferred from extension
        # see meshio-convert -h for all possible formats
    )

    points_orig = mesh.points
    cells = mesh.cells
    cells_dict = mesh.cells_dict

    triangles = cells_dict['triangle']

    # dup_indices = check_duplicate_nodes(points=points_orig)
    # print(dup_indices)

    # exit()
    points_orig, triangles, cell_adjacency, edges2cells, points2cells = find_cell_adjacency(points=points_orig, cells=triangles)
    # edges2cells = find_cell_adjacency(points=points_orig, cells=triangles)

    upper_TE_cells, lower_TE_cells, TE_edges, TE_node_indices = TE_detection(
        points=points_orig,
        cells=triangles,
        # cell_adjacency=cell_adjacency,
        edges2cells=edges2cells,
        threshold_theta=125.
    )

    start = time.time()
    projected_panel_mesh = geometry.project(
        points_orig, 
        grid_search_density_parameter=1, 
        projection_tolerance=1.e-3, 
        grid_search_density_cutoff=20,
        force_reprojection=False, 
        plot=False,
    )
    end_mesh_projection = time.time()
    print(f'panel mesh projection time: {end_mesh_projection-start} seconds')

    save_meshes = True
    shutdown_inline = True
    import pickle
    # project panel centers
    # if save_meshes:
    #     print('projecting panel centers')
    #     panel_centers = np.zeros((len(triangles), 3))
    #     for i, triangle in enumerate(triangles):
    #         panel_centers[i] = np.mean(points_orig[triangle], axis=0)

    #     panel_centers = oml.project(
    #             panel_centers, 
    #             grid_search_density_parameter=0.5, 
    #             projection_tolerance=0.05,
    #             grid_search_density_cutoff=20,
    #             force_reprojection=False, 
    #             plot=False
    #         )
    #     with open('mesh_projections/panel_centers.pkl', 'wb') as f:
    #         pickle.dump(panel_centers, f)
    #     print('done projecting panel centers')

    # # load the projected panel centers from a file
    # with open('mesh_projections/panel_centers.pkl', 'rb') as f:
    #     projected_panel_centers = pickle.load(f)

    # ===========================================================================================================================
    # =============================================== PANEL METHOD SET UP =======================================================
    recorder.inline = not shutdown_inline
    panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
    panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')
    # panel_center = geometry.evaluate(projected_panel_centers, plot=False)

    # SET UP VELOCITY, DENSITY, AND PITCH ARRAYS
    U0 = operating_reference['U0']
    p0 = operating_reference['p0']
    T0 = operating_reference['T0']
    A0        = 527           # Projected area of entire BWB. Used for normalizing CD and CL

    #################################### ONLY X IS NEGATIVE, BE CAREFUL WHEN IMPLMENTING AOA #################################
    p_inf = csdl.Variable(value=p0)
    t_inf = csdl.Variable(value=T0)
    
    V_inf_dv = csdl.Variable(value = np.array([U0, 0.0, 0.0]))
    V_inf = -V_inf_dv
    rho_inf = p_inf / (t_inf * 287.0)

    # For off-body
    rho_array = rho_inf
    V_inf_array = csdl.norm(V_inf_dv)

    # point_velocities = csdl.Variable(value=0.0, shape=panel_mesh.shape)
    # point_velocities = point_velocities.set()
    point_velocities = csdl.expand(V_inf, panel_mesh.shape, 'i->aji')

    TE_data = [TE_node_indices, TE_edges, (upper_TE_cells, lower_TE_cells)]
    connectivity_data = [triangles, cell_adjacency, points2cells]

    # print(f'num triangles: {len(triangles)}')
    # print(f'num cell adjancency: {len(cell_adjacency)}')
    # print(f'num points2cells: {len(points2cells)}')

    # triangles: (num_triangles, 3)
    # cell_adjacency: (num_triangles, 3) adjanct triangles
    # points to cells: dict with key = point index, value = list of cell indices that the point is a part of
    # print(panel_mesh.shape, point_velocities.shape)
    # exit()

    output_dict, mesh_dict, mu, sigma = steady_panel_solver(
        panel_mesh, 
        connectivity_data, 
        TE_data, 
        point_velocities, 
        mesh_mode='unstructured',
        batch_size=1,
        Cp_cutoff=-5,
        # M_inf=csdl.norm(V_inf_dv)/303,
        constant_geometry=True,
        rho = rho_inf,
    )

    # graph = csdl.get_current_recorder().active_graph
    # graph.visualize('ode_graph')
    # exit()

    # print('output keys', output_dict.keys())
    # print('mesh_dict keys', mesh_dict.keys())
    # print(mu)
    # print(sigma)
    # mesh_velocities = mesh_dict['nodal_velocity']
    mesh_velocity_x = output_dict['Vx']
    mesh_velocity_y = output_dict['Vy']
    mesh_velocity_z = output_dict['Vz']
    mesh_velocity_magnitudes = output_dict['V_mag']
    assert mesh_velocity_magnitudes.shape == mesh_velocity_x.shape == mesh_velocity_y.shape == mesh_velocity_z.shape, f"Velocity magnitude shape {mesh_velocity_magnitudes.shape} does not match mesh velocity shapes {mesh_velocity_x.shape}"

    Cp = output_dict['Cp']
    wake_dict = output_dict['wake_dict']

    flow_output_names = [
        'Cp',
        'mu',
        'sigma',
        'panel mesh',
        'mesh velocity',
    ]

    mesh_output_names = [
        'panel_corners',
        'panel_center',
        'panel_x_dir',
        'panel_y_dir',
        'panel_normal',
        'S',
        'SL',
        'SM',
        'panel_area'
    ]

    wake_output_names = [
        'panel_corners'
    ]

    flow_names = [
        'rho_array','V_inf_array','p_c',
    ]
    force_names = [
        'L', 'Di',
    ]

    output_names = flow_output_names + mesh_output_names + wake_output_names

    mesh_outputs = [mesh_dict[name] for name in mesh_output_names]
    wake_outputs = [wake_dict[name] for name in wake_output_names]

    for_off_body_outputs = [Cp, mu, sigma, panel_mesh, point_velocities]
    for_off_body_outputs.extend(mesh_outputs)
    for_off_body_outputs.extend(wake_outputs)
    for_off_body_outputs.extend([rho_array, V_inf_array, p_inf])
    for_off_body_outputs.extend([output_dict['L'], output_dict['Di']])

    # ===========================================================================================================================
    # =============================================== OFF BODY SET UP =======================================================

    # N_EVAL = 10_000
    N_EVAL = 1_000
    eval_pts_shape = (1,) + (N_EVAL, 3)
    # eval_pts_shape = (1,) + (n, 3)
    eval_points = csdl.Variable(shape=eval_pts_shape, value=0.)
    wake_mesh_dict = {'panel_corners': wake_dict[ 'panel_corners']}

    total_velocity = off_body_analysis(
        mesh_dict,
        wake_mesh_dict,
        eval_points,
        mu, 
        sigma,
        -V_inf.reshape(1,3),
        upper_TE_cells,
        lower_TE_cells
    )

    # ===========================================================================================================================
    # =============================================== Build Simulator =======================================================
    inputs = [geo_param for geo_param in geometry_config_map.values()] +  [eval_points]
    outputs = for_off_body_outputs + [total_velocity] + [mesh_velocity_x, mesh_velocity_y, mesh_velocity_z, mesh_velocity_magnitudes]

    flow_conditions_map = {
        'v_inf': V_inf_dv,
        'p_inf': p_inf,
        'T_inf': t_inf,
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
        
        if sample_points is None:
            sample_points = {}
            sample_points['eval_points'] = np.random.uniform(low=[-50, 0, -50], high=[50, 50, 50], size=(N_EVAL, 3))
            sample_points['sample_indices'] = np.zeros((N_EVAL,), dtype=int)

        # =====================SET INPUTS========================
        # Set inputs and run
        assert len(configuration) == len(geometry_config_map), f"Configuration length {len(configuration)} does not match expected {len(geometry_config_map)}"
        for geo_param_name, geo_param_value in configuration.items():
            geo_variable = geometry_config_map[geo_param_name]
            jax_sim[geo_variable] = geo_param_value['value']

        # for key in flow_condition.keys():
        assert len(flow_condition) == len(flow_conditions_map), f"Flow condition length {len(flow_condition)} does not match expected {len(flow_conditions_map)}"
        for flow_variable_name, flow_variable_value in flow_condition.items():
            flow_variable = flow_conditions_map[flow_variable_name]
            jax_sim[flow_variable] = flow_variable_value['value']
            print(f'Setting flow condition {flow_variable_name} to {flow_variable_value["value"]}')

        # Sample points
        points = sample_points['eval_points']
        assert points.shape == (N_EVAL, 3), f'Sample points shape {points.shape} does not match expected shape {(N_EVAL, 3)}'
        jax_sim[eval_points] = points.reshape((1, N_EVAL, 3))

        # freestreem
        # jax_sim[V_inf_dv] = flow_condition['v_inf']
        # print(flow_condition['v_inf'])
        # print(rho_inf.value)
        # exit()

        # ===================== RUN ========================
        start = time.time()
        jax_sim.run()
        end = time.time()

        L, Di = jax_sim[output_dict['L']][0], jax_sim[output_dict['Di']][0]
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
        sampled_velocity = jax_sim[total_velocity][0]
        assert sampled_velocity.shape == (N_EVAL, 3), f'Evaluated velocity shape {sampled_velocity.shape} does not match expected shape {(N_EVAL, 3)}'
        sampled_velocity = np.clip(sampled_velocity, [0, -300, -300], [350, 300, 300])
        sampled_velocity_magnitudes = np.linalg.norm(sampled_velocity, axis=1)
        sampled_pressure = p0 + 0.5 * rhoinf * (Vinf**2 - sampled_velocity_magnitudes**2)
        sampled_pressure = np.clip(sampled_pressure, 100, 50000)
        assert sampled_pressure.shape == (N_EVAL,), f'Sampled pressure shape {sampled_pressure.shape} does not match expected shape {(N_EVAL,)}'
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
        num_panels = triangles.shape[0]
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
            sample_velocity=sampled_velocity, sample_points=points, sample_pressure=sampled_pressure,
            L=L, Di=Di,
        )

        # Check if connectivity data is already saved as CONNECTIVITY.pkl in the storage directory, if not save it
        if name == '0':
            print('SAVED MESH TOPOLOGY AND CONNECTIVITY DATA')
            connectivity_save_path = os.path.join(dir, 'CONNECTIVITY.pkl')
            connectivity_data = {
                'triangles': triangles,
                'cell_adjacency': cell_adjacency,
                'panel_mesh': panel_mesh_val,
                'mesh_sample_indices': sample_points['sample_indices'],
                'pressure_baseline': pressure_phys,
            }
            with open(connectivity_save_path, 'wb') as f:
                pickle.dump(connectivity_data, f)


    return {
        'geometry_config_map': geometry_config_map,
        'model':run_simulation,
    }


if 0:
    # ================== Sample N from a range around the reference value (ref_value +/- delta) ===========
    # read pickle file from data storage directory
    import pickle
    STORAGE_DIRECTORY = 'PANEL_DATA_STORAGE'
    config_filepath = 'PANEL_CONFIG.pkl'
    file_path = os.path.join(STORAGE_DIRECTORY, config_filepath)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    data_generation_config = data

    # print(f'Data generation configuration keys: {list(data_generation_config[0].keys())}')
    for i, config in enumerate(data_generation_config):
        case_id = config['case_id']
        configuration = {}
        for key, value in geometry_config_ranges.items():
            geo_param_name = value['name']
            geo_param_value = np.asarray(config['aircraft_geometry'][geo_param_name])
            assert geo_param_value.shape == np.asarray(value['ref_value']).shape, f'Geometry parameter {geo_param_name} has shape {geo_param_value.shape} but expected shape {value["ref_value"].shape}'
            configuration[key] = {'value': geo_param_value, 'name': geo_param_name}
        # check that all keys in config are in geometry_config_ranges
        for geo_param_name in config['aircraft_geometry'].keys():
            assert geo_param_name in [v['name'] for v in geometry_config_ranges.values()], f'Geometry parameter {geo_param_name} in config is not in geometry config ranges'
        assert len(configuration) == len(geometry_config_ranges), f'Configuration length {len(configuration)} does not match geometry config ranges length {len(geometry_config_ranges)}'
        
        sample_points = {
            'eval_points': config['sample_points'],
            'sample_indices': config['sample_indices'],
        }
        flow_condition = config['flow_condition']
        print(f"generating {i+1}/{len(data_generation_config)}: case_id={case_id}")
        assert len(configuration) == len(geometry_config_ranges), f'Configuration length {len(configuration)} does not match geometry config ranges length {len(geometry_config_ranges)}'
        run_simulation(case_id, configuration, sample_points, flow_condition, STORAGE_DIRECTORY)

    exit()
    print(data_generation_config)

    run_simulation(0, configuration)
    exit()

    for i, configuration in enumerate(samples):
        run_simulation(i+1, configuration)

    exit()


    num_nodes = 1
    alpha = csdl.Variable(value=np.zeros((num_nodes,)))
    rho_c, a_c, mu_c, p_c = atmos_model(h_km) # properties at cruise altitude
    #endregion

    V_cruise = a_c*cruise_mach
    V_cruise_array = csdl.Variable(value=np.zeros((num_cruise,)))
    V_cruise_array = V_cruise_array.set(csdl.slice[:], V_cruise)

    V_structural = mach*340.3 # sea level (SL)

    V_inf_array = csdl.Variable(value=np.zeros((num_nodes,)))
    V_inf_array = V_inf_array.set(csdl.slice[:num_cruise], V_cruise_array) # cruise speeds
    V_inf_array = V_inf_array.set(csdl.slice[num_cruise:num_cruise+2], value=V_structural) # speed for structural analysis
    V_inf_array = V_inf_array.set(csdl.slice[-1], value=V_cruise) # perturbation for static margin at same altitude

    V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
    V_vec = V_vec.set(csdl.slice[:,0], value=-V_inf_array)

    rho_array = csdl.Variable(value=np.zeros((num_nodes,)))
    rho_array = rho_array.set(csdl.slice[:num_cruise], rho_c) # cruise altitude
    rho_array = rho_array.set(csdl.slice[num_cruise:num_cruise+2], 1.225) # SL
    rho_array = rho_array.set(csdl.slice[-1], rho_c) # cruise altitude

    pitch_array = csdl.Variable(value=np.zeros(rho_array.shape))
    pitch_array = pitch_array.set(csdl.slice[:num_nodes-1], pitch)
    pitch_array = pitch_array.set(csdl.slice[-1], pitch[0] + dalpha)

    # region: panel method

    pitch_rad = pitch_array*np.pi/180.
    V_rot_mat = csdl.Variable(value=0., shape=(num_nodes,3,3))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

    V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
    point_velocities = csdl.expand(V_vec_rot, (num_nodes,) + panel_mesh.shape[1:], 'ij->iaj')