import jax
jax.config.update("jax_enable_x64", False)

import aerodml as adml
from aerodml.postprocessing import compute_forces, integrate_forces

import os
import jax.numpy as jnp
import numpy as np
import time

from aerodml.utils import rho_physical
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

PARAM_DIR = HERE / "saved_parameters"
DATA_DIR = CASES_DIR / "datasets" / "BWB" / "data" 
print(f"CASE_DIR: {CASE_DIR}")
print(f"PARAM_DIR: {PARAM_DIR}")

def load_cfd_data(dir:str, split:str, analyze:bool=False, filter:Callable = None, num_cases:int=None)->dict:
    print("Retrieving CFD data...")
    return load_cfd(dir, split, analyze=analyze, filter=filter, num_cases=num_cases)

def get_graph(path_dir:str, mesh_load_settings:dict=None, dual_load_settings:dict=None, coarse_load_settings:dict=None)->dict:
    # Translates openfoam mesh data into Python arrays then processes them for GNNs
    mesh_load_settings = check_mesh_settings(mesh_load_settings)
    foam_mesh = load_foam_mesh(path_dir, mesh_load_settings)

    dual_load_settings = check_mesh_settings(dual_load_settings)
    dual_graph = load_dual_graph(path_dir, foam_mesh, dual_load_settings)

    coarse_load_settings = check_mesh_settings(coarse_load_settings)
    coarse_graph_l1 = load_coarse_dual_graph(path_dir, dual_graph, foam_mesh, coarse_load_settings, name = 'dual_coarse_mesh_l1', N = 2, max_edge_length=1.7) # equivalent to N=2 on original mesh
    coarse_graph_l2 = load_coarse_dual_graph(path_dir, coarse_graph_l1, foam_mesh, coarse_load_settings, name = 'dual_coarse_mesh_l2', N = 3, max_edge_length=25.7) # equivalent to N=5 on original mesh

    hierarchy = ['dual_graph', 'coarse_graph_l1', 'coarse_graph_l2']
    geometry = {'foam_mesh': foam_mesh, 'dual_graph': dual_graph, 'coarse_graph_l1': coarse_graph_l1, 'coarse_graph_l2': coarse_graph_l2, 'hierarchy': hierarchy}
    for key in hierarchy:
        assert key in geometry, f"Error: key {key} in hierarchy is not present in geometry"
    return geometry

def main(
        load:bool = True, 
        train:bool = False,
        save:bool = False, 
        dataset_dir = 'sampling_200',
        split = 'split_0',
        load_model_path:str=None,
        as_multifidelity_model:bool = True,
        save_dir_path:str = 'gnn',
        save_training_path:str = "gnn",
        with_pinns:bool = False,
        model_name:str = "",
    ):
    # High-level options
    save_params_path = save_training_path + f'_{save_dir_path}'
    device = jax.devices("gpu")[0]
    # device = jax.devices()[0]
    
    # ============== Training settings ==============
    if train:
        import optax
        schedule = optax.piecewise_constant_schedule(
            init_value=1e-3,
            boundaries_and_scales={
                45000: 0.1,
            }
        )
        schedule = optax.piecewise_constant_schedule(
            init_value=1e-3,
            boundaries_and_scales={
                30_000: 0.1,
            }
        )
        # schedule = 1e-5
    else:
        schedule = 'ERROR'

    # schedule = 1e-4
    training_settings = {
        'pinn_loss':with_pinns, 'max_iter': 65000, 'batch_size': 1,'lr': schedule,
        'seed':17, 'print_stride': 50, 'device': device, 'save':save_training_path,
    }

    # low-fidelity input:

    # ============= Pre-processing ==============:
    # device = jax.devices()[0]
    mesh_load_settings =   {'verbose': False, 'run_checks': True, 'visualize': False}
    dual_load_settings =   {'verbose': False, 'run_checks': True, 'visualize': False}
    coarse_load_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    lowfi_load_settings =  {'verbose': False, 'run_checks': True, 'visualize': False}

    # Flow conditions (SI units):
    # - References and normalization constants
    R = 287.05
    T_ref = 228.714
    nuTilda_ref = 4.5e-5
    p_ref = 30089.6
    v_ref = 227.38
    rho_ref = rho_physical(p_ref, T_ref, R)
    constants = {
        'T': {      'mean': np.float64(232),                'std': 3*np.float64(11.0),             'ref': T_ref,},
        'nuTilda': {'mean': np.float64(0.00559),            'std': 3*np.float64(0.0179),           'ref': nuTilda_ref,},
        'p': {      'mean': np.float64(3.05e+04),           'std': 3*np.float64(3.86e+03),         'ref': p_ref,},
        'v': {      'mean': np.array([202.8, 14.2, -8.01]), 'std': 3*np.array([48.15, 32.3, 44.6]),'ref': v_ref,},
        'rho': {                                                                                   'ref': rho_ref,},
        'R': R, 'gamma': 1.4,
        'device': device,
    }

    # ============= Main Setup and Preprocessing ==============:
    geometry_topology = get_graph('mesh_639k', mesh_load_settings=mesh_load_settings, dual_load_settings=dual_load_settings, coarse_load_settings=coarse_load_settings)
    
    if as_multifidelity_model == 'panel':
        # lowfi_topology = get_panel_mesh(f'{HERE}/CONNECTIVITY.pkl', geometry_topology, lowfi_load_settings)
        lowfi_topology = get_lowfi_panel_mesh(f'{HERE}/CONNECTIVITY.pkl', geometry_topology, lowfi_load_settings)
    elif as_multifidelity_model == 'euler':
        lowfi_topology = get_lowfi_CFD_mesh(f'mesh_112k', geometry_topology, mesh_load_settings, dual_load_settings, lowfi_load_settings)
    elif as_multifidelity_model == False:
        lowfi_topology = None
    else:
        raise ValueError(f'invalid as_multifidelity_model {as_multifidelity_model}')

    model, param_info = get_model(constants, multifidelity_model=as_multifidelity_model,root=PARAM_DIR)
    if load:
        if load_model_path is None:
            raise ValueError("load_model_path must be provided when load is True")
        params = param_info.load(save_params_path, custom_path=load_model_path)
    else:
        params = param_info.initialize()
    
    dataset_path = f'{DATA_DIR}/{dataset_dir}'
    split_path = f'{dataset_path}/{split}'
    cfd_data = load_cfd_data(dir = f'{dataset_path}/rans', split = split_path, num_cases = None)
    # cfd_data = load_cfd_data(dir = f'{DATA_DIR}/d0/cfd', split = f'{dataset_path}/{split}', num_cases = None)
    if as_multifidelity_model == 'panel':
        load_lowfi_panel_data(cfd_data, dir = f'{dataset_path}/panel', split_dir = split_path)
    if as_multifidelity_model == 'euler':
        load_lowfi_cfd_data(cfd_data, lf_dir = f'{dataset_path}/euler', split_dir = split_path)
    
    # cfd_data = [item for item in cfd_data if int(item['case_id']) == 0]
    train_data, test_data = build_geometry_data(
        geometry_topology, cfd_data, lowfi_topology,
        faces = with_pinns, verbose=1, visualize=0, device=device
    )

    
    res_func = build_residual_function(model)

    # print which cases are test
    # test_ids = [config['case_id'] for config in test_data]
    # print(f"Test cases: {test_ids}")

    # check_model(model, params, train_data[0], constants)
    # check_res_func(res_func, params, train_data[0], constants)

    # check_res_func(res_func, params, train_data[0], constants, grad = True, pinn_loss=with_pinns)
    # check_res_func(res_func, params, train_data[0], constants, grad = True)
    # exit()

    if train:
        params = train_model(model, res_func, params, train_data, test_data, constants, training_settings, param_info)
    if save:
        # save_params(params, save_dir_path)
        param_info.save(params, save_params_path)

    # exit()
    # ============== Post-processing ==============:
    # config = test_data[-1]
    # config = train_data[0]
    # topology = config['topology']
    # # model_eval = jax.jit(lambda x,y: model(x,y,topology), device=constants['device'])
    # model_eval = jax.jit(lambda p,x,y: model(p,x,y,topology), device=constants['device'])
    
    # from postprocessing_viz import visualize_volume_post
    # state = 'p'
    # for config in test_data + train_data:
    #     visualize_volume_post(model_eval, params, state, config, constants, distance_cutoff=1000.0)
    # exit()

    # from postprocessing_viz import visualize_surface_post
    # visualize_surface_post(model_eval, params, config, constants, window_size=(800, 2400), return_image=True)
    # exit()

    # ===================================================================================================================
    # ============================================ Visualization Preparation ============================================
    # ===================================================================================================================
    test_data = sorted(test_data, key=lambda x: int(x['case_id']))
    train_data = sorted(train_data, key=lambda x: int(x['case_id']))

    model_eval = jax.jit(lambda p,x,y: model(p,x,y,topology), device=constants['device'])

    vid_data = sorted(train_data, key=lambda x: int(x['case_id'])) + sorted(test_data, key=lambda x: int(x['case_id']))
    topology = vid_data[0]['topology']
    # ===================================================================================================================
    # ============================================ Visualization Preparation ============================================
    # ===================================================================================================================

    # vid_data = [vid_data[0], vid_data[1]]
    from aerodml.postprocessing_viz import visualize_surface_post

    # =========================== Analyze Errors ===========================
    from aerodml.postprocessing_viz import analyze_training, analyze_pde_errors
    res_func_full = build_residual_function(model, return_full_residuals=True)
    res_func_full_jit = jax.jit(lambda d,f,x: res_func_full(params,d,f,x, topology, constants), device=constants['device'])
    # analyze_training(model_eval, res_func_full_jit, params, train_data, test_data, constants)
    # analyze_pde_errors(model_eval, res_func_full, params, train_data, test_data, constants)
    # analyze_loss(model_eval, params, train_data, test_data, constants)
    # exit()

    # =========================== Visualize Force Error Histogram =========================== 
    from aerodml.postprocessing_viz import plot_case_histogram, analyze_aero_forces #, analyze_model
    # # plot_case_histogram(model_eval, params, vid_data, constants)
    # # analyze_model(model_eval, params, vid_data, constants, screenshot=True)
    analyze_aero_forces(model_eval, params, vid_data, constants, screenshot=True, model_name=model_name)

    # =========================== Visualize surfaces single =========================== 
    # from visualize_mesh import build_pyvista_mesh_topology
    # pv_mesh = build_pyvista_mesh_topology(topology['foam_mesh'])
    # for config in vid_data:
    #     if config['train']:
    #         title = f"Case {config['case_id']} (TRAINING)"
    #     else:
    #         title = f"Case {config['case_id']} (TEST)"
    #     visualize_surface_post(model_eval, params, config, constants, state = 'T', title = title,camera = 'TE', window_size=(993, 960), return_image=False, y_plane = 0.5, pyvista_mesh_topology=pv_mesh)

    # exit()
    from aerodml.visualize_mesh import build_pyvista_mesh_topology
    pv_mesh = build_pyvista_mesh_topology(topology['foam_mesh'])
    # =========================== Visualize PAPER FIGURES =========================== 
    # from aerodml.postprocessing_viz import visualize_paper
    # state, camera= 'p', 'pos_2'
    # # state, camera= 'T', 'pos_2'
    # # state, camera= 'v', 'pos_2'
    # for config in vid_data:
    #     print(f"Creating case id {config['case_id']} figures ...")
    #     from pathlib import Path
    #     save_dir = Path(f'paper_field_plots/{model_name}/case_{config["case_id"]}')
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     visualize_paper(
    #         model_eval, params, config, constants, save_dir, state=state,
    #         window_size=(993, 960), return_image=False, enable_picking=False, pyvista_mesh_topology=pv_mesh,
    #         camera = camera)

    # exit()
    # =========================== Visualize surface video =========================== 
    images = []
    # state, camera = 'v', 'TE'
    # state, camera, plane = 'T', 'TE_underside', 0.5
    # state, camera, plane = 'T', 'TE', 0.5
    state, camera, plane = 'p', 'pos_2', 0.1
    # state, camera, plane = 'nuTilda', 'pos_2', 0.5
    # state, camera, plane = 'v', 'pos_2', 12.5
    
    # If interactive or not
    # video = False
    video = True
    for config in vid_data:
        start = time.time()
        title = f"Case {config['case_id']} (train={config['train']})"
        image = visualize_surface_post(model_eval, params, config, constants, title = title, state=state, window_size=(993, 960), return_image=video, enable_picking=False, pyvista_mesh_topology=pv_mesh, camera = camera, y_plane = plane)
        images.append(image)
        elapsed = time.time() - start
        print(f"Visualized case {config['case_id']} ({elapsed:.2f} sec)...")

    import imageio.v2 as imageio
    fps = 2
    with imageio.get_writer(f"surface_analysis_video_{state}_{camera}_{model_name}.mp4", fps=fps, quality=10, pixelformat='yuvj444p') as writer:
        for img in images:
            # If screenshot has alpha channel (RGBA), drop it or composite it
            if img.shape[-1] == 4:
                img = img[..., :3]  # simplest: remove alpha

            # Ensure uint8
            img = np.asarray(img, dtype=np.uint8)
            writer.append_data(img)

if __name__ == "__main__":
    # ============================ Important Script Flags ============================:
    #                                               # Uncomment a line to:
    # load, train, save = False, True, True         # Train from scratch
    # load, train, save = True, True, True          # Train from existing0
    load, train, save = True, False, False        # Load results
    # load, train, save = False, False, False       # Load untrained model (for debugging)

    # DATA-ONLY MODEL:
    load_model_path = 'RANS_GNN_tune_sample_b_TEMP_BEST_61391.pickle'
    model_name = 'RANS-GNN'
    as_multifidelity_model = False
    dataset_dir, split = 'sample_case', 'split_0'

    main(
        load=load, 
        train=train, 
        save=save, 
        dataset_dir = dataset_dir,
        split = split,
        load_model_path=load_model_path,
        save_training_path='DEL',
        as_multifidelity_model=as_multifidelity_model,
        with_pinns=False,
        model_name=model_name
    )