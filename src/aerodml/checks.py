import jax
from typing import Callable
from .utils import print_content
import time

def check_model(model, params, config, constants, num_runs:int = 30):
    # extract inputs to GPU
    geometry = config['topology']
    pts = jax.device_put(config['warped_geo'], device=constants['device'])
    freestream = jax.device_put(config['flow_condition'], device=constants['device'])
    
    # Compile and run the model once to get output shapes and timing 
    single_model = jax.jit(lambda params, flow, graph_points: model(params, flow, graph_points, geometry), device=constants['device'])
    # single_model = (lambda params, graph_points: model(params, graph_points, geometry))
    test = single_model(params, freestream, pts)  # TODO: replace None with actual vertices input
    
    # time the model over multiple runs to get average inference time
    start = time.time()
    for _ in range(num_runs):
        test = single_model(params, freestream, pts)  # TODO: replace None with actual vertices input
        test[0].block_until_ready()
    end = time.time()
    
    print('====================== Model Stats ====================== ')
    print(f'Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}')
    print(f'Model output shape: {test[0].shape}, {test[1].shape}, {test[2].shape}, {test[3].shape}')
    print(f'Average inference time over {num_runs} runs: {(end - start)/num_runs:.4f} seconds')
    print('====================== Model Stats ======================\n')

def check_res_func(res_func:Callable, params:dict, config:dict, constants:dict, grad:bool = True, pinn_loss:bool=False):
    # extract inputs to GPU
    geometry = config['topology']
    data = jax.device_put(config['data'], device=constants['device'])
    pts = jax.device_put(config['warped_geo'], device=constants['device'])
    freestream = jax.device_put(config['flow_condition'], device=constants['device'])

    # Compile and run the residual function once to get output shapes and timing
    residual_func = jax.jit(lambda params, data_config, flow_conditions, graph_config: res_func(params, data_config, flow_conditions, graph_config, geometry, constants, compute_pinn_loss=pinn_loss), device=constants['device'])
    residual_func(params, data, freestream, pts)
    if grad:
        df_dx_func = jax.jit(jax.grad(lambda params, data_config, flow_conditions, graph_config: res_func(params, data_config, flow_conditions, graph_config, geometry, constants, compute_pinn_loss=pinn_loss)[0]), device=constants['device'])
        df_dx_func(params, data, freestream, pts)

    num_runs = 20
    start = time.time()
    for i in range(num_runs):
        data = jax.device_put(config['data'], device=constants['device'])
        pts = jax.device_put(config['warped_geo'], device=constants['device'])
        test_res = residual_func(params, data, freestream, pts)
        test_res[0].block_until_ready()
    end = time.time()
    
    if grad:
        start_adj = time.time()
        for i in range(num_runs):
            data = jax.device_put(config['data'], device=constants['device'])
            pts = jax.device_put(config['warped_geo'], device=constants['device'])
            grads = df_dx_func(params, data, freestream, pts)
            jax.tree_util.tree_leaves(grads)[0].block_until_ready()
        end_adj = time.time()

    print('====================== Model Stats ====================== ')
    print_content(test_res[1], 0)
    print(f'Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}')
    print(f'Average residual evaluation time over {num_runs} runs: {(end - start)/num_runs:.4f} seconds')
    if grad:
        print(f'Average adjoint (gradient) time over {num_runs} runs: {(end_adj - start_adj)/num_runs:.4f} seconds')
    else: print(f'Gradient check skipped.')
    print('====================== Model Stats ======================\n')

    # num_runs = 100
    # # start = time.time()
    # # for i in range(num_runs):
    # #     data = jax.device_put(config['data'], device=constants['device'])
    # #     pts = jax.device_put(config['warped_geo'], device=constants['device'])
    # #     pts['coarse_graph_l1']['dual_cells'] = pts['coarse_graph_l1']['dual_cells'] + 0.001 * i 
    # #     test_res = residual_func(params, data, pts)
    # #     test_res[0].block_until_ready()
    # # end = time.time()
    
    # data = jax.device_put(config['data'], device=constants['device'])
    # pts = jax.device_put(config['warped_geo'], device=constants['device'])

    # start_adj = time.time()
    # for i in range(num_runs):

    #     jax.tree_util.tree_leaves(pts)[0].block_until_ready().block_until_ready()
    #     jax.tree_util.tree_leaves(data)[0].block_until_ready().block_until_ready()
    #     start_iter = time.time()
    #     pts['coarse_graph_l1']['dual_cells'] = pts['coarse_graph_l1']['dual_cells'] + 0.00000000000001 * i 
    #     grads = df_dx_func(params, data, pts)
    #     jax.block_until_ready(grads)
    #     print(f'Adjoint run {i+1}/{num_runs} time: {time.time() - start_iter:.4f} seconds')

    # end_adj = time.time()