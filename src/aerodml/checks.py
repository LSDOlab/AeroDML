import jax
from typing import Callable
from .utils import print_content
import time
import numpy as np

def check_model(model, params, config, constants, num_runs:int = 30):
    # extract inputs to GPU
    geometry = config['topology']
    pts = jax.device_put(config['warped_geo'], device=constants['device'])
    freestream = jax.device_put(config['flow_condition'], device=constants['device'])
    
    # Compile and run the model once to get output shapes and timing 
    single_model = jax.jit(lambda params, flow, graph_points: model(params, flow, graph_points, geometry), device=constants['device'])
    # single_model = (lambda params, graph_points: model(params, graph_points, geometry))
    start_compile = time.time()
    test = single_model(params, freestream, pts)  # TODO: replace None with actual vertices input
    test[0].block_until_ready()
    end_compile = time.time()
    compile_time = end_compile-start_compile
    print(f'[MODEL CHECK] Compile (+run) time: {compile_time:.4f} seconds \t ')

    # time the model over multiple runs to get average inference time
    outputs = []
    start = time.time()

    for i in range(num_runs):
        single_time = time.time()

        test = single_model(params, freestream, pts)
        test[0].block_until_ready()

        single_time = time.time() - single_time
        outputs.append(np.asarray(test[0]))

        print(
            f'[MODEL CHECK] Run {i+1}/{num_runs} time: {single_time:.4f} seconds \t '
            f'avg output[0]:{np.mean(test[0]):.6f}, '
            f'max output[0]:{np.max(test[0]):.6f}, '
            f'min output[0]:{np.min(test[0]):.6f}'
        )

    end = time.time()

    # nondeterminism relative to first measured run
    reference = outputs[0]
    max_abs_diff = max(np.max(np.abs(out - reference)) for out in outputs[1:])
    mean_abs_diff = np.mean([np.mean(np.abs(out - reference)) for out in outputs[1:]])

    print('====================== Model Stats ====================== ')
    print(f'Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}')
    print(f'Model output shape: {test[0].shape}, {test[1].shape}, {test[2].shape}, {test[3].shape}')
    print(f'Average inference time over {num_runs} runs: {(end - start)/num_runs:.3f} seconds \t(compile={compile_time:.3f})')
    print(f'Non-determinism: max abs diff = {max_abs_diff:.8e}, mean abs diff = {mean_abs_diff:.8e}')
    print('====================== Model Stats ======================\n')

def check_res_func(res_func:Callable, params:dict, config:dict, constants:dict, grad:bool = True, pinn_loss:bool=False):
    # extract inputs to GPU
    geometry = config['topology']
    data = jax.device_put(config['data'], device=constants['device'])
    pts = jax.device_put(config['warped_geo'], device=constants['device'])
    freestream = jax.device_put(config['flow_condition'], device=constants['device'])

    # Compile and run the residual function once to get output shapes and timing
    residual_func = jax.jit(lambda params, data_config, flow_conditions, graph_config: res_func(params, data_config, flow_conditions, graph_config, geometry, constants, compute_pinn_loss=pinn_loss), device=constants['device'])
    start_compile = time.time()
    test_res = residual_func(params, data, freestream, pts)
    test_res[0].block_until_ready()
    compile_time = time.time() - start_compile
    print(f'[RES CHECK] Compile (+run) time: {compile_time:.4f} seconds \t ')

    if grad:
        df_dx_func = jax.jit(jax.grad(lambda params, data_config, flow_conditions, graph_config: res_func(params, data_config, flow_conditions, graph_config, geometry, constants, compute_pinn_loss=pinn_loss)[0], argnums=0), device=constants['device'])
        start_compile_adj = time.time()
        grads = df_dx_func(params, data, freestream, pts)
        jax.tree_util.tree_leaves(grads)[0].block_until_ready()
        end_compile_adj = time.time()
        compile_time_adj = end_compile_adj-start_compile_adj
        print(f'[RES CHECK] Adjoint compile (+run) time: {compile_time_adj:.4f} seconds \t ')

    num_runs = 20
    outputs = []
    start = time.time()
    for i in range(num_runs):
        data = jax.device_put(config['data'], device=constants['device'])
        pts = jax.device_put(config['warped_geo'], device=constants['device'])
        single_time = time.time()

        test_res = residual_func(params, data, freestream, pts)
        test_res[0].block_until_ready()
        outputs.append(np.asarray(test_res[0]))

        print(
            f'[RES CHECK] Run {i+1}/{num_runs} time: {time.time() - single_time:.4f} seconds \t '
            f'avg residual:{np.mean(test_res[0]):.6f}, '
            f'max residual:{np.max(test_res[0]):.6f}, '
            f'min residual:{np.min(test_res[0]):.6f}'
        )
    end = time.time()

    reference = outputs[0]
    max_abs_diff = max(np.max(np.abs(out - reference)) for out in outputs[1:])
    mean_abs_diff = np.mean([np.mean(np.abs(out - reference)) for out in outputs[1:]])

    if grad:
        grad_outputs = []
        start_adj = time.time()
        for i in range(num_runs):
            data = jax.device_put(config['data'], device=constants['device'])
            pts = jax.device_put(config['warped_geo'], device=constants['device'])
            single_time = time.time()

            grads = df_dx_func(params, data, freestream, pts)
            grad_leaves = jax.tree_util.tree_leaves(grads)
            grad_leaves[0].block_until_ready()
            grad_flat = np.concatenate([np.asarray(x).ravel() for x in grad_leaves])
            grad_outputs.append(grad_flat)

            print(f'[RES CHECK] Adjoint {i+1}/{num_runs} time: {time.time() - single_time:.4f} seconds')
        end_adj = time.time()

        grad_reference = grad_outputs[0]
        grad_diffs = [np.abs(out - grad_reference) for out in grad_outputs[1:]]
        grad_max_abs_diff = max(np.max(diff) for diff in grad_diffs)
        grad_mean_abs_diff = np.mean([np.mean(diff) for diff in grad_diffs])
        grad_rms_diff = np.sqrt(np.mean([np.mean(diff**2) for diff in grad_diffs]))

    print('====================== Model Stats ====================== ')
    print_content(test_res[1], 0)
    print(f'Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}')
    print(f'Average residual evaluation time over {num_runs} runs: {(end - start)/num_runs:.4f} seconds \t(compile={compile_time:.3f})')
    print(f'Residual non-determinism: max abs diff = {max_abs_diff:.8e}, mean abs diff = {mean_abs_diff:.8e}')
    if grad:
        print(f'Average adjoint (gradient) time over {num_runs} runs: {(end_adj - start_adj)/num_runs:.4f} seconds \t(compile={compile_time_adj:.3f})')
        print(f'Adjoint non-determinism: max abs diff = {grad_max_abs_diff:.8e}, mean abs diff = {grad_mean_abs_diff:.8e}')
        print(f'Gradients checked: {len(grad_leaves)} arrays and {grad_reference.size} scalars')
        print(f'Adjoint non-determinism: max={grad_max_abs_diff:.8e}, mean={grad_mean_abs_diff:.8e}, RMS={grad_rms_diff:.8e}')
    else: print(f'Gradient check skipped.')
    print('====================== Model Stats ======================\n')