import jax

import jax.numpy as jnp
import numpy as np

from typing import Callable

# model 
from .networks import build_network, ParameterInfo

from .utils import stacked_normalize

def get_panel_base(panel:dict, constants:dict):
    v_base = stacked_normalize(panel['U'], constants['v']['mean'], constants['v']['std'])
    T_base = stacked_normalize(panel['T'], constants['T']['mean'], constants['T']['std'])
    p_base = stacked_normalize(panel['p'], constants['p']['mean'], constants['p']['std'])
    nuTilda_base = stacked_normalize(panel['nuTilda'], constants['nuTilda']['mean'], constants['nuTilda']['std'])
    return v_base, T_base, p_base, nuTilda_base

def build_graph_features(graph:dict, states_input:dict, model_constants:dict, add_cell_sdf, return_nodes:bool=True):
    # unpack constants
    NODE_NUM_X, EDGE_NUM_X = model_constants['NODE_NUM_X'], model_constants['EDGE_NUM_X']

    # v_inf_sc, T_inf_sc, p_inf_sc, nuTilda_inf_sc = np.array([1.0,1.0,1.0]), 1.0, 1.0, 1.0
    v_input = states_input['v']
    t_input = states_input['T']
    p_input = states_input['p']
    nuTilda_input = states_input['nuTilda']

    # build edge and node feature vectors
    # - edge features: (num_edges, EDGE_NUM_X): [edge distance x, edge distance y, edge distance z, edge length, edge type one-hot (4)]
    # - node features: (num_nodes, NODE_NUM_X): 
    #   - freestream vx, vy, vz, p, T, nuTilda
    #   - node type 0: fluid
    #   - node type 2: wall
    #   - node type 1: freestream
    #   - node type 3: symmetry
    #   - volume? (not now)
    dual_edges, dual_cells = graph['dual_edges'], graph['dual_cells']
    dual_edge_types, dual_cell_types = graph['dual_edge_types'], graph['dual_cell_types']
    is_interior = graph['dual_cell_types'] == 0
    assert is_interior.shape == (dual_cells.shape[0],), f'Error: is_interior should have shape (num_nodes,), but got {is_interior.shape}'

    # Edge features:
    N_edges = dual_edges.shape[0]
    edge_1, edge_2 = dual_edges[:, 0], dual_edges[:, 1]
    edge_vector = dual_cells[edge_2] - dual_cells[edge_1]
    # print('edge_vector mean, std:', np.mean(edge_vector, axis=0), np.std(edge_vector, axis=0))
    # print('edge_length mean, std:', np.mean(np.linalg.norm(edge_vector, axis=1, keepdims=True), axis=0), np.std(np.linalg.norm(edge_vector, axis=1, keepdims=True), axis=0))
    # exit()
    assert edge_vector.shape == (N_edges, 3), f'Error: edge_vector should have shape (N_edges, 3), but got {edge_vector.shape}'
    edge_length = jnp.linalg.norm(edge_vector, axis=1, keepdims=True)
    eps = 1e-7
    edge_length = jnp.maximum(edge_length, eps)

    assert edge_length.shape == (N_edges, 1), f'Error: edge_length should have shape (N_edges, 1), but got {edge_length.shape}'
    known_types = [0, 1, 2, 3]  # fluid, wall, freestream, symmetry
    edge_one_hots = np.zeros((N_edges, len(known_types)))
    for et in known_types:
        edge_one_hots[:, et] = (dual_edge_types == et)
    assert edge_one_hots.shape == (N_edges, len(known_types)), f'Error: edge_one_hots should have shape (N_edges, {len(known_types)}), but got {edge_one_hots.shape}'
    edge_X_1 = jnp.concatenate([edge_vector/edge_length, edge_length, edge_one_hots], axis=1)
    edge_X_2 = jnp.concatenate([-edge_vector/edge_length, edge_length, edge_one_hots], axis=1)
    assert edge_X_1.shape == edge_X_2.shape == (N_edges, EDGE_NUM_X), f'Error: edge_X_1 should have shape (N_edges, {EDGE_NUM_X}), but got {edge_X_1.shape} and {edge_X_2.shape}'

    if not return_nodes:
        return edge_X_1, edge_X_2
    
    # If needed, node features:
    N_nodes = dual_cells.shape[0]

    node_one_hots = np.zeros((N_nodes, len(known_types)))
    for nt in known_types:
        node_one_hots[:, nt] = (dual_cell_types == nt)
    assert node_one_hots.shape == (N_nodes, len(known_types)), f'Error: node_one_hots should have shape (N_nodes, {len(known_types)}), but got {node_one_hots.shape}'
    node_X = jnp.concatenate([v_input, t_input[:, None], p_input[:, None], nuTilda_input[:, None], node_one_hots], axis=1)

    # (OPTIONAL) add centroids as node features (remember to change NODE_NUM_X)
    # centroids = graph['dual_cells']
    # node_X = jnp.concatenate([node_X, centroids/100], axis=1)
    # assert node_X.shape == (N_nodes, NODE_NUM_X), f'Error: node_X should have shape (N_nodes, {NODE_NUM_X + 3}), but got {node_X.shape}'

    # (OPTIONAL) add cell surface area as node features (remember to change NODE_NUM_X)

    # (OPTIONAL) add *APPROXIMATE* sdf (remember to change NODE_NUM_X)
    if add_cell_sdf:
        wall_pos = dual_cells[graph['closest_wall_id']]  # (N,3)
        diff = dual_cells - wall_pos
        # d = jnp.linalg.norm(diff, axis=1) #.astype(np.float32)   # OLD
        d = jnp.sqrt(jnp.sum(diff**2, axis=1) + eps)               # NEW
        sigma = 5
        
        # inputs
        d_tilde = 1.0 - jnp.exp(-d / sigma)
        # diff_unit = diff / (d[:, None] + 1e-8)   # OLD
        diff_unit = diff / (d[:, None] + eps)      # NEW
        node_X = jnp.concatenate([node_X, d_tilde[:, None], diff_unit], axis=1)
    assert node_X.shape == (N_nodes, NODE_NUM_X), f'Error: node_X should have shape (N_nodes, {NODE_NUM_X}), but got {node_X.shape}'

    return node_X, edge_X_1, edge_X_2

def pass_message(params, edge_1, edge_2, edge_X, cell_latents, processor_network, constants):
    # build inputs to processor network
    source_latents = cell_latents[edge_1]
    target_latents = cell_latents[edge_2]
    inputs = jnp.hstack([source_latents, target_latents, edge_X])  # (N_edges, 2*CELL_LATENT_DIM + 3)
    assert inputs.shape == (edge_X.shape[0], constants['PROCESSOR_IN_DIM']), f"Expected {(edge_X.shape[0], constants['PROCESSOR_IN_DIM'])}, got {inputs.shape}"
    
    # process messages
    # This has message from edge_1 to edge_2 for each edge
    messages = jax.vmap(lambda x: processor_network(params['processor_network'], x), in_axes=(0,))(inputs)  # (N_edges, CELL_LATENT_DIM)
    assert messages.shape == (edge_X.shape[0], constants['CELL_LATENT_DIM']), f"Expected {(edge_X.shape[0], constants['CELL_LATENT_DIM'])}, got {messages.shape}"

    # aggregate messages at TARGET nodes. We do NOT return to source nodes here OR GHOST NODES
    aggregated_messages = jnp.zeros((constants['N_nodes'], constants['CELL_LATENT_DIM']))  # placeholder
    aggregated_messages = aggregated_messages.at[edge_2].add(messages)  # sum aggregation
    return aggregated_messages

def pass_messages(
        N:int,
        processor_network:Callable,
        update_network:Callable,
        params:dict,
        cell_latents:jax.Array, edge_X_1:jax.Array, edge_X_2:jax.Array,
        graph:dict,
        model_constants:dict,
        checkpoint_mp:bool = True,
        checkpoint_iter:bool = True,
    )->jax.Array:
    dual_edges, dual_cells = graph['dual_edges'], graph['dual_cells']
    is_interior = graph['dual_cell_types'] == 0
    edge_1, edge_2 = dual_edges[:, 0], dual_edges[:, 1]

    # integers
    N_nodes = graph['nDualCells']
    CELL_LATENT_DIM, PROCESSOR_IN_DIM = model_constants['CELL_LATENT_DIM'], model_constants['PROCESSOR_IN_DIM']

    # checks:
    assert cell_latents.shape == (N_nodes, CELL_LATENT_DIM), f'Error: cell_latents should have shape ({N_nodes}, {CELL_LATENT_DIM}), but got {cell_latents.shape}'
    assert edge_X_1.shape == edge_X_2.shape == (edge_1.shape[0], PROCESSOR_IN_DIM - 2*CELL_LATENT_DIM), f'Error: edge_X_1 and edge_X_2 should have shape ({edge_1.shape[0]}, {PROCESSOR_IN_DIM - 2*CELL_LATENT_DIM}), but got {edge_X_1.shape} and {edge_X_2.shape}'

    # prep
    constants={'CELL_LATENT_DIM': CELL_LATENT_DIM, 'PROCESSOR_IN_DIM': PROCESSOR_IN_DIM, 'N_nodes': N_nodes}
    pass_message_temp = lambda params, edge_1, edge_2, edge_X, cell_latents: pass_message(params, edge_1, edge_2, edge_X, cell_latents, processor_network, constants)
    if checkpoint_mp:
        pass_message_temp = jax.checkpoint(pass_message_temp)  # checkpointing to save memory since we will recompute in backward pass anyway

    cell_latents_original = cell_latents
    original_ghost_latents = cell_latents_original[~is_interior]
    deg = graph['deg']

    def mp_step(cell_latents, _):
        latent_fwd  = pass_message_temp(params, edge_1, edge_2, edge_X_1, cell_latents)
        latent_back = pass_message_temp(params, edge_2, edge_1, edge_X_2, cell_latents)

        cell_latent_aggregates = (latent_fwd + latent_back) / deg[:, None]

        # weighted mean instead of boolean indexing (more JIT-friendly)
        g = jnp.mean(cell_latents[is_interior], axis=0) # (CELL_LATENT_DIM,)
        g_tiled = jnp.broadcast_to(g, (N_nodes, CELL_LATENT_DIM))  # (N_nodes, CELL_LATENT_DIM)
        assert g_tiled.shape == (N_nodes, CELL_LATENT_DIM), f'Error: g_tiled should have shape (N_nodes, {CELL_LATENT_DIM}), but got {g_tiled.shape}'

        update_input = jnp.hstack([cell_latents, cell_latent_aggregates, g_tiled])
        delta = jax.vmap(lambda x: update_network(params['update_network'], x))(update_input)

        new_latents = cell_latents + delta

        # reset ghost cells without fancy indexing
        # new_latents = new_latents.at[~is_interior].set(original_ghost_latents)
        return new_latents, None

    if checkpoint_iter:
        mp_step = jax.checkpoint(mp_step)   # biggest memory saver
    for _ in range(N):
        cell_latents, _ = mp_step(cell_latents, None)
    return cell_latents

def latent_pool(fine_latents:jax.Array, coarse_graph:dict):
    K = coarse_graph['nDualCells']
    counts = coarse_graph['counts']
    mapping = coarse_graph['mapping']
    sums = jax.ops.segment_sum(fine_latents, mapping, num_segments=K)
    coarse_cell_latent = sums / counts[:, None]
    assert coarse_cell_latent.shape == (K, fine_latents.shape[1]), f'Error: coarse_cell_latent should have shape ({K}, {fine_latents.shape[1]}), but got {coarse_cell_latent.shape}'
    return coarse_cell_latent

def latent_unpool_bcast(finer_latents_up:jax.Array, coarser_latents:jax.Array, coarser_graph:dict, bcast_network:Callable, params:dict):
    finer_cell_unpooled = coarser_latents[coarser_graph['mapping']]
    inp = jnp.concatenate([finer_latents_up, finer_cell_unpooled], axis=-1)
    delta = jax.vmap(lambda x: bcast_network(params['broadcast_network'], x), in_axes=(0,))(inp)
    finer_cell_latent = finer_latents_up + delta
    # finer_cell_latent = delta
    assert finer_cell_latent.shape == finer_latents_up.shape, f'Error: finer_cell_latent should have shape {finer_latents_up.shape}, but got {finer_cell_latent.shape}'
    return finer_cell_latent

def get_model(constants:dict[str], multifidelity_model:bool = False, add_cell_sdf:bool = True, root:str=None):
    NODE_NUM_X, EDGE_NUM_X = 10, 8
    if add_cell_sdf: NODE_NUM_X += 4
    CELL_LATENT_DIM = 45
    PROCESSOR_IN_DIM = 2*CELL_LATENT_DIM + EDGE_NUM_X
    model_constants={'CELL_LATENT_DIM': CELL_LATENT_DIM, 'PROCESSOR_IN_DIM': PROCESSOR_IN_DIM, 'NODE_NUM_X': NODE_NUM_X, 'EDGE_NUM_X': EDGE_NUM_X}

    # Architecture:
    # 1 fine_X = feature_encoder(node_features)     (NN0)
    # 2 fine_XA = MP_L0(fine_X)                     (NN1,NN2)
    #       mid_XA = POOL_L1(fine_XA)               (Averaging)
    #       mid_XA = MP_L1(mid_XA)                  (NN1,NN2)
    #           coarse_XA = POOL_L2(mid_XA)         (Averaging)
    #           coarse_XA = MP_L2(coarse_XA)        (NN1,NN2)
    #       mid_XB = UNPOOL_L1(coarse_XA, mid_XA)   (Bcast)
    #       mid_XB = MP_L1(mid_XB)                  (NN1,NN2)
    # 2 fine_XB = UNPOOL_L0(mid_XB, fine_XA)        (Bcast)
    # 3 output = field_decoder(fine_XB)             (NN3)

    encoder_network, encoder_params_init = build_network('encoder_network', layers = [NODE_NUM_X, 48, 48, CELL_LATENT_DIM], activation = jax.nn.silu, weight_normalization=False)
    processor_network, processor_params_init = build_network('processor_network', layers = [PROCESSOR_IN_DIM, 85, 85, CELL_LATENT_DIM], activation = jax.nn.silu, weight_normalization=False)
    update_network, update_params_init = build_network('update_network', layers = [3*CELL_LATENT_DIM, 85, 85, CELL_LATENT_DIM], activation = jax.nn.silu, weight_normalization=False)
    broadcast_network, broadcast_params_init = build_network('broadcast_network', layers = [2*CELL_LATENT_DIM, 48, 48, CELL_LATENT_DIM], activation = jax.nn.silu, weight_normalization=False)
    decoder_network, decoder_params_init = build_network('decoder_network', layers = [CELL_LATENT_DIM, 48, 48, 6], activation = jax.nn.silu, weight_normalization=False)

    def predict(params:dict, flow_condition:dict, graph_points:dict, geometry:dict)->tuple:
        print("COMPILING MODEL...")
        # Set up:
        geometry['foam_mesh']['points'] = graph_points['foam_mesh']['points']
        geometry['foam_mesh']['cells']['centroids'] = graph_points['foam_mesh']['cells']['centroids']
        for graph_key in geometry['hierarchy']:
            geometry[graph_key]['dual_cells'] = graph_points[graph_key]['dual_cells']

        # Normalize freestream conditions and add to node features
        if not multifidelity_model:
            v_inf_sc = flow_condition['v_inf'] / constants['v']['ref']
            T_inf_sc = flow_condition['T_inf'] / constants['T']['ref']
            p_inf_sc = flow_condition['p_inf']  / constants['p']['ref']
            nuTilda_inf_sc = flow_condition['nuTilda_inf'] / constants['nuTilda']['ref']
            # set y and z velocity == 1
            # nd_state_input['v_inf'] = nd_state_input['v_inf'].at[([1,2])].set(1.0)
            # v_inf_sc = v_inf_sc.at[1:].set(1.0)

            # v_inf_sc = jnp.ones(3)
            # T_inf_sc, p_inf_sc, nuTilda_inf_sc = 1.0, 1.0, 1.0
            N_nodes = geometry['dual_graph']['nDualCells']
            v_inf, T_inf, p_inf, nuTilda_inf = jnp.broadcast_to(v_inf_sc, (N_nodes, 3)), jnp.ones(N_nodes)*T_inf_sc, jnp.ones(N_nodes)*p_inf_sc, jnp.ones(N_nodes)*nuTilda_inf_sc
            nd_state_input = {
                'v': v_inf,
                'T': T_inf,
                'p': p_inf,
                'nuTilda': nuTilda_inf,
            }

        else:
            # nd_state_input = {
            #     'v': graph_points['panel']['v'] / constants['v']['ref'],
            #     'T': graph_points['panel']['T'] / constants['T']['ref'],
            #     'p': graph_points['panel']['p'] / constants['p']['ref'],
            #     'nuTilda': graph_points['panel']['nuTilda'] / constants['nuTilda']['ref'],
            # }
            nd_state_input = {
                'v': graph_points['low_fidelity_inputs']['U'] / constants['v']['ref'],
                'T': graph_points['low_fidelity_inputs']['T'] / constants['T']['ref'],
                'p': graph_points['low_fidelity_inputs']['p'] / constants['p']['ref'],
                'nuTilda': graph_points['low_fidelity_inputs']['nuTilda'] / constants['nuTilda']['ref'],
            }

        nCells = geometry['foam_mesh']['cells']['nCells']
        node_X, edge_X_1, edge_X_2 = build_graph_features(geometry['dual_graph'], nd_state_input, model_constants, add_cell_sdf)
        edge_Xl1_1, edge_Xl1_2 = build_graph_features(geometry['coarse_graph_l1'], nd_state_input, model_constants, add_cell_sdf, return_nodes=False)
        edge_Xl2_1, edge_Xl2_2 = build_graph_features(geometry['coarse_graph_l2'], nd_state_input, model_constants, add_cell_sdf, return_nodes=False)

        def MP(N, cell_X, edge_X_1, edge_X_2, graph, checkpoint_mp=True, checkpoint_iter=True):
            return pass_messages(N, processor_network, update_network, params, cell_X, edge_X_1, edge_X_2, graph, model_constants, checkpoint_mp, checkpoint_iter)
        
        def latent_unpool(finer_latents_up, coarser_latents, coarser_graph):
            return latent_unpool_bcast(finer_latents_up, coarser_latents, coarser_graph, broadcast_network, params)

        # ================== 1 Initial encoding of cell features
        l0_cell_X_up = jax.vmap(lambda x: encoder_network(params['encoder_network'], x), in_axes=(0,))(node_X)

        # ================== 2 Main U-shaped message passing with pooling and unpooling ==================
        # up-scale fine
        l0_cell_X_up = MP(1, l0_cell_X_up, edge_X_1, edge_X_2, geometry['dual_graph'])

        # up-scale mid
        l1_cell_X_up = latent_pool(l0_cell_X_up, geometry['coarse_graph_l1'])
        l1_cell_X_up = MP(3, l1_cell_X_up, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1']) 

        # up/down-scale coarse
        l2_cell_X = latent_pool(l1_cell_X_up, geometry['coarse_graph_l2'])
        l2_cell_X = MP(5, l2_cell_X, edge_Xl2_1, edge_Xl2_2, geometry['coarse_graph_l2'])  # no checkpointing at coarsest level since it's cheap

        # down-scale mid
        l1_cell_X_down = latent_unpool(l1_cell_X_up, l2_cell_X, geometry['coarse_graph_l2'])
        l1_cell_X_down = MP(3, l1_cell_X_down, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1'])

        # down-scale fine
        l0_cell_X_down = latent_unpool(l0_cell_X_up, l1_cell_X_down, geometry['coarse_graph_l1'])
        l0_cell_X_down = MP(1, l0_cell_X_down, edge_X_1, edge_X_2, geometry['dual_graph'])

        # # up-scale fine
        # l0_cell_X_up = MP(1, l0_cell_X_up, edge_X_1, edge_X_2, geometry['dual_graph'])

        # # up-scale mid
        # l1_cell_X_up = latent_pool(l0_cell_X_up, geometry['coarse_graph_l1'])
        # l1_cell_X_up = MP(3, l1_cell_X_up, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1']) 

        # # # # up/down-scale coarse
        # # # l2_cell_X = latent_pool(l1_cell_X_up, geometry['coarse_graph_l2'])
        # # # l2_cell_X = MP(7, l2_cell_X, edge_Xl2_1, edge_Xl2_2, geometry['coarse_graph_l2'])  # no checkpointing at coarsest level since it's cheap

        # # # down-scale mid
        # # l1_cell_X_down = latent_unpool(l1_cell_X_up, l2_cell_X, geometry['coarse_graph_l2'])
        # l1_cell_X_down = MP(3, l1_cell_X_up, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1'])

        # # # down-scale fine
        # l0_cell_X_down = latent_unpool(l0_cell_X_up, l1_cell_X_down, geometry['coarse_graph_l1'])
        # l0_cell_X_down = MP(1, l0_cell_X_down, edge_X_1, edge_X_2, geometry['dual_graph'])

        # ================== 3 Final decoding of cell features to get field predictions
        is_interior = geometry['dual_graph']['dual_cell_types'] == 0
        primitive_predictions = jax.vmap(lambda x: decoder_network(params['decoder_network'], x), in_axes=(0,))(l0_cell_X_down[is_interior])
        assert primitive_predictions.shape == (nCells, 6), f"Expected ({nCells},6), got {primitive_predictions.shape}"

        # predict corrections to freestream state variables
        delta = primitive_predictions  # tunable
        if not multifidelity_model:
            v = delta[:, :3]
            T = delta[:, 3]
            p = delta[:, 4]
            nuTilda = delta[:, 5]
        else:
            v_panel_base, T_panel_base, p_panel_base, nuTilda_panel_base = get_panel_base(graph_points['low_fidelity_inputs'], constants)
            v = delta[:, :3] + v_panel_base[is_interior]
            T = delta[:, 3] + T_panel_base[is_interior]
            p = delta[:, 4] + p_panel_base[is_interior]
            nuTilda = delta[:, 5] + nuTilda_panel_base[is_interior]

            # v = delta[:, :3]
            # T = delta[:, 3]
            # p = delta[:, 4]
            # nuTilda = delta[:, 5]
        return v, T, p, nuTilda

    param_info = ParameterInfo('all_params', root)
    param_info.add(encoder_params_init)
    param_info.add(processor_params_init)
    param_info.add(decoder_params_init)
    param_info.add(update_params_init)
    param_info.add(broadcast_params_init)
    return predict, param_info
