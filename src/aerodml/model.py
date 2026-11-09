import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np

from typing import Callable

# model 
from .networks import build_network, ParameterInfo

from .utils import stacked_normalize, aggregate_sum, build_aggregate_info, print_content

def get_panel_base(panel:dict, constants:dict):
    v_base = jnp.asarray(stacked_normalize(jnp.asarray(panel['U'], dtype=jnp.float32), constants['v']['mean'], constants['v']['std']), dtype=jnp.float32)
    T_base = jnp.asarray(stacked_normalize(jnp.asarray(panel['T'], dtype=jnp.float32), constants['T']['mean'], constants['T']['std']), dtype=jnp.float32)
    p_base = jnp.asarray(stacked_normalize(jnp.asarray(panel['p'], dtype=jnp.float32), constants['p']['mean'], constants['p']['std']), dtype=jnp.float32)
    nuTilda_base = jnp.asarray(stacked_normalize(jnp.asarray(panel['nuTilda'], dtype=jnp.float32), constants['nuTilda']['mean'], constants['nuTilda']['std']), dtype=jnp.float32)
    return v_base, T_base, p_base, nuTilda_base

def build_graph_features(geometry:dict, graph_points:dict, states_input:dict, model_constants:dict, add_cell_sdf, return_nodes:bool=True):
    # unpack constants
    NODE_NUM_X, EDGE_NUM_X = model_constants['NODE_NUM_X'], model_constants['EDGE_NUM_X']
    DTYPE = model_constants['DTYPE']

    # v_inf_sc, T_inf_sc, p_inf_sc, nuTilda_inf_sc = np.array([1.0,1.0,1.0]), 1.0, 1.0, 1.0
    v_input = jnp.asarray(states_input['v'], dtype=DTYPE)
    t_input = jnp.asarray(states_input['T'], dtype=DTYPE)
    p_input = jnp.asarray(states_input['p'], dtype=DTYPE)
    nuTilda_input = jnp.asarray(states_input['nuTilda'], dtype=DTYPE)

    # build edge and node feature vectors
    # - edge features: (num_edges, EDGE_NUM_X): [edge distance x, edge distance y, edge distance z, edge length, edge type one-hot (4)]
    # - node features: (num_nodes, NODE_NUM_X): 
    #   - freestream vx, vy, vz, p, T, nuTilda
    #   - node type 0: fluid
    #   - node type 1: wall
    #   - node type 2: freestream
    #   - node type 3: symmetry
    #   - volume? (not now)
    dual_edges = geometry['dual_edges']
    dual_cells = jnp.asarray(graph_points['dual_cells'], dtype=DTYPE)
    dual_edge_types, dual_cell_types = geometry['dual_edge_types'], geometry['dual_cell_types']
    is_interior = geometry['dual_cell_types'] == 0
    assert is_interior.shape == (dual_cells.shape[0],), f'Error: is_interior should have shape (num_nodes,), but got {is_interior.shape}'

    # Edge features:
    N_edges = dual_edges.shape[0]
    edge_1, edge_2 = dual_edges[:, 0], dual_edges[:, 1]
    edge_vector = dual_cells[edge_2] - dual_cells[edge_1]
    # print('edge_vector mean, std:', np.mean(edge_vector, axis=0), np.std(edge_vector, axis=0))
    # print('edge_length mean, std:', np.mean(np.linalg.norm(edge_vector, axis=1, keepdims=True), axis=0), np.std(np.linalg.norm(edge_vector, axis=1, keepdims=True), axis=0))
    # exit()
    assert edge_vector.shape == (N_edges, 3), f'Error: edge_vector should have shape (N_edges, 3), but got {edge_vector.shape}'
    eps = jnp.asarray(1e-7, dtype=edge_vector.dtype)
    edge_length = jnp.sqrt(jnp.sum(edge_vector * edge_vector, axis=1, keepdims=True) + eps**2)

    assert edge_length.shape == (N_edges, 1), f'Error: edge_length should have shape (N_edges, 1), but got {edge_length.shape}'
    known_types = [0, 1, 2, 3]  # fluid, wall, freestream, symmetry
    edge_one_hots = np.zeros((N_edges, len(known_types)), dtype=np.float32)
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

    node_one_hots = np.zeros((N_nodes, len(known_types)), dtype=np.float32)
    for nt in known_types:
        node_one_hots[:, nt] = (dual_cell_types == nt)
    assert node_one_hots.shape == (N_nodes, len(known_types)), f'Error: node_one_hots should have shape (N_nodes, {len(known_types)}), but got {node_one_hots.shape}'
    node_X = jnp.concatenate([v_input, t_input[:, None], p_input[:, None], nuTilda_input[:, None], node_one_hots], axis=1)

    # (OPTIONAL) add centroids as node features (remember to change NODE_NUM_X)
    # centroids = graph_points['dual_cells']
    # node_X = jnp.concatenate([node_X, centroids/100], axis=1)
    # assert node_X.shape == (N_nodes, NODE_NUM_X), f'Error: node_X should have shape (N_nodes, {NODE_NUM_X + 3}), but got {node_X.shape}'

    # (OPTIONAL) add *APPROXIMATE* sdf (remember to change NODE_NUM_X)
    if add_cell_sdf:
        wall_pos = dual_cells[geometry['closest_wall_id']]  # (N,3)
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

def pass_message(params, edge_1, edge_2, edge_X, cell_latents, processor_network, constants, aggregate_method, aggregate_info):
    # build inputs to processor network
    source_latents = cell_latents[edge_1]
    target_latents = cell_latents[edge_2]
    inputs = jnp.hstack([source_latents, target_latents, edge_X])  # (N_edges, 2*CELL_LATENT_DIM + 3)
    assert inputs.shape == (edge_X.shape[0], constants['PROCESSOR_IN_DIM']), f"Expected {(edge_X.shape[0], constants['PROCESSOR_IN_DIM'])}, got {inputs.shape}"
    
    # process messages
    # This has message from edge_1 to edge_2 for each edge
    messages = jax.vmap(lambda x: processor_network(params['processor_network'], x), in_axes=(0,))(inputs)  # (N_edges, CELL_LATENT_DIM)
    assert messages.shape == (edge_X.shape[0], constants['CELL_LATENT_DIM']), f"Expected {(edge_X.shape[0], constants['CELL_LATENT_DIM'])}, got {messages.shape}"

    # aggregate messages at TARGET nodes.
    # aggregated_messages = jnp.zeros((constants['N_nodes'], constants['CELL_LATENT_DIM']))  # placeholder
    # aggregated_messages = aggregated_messages.at[edge_2].add(messages)  # sum aggregation
    # aggregated_messages = jax.lax.fori_loop(0, messages.shape[0], lambda i, a: a.at[edge_2[i]].set(a[edge_2[i]] + messages[i]), aggregated_messages)
    
    aggregated_messages = aggregate_sum(messages, edge_2, constants['N_nodes'], method=aggregate_method, info=aggregate_info)
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
        aggregate_method:str = 'grouped',
    )->jax.Array:
    dual_edges = graph['dual_edges']
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
    aggregate_info_fwd = build_aggregate_info(edge_2, N_nodes) if aggregate_method == 'grouped' else None
    aggregate_info_bwd = build_aggregate_info(edge_1, N_nodes) if aggregate_method == 'grouped' else None
    pass_message_fwd_temp = lambda params, edge_X, cell_latents: pass_message(params, edge_1, edge_2, edge_X, cell_latents, processor_network, constants, aggregate_method, aggregate_info_fwd)
    pass_message_bwd_temp = lambda params, edge_X, cell_latents: pass_message(params, edge_2, edge_1, edge_X, cell_latents, processor_network, constants, aggregate_method, aggregate_info_bwd)
    if checkpoint_mp:
        pass_message_fwd_temp = jax.checkpoint(pass_message_fwd_temp)  # checkpointing to save memory since we will recompute in backward pass anyway
        pass_message_bwd_temp = jax.checkpoint(pass_message_bwd_temp)  # checkpointing to save memory since we will recompute in backward pass anyway
    
    cell_latents_original = cell_latents
    original_ghost_latents = cell_latents_original[~is_interior]
    deg = graph['deg']

    def mp_step(cell_latents, _):
        latent_fwd  = pass_message_fwd_temp(params, edge_X_1, cell_latents)
        # latent_back  = pass_message_fwd_temp(params, edge_X_2, cell_latents)
        latent_back = pass_message_bwd_temp(params, edge_X_2, cell_latents)

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

def latent_pool(fine_latents:jax.Array, coarse_graph:dict, aggregate_method:str='grouped'):
    K = coarse_graph['nDualCells']
    counts = coarse_graph['counts']
    mapping = coarse_graph['mapping']
    # sums = jax.ops.segment_sum(fine_latents, mapping, num_segments=K)
    sums = aggregate_sum(fine_latents, mapping, K, method=aggregate_method, info=None)
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

def get_model(constants:dict[str], multifidelity_model:bool = False, add_cell_sdf:bool = True, root:str=None, model_config:dict=None):
    # Defaults reproduce the current model.
    default_model_config = {
        'cell_latent_dim': 45,
        'encoder_width': 48,
        'processor_width': 85,
        'broadcast_width': 48,
        'decoder_width': 48,
        'mlp_hidden_layers': 2,
        'mp_steps': (1, 3, 5, 3, 1),
        'share_mp_weights': False,
        # 'share_mp_weights': True,
        'aggregate_method': 'grouped',
    }
    if model_config is None:
        model_config = {}
    model_config = {**default_model_config, **model_config}

    DTYPE = jnp.float32
    NODE_NUM_X, EDGE_NUM_X = 10, 8
    if add_cell_sdf: NODE_NUM_X += 4
    SURFACE_NODE_NUM_X = NODE_NUM_X + 4  # surface area and unit normal
    CELL_LATENT_DIM = model_config['cell_latent_dim']
    PROCESSOR_IN_DIM = 2*CELL_LATENT_DIM + EDGE_NUM_X
    ENCODER_WIDTH = model_config['encoder_width']
    PROCESSOR_WIDTH = model_config['processor_width']
    BROADCAST_WIDTH = model_config['broadcast_width']
    DECODER_WIDTH = model_config['decoder_width']
    MLP_HIDDEN_LAYERS = model_config['mlp_hidden_layers']
    SHARE_MP_WEIGHTS = model_config['share_mp_weights']
    AGGREGATE_METHOD = model_config['aggregate_method']
    assert AGGREGATE_METHOD in ('grouped', 'sequential', 'scatter'), f'Unknown aggregate method: {AGGREGATE_METHOD}'
    MP_STEPS = tuple(model_config['mp_steps'])
    assert len(MP_STEPS) == 5, f'Expected 5 MP step counts, got {len(MP_STEPS)}'
    MP_L0_UP, MP_L1_UP, MP_L2, MP_L1_DOWN, MP_L0_DOWN = MP_STEPS

    model_constants={
        'CELL_LATENT_DIM': CELL_LATENT_DIM,
        'PROCESSOR_IN_DIM': PROCESSOR_IN_DIM,
        'NODE_NUM_X': NODE_NUM_X,
        'EDGE_NUM_X': EDGE_NUM_X,
        'DTYPE': DTYPE,
    }

    # Architecture:
    # 1 fine_X = type-specific feature encoders(node_features) (NN0)
    # 2 fine_XA = MP_L0(fine_X)                     (NN1,NN2)
    #       mid_XA = POOL_L1(fine_XA)               (Averaging)
    #       mid_XA = MP_L1(mid_XA)                  (NN1,NN2)
    #           coarse_XA = POOL_L2(mid_XA)         (Averaging)
    #           coarse_XA = MP_L2(coarse_XA)        (NN1,NN2)
    #       mid_XB = UNPOOL_L1(coarse_XA, mid_XA)   (Bcast)
    #       mid_XB = MP_L1(mid_XB)                  (NN1,NN2)
    # 2 fine_XB = UNPOOL_L0(mid_XB, fine_XA)        (Bcast)
    # 3 output = field_decoder(fine_XB)             (NN3)

    def make_layers(in_dim, width, out_dim):
        return [in_dim] + [width]*MLP_HIDDEN_LAYERS + [out_dim]

    encoder_network, encoder_params_init = build_network('encoder_network', layers=make_layers(NODE_NUM_X, ENCODER_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
    surface_encoder_network, surface_encoder_params_init = build_network('surface_encoder_network', layers=make_layers(SURFACE_NODE_NUM_X, ENCODER_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)

    if SHARE_MP_WEIGHTS:
        processor_network_shared, processor_params_init_shared = build_network('processor_network_shared', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_shared, update_params_init_shared = build_network('update_network_shared', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)

        processor_network_l0_up = processor_network_l1_up = processor_network_l2 = processor_network_l1_down = processor_network_l0_down = processor_network_shared
        update_network_l0_up = update_network_l1_up = update_network_l2 = update_network_l1_down = update_network_l0_down = update_network_shared

        processor_params_key_l0_up = processor_params_key_l1_up = processor_params_key_l2 = processor_params_key_l1_down = processor_params_key_l0_down = 'processor_network_shared'
        update_params_key_l0_up = update_params_key_l1_up = update_params_key_l2 = update_params_key_l1_down = update_params_key_l0_down = 'update_network_shared'
    else:
        processor_network_l0_up, processor_params_init_l0_up = build_network('processor_network_l0_up', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        processor_network_l1_up, processor_params_init_l1_up = build_network('processor_network_l1_up', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        processor_network_l2, processor_params_init_l2 = build_network('processor_network_l2', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        processor_network_l1_down, processor_params_init_l1_down = build_network('processor_network_l1_down', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        processor_network_l0_down, processor_params_init_l0_down = build_network('processor_network_l0_down', layers=make_layers(PROCESSOR_IN_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_l0_up, update_params_init_l0_up = build_network('update_network_l0_up', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_l1_up, update_params_init_l1_up = build_network('update_network_l1_up', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_l2, update_params_init_l2 = build_network('update_network_l2', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_l1_down, update_params_init_l1_down = build_network('update_network_l1_down', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
        update_network_l0_down, update_params_init_l0_down = build_network('update_network_l0_down', layers=make_layers(3*CELL_LATENT_DIM, PROCESSOR_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)

        processor_params_key_l0_up = 'processor_network_l0_up'
        processor_params_key_l1_up = 'processor_network_l1_up'
        processor_params_key_l2 = 'processor_network_l2'
        processor_params_key_l1_down = 'processor_network_l1_down'
        processor_params_key_l0_down = 'processor_network_l0_down'
        update_params_key_l0_up = 'update_network_l0_up'
        update_params_key_l1_up = 'update_network_l1_up'
        update_params_key_l2 = 'update_network_l2'
        update_params_key_l1_down = 'update_network_l1_down'
        update_params_key_l0_down = 'update_network_l0_down'

    broadcast_network_l1_down, broadcast_params_init_l1_down = build_network('broadcast_network_l1_down', layers=make_layers(2*CELL_LATENT_DIM, BROADCAST_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
    broadcast_network_l0_down, broadcast_params_init_l0_down = build_network('broadcast_network_l0_down', layers=make_layers(2*CELL_LATENT_DIM, BROADCAST_WIDTH, CELL_LATENT_DIM), activation=jax.nn.silu, weight_normalization=False)
    decoder_network, decoder_params_init = build_network('decoder_network', layers=make_layers(CELL_LATENT_DIM, DECODER_WIDTH, 6), activation=jax.nn.silu, weight_normalization=False)

    def predict(params:dict, flow_condition:dict, graph_points:dict, geometry:dict)->tuple:
        print(f"COMPILING MODEL... ()")

        # Normalize freestream conditions and add to node features
        if not multifidelity_model:
            v_inf_sc = jnp.asarray(flow_condition['v_inf'], dtype=DTYPE) / constants['v']['ref']
            T_inf_sc = jnp.asarray(flow_condition['T_inf'], dtype=DTYPE) / constants['T']['ref']
            p_inf_sc = jnp.asarray(flow_condition['p_inf'], dtype=DTYPE) / constants['p']['ref']
            nuTilda_inf_sc = jnp.asarray(flow_condition['nuTilda_inf'], dtype=DTYPE) / constants['nuTilda']['ref']
            # print(f"v_inf_sc: {v_inf_sc}, T_inf_sc: {T_inf_sc}, p_inf_sc: {p_inf_sc}, nuTilda_inf_sc: {nuTilda_inf_sc}")
            
            # v_inf_sc = constants['v']['ref'] / constants['v']['ref']
            # T_inf_sc = constants['T']['ref'] / constants['T']['ref']
            # p_inf_sc = constants['p']['ref']  / constants['p']['ref']
            # nuTilda_inf_sc = constants['nuTilda']['ref'] / constants['nuTilda']['ref']

            # set y and z velocity == 1
            # nd_state_input['v_inf'] = nd_state_input['v_inf'].at[([1,2])].set(1.0)
            # v_inf_sc = v_inf_sc.at[1:].set(1.0)

            # v_inf_sc = jnp.ones(3)
            # T_inf_sc, p_inf_sc, nuTilda_inf_sc = 1.0, 1.0, 1.0
            N_nodes = geometry['dual_graph']['nDualCells']
            v_inf, T_inf, p_inf, nuTilda_inf = jnp.broadcast_to(v_inf_sc, (N_nodes, 3)), jnp.ones(N_nodes, dtype=DTYPE)*T_inf_sc, jnp.ones(N_nodes, dtype=DTYPE)*p_inf_sc, jnp.ones(N_nodes, dtype=DTYPE)*nuTilda_inf_sc
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
                'v': jnp.asarray(graph_points['low_fidelity_inputs']['U'], dtype=DTYPE) / constants['v']['ref'],
                'T': jnp.asarray(graph_points['low_fidelity_inputs']['T'], dtype=DTYPE) / constants['T']['ref'],
                'p': jnp.asarray(graph_points['low_fidelity_inputs']['p'], dtype=DTYPE) / constants['p']['ref'],
                'nuTilda': jnp.asarray(graph_points['low_fidelity_inputs']['nuTilda'], dtype=DTYPE) / constants['nuTilda']['ref'],
            }

        nCells = geometry['foam_mesh']['cells']['nCells']
        node_X, edge_X_1, edge_X_2 = build_graph_features(geometry['dual_graph'], graph_points['dual_graph'], nd_state_input, model_constants, add_cell_sdf)
        edge_Xl1_1, edge_Xl1_2 = build_graph_features(geometry['coarse_graph_l1'], graph_points['coarse_graph_l1'], nd_state_input, model_constants, add_cell_sdf, return_nodes=False)
        edge_Xl2_1, edge_Xl2_2 = build_graph_features(geometry['coarse_graph_l2'], graph_points['coarse_graph_l2'], nd_state_input, model_constants, add_cell_sdf, return_nodes=False)

        def MP(N, cell_X, edge_X_1, edge_X_2, graph, processor_network, update_network, processor_params_key, update_params_key, checkpoint_mp=True, checkpoint_iter=True):
            stage_params = {'processor_network': params[processor_params_key], 'update_network': params[update_params_key]}
            return pass_messages(N, processor_network, update_network, stage_params, cell_X, edge_X_1, edge_X_2, graph, model_constants, checkpoint_mp, checkpoint_iter, AGGREGATE_METHOD)
        
        def latent_unpool(finer_latents_up, coarser_latents, coarser_graph, broadcast_network, broadcast_params_key):
            stage_params = {'broadcast_network': params[broadcast_params_key]}
            return latent_unpool_bcast(finer_latents_up, coarser_latents, coarser_graph, broadcast_network, stage_params)

        # ================== 1 Initial encoding of cell features ==================
        is_surface = geometry['dual_graph']['dual_cell_types'] == 1
        surface_areas = jnp.asarray(graph_points['foam_mesh']['faces']['surface_areas'], dtype=DTYPE)
        surface_normals = jnp.asarray(graph_points['foam_mesh']['faces']['surface_normals'], dtype=DTYPE)
        num_surface_nodes = int(np.sum(is_surface))

        assert surface_areas.shape == (num_surface_nodes,)
        assert surface_normals.shape == (num_surface_nodes, 3)

        surface_area_features = (jnp.log1p(surface_areas / jnp.asarray(0.0017, dtype=DTYPE)) - jnp.asarray(1.22, dtype=DTYPE)) / jnp.asarray(1.29, dtype=DTYPE)
        surface_node_X = jnp.concatenate([node_X[is_surface], surface_area_features[:, None], surface_normals], axis=1,)
        assert surface_node_X.shape == (num_surface_nodes, SURFACE_NODE_NUM_X)

        non_surface_latents = jax.vmap(lambda x: encoder_network(params['encoder_network'], x), in_axes=(0,))(node_X[~is_surface])
        surface_latents = jax.vmap(
            lambda x: surface_encoder_network(params['surface_encoder_network'], x), in_axes=(0,)
        )(surface_node_X)

        l0_cell_X_up = jnp.zeros((node_X.shape[0], CELL_LATENT_DIM), dtype=DTYPE)
        l0_cell_X_up = l0_cell_X_up.at[~is_surface].set(non_surface_latents)
        l0_cell_X_up = l0_cell_X_up.at[is_surface].set(surface_latents)

        # OLD model behavior (no surface encoder, just use the same encoder for all nodes)
        # l0_cell_X_up = jax.vmap(lambda x: encoder_network(params['encoder_network'], x), in_axes=(0,))(node_X)
        # ================== 2 Main U-shaped message passing with pooling and unpooling ==================
        # up-scale fine
        l0_cell_X_up = MP(MP_L0_UP, l0_cell_X_up, edge_X_1, edge_X_2, geometry['dual_graph'], processor_network_l0_up, update_network_l0_up, processor_params_key_l0_up, update_params_key_l0_up)

        # up-scale mid
        l1_cell_X_up = latent_pool(l0_cell_X_up, geometry['coarse_graph_l1'], AGGREGATE_METHOD)
        l1_cell_X_up = MP(MP_L1_UP, l1_cell_X_up, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1'], processor_network_l1_up, update_network_l1_up, processor_params_key_l1_up, update_params_key_l1_up) 

        # up/down-scale coarse
        l2_cell_X = latent_pool(l1_cell_X_up, geometry['coarse_graph_l2'], AGGREGATE_METHOD)
        l2_cell_X = MP(MP_L2, l2_cell_X, edge_Xl2_1, edge_Xl2_2, geometry['coarse_graph_l2'], processor_network_l2, update_network_l2, processor_params_key_l2, update_params_key_l2)  # no checkpointing at coarsest level since it's cheap

        # down-scale mid
        l1_cell_X_down = latent_unpool(l1_cell_X_up, l2_cell_X, geometry['coarse_graph_l2'], broadcast_network_l1_down, 'broadcast_network_l1_down')
        l1_cell_X_down = MP(MP_L1_DOWN, l1_cell_X_down, edge_Xl1_1, edge_Xl1_2, geometry['coarse_graph_l1'], processor_network_l1_down, update_network_l1_down, processor_params_key_l1_down, update_params_key_l1_down)

        # down-scale fine
        l0_cell_X_down = latent_unpool(l0_cell_X_up, l1_cell_X_down, geometry['coarse_graph_l1'], broadcast_network_l0_down, 'broadcast_network_l0_down')
        l0_cell_X_down = MP(MP_L0_DOWN, l0_cell_X_down, edge_X_1, edge_X_2, geometry['dual_graph'], processor_network_l0_down, update_network_l0_down, processor_params_key_l0_down, update_params_key_l0_down)

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
    param_info.add(surface_encoder_params_init)
    if SHARE_MP_WEIGHTS:
        param_info.add(processor_params_init_shared)
    else:
        param_info.add(processor_params_init_l0_up)
        param_info.add(processor_params_init_l1_up)
        param_info.add(processor_params_init_l2)
        param_info.add(processor_params_init_l1_down)
        param_info.add(processor_params_init_l0_down)
    param_info.add(decoder_params_init)
    if SHARE_MP_WEIGHTS:
        param_info.add(update_params_init_shared)
    else:
        param_info.add(update_params_init_l0_up)
        param_info.add(update_params_init_l1_up)
        param_info.add(update_params_init_l2)
        param_info.add(update_params_init_l1_down)
        param_info.add(update_params_init_l0_down)
    param_info.add(broadcast_params_init_l1_down)
    param_info.add(broadcast_params_init_l0_down)
    return predict, param_info
