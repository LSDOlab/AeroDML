import pyvista as pv
import numpy as np
import copy
import time 
from .visualize_mesh import visualize_dual, visualize_foam_mesh, visualize_scalars, visualize_coarse
from .utils import print_content

import jax
import jax.numpy as jnp

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def check_mesh_settings(load_settings:dict)->dict:
    default_settings = {'verbose': False, 'run_checks': True, 'visualize': False}
    if load_settings is None:
        return default_settings
    load_settings = load_settings.copy()
    for key, default_value in default_settings.items():
        if key not in load_settings:
            load_settings[key] = default_value
    for key in load_settings:
        if key not in default_settings:
            raise ValueError(f"Invalid mesh load setting: '{key}'. Valid keys are: {list(default_settings.keys())}")
    return load_settings

def load_foam_mesh(path_dir:str, load_settings:dict)->dict:
    return get_mesh(path_dir, load_settings)

def load_dual_graph(path_dir:str, foam_mesh:dict, load_settings)->dict:
    return get_dual_graph(path_dir, foam_mesh, load_settings)

def load_coarse_dual_graph(path_dir:str, dual_mesh:dict, foam_mesh:dict, load_settings:dict, name:str, N:int, max_edge_length:float)->dict:
    return get_coarse_dual_graph(path_dir, dual_mesh, foam_mesh, load_settings, name, N, max_edge_length)

import numpy as np
import time
import os
import pickle

def find_closest_wall_node_id(dual_cells:np.ndarray, dual_cell_types:np.ndarray)->np.ndarray:
    cell_centers = dual_cells                       # (N,3)
    wall_mask = (dual_cell_types == 1)             
    wall_node_ids = np.where(wall_mask)[0]          # indices into dual_cells
    wall_points = cell_centers[wall_mask]           # (Nwall,3)
    tree = cKDTree(wall_points)
    d_wall, idx_local = tree.query(cell_centers, k=1, workers=-1)
    # d_wall:     (N,)      float distances
    # idx_local:  (N,)      int indices into wall_points (0..Nwall-1)
    closest_wall_node_id = wall_node_ids[idx_local]  # (N,) indices into dual_cells
    assert closest_wall_node_id.shape == (dual_cells.shape[0],), f"Error: closest_wall_node_id shape {closest_wall_node_id.shape} does not match expected shape {(dual_cells.shape[0],)}"
    return closest_wall_node_id

def find_surface_neighbors(foam_mesh:dict)->np.ndarray:
    # for each surface, gets the cell ids of the dual cells that are adjacent to that surface, and stores in dual_dict['surface_neighbor_cells']
    face_type = foam_mesh['faces']['type']
    owner = foam_mesh['faces']['owner']

    surface_face_mask = (face_type == 1)
    # Cell ID for each surface face, in the SAME order as surface_face_ids
    surface_face_owner_cells = owner[surface_face_mask]
    assert surface_face_owner_cells.shape[0] == np.sum(surface_face_mask), f"Error: number of surface faces {np.sum(surface_face_mask)} does not match expected {surface_face_owner_cells.shape[0]}"
    return surface_face_owner_cells

def build_dual_from_vertices(foam_mesh:dict)->dict:
    cell_points = foam_mesh['cells']['centroids']
    points = foam_mesh['points'] # shape (nPoints, 3)
    
    # Build ghost points across boundary faces, then build dual graph based on these points and the cell connectivity
    boundary_face_mask = foam_mesh['faces']['type'] > 0 # for each boundary face, we create a ghost point on the face center
    num_real_cells = foam_mesh['cells']['nCells']
    num_ghost_cells = int(np.sum(boundary_face_mask))
    nDualCells = num_real_cells + num_ghost_cells

    num_faces = foam_mesh['faces']['nFaces']

    # create ghosts:
    ghost_points = np.zeros((num_ghost_cells, 3), dtype=np.float32)
    ghost_tags = np.zeros(num_ghost_cells, dtype=np.int16) # 1: wall, 2: freestream, 3: symmetry
    ghost_ids = -np.ones(num_faces, dtype=np.int32)

    ghost_counter = 0
    for face_index, face in enumerate(foam_mesh['faces']['points']):
        if not boundary_face_mask[face_index]: # ignore internal faces
            continue
        
        # Keep track of which ghost node corresponds to which boundary face, so we can connect them in the dual graph
        ghost_node_index = num_real_cells + ghost_counter
        ghost_ids[face_index] = ghost_node_index

        # face center = average of its points
        face_center = np.mean(points[face], axis=0)
        ghost_points[ghost_counter] = face_center
        ghost_tags[ghost_counter] = foam_mesh['faces']['type'][face_index]
        ghost_counter += 1

    assert ghost_counter == num_ghost_cells, f"Error: ghost counter {ghost_counter} does not match expected number of ghost cells {num_ghost_cells}"

    # combine to build dual graph:
    dual_cells = np.vstack([cell_points, ghost_points]) # shape (num_real_cells + num_ghost_cells, 3)
    assert dual_cells.shape == (nDualCells, 3), f"Error: dual_cells shape {dual_cells.shape} does not match expected shape {(nDualCells, 3)}"
    
    dual_cell_types = np.zeros(nDualCells, dtype=np.int16)
    dual_cell_types[:num_real_cells] = 0 # real cells
    dual_cell_types[num_real_cells:] = ghost_tags # ghost cells
    assert dual_cell_types.shape == (nDualCells,), f"Error: dual_cell_types shape {dual_cell_types.shape} does not match expected shape {(nDualCells,)}"
    assert np.all(np.isin(dual_cell_types, [0, 1, 2, 3])), f"Error: dual_cell_types contains invalid values {np.unique(dual_cell_types)}"
    assert np.sum(dual_cell_types == 0) == num_real_cells, f"Error: number of real cells {np.sum(dual_cell_types == 0)} does not match expected {num_real_cells}"
    assert np.sum(dual_cell_types == 1) == np.sum(ghost_tags == 1), f"Error: number of wall ghost cells {np.sum(dual_cell_types == 1)} does not match expected {np.sum(ghost_tags == 1)}"
    assert np.sum(dual_cell_types == 2) == np.sum(ghost_tags == 2), f"Error: number of freestream ghost cells {np.sum(dual_cell_types == 2)} does not match expected {np.sum(ghost_tags == 2)}"
    assert np.sum(dual_cell_types == 3) == np.sum(ghost_tags == 3), f"Error: number of symmetry ghost cells {np.sum(dual_cell_types == 3)} does not match expected {np.sum(ghost_tags == 3)}"

    # build edges (one per face):
    graph_edges = np.zeros((num_faces, 2), dtype=np.int32) # (num_faces, 2) array of [cell1, cell2] indices for each face. cell2 = -1 for boundary faces
    for face_index in range(num_faces):
        owner_cell = foam_mesh['faces']['owner'][face_index]
        neighbour_cell = foam_mesh['faces']['neighbour'][face_index]
        assert owner_cell >= 0 and owner_cell < num_real_cells, f"Error: owner cell index {owner_cell} out of bounds for face {face_index}"
        
        if neighbour_cell >= 0:
            # if internal face, connect owner and neighbour
            u, v = owner_cell, neighbour_cell
        else:
            # Now this is a boundary face, we connect the owner cell to the corresponding ghost cell
            ghost_index = int(ghost_ids[face_index])
            assert ghost_index >= num_real_cells 
            u,v = owner_cell, ghost_index

        graph_edges[face_index, 0] = u
        graph_edges[face_index, 1] = v
    graph_edge_types = foam_mesh['faces']['type']
    assert graph_edges.shape == (num_faces, 2), f"Error: graph_edges shape {graph_edges.shape} does not match expected shape {(num_faces, 2)}"
    assert graph_edge_types.shape == (num_faces,), f"Error: graph_edge_types shape {graph_edge_types.shape} does not match expected shape {(num_faces,)}"

    # deg per cell/node:
    final_edges = graph_edges
    deg = np.bincount(final_edges.ravel(), minlength=nDualCells)
    assert deg.shape == (nDualCells,), f"Error: degree array shape {deg.shape} does not match expected shape {(nDualCells,)}"
    assert deg.min() >= 1, f"Error: found cell with degree {deg.min()}, expected all cells to have degree >= 1"

    # build approximate SDF values
    closest_wall_node_id = find_closest_wall_node_id(dual_cells, dual_cell_types)

    # finally, organize into dict:
    dual_dict = {
        'nDualCells': nDualCells,
        'dual_cells': dual_cells,
        'closest_wall_id': closest_wall_node_id,
        'dual_cell_types': dual_cell_types,
        'dual_edges': graph_edges,
        'dual_edge_types': graph_edge_types,
        'num_real_cells': num_real_cells,
        'num_ghost_cells': num_ghost_cells,
        'deg': deg,
    }
    return dual_dict

def cluster_graph(graph:dict, max_edge_length:float)->dict:
    graph_nodes = graph['nodes']
    graph_edges = graph['edges']
    original_mapping = graph['original_mapping']

    M = graph_nodes.shape[0]
    parent = np.arange(M, dtype=np.int64)
    used = np.zeros(M, dtype=bool) # to keep track of which nodes have been merged already
    
    is_ghost = (graph['node_types'] != 0)

    u = graph_edges[:, 0]
    v = graph_edges[:, 1]
    next_to_ghost = np.zeros(M, dtype=bool)
    np.logical_or.at(next_to_ghost, u, is_ghost[v])
    np.logical_or.at(next_to_ghost, v, is_ghost[u])

    def find(x): # finds the leader
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b): # merge the cluster
        ra, rb = find(a), find(b)
        parent[ra] = rb

    def valid(i):
        a,b = graph_edges[i]
        if used[a] or used[b]: # if either node has already been merged, skip to avoid creating clusters larger than size 2
             return False
        
        # 1) Do not merge if either node is a ghost cell
        if graph['node_types'][a] != 0 or graph['node_types'][b] != 0:
            return False

        # 2) Make sure that the nodes are not next to any ghost cells
        if next_to_ghost[a] or next_to_ghost[b]:
            return False

        # 3) Prioritize merging cells that are close together in space
        edge_length = np.linalg.norm(graph_nodes[a] - graph_nodes[b])
        if edge_length >= max_edge_length:
            return False
        
        return True

    # Union all edges
    # sort edges by length, so that we prioritize merging closer nodes first
    edge_lengths = np.linalg.norm(graph_nodes[u] - graph_nodes[v], axis=1)
    edge_order = np.argsort(edge_lengths)
    for i in edge_order:
        if valid(i):
            a,b = graph_edges[i]
            union(a, b)
            used[a] = True
            used[b] = True

    # Convert roots to consecutive cluster ids 0..K-1
    roots = np.array([find(i) for i in range(M)], dtype=np.int64)
    _, mapping = np.unique(roots, return_inverse=True)
    assert mapping.min() == 0 and mapping.max() < M, f"Error: mapping values should be in range [0, {M-1}] but got min {mapping.min()} and max {mapping.max()}"
    K = int(mapping.max()) + 1
    # mapping[i] says which cluster node i belongs to, with cluster ids in range [0, K-1] where K is the number of clusters formed

    counts = np.bincount(mapping, minlength=K).astype(np.float64)  # how many original nodes in each cluster
    assert counts.shape == (K,), f"Error: counts shape {counts.shape} does not match expected shape {(K,)}"
    assert np.all(counts > 0), f"Error: found empty cluster with count 0, counts: {counts}"
    assert np.all(counts <= 2), f"Error: found cluster with more than 2 nodes, counts: {counts}" 

    sums = np.zeros((K,3))
    np.add.at(sums, mapping, graph_nodes)

    # new nodes and edges
    new_nodes = sums / counts.reshape((K,) + (1,) * (graph_nodes.ndim - 1)) # Average the node features within each cluster to get the new node features
    new_edges = mapping[graph_edges]  # map the original edge endpoints to their respective cluster ids to get the new edges.
    new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]  # drop self-loops
    new_edges = np.unique(new_edges, axis=0)
    assert new_edges.shape[1] == 2, f"Error: new_edges should have shape (num_new_edges, 2) but got {new_edges.shape}"
    assert np.all(new_edges >= 0) and np.all(new_edges < K), f"Error: new_edges contain invalid cluster ids, should be in range [0, {K-1}] but got min {new_edges.min()} and max {new_edges.max()}"

    # new types: only real cells are merged, so 
    rep = np.full(K, M, dtype=np.int64)
    np.minimum.at(rep, mapping, np.arange(M, dtype=np.int64))
    new_types = graph['node_types'][rep]
    assert new_types.shape == (K,), f"Error: new_types shape {new_types.shape} does not match expected shape {(K,)}"
    # make sure number of types is the same and that we don't accidentally merge ghost cells
    for t in [1, 2, 3]:
        num_before, num_after = np.sum(graph['node_types'] == t), np.sum(new_types == t)
        assert num_after == num_before, f"Error: number of nodes of type {t} should be the same before and after clustering, but got {num_before} before and {num_after} after. This likely means that we accidentally merged some ghost cells or merged real cells with ghost cells."

    # Finally update the original mapping to point to the new cluster ids
    new_original_mapping = mapping[original_mapping]  # shape (M,)
    assert new_original_mapping.shape == original_mapping.shape, f"Error: new_original_mapping shape {new_original_mapping.shape} does not match original_mapping shape {original_mapping.shape}"
    assert np.all(new_original_mapping >= 0) and np.all(new_original_mapping < K), f"Error: new_original_mapping contains invalid cluster ids, should be in range [0, {K-1}] but got min {new_original_mapping.min()} and max {new_original_mapping.max()}"

    coarsened_graph = {
        'nodes': new_nodes,
        'edges': new_edges,
        'original_mapping': new_original_mapping,
        'node_types': new_types,
    }
    return coarsened_graph

def build_coarse_dual(dual_mesh:dict, N:int, max_edge_length:float)->dict:
    # We want the coarse dual graph to have the same information as the dual graph (with obviously less nodes and edges) 
    # The only difference is that there will be a mapping from nodes in the coarse dual graph to sets of nodes in the original dual graph
    # Requirements:
    # 1) Do no merge ghost cells at all
    # 2) Only merge real cells that are not adjacent to any ghost cells (perhaps N cells away from any ghost cell?)
    # Prioritize merging cells that are close together in space (short edges)

    # General steps:
    # 1) Identify which real edges are eligible for merging (those that are at least N cells away from any ghost cell)
    # 2) Iterate through all edges and do N passes (1 pass will in theory reduce number of nodes by up to a factor of max 2)

    # 2) 
    coarsened = {
        'nodes': dual_mesh['dual_cells'],
        'edges': dual_mesh['dual_edges'],
        'original_mapping': np.arange(dual_mesh['nDualCells']),
        'node_types': dual_mesh['dual_cell_types'],
    }

    print(f"Pass 0/{N} completed. nodes: {coarsened['nodes'].shape[0]}, edges: {coarsened['edges'].shape[0]}")
    for i in range(N):
        coarsened = cluster_graph(coarsened, max_edge_length)
        print(f"Pass {i+1}/{N} completed. nodes: {coarsened['nodes'].shape[0]}, edges: {coarsened['edges'].shape[0]}")
    
    # Now rebuild the coarse dual dict with the same format as the original dual dict, but with the coarsened graph information
    mapping = coarsened['original_mapping']
    nDualCells = coarsened['nodes'].shape[0]
    dual_cell_types = coarsened['node_types']
    deg = np.bincount(coarsened['edges'].ravel(), minlength=nDualCells)
    counts = np.bincount(mapping, minlength=nDualCells)
    assert counts.shape == (nDualCells,), f"Error: counts shape {counts.shape} does not match expected shape {(nDualCells,)}"

    # dual_cells = avg of the original dual cells that belong to each cluster (recompute)
    dual_cells = np.zeros_like(coarsened['nodes'])
    np.add.at(dual_cells, mapping, dual_mesh['dual_cells'])
    dual_cells = dual_cells / counts.reshape((nDualCells, 1))

    # wall ID
    closest_wall_node_id = find_closest_wall_node_id(dual_cells, dual_cell_types)

    # edge types
    u = coarsened['edges'][:, 0]
    v = coarsened['edges'][:, 1]
    tu = dual_cell_types[u]
    tv = dual_cell_types[v]
    new_edge_types = np.maximum(tu, tv)
    assert new_edge_types.shape == (coarsened['edges'].shape[0],), f"Error: new_edge_types shape {new_edge_types.shape} does not match expected shape {(coarsened['edges'].shape[0],)}"
    for t in [1, 2, 3]:
        num_before, num_after = np.sum(dual_mesh['dual_edge_types'] == t), np.sum(new_edge_types == t)
        assert num_after == num_before, f"Error: number of edges of type {t} should be the same before and after clustering, but got {num_before} before and {num_after} after. This likely means that we accidentally merged some ghost cells or merged real cells with ghost cells."

    coarse_dual_dict = {
        'mapping': mapping, 
        'counts': counts,
        'nDualCells': nDualCells,
        'dual_cells': dual_cells,
        'closest_wall_id': closest_wall_node_id,
        'dual_cell_types': dual_cell_types,
        'dual_edges': coarsened['edges'],
        'dual_edge_types': new_edge_types,
        'num_real_cells':  np.sum(coarsened['node_types'] == 0),
        'num_ghost_cells': np.sum(coarsened['node_types'] != 0),
        'deg': deg,
    }
    return coarse_dual_dict

def get_coarse_dual_graph(name:str, dual_mesh:dict, foam_mesh:dict, load_settings:dict, fname:str, N:int, max_edge_length:float)->dict:
    pickle_file = f"{name}/{fname}.pkl"
    print(f"Loading Coarse dual graph '{fname}' from case '{name}'...")

    if os.path.exists(pickle_file):
        print(f"Pickle file '{fname}' found, loading coarse from pickle...")
        with open(pickle_file, 'rb') as f:
            coarse_dual = pickle.load(f)
    else:
        coarse_dual = build_coarse_dual(dual_mesh, N, max_edge_length)
        with open(pickle_file, 'wb') as f:
            pickle.dump(coarse_dual, f)
        print(f"Coarse dual graph '{fname}' saved to pickle for faster loading next time.")
    
    if load_settings['verbose']:
        # print full information about the mesh dict, types and shapes even for lists, dicts etc
        print(f"\n=== Coarse Dual Graph '{fname}' Data Summary ===")
        print(print_content(coarse_dual,0))
        print("=========================\n")

    if load_settings['run_checks']:
        # Checks:
        pass
    
    if load_settings['visualize']:
        visualize_coarse(coarse_dual, dual_mesh, foam_mesh)

        X = np.asarray(coarse_dual["dual_cells"])                  # (K,3)
        E = np.asarray(coarse_dual["dual_edges"], dtype=np.int64)  # (nE,2)

        # Build the vtk "lines" array: [2,u0,v0, 2,u1,v1, ...]
        lines = np.hstack([np.full((E.shape[0], 1), 2, dtype=np.int64), E]).ravel()

        edge_mesh = pv.PolyData(X)
        edge_mesh.lines = lines 

        from .visualize_mesh import _visualize_surface
        p = pv.Plotter()
        p.add_mesh(edge_mesh, color="grey", line_width=1, point_size=0.5)
        _visualize_surface(p, {'foam_mesh': foam_mesh}, {'foam_mesh': foam_mesh}, scalars=None)
        inverted_pts = foam_mesh['points'].copy()
        inverted_pts[:, 1] *= -1
        _visualize_surface(p, {'foam_mesh': foam_mesh}, {'foam_mesh': { 'points': inverted_pts }}, scalars=None)

        p.camera_position = [(-30.9105, -62.9705, 53.1841), (24.7327, 5.65883, -2.86862), (0.269591, 0.468548, 0.841299)]
        p.camera.view_angle = 30
        p.camera.clipping_range = (2.90825, 2908.25)
        p.window_size = [2610, 1578]

    return coarse_dual

def get_dual_graph(name:str, foam_mesh:dict, load_settings:dict)->dict:
    pickle_file = f"{name}/dual_mesh.pkl"
    start = time.time()
    print(f"Loading dual graph from case '{name}'...")

    if os.path.exists(pickle_file):
        print("Pickle file found, loading dual from pickle...")
        with open(pickle_file, 'rb') as f:
            dual_dict = pickle.load(f)
    else:
        dual_dict = build_dual_from_vertices(foam_mesh)
        with open(pickle_file, 'wb') as f:
            pickle.dump(dual_dict, f)
        print("Dual graph saved to pickle for faster loading next time.")
    if load_settings['verbose']:
        # print full information about the mesh dict, types and shapes even for lists, dicts etc
        print("\n=== Dual Graph Data Summary ===")
        print(print_content(dual_dict,0))
        print("=========================\n")

    if load_settings['run_checks']:
        # Checks:
        pass
    
    if load_settings['visualize']:
        # ==========================Visualize==========================
        # Visualize random graph nodes and respective openfoam cells
        visualize_dual(dual_dict,foam_mesh)

        dual_cells = dual_dict['dual_cells']  # (N,3)
        wall_pos = dual_cells[dual_dict['closest_wall_id']]  # (N,3)
        d = np.linalg.norm(dual_cells - wall_pos, axis=1).astype(np.float32)  # (N,)

        # Pick a scale. Smaller => more mass near 1 for very near-wall points.
        sigma = np.percentile(d, 80).astype(np.float32) + 1e-12
        sigma = 5

        # Option A: exponential squash into [0, 1)
        d_tilde = 1.0 - np.exp(-d / sigma)

        # d_tilde = d_tilde / (1.0 - np.exp(-d.max() / sigma) + 1e-12)
        # d_tilde = np.clip(d_tilde, 0.0, 1.0)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        ax[0].hist(d, bins=50, log=True)
        ax[0].set_title("distance to nearest wall node")
        ax[0].set_xlabel("d")
        ax[0].set_ylabel("Node count")

        ax[1].hist(d_tilde, bins=50, log=False)
        ax[1].set_title(f"Transformed distance: 1.0 - np.exp(-d / {sigma})")
        ax[1].set_xlabel(r"$\tilde d \in [0,1]$")
        ax[1].set_ylabel("Node count")

        # plot the function itself
        d_vals = np.linspace(0, d.max(), 1000)
        d_tilde_vals = 1.0 - np.exp(-d_vals / sigma)
        ax[2].plot(d_vals, d_tilde_vals)
        ax[2].set_title("Squash function")
        ax[2].set_xlabel("d")
        ax[2].set_ylabel(r"$\tilde d$")
        plt.show()

        # Visualize the distribution of distances to the nearest wall node, and the effect of the squashing function
        p = visualize_scalars(
            dual_cells,
            d_tilde,
            title="P scalar field",
            cmap="viridis",
        )
        p.show()


        X = np.asarray(dual_dict["dual_cells"])                  # (K,3)
        E = np.asarray(dual_dict["dual_edges"], dtype=np.int64)  # (nE,2)

        # Build the vtk "lines" array: [2,u0,v0, 2,u1,v1, ...]
        lines = np.hstack([np.full((E.shape[0], 1), 2, dtype=np.int64), E]).ravel()

        edge_mesh = pv.PolyData(X)
        edge_mesh.lines = lines 

        from .visualize_mesh import _visualize_surface
        p = pv.Plotter()
        p.add_mesh(edge_mesh, color="grey", line_width=1, point_size=0.5)
        _visualize_surface(p, {'foam_mesh': foam_mesh}, {'foam_mesh': foam_mesh}, scalars=None)
        inverted_pts = foam_mesh['points'].copy()
        inverted_pts[:, 1] *= -1
        _visualize_surface(p, {'foam_mesh': foam_mesh}, {'foam_mesh': { 'points': inverted_pts }}, scalars=None)

        p.camera_position = [(-30.9105, -62.9705, 53.1841), (24.7327, 5.65883, -2.86862), (0.269591, 0.468548, 0.841299)]
        p.camera.view_angle = 30
        p.camera.clipping_range = (2.90825, 2908.25)
        p.window_size = [2610, 1578]
        p.show()

    end = time.time()
    print("Dual graph loaded and transferred in {:.2f} seconds.".format(end - start))
    return dual_dict

def build_cell_faces(nCells, nFaces, owner, neighbour):
    cell_faces = [[] for _ in range(nCells)]

    for f in range(nFaces):
        o = owner[f]
        cell_faces[o].append(f)

        if f < len(neighbour):  # internal face
            n = neighbour[f]
            if n >= 0:
                cell_faces[n].append(f)
    return cell_faces

def build_cell_points(faces, cell_faces):
    cell_points = []
    for cfaces in cell_faces:
        pts = set()
        for f in cfaces:
            pts.update(faces[f])
        cell_points.append(sorted(pts))
    return cell_points

def build_face_types(faces, nei, bnd):
    nFaces = len(faces)
    face_type = np.zeros(nFaces, dtype=np.int32)  # default: fluid
    def k2s(x):
        return x.decode(errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)

    for k in bnd.keys():
        bd = bnd[k]              # has: type, start, num
        ptype = k2s(bd.type)     # "wall", "symmetry", "patch"
        start = int(bd.start)
        num   = int(bd.num)

        if ptype == "wall":
            code = 1
        elif ptype == "patch":
            code = 2
        elif ptype == "symmetry":
            code = 3
        else:
            code = 4  # SHOULD NOT HAPPEN

        face_type[start:start+num] = code

    if np.any(face_type == 4):
        raise ValueError("Error: Found unknown boundary type code 4")

    # ensure internal faces are fluid
    face_type[nei >= 0] = 0
    return face_type

def get_mesh(name:str, load_settings:dict)->dict:
    pickle_file = f"{name}/numpy_mesh.pkl"
    
    start = time.time()
    print(f"Loading mesh from case '{name}'...")
    # if 0:
    if os.path.exists(pickle_file):
        print("Pickle file found, loading mesh from pickle...")
        with open(pickle_file, 'rb') as f:
            mesh_dict = pickle.load(f)
    else:
        import Ofpp
        mesh = Ofpp.FoamMesh(name)
        print("Mesh loaded, transferring...")

        # Points: already (N,3) float array
        points = np.asarray(mesh.points)              # shape (nPoints, 3)

        # Faces: list-of-lists (variable length) -> keep as Python lists
        faces = mesh.faces                            # len = nFaces

        # Owner/neighbour: convert to numpy int arrays
        owner = np.asarray(mesh.owner, dtype=np.int32)
        neighbour = np.asarray(mesh.neighbour, dtype=np.int32)

        # Boundary patches: dict
        boundary = mesh.boundary

        nCells = mesh.num_cell
        nPoints = mesh.num_point
        nFaces = len(faces)

        cells = build_cell_faces(nCells, nFaces, owner, neighbour)
        cell_points = build_cell_points(faces, cells)
        cell_centroids = []
        for c in range(nCells):
            cell_centroids.append(np.mean(points[cell_points[c]], axis=0))
        cell_centroids = np.array(cell_centroids) 

        types = build_face_types(faces, neighbour, boundary)
        # Organize cell info
        cells_dict = {
            'nCells': nCells,             # total number of cells
            'faces': cells,          # list of lists of face indices for each cell
            'points': cell_points,   # list of lists of point indices for each cell
            'centroids': cell_centroids, # shape (nCells, 3)
        }

        # Organize face info
        faces_dict = {
            'nFaces': nFaces,            # total number of faces
            'points': faces,        # list of lists of point indices for each face
            'owner': owner,         # array of owner cell indices for each face
            'neighbour': neighbour, # array of neighbour cell indices for internal faces
            'type': types, # integer array indicating face type (0: internal, 1: freestream boundary, 2: wall boundary, 3: symmetry boundary)
        }

        mesh_dict = {
            'points': points,
            'nPoints': nPoints,
            'cells': cells_dict,
            'faces': faces_dict,
        }

        # save mesh_dict to file for faster loading next time
        with open(pickle_file, 'wb') as f:
                    pickle.dump(mesh_dict, f)
        print("Mesh saved to pickle for faster loading next time.")
        
    # TODO: we need to move this into initial mesh builder later
    surface_neighbors = find_surface_neighbors(mesh_dict)
    mesh_dict['faces']['surface_neighbors'] = surface_neighbors

    if load_settings['verbose']:
        # print full information about the mesh dict, types and shapes even for lists, dicts etc
        print("\n=== Mesh Data Summary ===")
        print(print_content(mesh_dict,0))
        print("=========================\n")

        # find how many points do NOT exist in any face/cell
        # points_in_faces = set()
        # for face in mesh_dict['faces']['points']:
        #     points_in_faces.update(face)
        # points_in_cells = set()
        # for cell in mesh_dict['cells']['points']:
        #     points_in_cells.update(cell)
        # points_in_faces_or_cells = points_in_faces.union(points_in_cells)
        # all_points = set(range(mesh_dict['nPoints']))
        # points_not_in_faces_or_cells = all_points - points_in_faces_or_cells
        # print(f"Number of points not in any face or cell: {len(points_not_in_faces_or_cells)}")


    if load_settings['run_checks']:
        # Check cells
        assert mesh_dict['cells']['nCells'] == len(mesh_dict['cells']['faces']) == len(mesh_dict['cells']['points']) == len(mesh_dict['cells']['centroids']), f"Error: Inconsistent number of cells in mesh_dict['cells']"
        assert mesh_dict['cells']['centroids'].shape == (mesh_dict['cells']['nCells'], 3), f"Error: cell centroids shape {mesh_dict['cells']['centroids'].shape} does not match expected {(mesh_dict['cells']['nCells'], 3)}"

        # print number of negatives values in neighbor and make sure it matches with the number of boundary faces (type > 0)
        owner = mesh_dict['faces']['owner']
        neighbour = mesh_dict['faces']['neighbour']
        num_boundary_faces = np.sum(mesh_dict['faces']['type'] > 0)
        num_negative_neighbors = np.sum(neighbour < 0)
        num_negative_owners = np.sum(owner < 0)
        assert num_negative_neighbors == num_boundary_faces, f"Error: number of negative neighbors {num_negative_neighbors} does not match expected number of boundary faces {num_boundary_faces}"
        assert num_negative_owners == 0, f"Error: found {num_negative_owners} negative owner indices, expected 0"

    if load_settings['visualize']:
        # Visualize:
        # 1) Visualize surface faces by coloring them by type (internal (ignore), freestream boundary, wall boundary, symmetry boundary)
        visualize_foam_mesh(mesh_dict)

        from .visualize_mesh import _visualize_mesh, visualize_scalars
        near_surface = jnp.ones(mesh_dict['cells']['centroids'].shape[0]) # visualize all cells
        near_surface = near_surface.at[mesh_dict['faces']['surface_neighbors']].set(2.0) # highlight near-surface cells
        p = visualize_scalars(pos = mesh_dict['cells']['centroids'], scalars = near_surface)
        _visualize_mesh(mesh_dict ,p)
        p.show()

    end = time.time()
    print("Mesh loaded and transferred in {:.2f} seconds.".format(end - start))
    return mesh_dict

def _compute_centroids(points:jax.Array, cell_points:list[list])->jax.Array:
    num_cells = len(cell_points)
    flat_idx = np.concatenate([np.asarray(c, dtype=np.int32) for c in cell_points], axis=0)
    cell_id  = np.concatenate([np.full(len(c), i, dtype=np.int32) for i, c in enumerate(cell_points)], axis=0)
    counts = np.bincount(cell_id, minlength=num_cells).astype(np.int32)  # (C,)
    summed = jnp.zeros((num_cells, 3)).at[cell_id].add(points[flat_idx])
    return summed / counts[:, None].astype(points.dtype)  # (C, D)

def _compute_new_ghost_points(new_cell_centroids, topology)->jax.Array:
    new_points = new_cell_centroids
    face_points = topology['foam_mesh']['faces']['points']
    boundary_face_mask = topology['foam_mesh']['faces']['type'] > 0 # for each boundary face, we create a ghost point on the face center
    
    boundary_faces = [
        np.asarray(face_points[f], dtype=np.int32)
        for f in range(len(face_points))
        if boundary_face_mask[f]
    ]
    num_ghost = topology['dual_graph']['num_ghost_cells']
    flat_idx = np.concatenate(boundary_faces, axis=0) # all point indices that belong to any boundary face, concatenated into one long array
    ghost_id = np.concatenate(
        [np.full(len(face), g, dtype=np.int32) for g, face in enumerate(boundary_faces)],
        axis=0,
    ) # points each boundary face's points to the corresponding ghost cell id 
    counts = np.bincount(ghost_id, minlength=num_ghost).astype(np.int32) # number of points that belong to each ghost cell, used for averaging later. Should be >0 for all ghost cells.
    assert counts.shape == (num_ghost,), f"Error: counts shape {counts.shape} does not match expected shape {(num_ghost,)}"
    assert np.all(counts > 0), f"Error: found ghost cell with count 0, counts: {counts}"
    
    n_ghost = counts.shape[0]
    summed = jnp.zeros((n_ghost, 3)).at[ghost_id].add(new_points[flat_idx])
    ghost_points = summed / counts[:, None].astype(new_points.dtype)
    return ghost_points

def _compute_new_coarse_points(new_finer_points, fine_graph_key, coarse_graph_key, topology)->jax.Array:
    mapping = topology[coarse_graph_key]['mapping']          # (nDualCells,)
    nDualCells = topology[fine_graph_key]['nDualCells']
    nCoarseDualCells = topology[coarse_graph_key]['nDualCells']

    assert mapping.shape == (nDualCells,)
    assert new_finer_points.shape[0] == nDualCells

    summed = jnp.zeros((nCoarseDualCells, 3), dtype=new_finer_points.dtype).at[mapping].add(new_finer_points)
    counts = jnp.zeros((nCoarseDualCells,), dtype=jnp.int32).at[mapping].add(1)

    return summed / counts[:, None].astype(new_finer_points.dtype)

def _print_diff(name, new, old):
    error = np.abs(new - old)
    print(f"{name} diff (max, mean, min, std): {error.max()}, {error.mean()}, {error.min()}, {error.std()}")

def compute_face_geometry(foam_mesh:dict, vertices:jax.Array)->tuple[jax.Array, jax.Array, jax.Array]:
    faces = foam_mesh["faces"]
    face_points = faces["points"]                    # ragged list[list[int]]
    owner = faces["owner"]

    # ===================== PREPARATION =====================
    counts = np.asarray([len(fp) for fp in face_points], dtype=np.int32) # number of vertices per surface face

    # Flattened vertex ids for p_i
    flat_idx = np.concatenate(
        [np.asarray(fp, dtype=np.int32) for fp in face_points], axis=0
    )

    # Flattened vertex ids for p_{i+1} offset by one for cross product later with flat_idx
    next_idx = np.concatenate(
        [np.roll(np.asarray(fp, dtype=np.int32), -1) for fp in face_points], axis=0
    )

    # We need to keep track of which face each vertex belongs to for the cross product later, so we create an array mapping each vertex -> face id
    face_id = np.concatenate(
        [np.full(len(fp), i, dtype=np.int32) for i, fp in enumerate(face_points)],
        axis=0,
    )

    # ===================== REAL CALCUALTION =====================
    num_faces = counts.shape[0]

    # p_i and p_{i+1} for every polygon edge on every surface face
    pi = vertices[flat_idx]      # (K, 3)
    pj = vertices[next_idx]      # (K, 3)

    # Sum of all cross-products per face in order leads to the area vector, whose magnitude is the area and direction is the normal
    # 0.5 * sum_i cross(p_i, p_{i+1}) = oriented area vector
    area_vec = 0.5 * jnp.zeros((num_faces, 3)).at[face_id].add(jnp.cross(pi, pj))  # (Fsurf, 3)

    # Face areas
    areas = jnp.linalg.norm(area_vec, axis=1)  # (Fsurf,)

    # Unit normals from area vectors
    eps = 1e-8
    normals = -area_vec / jnp.maximum(areas[:, None], eps)

    # Finally, compute face centroids as the average of their vertices
    face_centers = jnp.zeros((num_faces, 3)).at[face_id].add(pi)
    face_centers = face_centers / counts[:, None]

    assert areas.shape == (num_faces,)
    assert normals.shape == (num_faces, 3)
    assert face_centers.shape == (num_faces, 3)
    return normals, areas, face_centers

def build_geometry_case(topology:dict, vertices:np.ndarray, faces:bool)->dict:
    new_points = {'foam_mesh': {'cells':{}, 'faces':{}}}
    for key in topology['hierarchy']:
        new_points[key] = {}
    new_vertices = vertices

    # First we need to update the foam mesh: (points and centroids)
    assert topology['foam_mesh']['points'].shape == vertices.shape, f"Error: topology foam mesh points shape {topology['foam_mesh']['points'].shape} does not match vertices shape {vertices.shape}"    
    new_centroids = _compute_centroids(new_vertices, topology['foam_mesh']['cells']['points'])
    assert topology['foam_mesh']['cells']['centroids'].shape == new_centroids.shape, f"Error: computed centroids shape {new_centroids.shape} does not match expected shape {topology['foam_mesh']['cells']['centroids'].shape}"

    # Then we need to update the dual mesh: 'dual cells'
    ghost_points = _compute_new_ghost_points(new_vertices, topology)
    dual_cells = jnp.vstack([new_centroids, ghost_points]) # shape (num_real_cells + num_ghost_cells, 3)
    assert dual_cells.shape == (topology['dual_graph']['nDualCells'], 3), f"Error: dual_cells shape {dual_cells.shape} does not match expected shape {(topology['dual_graph']['nDualCells'], 3)}"
    new_points['dual_graph']['dual_cells'] = dual_cells

    # Then we need to update the coarse dual meshes: 'dual cells'
    num_levels = len(topology['hierarchy'])
    for i in range(num_levels - 1):
        fine_graph_key, coarse_graph_key = topology['hierarchy'][i], topology['hierarchy'][i+1]
        fine_points = new_points[fine_graph_key]['dual_cells']
        coarse_points = _compute_new_coarse_points(fine_points, fine_graph_key, coarse_graph_key, topology)
        new_points[coarse_graph_key]['dual_cells'] = coarse_points

    # New points
    new_points['foam_mesh']['points'] = new_vertices
    new_points['foam_mesh']['cells']['centroids'] = new_centroids
    
    # Additional values
    if faces:
        normals, face_areas, face_centroids = compute_face_geometry(topology['foam_mesh'], new_vertices)
    else:
        normals, face_areas, face_centroids = None, None, None
    new_points['foam_mesh']['faces']['normals'] = normals
    new_points['foam_mesh']['faces']['areas'] = face_areas
    new_points['foam_mesh']['faces']['centroids'] = face_centroids
    new_points['foam_mesh']['cells']['volumes'] = None
    return new_points

def build_panel_case(topology:dict, panel_topology:dict, warped_geo:dict, panel_data:dict, farfield:dict):

    graph = topology['dual_graph']
    edges = graph['dual_edges']
    points = warped_geo['dual_graph']['dual_cells']

    cell_ids = panel_topology["interp_cell_ids"]          # (N, 4)
    cell_weights = panel_topology["interp_cell_weights"]  # (N, 4)

    # ======= Known nodes =======
    # 1: surface
    surface_nodes = graph['dual_cell_types'] == 1
    # surface_pressures = panel_data['pressure'][panel_topology['projected_surface_map']]
    # surface_vels = panel_data['velocity'][panel_topology['projected_surface_map']]
    surface_pressures = jnp.sum(
        panel_data['pressure'][cell_ids] * cell_weights,
        axis=1,
    )
    surface_vels = jnp.sum(
        panel_data['velocity'][cell_ids] * cell_weights[:, :, None],
        axis=1,
    )

    # 2: far-field
    ff_nodes = graph['dual_cell_types'] == 2
    ff_pressures = farfield['p_inf'] * np.ones(np.sum(ff_nodes))
    ff_vels = farfield['v_inf'] * np.ones((np.sum(ff_nodes), 3))

    # 3: N sampled points
    sampled_indices = panel_topology['sample_indices']
    sampled_pressures = panel_data['sample_pressure']
    sampled_vels = panel_data['sample_velocity']

    # =============== Interpolate ==============
    assert surface_pressures.shape == (np.sum(surface_nodes),)
    assert ff_pressures.shape == (np.sum(ff_nodes),)
    assert sampled_pressures.shape == sampled_indices.shape

    # sampled points should not already be known
    assert not np.any(surface_nodes[sampled_indices])
    assert not np.any(ff_nodes[sampled_indices])

    power = 2.0
    eps = 1e-12

    nbr_idx = panel_topology['nbr_idx']
    known_indices = panel_topology['known_indices']

    # convert local-known indexing -> global node indexing
    nbr_global_idx = known_indices[nbr_idx]       # shape (N, k)
    # warped mesh coordinates
    points = warped_geo['dual_graph']['dual_cells']   # shape (N, 3)
    # coordinates of each node's precomputed neighbors
    nbr_points = points[nbr_global_idx]           # shape (N, k, 3)
    # distance from each node to its k neighbors
    # OLD:
    # dists = jnp.linalg.norm(points[:, None, :] - nbr_points, axis=2)   # shape (N, k)
    # weights = 1.0 / jnp.maximum(dists, eps) ** power

    # NEW:
    diff = points[:, None, :] - nbr_points
    sq_dists = jnp.sum(diff * diff, axis=2)
    # pick epsilon based on dtype
    eps2 =  1e-8
    # since power = 2, inverse-distance weighting becomes inverse squared distance
    weights = 1.0 / (sq_dists + eps2)

    # Interpolate pressures at unknown nodes using inverse distance weighting from known neighbors
    known_pressures = jnp.concatenate([surface_pressures,ff_pressures,sampled_pressures,])
    nbr_pressures = known_pressures[nbr_idx]
    interpolated_pressure = jnp.sum(weights * nbr_pressures, axis=1) / jnp.sum(weights, axis=1)
    assert known_indices.shape == known_pressures.shape
    interpolated_pressure = interpolated_pressure.at[known_indices].set(known_pressures)
    assert interpolated_pressure.shape == (points.shape[0],)

    # Interpolate velocities at unknown nodes using inverse distance weighting from known neighbors
    known_vels = jnp.concatenate([surface_vels,ff_vels,sampled_vels,])
    nbr_vels = known_vels[nbr_idx]
    interpolated_vels = jnp.sum(weights[:, :, None] * nbr_vels, axis=1) / jnp.sum(weights, axis=1)[:, None]
    interpolated_vels = interpolated_vels.at[known_indices].set(known_vels)
    assert interpolated_vels.shape == (points.shape[0], 3)

    # nutilda and T are just the flow condition 
    num_points = points.shape[0]
    interpolated_nuTilda = farfield['nuTilda_inf'] * np.ones(num_points)
    interpolated_T = farfield['T_inf'] * np.ones(num_points)
    return {'p': interpolated_pressure,'v': interpolated_vels, 'nuTilda': interpolated_nuTilda, 'T': interpolated_T}


def get_lowfi_CFD_mesh(path_dir, graph_topology, mesh_load_settings, dual_load_settings, lowfi_load_settings):
    # Translates openfoam mesh data into Python arrays then processes them for GNNs
    mesh_load_settings = check_mesh_settings(mesh_load_settings)
    foam_mesh = load_foam_mesh(path_dir, mesh_load_settings)

    dual_load_settings = check_mesh_settings(dual_load_settings)
    lofi_dual_graph = load_dual_graph(path_dir, foam_mesh, dual_load_settings)
    
    # KD Tree
    from scipy.spatial import cKDTree
    k = 3 #HYPERPARAMETER

    src = np.asarray(lofi_dual_graph["dual_cells"][:lofi_dual_graph["num_real_cells"]], np.float32)
    dst = np.asarray(graph_topology["dual_graph"]["dual_cells"], np.float32)
    _, nn = cKDTree(src).query(dst, k=k)
    nn, src = jnp.asarray(nn, jnp.int32), jnp.asarray(src)

    # Must be jittable wrt graph_x, euler_data, ff
    def data_transfer_function(graph_topology, euler_topology, graph_x, euler_data, ff):
        # states = 12
        # print_content(graph_topology)
        # print("=== euler_topology ===")
        # print_content(euler_topology)
        # print("=== graph_x ===")
        # print_content(graph_x)
        # print("=== euler_data ===")
        # print_content(euler_data)
        # print("=== ff ===")
        # print_content(ff)
        # return states
    
        x = graph_x["dual_graph"]["dual_cells"]
        r2 = jnp.sum((x[:, None, :] - src[nn]) ** 2, axis=-1)
        w = 1.0 / (r2 + 1e-12)
        w = w / jnp.sum(w, axis=1, keepdims=True)
        take = lambda a: jnp.sum(jnp.asarray(a)[nn] * w[..., None], axis=1)
        return {
            "p": take(euler_data["p"][:, None])[:, 0],
            "U": take(euler_data["U"]),
            "T": take(euler_data["T"][:, None])[:, 0],
            "nuTilda": jnp.full((x.shape[0],), ff["nuTilda_inf"]),
        }

    lowfi_full_topology = {
        'foam_mesh': foam_mesh,
        'dual_graph': lofi_dual_graph,
    }
    return {'mapping_function': data_transfer_function, 'topology': lowfi_full_topology}

def build_geometry_data(
        topology:dict,
        cfd_data:list[dict],
        lowfi_dict:dict = None,
        faces:bool = True,
        verbose:bool = False,
        visualize:bool = False,
        device=None)->tuple[list[dict]]:
    if device is None:
        device = jax.devices("cpu")[0]

    data = []
    build_geometry_case_jit = jax.jit(lambda x: build_geometry_case(topology, x, faces), device=device)
    
    # build_panel_case_jit = jax.jit(lambda x,p,ff: build_panel_case(topology, lowfi_topology, x, p, ff), device=device)
    # build_panel_case_jit = lambda x,p,ff: build_panel_case(topology, panel_topology, x, p, ff)

    if lowfi_dict is not None:
        lf_map = lowfi_dict['mapping_function'] 
        lf_topology = lowfi_dict['topology']
        # lf_map_jit = jax.jit(lambda x,p,ff: lf_map(topology, lf_topology, x, p, ff), device=device)
        lf_map_jit = lambda x,p,ff: lf_map(topology, lf_topology, x, p, ff)

    print(f"* Processing {len(cfd_data)} data configurations...")
    for case in cfd_data:
        flow_condition = case['flow_condition']

        # ============================ Graph construction ============================
        vertices = case['vertex_coordinates']
        vertices = jnp.array(vertices)
        start = time.time()
        case_geometry = build_geometry_case_jit(vertices)
        jax.block_until_ready(case_geometry)

        # ============================ OLD Panel Method interpolation ==========================
        # if lowfi_topology is not None:
        #     panel_data = case['panel']
        #     panel_states = build_panel_case_jit(case_geometry, panel_data, flow_condition)
        #     jax.block_until_ready(panel_states)
        #     # panel_states = jax.device_get(panel_states)
        #     # if verbose:
        #     #     print(f"  ** Case {case['case_id']} panel method interpolation done in {time.time() - start:.2f} seconds.")
        #     case_geometry['panel'] = panel_states
        #     panel_forces = {'lift': np.asarray(case['panel']['lift']),'drag': np.asarray(case['panel']['drag']),}
        # else:
        #     panel_states = None
        #     panel_forces = None
        
        # ============================ NEW Lowfidelity interpolation ==========================
        if lowfi_dict is not None:
        # if 0:
            low_fidelity_data = case['low_fidelity_data']
            low_fidelity_inputs = lf_map_jit(case_geometry, low_fidelity_data, flow_condition)
            jax.block_until_ready(low_fidelity_inputs)
            case_geometry['low_fidelity_inputs'] = low_fidelity_inputs
            lofi_info = {
                'lift': low_fidelity_data['lift'],
                'drag': low_fidelity_data['drag'],
            }
        else:
            lofi_info = None

        if verbose:
            print(f"  * Case {case['case_id']} geometry (and panel interpolation) built in {time.time() - start:.2f} seconds.")
        case_geometry = jax.device_get(case_geometry)

        state_data = {
            'p': np.asarray(case['p']),
            'U': np.asarray(case['U']),
            'T': np.asarray(case['T']),
            'nuTilda': np.asarray(case['nuTilda']),
            'lift': np.asarray(case['lift']),
            'drag': np.asarray(case['drag']),
        }
        data.append(
            {
                'topology': topology, 'warped_geo': case_geometry,
                'data': state_data, 'low_fidelity_info': lofi_info, 'flow_condition': flow_condition, 
                'case_id': case['case_id'], 'info': case['info'], 'train': case['train'],
                'aircraft_geometry': case['aircraft_geometry'],
            }
        )

    if verbose == 2:
        print('Final case differences:')
        # Let move all the errors here so that we can see them for each case, and they won't affect the timing of the geometry building
        # _print_diff("\tvertex coordinates ", case_geometry['foam_mesh']['points'], topology['foam_mesh']['points'])
        # _print_diff("\tcell centroids     ", case_geometry['foam_mesh']['cells']['centroids'], topology['foam_mesh']['cells']['centroids'])
        # _print_diff("\tdual cells         ", case_geometry['dual_graph']['dual_cells'], topology['dual_graph']['dual_cells'])
        # _print_diff("\tcoarse dual cells  ", case_geometry['coarse_graph']['dual_cells'], topology['coarse_graph']['dual_cells'])

        print("\n=== Data Config Summary ===")
        print_content(data[-1], 1)
        print("=========================\n")

    train = [config for config in data if config['train']]
    test = [config for config in data if not config['train']]
    num_train, num_test, num_original = len(train), len(test), len(data)
    assert num_train + num_test == num_original, f"Error: train and test split does not sum up to original data size, got {num_train} train and {num_test} test but expected {num_original} total"
    print(f"* Data split: {num_train}:{num_test}:{num_original} (train:test:original)")

    if visualize:
        # lets just print out the first test case and baseline case pressures on centroids
        # find baseline case where (config['case_id] == 0)
        for config in data:
            if config['case_id'] == '0':
                baseline_case = config
            else:
                first_test_config = config
        # plot_configs = [baseline_case, first_test_config]
        plot_configs = train + test

        if lowfi_dict is not None:
            for plot_config in plot_configs:
                points = np.asarray(plot_config['warped_geo']['dual_graph']['dual_cells'])
                interpolated_vels = plot_config['warped_geo']['low_fidelity_inputs']['U']
                interpolated_pressure = plot_config['warped_geo']['low_fidelity_inputs']['p']
                # points_viz = points[surface_nodes]
                # scalars_viz = surface_pressures
                # sampled_indices = panel_topology['sample_indices']
                # points_viz = points[sampled_indices]
                # scalars_viz = interpolated_pressure[sampled_indices]
                # points_viz = points
                # scalars_viz = interpolated_vels[:, 1]

                foam_mask = plot_config['topology']['dual_graph']['dual_cell_types']==0
                p_data = plot_config['data']['p']
                # points_viz = points[foam_mask] #plot error pressure
                points_viz = points #plot error pressure
                scalars_viz = np.abs(interpolated_pressure[foam_mask] - p_data) # plot error pressure
                # scalars_viz = interpolated_pressure[foam_mask]
                scalars_viz = interpolated_pressure
                p = visualize_scalars(
                    points_viz,
                    scalars_viz,
                    title=f"Interpolated panel field",
                    cmap="viridis",
                    # clim=(0,10000),
                )
                p.show()

        from visualize_mesh import visualize_normals
        for plot_config in plot_configs:
            visualize_normals(plot_config['topology'], plot_config['warped_geo'], title=f"Faces+normals (case {plot_config['case_id']})")

        from visualize_mesh import visualize_graph_hierarchy
        for plot_config in plot_configs:
            visualize_graph_hierarchy(plot_config['topology'], plot_config['warped_geo'], title=f"Graph hierarchy (case {plot_config['case_id']})")

        for plot_config in plot_configs:
            # points = plot_config['geometry']['foam_mesh']['cells']['centroids']
            scalars = plot_config['data']['p']

            # mask
            from utils import mask_set
            points = np.asarray(plot_config['warped_geo']['dual_graph']['dual_cells'])
            wall_pos = points[plot_config['topology']['dual_graph']['closest_wall_id']]  # (N,3)
            d = np.linalg.norm(points - wall_pos, axis=1).astype(np.float32)  # (N,)
            mask = (d[plot_config['topology']['dual_graph']['dual_cell_types']==0] < 0.1)  # only visualize interior cells within distance of 5 from wall
            points, scalars =  mask_set(mask, (points[plot_config['topology']['dual_graph']['dual_cell_types']==0], scalars))
            p = visualize_scalars(
                points,
                scalars,
                title=f"P data field ({plot_config['case_id']})",
                cmap="viridis",
                clim=(1e4, 4.5e4),
            )
            p.show()

    return train, test
