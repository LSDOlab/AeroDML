import numpy as np
from aerodml.load_mesh import load_foam_mesh, load_dual_graph, load_coarse_dual_graph, load_coarse_dual_graph
import time

def read_mesh(dir:str, load_settings):
    foam_mesh = load_foam_mesh(dir, load_settings)
    dual_graph = load_dual_graph(dir, foam_mesh, load_settings)
    coarse_graph_l1 = load_coarse_dual_graph(dir, dual_graph, foam_mesh, load_settings, name = 'dual_coarse_mesh_l1', N = 2, max_edge_length=1.7) # equivalent to N=2 on original mesh
    coarse_graph_l2 = load_coarse_dual_graph(dir, coarse_graph_l1, foam_mesh, load_settings, name = 'dual_coarse_mesh_l2', N = 3, max_edge_length=25.7) # equivalent to N=5 on original mesh

    hierarchy = ['dual_graph', 'coarse_graph_l1', 'coarse_graph_l2']
    geometry = {'foam_mesh': foam_mesh, 'dual_graph': dual_graph, 'coarse_graph_l1': coarse_graph_l1, 'coarse_graph_l2': coarse_graph_l2, 'hierarchy': hierarchy}
    for key in hierarchy:
        assert key in geometry, f"Error: key {key} in hierarchy is not present in geometry"

    numPoints = foam_mesh['points'].shape[0]

    # Build the moving surface once, on root only. This returns LOCAL information of just the surface points/faces
    surf_pts_root, surf_conn_root, surf_faceSizes_root = build_surface_definition_from_mesh(foam_mesh)

    local_mesh_idwarp = {
        'points': foam_mesh["points"],
        'surface_connectivity':{
            'pts':surf_pts_root,
            'conn':surf_conn_root,
            'faceSizes':surf_faceSizes_root
        },
    }
    mesh_topo = {}
    mesh_topo['idwarp_USMesh'] = local_mesh_idwarp
    mesh_topo['global_surface_pts'] = surf_pts_root
    mesh_topo['geometry_topology'] = geometry
    return mesh_topo

def read_and_split(comm, dir:str, load_settings):
    
    # Rank == 0 reads the mesh and splits points up across ranks 
    comm.Barrier()
    if comm.rank == 0:
        foam_mesh = load_foam_mesh(dir, load_settings)
        dual_graph = load_dual_graph(dir, foam_mesh, load_settings)
        coarse_graph_l1 = load_coarse_dual_graph(dir, dual_graph, foam_mesh, load_settings, name = 'dual_coarse_mesh_l1', N = 2, max_edge_length=1.7) # equivalent to N=2 on original mesh
        coarse_graph_l2 = load_coarse_dual_graph(dir, coarse_graph_l1, foam_mesh, load_settings, name = 'dual_coarse_mesh_l2', N = 3, max_edge_length=25.7) # equivalent to N=5 on original mesh

        hierarchy = ['dual_graph', 'coarse_graph_l1', 'coarse_graph_l2']
        geometry = {'foam_mesh': foam_mesh, 'dual_graph': dual_graph, 'coarse_graph_l1': coarse_graph_l1, 'coarse_graph_l2': coarse_graph_l2, 'hierarchy': hierarchy}
        for key in hierarchy:
            assert key in geometry, f"Error: key {key} in hierarchy is not present in geometry"

        numPoints = foam_mesh['points'].shape[0]
        print(f"* Distributing {numPoints} points across {comm.size} rank(s) ...")
    else:
        geometry = None
        foam_mesh = None

    local_pts, surf_pts, surf_conn, surf_faceSizes, mesh_topo = split_mesh(foam_mesh, comm)
    global_surf_pts = comm.bcast(surf_pts, root=0)
    print(f'* RANK {comm.rank} assigned {local_pts.shape[0]} points')
    local_mesh_idwarp = {
        'points': local_pts,
        'surface_connectivity':{
            'pts':surf_pts,
            'conn':surf_conn,
            'faceSizes':surf_faceSizes
        },
    }
    mesh_topo['idwarp_USMesh'] = local_mesh_idwarp
    mesh_topo['global_surface_pts'] = global_surf_pts
    mesh_topo['geometry_topology'] = geometry

    comm.Barrier()
    return mesh_topo



def split_mesh(mesh_dict, comm):
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        all_points = np.ascontiguousarray(mesh_dict["points"], dtype=float)
        nGlobalPoints = all_points.shape[0]

        # Split global point ids as evenly as possible
        global_ids = np.arange(nGlobalPoints, dtype=np.int64)
        id_chunks = [np.asarray(chunk, dtype=np.int64)
                     for chunk in np.array_split(global_ids, size)]

        # Build local point chunks
        point_chunks = [np.ascontiguousarray(all_points[idx], dtype=float)
                        for idx in id_chunks]

        # Build the moving surface once, on root only. This returns LOCAL information of just the surface points/faces
        surf_pts_root, surf_conn_root, surf_faceSizes_root = build_surface_definition_from_mesh(mesh_dict)
    else:
        # Will be filled up by the MPI
        nGlobalPoints = None
        id_chunks = None
        point_chunks = None
        surf_pts_root = None
        surf_conn_root = None
        surf_faceSizes_root = None

    # Scatter local volume points and their original global ids
    local_global_ids = comm.scatter(id_chunks, root=0)
    local_pts = comm.scatter(point_chunks, root=0)

    # Surface lives only on root
    if rank == 0:
        surf_pts = surf_pts_root
        surf_conn = surf_conn_root
        surf_faceSizes = surf_faceSizes_root
    else:
        surf_pts = np.zeros((0, 3), dtype=float)
        surf_conn = np.zeros((0,), dtype=np.intc)
        surf_faceSizes = np.zeros((0,), dtype=np.intc)

    # Metadata for reassembly later
    nGlobalPoints = comm.bcast(nGlobalPoints, root=0)
    mesh_topo = {
        "local_global_ids": local_global_ids,
        "nGlobalPoints": nGlobalPoints,
        "surf_pts": surf_pts,
        "surf_conn": surf_conn,
        "surf_faceSizes": surf_faceSizes,
    }

    return local_pts, surf_pts, surf_conn, surf_faceSizes, mesh_topo

def build_surface_definition_from_mesh(mesh):
    all_points = np.asarray(mesh["points"])                  # (nPoints, 3)
    face_point_lists = mesh["faces"]["points"]               # list[list[int]]
    face_type = np.asarray(mesh["faces"]["type"])

    # 1) Pick the faces you want on the moving surface
    surf_faces = [face_point_lists[i] for i in range(len(face_point_lists))
                  if face_type[i] == 1]

    if len(surf_faces) == 0:
        pts = np.zeros((0, 3), dtype=all_points.dtype)
        conn = np.zeros((0,), dtype=np.intc)
        faceSizes = np.zeros((0,), dtype=np.intc)
        return pts, conn, faceSizes

    # 2) Flatten their global point ids
    flat_global_ids = np.array([p for face in surf_faces for p in face], dtype=np.int64)

    # 3) Build compact local numbering for just the surface points
    unique_global_ids, inverse = np.unique(flat_global_ids, return_inverse=True)

    # 4) Surface point coordinates
    pts = np.ascontiguousarray(all_points[unique_global_ids], dtype=float)

    # 5) Flat connectivity into pts (0-based, exactly what IDWarp wants)
    conn = np.asarray(inverse, dtype=np.intc)

    # 6) Face sizes
    faceSizes = np.asarray([len(face) for face in surf_faces], dtype=np.intc)

    return pts, conn, faceSizes

def find_projected_points(query, target, visualize=False, stats=False):
    from scipy.spatial import cKDTree

    start = time.time()
    tree = cKDTree(target)
    distances, indices = tree.query(query, k=1)
    assert np.unique(indices).size == indices.size, f"Duplicate target indices found: multiple surf_points map to the same x_surf_dafoam_full point ({np.unique(indices).size} != {indices.size})"
    print(f"Surface->projected mesh query completed in {time.time() - start:.2f} seconds.")
    
    # From the already projected mesh, get the points corresponding to OUR surface mesh
    # The geometry MUST be the basline geometry at this point
    # from get_geometry_topology import find_projected_points

    query_proxy = target[indices]
    proxy = query_proxy

    # Euclidean error for each matched pair
    errors = np.linalg.norm(query - proxy, axis=1)
    if stats:
        # Summary
        print(f"n_points           = {len(errors)}")
        print(f"max error          = {errors.max():.6e}")
        print(f"mean error         = {errors.mean():.6e}")
        print(f"median error       = {np.median(errors):.6e}")
        print(f"95th percentile    = {np.percentile(errors, 95):.6e}")
        print(f"99th percentile    = {np.percentile(errors, 99):.6e}")

    if visualize:
        # Worst matches
        n_show = 50
        worst_idx = np.argsort(errors)[-n_show:][::-1]

        print("\nWorst matches:")
        for rank, i in enumerate(worst_idx[:10], start=1):
            print(
                f"{rank:2d}: idx={i:6d}, err={errors[i]:.6e}, "
                f"surf={query[i]}, proxy={proxy[i]}"
            )

        import pyvista as pv
        p = pv.Plotter()

        # All points
        p.add_points(
            query,
            color="red",
            point_size=4,
            render_points_as_spheres=True,
        )
        p.add_points(
            proxy,
            color="blue",
            point_size=4,
            render_points_as_spheres=True,
        )

        # Draw lines only for the worst matches
        for i in worst_idx:
            line = pv.Line(query[i], proxy[i])
            p.add_mesh(line, color="lime", line_width=2)

        # Highlight the worst source points and color them by error
        bad_pts = pv.PolyData(query[worst_idx])
        bad_pts["error"] = errors[worst_idx]
        p.add_mesh(
            bad_pts,
            scalars="error",
            point_size=12,
            render_points_as_spheres=True,
        )

        # Optional labels for a few worst ones
        n_label = 10
        label_idx = worst_idx[:n_label]
        labels = [f"{errors[i]:.2e}" for i in label_idx]
        p.add_point_labels(
            query[label_idx],
            labels,
            font_size=12,
            text_color="white",
            shape=None,
            margin=0,
        )

        p.add_axes()
        p.show()

    return indices