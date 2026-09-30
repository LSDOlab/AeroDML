import pyvista as pv
import numpy as np


available_cameras = {
    'pos_1':{ 
        'position': [(-22.2135, -50.2221, 56.7816), (10.9249, 0.147666, 8.07815), (0.340515, 0.528137, 0.777896)],
        'view_angle': 30,
        'clipping_range': (2.8418, 2841.8),
    },
    'pos_2':{
        'position':  [(1.37402, 47.2993, 34.1624), (20.2279, 14.4275, 3.39566), (0.434551, -0.469903, 0.768346)],
        'view_angle': 30,
        'clipping_range': (1.71497, 1714.97),
    },
    'TE':{
        'position': [(50.5074, 30.5439, 9.23596), (24.6976, 12.6643, -1.97501), (-0.292625, -0.167857, 0.941379)],
        'view_angle': 30,
        'clipping_range': (1.81682, 1816.82),
    },
    'TE_underside':{
        'position': [(33.1122, 36.44, -24.4051), (28.8866, 19.0919, 3.74982), (-0.973712, 0.227708, -0.00583269)],
        'view_angle': 30,
        'clipping_range': (1.60824, 1608.24),
    }
}

def set_focal_point(picked_point, pl):
    if picked_point is None:
        return
    cam = pl.camera
    cam.focal_point = picked_point
    print(f"Camera focal point set to: {picked_point}")
    pl.render()

def fmt3(v, n=6):
    return tuple(float(f"{x:.{n}g}") for x in v)

def get_cam_angle(p):
    pos, focal, up = p.camera_position
    pos, focal, up = fmt3(pos), fmt3(focal), fmt3(up)
    w, h = p.window_size
    
    print("\n# paste this:")
    print(f"p.camera_position = [{pos}, {focal}, {up}]")
    print(f"p.camera.view_angle = {float(p.camera.view_angle):.6g}")
    print(f"p.camera.clipping_range = {tuple(float(f'{x:.6g}') for x in p.camera.clipping_range)}")
    print(f"p.window_size = [{w}, {h}]")

def visualize_dual(dual_dict:dict, foam_mesh:dict):
    pts = foam_mesh["points"]                    # (nPoints, 3)
    face_pts = foam_mesh["faces"]["points"]      # list/array: face -> point indices
    ftype = foam_mesh["faces"]["type"]           # (nFaces,)
    nFaces = foam_mesh["faces"]["nFaces"]

    X = dual_dict["dual_cells"]                  # (nDual, 3)
    ctype = dual_dict["dual_cell_types"]         # (nDual,)
    E = dual_dict["dual_edges"]                  # (nFaces, 2)
    n_real = dual_dict["num_real_cells"]

    # ---- boundary mask 
    bmask = ftype > 0

    # ---- ghost points (should be at face centers)
    ghost_mask = np.arange(X.shape[0]) >= n_real
    ghost_pts = pv.PolyData(X[ghost_mask])
    ghost_pts["code"] = ctype[ghost_mask]  # 1,2,3

    # ---- arrows: owner -> ghost(face center)
    u = E[bmask, 0]
    v = E[bmask, 1]
    starts = X[u]
    vecs = X[v] - X[u]

    arrows_src = pv.PolyData(starts)
    arrows_src["vec"] = vecs
    # arrows_src["mag"] = np.linalg.norm(vecs, axis=1).astype(np.float32)
    arrows_src["mag"] = 0.5*np.ones(vecs.shape[0], dtype=np.float32)  # fixed length for better visualization
    arrows_src["code"] = ftype[bmask].astype(np.int16)
    arrows = arrows_src.glyph(orient="vec", scale="mag", factor=1.0, geom=pv.Arrow())

    # ---- boundary faces as polygons
    # Build a PolyData from faces: each face is [n, i0, i1, ...]
    faces = []
    face_codes = []
    for fi in range(nFaces):
        if not bmask[fi]:
            continue
        idx = face_pts[fi]
        faces.append(np.r_[len(idx), np.asarray(idx, dtype=np.int64)])
        face_codes.append(int(ftype[fi]))

    bfaces = pv.PolyData(pts, np.concatenate(faces))
    bfaces["code"] = np.asarray(face_codes, dtype=np.int16)

    # ---- plot (separate from your original mesh)
    p = pv.Plotter()
    p.add_mesh(bfaces, scalars="code", categories=True,
               cmap=["red", "dodgerblue", "limegreen"],  # 1,2,3
               show_edges=True, opacity=0.35,
               scalar_bar_args={"title": "face code"})
    p.add_mesh(ghost_pts, scalars="code", categories=True,
               cmap=["red", "dodgerblue", "limegreen"],
               render_points_as_spheres=True, point_size=1.0)
    # p.add_mesh(arrows, scalars="code", categories=True,
    #            cmap=["red", "dodgerblue", "limegreen"])
    p.add_legend([("1 wall", "red"), ("2 freestream", "dodgerblue"), ("3 symmetry", "limegreen")])
    # p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.025)
    # p.show()

    # Visualize random edges:
    pts = foam_mesh["points"]
    face_pts = foam_mesh["faces"]["points"]
    owner = foam_mesh["faces"]["owner"]
    neigh = foam_mesh["faces"].get("neighbour", [])
    etype = foam_mesh["faces"]["type"]
    et = dual_dict["dual_edge_types"]
    k, seed, arrow_factor = 25, 0, 0.8
    rng = np.random.default_rng(seed)
    cells = rng.choice(n_real, size=min(k, n_real), replace=False)

    # p = pv.Plotter()
    for c in cells:
        # faces incident to cell c (owner==c OR neighbour==c)
        fids = np.flatnonzero(owner == c)
        if len(neigh):
            fids = np.unique(np.r_[fids, np.flatnonzero(np.asarray(neigh) == c)])

        # cell surface as those faces
        faces = [np.r_[len(face_pts[f]), np.asarray(face_pts[f], dtype=np.int64)] for f in fids]
        cell_surf = pv.PolyData(pts, np.concatenate(faces))
        p.add_mesh(cell_surf, color="white", opacity=0.25, show_edges=True)

        # dual node point (centroid)
        cent = pv.PolyData(X[c:c+1]).glyph(geom=pv.Sphere(radius=0.01))
        p.add_mesh(cent, color="yellow")

        # incident dual edges as arrows from centroid to the other node
        em = np.flatnonzero((E[:, 0] == c) | (E[:, 1] == c))
        others = np.where(E[em, 0] == c, E[em, 1], E[em, 0])
        starts = np.repeat(X[c:c+1], len(others), axis=0)
        vecs = X[others] - starts

        src = pv.PolyData(starts)
        src["vec"] = vecs
        src["mag"] = np.linalg.norm(vecs, axis=1).astype(np.float32)
        src["code"] = et[em].astype(np.int16)  # 0 internal, 1/2/3 boundary types
        arrows = src.glyph(orient="vec", scale="mag", factor=arrow_factor, geom=pv.Arrow())
        p.add_mesh(arrows, scalars="code", categories=True,
                   cmap=["lightgray", "red", "dodgerblue", "limegreen"])

    p.add_legend([("0 internal", "lightgray"),
                  ("1 wall", "red"), ("2 freestream", "dodgerblue"), ("3 symmetry", "limegreen")])
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    p.show()

def visualize_coarse(coarse_dict: dict, dual_dict: dict, mesh_dict):
    X = np.asarray(coarse_dict["dual_cells"])                  # (K,3)
    E = np.asarray(coarse_dict["dual_edges"], dtype=np.int64)  # (nE,2)

    # Build the vtk "lines" array: [2,u0,v0, 2,u1,v1, ...]
    lines = np.hstack([np.full((E.shape[0], 1), 2, dtype=np.int64), E]).ravel()

    edge_mesh = pv.PolyData(X)
    edge_mesh.lines = lines 

    p = pv.Plotter()
    p.add_mesh(edge_mesh, color="grey", line_width=1)

    # add points for nodes
    p.add_mesh(pv.PolyData(X), color="black", render_points_as_spheres=True, point_size=5)

    # plot the actual faces from the original mesh, colored by type
    # _visualize_mesh(mesh_dict, p)
    p=visualize_surface(
        {'foam_mesh': mesh_dict},
        {'foam_mesh': mesh_dict},
        None,
        title=None,
        p=p,
        enable_picking=False,
        show_scalar_bar=False,
        smooth_scalars=True,
        add_axes=False,
    )
    y_inverted_pts = mesh_dict['points'].copy()
    y_inverted_pts[:, 1] *= -1
    pts2 = {'foam_mesh': {'points': y_inverted_pts}}
    p=visualize_surface(
        {'foam_mesh': mesh_dict},
        pts2,
        None,
        title=None,
        p=p,
        enable_picking=False,
        show_scalar_bar=False,
        smooth_scalars=True,
        add_axes=False,
    )

    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    p.show()

    # Second plot showing all nodes (including ghost) colored by SDF
    p = pv.Plotter()
    points = X
    vec = X-X[coarse_dict['closest_wall_id']]
    sdf = np.linalg.norm(vec, axis=1)
    mesh = pv.PolyData(points)
    mesh["sdf"] = sdf
    p.add_mesh(mesh, scalars="sdf", cmap="viridis", show_edges=False, render_points_as_spheres=True, point_size=5)

    # plot the actual faces from the original mesh, colored by type
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    p.show() 

def visualize_graph_hierarchy(topology:dict, geometry:dict, title="Graph hierarchy"):
    M = 300
    # pick M random points from the coarsest mesh.
    # visualize that point with a single color, and then visualize the corresponding finer points in the next level with a different color, and so on until the dual mesh.
    fine = topology["hierarchy"][0]      # 'dual_graph'
    mid = topology["hierarchy"][1]       # 'coarse_graph_l1'
    coarse = topology["hierarchy"][2]    # 'coarse_graph_l2'

    # warped cell centers
    x_fine = geometry[fine]["dual_cells"]
    x_mid = geometry[mid]["dual_cells"]
    x_coarse = geometry[coarse]["dual_cells"]

    # mappings
    # mid["mapping"]: fine index  -> mid index
    # coarse["mapping"]: mid index -> coarse index
    fine_to_mid = topology[mid]["mapping"]
    mid_to_coarse = topology[coarse]["mapping"]

    # sample coarse cells
    rng = np.random.default_rng(2)
    M = min(M, len(x_coarse))
    coarse_ids = rng.choice(len(x_coarse), size=M, replace=False)

    p = pv.Plotter()
    p.add_title(title)

    colors = [
            "red", "lime", "cyan", "yellow", "magenta", "orange",
            "blue", "white", "purple", "pink"
        ]

    for i, cid in enumerate(coarse_ids):

        # mid cells belonging to this coarse cell
        mid_ids = np.where(mid_to_coarse == cid)[0]

        # fine cells belonging to those mid cells
        fine_ids = np.where(np.isin(fine_to_mid, mid_ids))[0]

        # plot coarse point
        p.add_mesh(
            pv.PolyData(x_coarse[[cid]]),
            color=colors[0],
            point_size=6,
            render_points_as_spheres=True,
        )

        # plot mid points
        if len(mid_ids) > 0:
            p.add_mesh(
                pv.PolyData(x_mid[mid_ids]),
                color=colors[1],
                point_size=6,
                render_points_as_spheres=True,
            )

        # plot fine points
        if len(fine_ids) > 0:
            p.add_mesh(
                pv.PolyData(x_fine[fine_ids]),
                color=colors[5],
                point_size=6,
                render_points_as_spheres=True,
            )
    p.add_legend([
        ("coarse", colors[0]),
        ("mid", colors[1]),
        ("fine", colors[5]),
    ])
    _visualize_surface(p, topology, geometry, scalars=None)
    p.camera_position = available_cameras['pos_2']['position']
    p.camera.view_angle = available_cameras['pos_2']['view_angle']
    p.camera.clipping_range = available_cameras['pos_2']['clipping_range']
    p.add_title(title)
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    p.show()
    
def _visualize_mesh(mesh_dict:dict, p:pv.Plotter):
    mask = mesh_dict['faces']['type'] > 0
    idx  = np.flatnonzero(mask)
    sel_faces = [mesh_dict['faces']['points'][i] for i in idx]
    pv_faces = np.hstack([np.r_[len(f), f] for f in sel_faces]).astype(np.int64)
    mesh = pv.PolyData(mesh_dict['points'], pv_faces)
    mesh.cell_data["code"] = mesh_dict['faces']['type'][idx].astype(np.int32)

    p.add_mesh(mesh, scalars="code", categories=True,
            cmap=["red", "dodgerblue", "limegreen"],  # 1,2,3
            show_edges=True, scalar_bar_args={"title": "face code"})
    p.add_legend([("1 wall", "red"), ("2 freestream", "dodgerblue"), ("3 symmetry", "limegreen")])

def visualize_foam_mesh(mesh_dict:dict):
    p = pv.Plotter()
    _visualize_mesh(mesh_dict, p)
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    p.camera_position = available_cameras['pos_2']['position']
    p.camera.view_angle = available_cameras['pos_2']['view_angle']
    p.camera.clipping_range = available_cameras['pos_2']['clipping_range']
    p.show()

def visualize_scalars(pos, scalars, title="scalars", cmap="viridis", clim = None, opacity = None,camera = 'pos_1')->pv.Plotter:
    if not isinstance(pos, np.ndarray):
        pos = np.asarray(pos)
    npoints = pos.shape[0]
    assert scalars.shape[0] == npoints, "Number of scalars must match number of points"
    assert pos.shape[1] == 3, "Position array must have shape (npoints, 3)"
    if clim is None:
        clim = (np.min(scalars), np.max(scalars))
    assert isinstance(clim, (tuple, list)) and len(clim) == 2, "clim must be a tuple or list of (min, max)"
    assert clim[0] < clim[1], "Invalid color limits: min must be less than max"

    if isinstance(camera, str):
        cam_params = available_cameras.get(camera)
        if cam_params is None:
            raise ValueError(f"Camera '{camera}' not found in available_cameras ({available_cameras.keys()})")
    elif isinstance(camera, dict):
        cam_params = camera
    else:
        raise ValueError(f"Camera must be a string key ({available_cameras.keys()}) or a dictionary of parameters")

    if opacity is not None:
        # normalize between 0 and 1 based on min and max of opacity
        op_min, op_max = np.min(opacity), np.max(opacity)
        opacity = (opacity - op_min) / (op_max - op_min + 1e-8)
        
    p = pv.Plotter()
    p.add_points(
        pos,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
    )
    p.camera_position = cam_params['position']
    p.camera.view_angle = cam_params['view_angle']
    p.camera.clipping_range = cam_params['clipping_range']
    p.add_title(title)
    p.add_axes()
    p.add_key_event("c", lambda: get_cam_angle(p))
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    return p

def visualize_surface(
        topology:dict,
        pts:dict,
        scalars:np.ndarray,
        title="surface scalars",
        cmap="viridis",
        clim=None,
        camera = 'pos_2',
        p:pv.Plotter = None,
        enable_picking=True,
        lighting=None,
        show_scalar_bar=True,
        smooth_scalars=False,
        add_axes=True,
    ):
    if p is None:
        p = pv.Plotter()
    _visualize_surface(p, topology, pts, scalars, cmap, clim, lighting=lighting,show_scalar_bar=show_scalar_bar, smooth_scalars=smooth_scalars)
    p.camera_position = available_cameras[camera]['position']
    p.camera.view_angle = available_cameras[camera]['view_angle']
    p.camera.clipping_range = available_cameras[camera]['clipping_range']
    if title is not None:
        p.add_title(title, font_size=10)
    if add_axes:
        p.add_axes()
    p.add_key_event("c", lambda: get_cam_angle(p))
    if enable_picking:
        p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
    return p

def _visualize_surface(
        p:pv.Plotter, topology:dict, pts:dict,
        scalars:np.ndarray, cmap="viridis", clim=None,
        lighting=None, show_scalar_bar=True, smooth_scalars=False,
    ):
    mask = topology['foam_mesh']['faces']['type'] == 1
    idx  = np.flatnonzero(mask)
    sel_faces = [topology['foam_mesh']['faces']['points'][i] for i in idx]
    pv_faces = np.hstack([np.r_[len(f), f] for f in sel_faces]).astype(np.int64)
    mesh = pv.PolyData(np.asarray(pts['foam_mesh']['points']), pv_faces)

    if scalars is None:
        surface_scalars = scalars
    elif scalars.shape[0] != np.sum(mask):
        scalar_idx = topology['foam_mesh']['faces']['surface_neighbors']  # precomputed mapping from surface face to neighboring cell
        surface_scalars = scalars[scalar_idx]
    else:
        surface_scalars = scalars

    if surface_scalars is not None:
        mesh.cell_data["field"] = surface_scalars

        if smooth_scalars:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
            scalars_to_plot = "field"
        else:
            scalars_to_plot = "field"
    else:
        scalars_to_plot = None

    p.add_mesh(mesh, scalars=scalars_to_plot, categories=True,
            cmap=cmap,  # 1,2,3
            clim=clim,
            show_edges=False,
            lighting=lighting,
            show_scalar_bar =show_scalar_bar,
        )
    
def visualize_normals(topology:dict, geometry:dict, title="Normals", camera='pos_2'):
    p = pv.Plotter()
    _visualize_surface(p, topology, geometry, None, show_scalar_bar=False)

    face_ids = np.random.choice( topology['foam_mesh']['faces']['nFaces'], size=1000, replace=False)
    # face_ids = topology['foam_mesh']['faces']['type'] == 1  # precomputed mapping from surface face to neighboring cell
    
    # Add normal vectors
    face_centers = geometry['foam_mesh']['faces']['centroids']  # (nFaces, 3)
    normals = geometry['foam_mesh']['faces']['normals']  # (nFaces, 3)
    face_centers = face_centers[face_ids]
    normals = normals[face_ids]
    p.add_arrows(face_centers, normals, mag=0.3, color='red')

    # Add corresponing faces from mesh
    face_pts = topology['foam_mesh']['faces']['points']
    faces = np.hstack([
        np.r_[len(face_pts[f]), np.asarray(face_pts[f], dtype=np.int64)]
        for f in face_ids
    ]).astype(np.int64)
    cell_surf = pv.PolyData(geometry['foam_mesh']['points'], faces)
    areas = geometry['foam_mesh']['faces']['areas'][face_ids]
    cell_surf["surface_areas"] = areas
    p.add_mesh(cell_surf, color="white", opacity=0.25, show_edges=True, scalars='surface_areas', cmap='viridis', scalar_bar_args={"title": "face area"}, clim=(areas.min(), areas.max()))

    p.add_title(title)
    p.add_axes()
    p.add_key_event("c", lambda: get_cam_angle(p))
    p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)

    p.camera_position = available_cameras[camera]['position']
    p.camera.view_angle = available_cameras[camera]['view_angle']
    p.camera.clipping_range = available_cameras[camera]['clipping_range']
    
    p.show()

def build_pyvista_mesh_topology(foam)->tuple[np.ndarray, np.ndarray]:
    cell_face_ids = foam["cells"]["faces"]                           # list[list[int]]
    face_point_ids = foam["faces"]["points"]                         # list[list[int]]
    owner = np.asarray(foam["faces"]["owner"], dtype=np.int64)       # (nFaces,)
    neighbour = np.asarray(foam["faces"]["neighbour"], dtype=np.int64)  # may be -1 on boundary, or shorter in some exports

    n_cells = foam["cells"]["nCells"]

    # All cells as general polyhedra
    celltypes = np.full(n_cells, pv.CellType.POLYHEDRON, dtype=np.uint8)

    cells_flat = []

    for c in range(n_cells):
        fids = cell_face_ids[c]

        # polyhedron payload begins with number of faces
        poly = [len(fids)]

        for fid in fids:
            face_pts = face_point_ids[fid]

            # Make the face winding outward for THIS cell.
            # OpenFOAM-style rule of thumb:
            # - owner side: keep order
            # - neighbour side: reverse order
            if fid < len(neighbour) and neighbour[fid] == c:
                face_pts = face_pts[::-1]

            poly.append(len(face_pts))
            poly.extend(face_pts)

        # Prepend NItems for this polyhedron
        cells_flat.append(len(poly))
        cells_flat.extend(poly)

    cells_flat = np.asarray(cells_flat, dtype=np.int64)
    return cells_flat, celltypes 


def visualize_raw_panel_data(panel_topology:dict, panel_data:dict, state:str, screenshot_path=None):
    tris = panel_topology['triangles']  # (nTris, 3)
    points = panel_topology['surface_points']     # (nPoints, 3)
    num_cells = tris.shape[0]

    if state == 'p':
        surface_field = panel_data['pressure']
        clim = (1e4, 4e4)

    # build triangular surface mesh
    faces = np.hstack([
        np.full((num_cells, 1), 3, dtype=np.int64),
        tris.astype(np.int64),
    ]).ravel()

    surf = pv.PolyData(points, faces)
    surf["field"] = surface_field
    if screenshot_path is not None:
        p = pv.Plotter(off_screen=True)
    else:
        p = pv.Plotter()
    p.add_mesh(surf, scalars='field', clim=clim)
    edges = surf.extract_all_edges()
    p.add_mesh(edges, line_width=3.5, color='k')

    p.camera_position = [(-3.85837, 52.2268, 41.954), (18.9548, 12.4519, 4.72624), (0.434551, -0.469903, 0.768347)]
    p.camera.view_angle = 30
    p.camera.clipping_range = (18.6938, 133.043)
    p.window_size = [1986, 1376]

    if screenshot_path is not None:
        p.screenshot(screenshot_path, transparent_background=True)
    else:
        p.add_key_event("c", lambda: get_cam_angle(p))
        p.enable_surface_point_picking(callback=lambda pp: set_focal_point(pp, p), show_point=True, tolerance=0.0025)
        p.show()
    p.close()
def add_sectional_planes(
        p,
        topology,
        pts2,
        sectional_data,
    ):
    p = add_section_planes(
        p,
        sectional_data,
        mirror_y=False,
        opacity=0.25,
    )

    p = add_y_plane_intersection_lines(
        p,
        topology,
        pts2,
        sectional_data,
        y_signs=(-1,),
        color="black",
        line_width=12,
        tube_radius=0.06,
    )
    return p


def make_y_section_rect(y, xlim, zlim):
    x0, x1 = xlim
    z0, z1 = zlim
    points = np.array([
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ])
    faces = np.array([4, 0, 1, 2, 3])
    import pyvista as pv
    return pv.PolyData(points, faces)

def make_y_section_border(y, xlim, zlim):
    x0, x1 = xlim
    z0, z1 = zlim
    points = np.array([
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ])
    lines = np.array([
        2, 0, 1,
        2, 1, 2,
        2, 2, 3,
        2, 3, 0,
    ])
    import pyvista as pv
    return pv.PolyData(points, lines=lines)

def add_section_planes(p, sectional_data, mirror_y=True, opacity=0.25):
    i = 0
    for y_plane, xlim, zlim in sectional_data:
        # if i == 0:
        #     y_plane = -y_plane
        i += 1
        for y in ([y_plane, -y_plane] if mirror_y else [-y_plane]):
            rect = make_y_section_rect(y, xlim, zlim)
            border = make_y_section_border(y, xlim, zlim)

            # translucent filled rectangle
            p.add_mesh(
                rect,
                color="black",
                opacity=opacity,
                show_edges=False,
                lighting=False,
                pickable=False,
            )

            # bold outer border
            p.add_mesh(
                border,
                color="black",
                line_width=2,
                lighting=False,
                pickable=False,
                render_lines_as_tubes=True,
            )

    return p

def make_surface_mesh(topology, pts):
    mask = topology["foam_mesh"]["faces"]["type"] == 1
    idx = np.flatnonzero(mask)

    sel_faces = [topology["foam_mesh"]["faces"]["points"][i] for i in idx]
    pv_faces = np.hstack([np.r_[len(f), f] for f in sel_faces]).astype(np.int64)
    import pyvista as pv
    return pv.PolyData(np.asarray(pts["foam_mesh"]["points"]), pv_faces)

def add_y_plane_intersection_lines(
    p,
    topology,
    pts,
    sectional_data,
    y_signs=(1,),
    color="black",
    line_width=10,
    tube_radius=None,
):
    surface = make_surface_mesh(topology, pts)

    for y_plane, xlim, zlim in sectional_data:
        for sign in y_signs:
            y = sign * y_plane

            line = surface.slice(
                normal="y",
                origin=(0, y, 0),
            )

            if line.n_points == 0:
                continue

            if tube_radius is not None:
                line = line.tube(radius=tube_radius)

            p.add_mesh(
                line,
                color=color,
                line_width=line_width,
                lighting=False,
                pickable=False,
                render_lines_as_tubes=(tube_radius is None),
            )

    return p