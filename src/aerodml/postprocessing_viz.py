from .utils import plot_states, normalize, unnormalize, stack_func, mask_set, print_content
from dataclasses import dataclass, replace

stacked_normalize = stack_func(normalize, in_axes=(0, None, None))
stacked_unnormalize = stack_func(unnormalize, in_axes=(0, None, None))
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from .visualize_mesh import visualize_scalars, visualize_surface, build_pyvista_mesh_topology
import time
import math

def visualize_volume_post(
        model_eval,
        params:dict,
        state_name:str,
        config,
        constants,
        distance_cutoff=0.1,
    ):
    import jax
    pts = config['warped_geo']
    data = config['data']
    topology = config['topology']

    # v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, pts)
    case_flow_condition = jax.device_put(config['flow_condition'], device=constants['device'])
    v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, case_flow_condition, pts)

    state_name_mapping = {
        'v': jax.device_get(v_pred),
        'T': jax.device_get(T_pred),
        'p': jax.device_get(p_pred),
        'nuTilda': jax.device_get(nuTilda_pred)
    }
    state_pred = state_name_mapping[state_name]

    # unnormalize predictions
    # coords = data['centroid_coordinates']
    coords = pts['dual_graph']['dual_cells'][topology['dual_graph']['dual_cell_types']==0]
    dual_cells = pts['dual_graph']['dual_cells']
    wall_pos = dual_cells[topology['dual_graph']['closest_wall_id']]  # (N,3)
    d = np.linalg.norm(dual_cells - wall_pos, axis=1).astype(np.float32)  # (N,)
    mask = (d < distance_cutoff)[topology['dual_graph']['dual_cell_types']==0]  # only visualize interior cells within distance of 5 from wall
    # mask = (d < 10.0)[topology['dual_graph']['dual_cell_types']==0]  # only visualize interior cells within distance of 5 from wall
    # mask = p_pred>32000
    # coords_mask, v_pred_mask, T_pred_mask, p_pred_mask, nuTilda_pred_mask = mask_set(mask, (coords, v_pred, T_pred, p_pred, nuTilda_pred))
    p_pred = stacked_unnormalize(state_pred, constants[state_name]['mean'], constants[state_name]['std'])
    coords_mask, p_pred_mask = mask_set(mask, (coords, p_pred))

    # p_pred = jax.device_get(p_pred)*1.0
    # p_pred[topology['foam_mesh']['faces']['surface_neighbors']] = 0.0
    # p = visualize_scalars(
    #     coords,
    #     p_pred,
    #     title=f"{state_name} scalar field",
    #     cmap="viridis",
    #     clim=(0.0, 4.5e4),
    #     camera = 'pos_1'
    # )
    # p.show()

    p = visualize_scalars(
        coords_mask,
        p_pred_mask,
        title=f"{state_name} scalar field",
        cmap="viridis",
        clim=(1e4, 4.5e4),
        camera = 'pos_1'
    )
    # p.show()

    # mask = data['p']>32000
    coords_mask, p_error_mask= mask_set(mask, (coords, data[state_name]))
    # Pressure data near body
    p = visualize_scalars(
        coords_mask,
        p_error_mask,
        title=f"{state_name} data field",
        cmap="viridis",
        clim=(1e4, 4.5e4),
    )
    p.show()

    error = np.abs(data[state_name]-p_pred)
    mask = (d < distance_cutoff)[topology['dual_graph']['dual_cell_types']==0]  # only visualize interior cells within distance of 5 from wall

    coords_mask, error_mask= mask_set(mask, (coords,error))
    coords_mask, error_mask= mask_set(error_mask<1e4, (coords_mask,error_mask))
    # Pressure near body
    p = visualize_scalars(
        coords_mask,
        error_mask,
        title=f"{state_name} error field",
        cmap="Oranges",
        clim= (0, 1e4),
        opacity=error_mask
    )
    # p.show()

    # p = visualize_surface(
    #     topology,
    #     scalars,
    #     title="P surface field",
    #     cmap="viridis",
    #     clim=(1e4, 4.5e4),
    # )

def visualize_surface_post(
        model_eval,
        params,
        config,
        constants,
        title = "",
        state = 'p',
        camera = 'pos_2',
        y_plane = 0.5,
        window_size= None,
        enable_picking = True,
        pyvista_mesh_topology = None,
        return_image = False,
        print_l2 = False,
        plot_error = True,
    ):
    freestream = config['flow_condition']
    pts = config['warped_geo']
    data = config['data']
    topology = config['topology']

    freestream_config = {**freestream}
    # import jax.numpy as jnp
    # freestream_config['v_inf'] = jnp.array([197.04, 0.0, 0.0])
    # start = time.time()
    v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, freestream_config, pts)
    # v_pred.block_until_ready()
    # print(time.time()-start)

    # unnormalize predictions
    # coords = data['centroid_coordinates']
    coords = pts['dual_graph']['dual_cells'][topology['dual_graph']['dual_cell_types']==0]
    v_pred = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])
    T_pred = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
    p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
    nuTilda_pred = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])

    # data = {
    #     'p': p_pred,
    #     'T':T_pred,
    # }

    cmap = "viridis"
    if state == 'p':
        pred_val, data_val, clim = p_pred, data['p'], (1e4, 4.e4)
        # pred_val, data_val, clim = np.abs(p_pred-data['p']), data['p'], (0, 1e4)
        # clim = (-1e4,1e4)
    elif state == 'T':
        pred_val, data_val, clim = T_pred, data['T'], (180, 290)
        cmap = "coolwarm"
    elif state == 'nuTilda':
        pred_val, data_val, clim = nuTilda_pred, data['nuTilda'], (0, 0.1)
    elif state == 'v':
        pred_val, data_val, clim = np.linalg.norm(v_pred, axis=1), np.linalg.norm(data['U'], axis=1), (50, 340)

    # Signed error: red is overprediction, blue is underprediction, and the
    # light-gray midpoint of coolwarm is zero error. Symmetric limits keep zero
    # at the center of the color scale.
    error_limit = {
        'p': 3e3,
        'T': 50,
        'nuTilda': 0.05,
        'v': 50,
    }[state]
    error_val = pred_val - data_val
    error_clim = (-error_limit, error_limit)

    if print_l2:
        l2_error = np.linalg.norm(pred_val - data_val) / np.linalg.norm(data_val)
        print(f"Relative L2 error {state}: {l2_error:.4f}")

    import pyvista as pv

    # Build slice
    if pyvista_mesh_topology is None:
        cells_flat, celltypes = build_pyvista_mesh_topology(topology['foam_mesh'])
    else:
        cells_flat, celltypes = pyvista_mesh_topology
    grid = pv.UnstructuredGrid(cells_flat, celltypes, pts['foam_mesh']['points'] )
    grid.cell_data["data"] = data_val*1.0
    slc_data = grid.slice(normal="y", origin=(00, y_plane, 0), generate_triangles=True)
    grid.cell_data["pred"] = pred_val*1.0
    slc_pred = grid.slice(normal="y", origin=(00, y_plane, 0), generate_triangles=True)

    off_screen = return_image
    if title is None:
        f_title = ""
    else:
        f_title = f"{title} (Predicted)"

    if plot_error:
        pl_shape = (3,1)
    else:
        pl_shape = (2,1)
    
    pl = pv.Plotter(shape=pl_shape, window_size=window_size, off_screen=off_screen)
    pl.subplot(0,0)
    p=visualize_surface(
        topology,
        pts,
        pred_val,
        title=f_title,
        cmap=cmap,
        clim=clim,
        camera = camera,
        p=pl,
        enable_picking=enable_picking,
        show_scalar_bar=False,
    )
    p.add_mesh(slc_pred, scalars="pred", cmap=cmap, clim = clim, lighting = False, show_scalar_bar=False)
    
    if title is None:
        f_title = ""
    else:
        f_title = f"{title} (Data)"
    pl.subplot(1,0)
    p = visualize_surface(
        topology,
        pts,
        data_val,
        title=f_title,
        cmap=cmap,
        clim=clim,
        camera = camera,
        p=pl,
        enable_picking=False,
        lighting = None,
        show_scalar_bar=False,
    )
    p.add_mesh(slc_data, scalars="data", cmap=cmap, clim = clim, lighting = False)
    if plot_error:
        grid.cell_data["error"] = error_val*1.0
        slc_pred = grid.slice(normal="y", origin=(00, y_plane, 0), generate_triangles=True)
        pl.subplot(2,0)
        p = visualize_surface(
            topology,
            pts,
            error_val,
            title=f"{title} (Error: Prediction - Data)",
            cmap="coolwarm",
            clim=error_clim,
            camera = camera,
            p=pl,
            enable_picking=False,
            lighting = None,
            show_scalar_bar=False,
        )
        p.add_mesh(slc_pred, scalars="error", cmap="coolwarm", clim = error_clim, lighting = False)

    # light = pv.Light(position=(0, 100, 500), focal_point=(0, 0, 0), light_type="scene light")
    # pl.add_light(light)

    pl.link_views()
    if return_image:
        img = pl.screenshot(None, transparent_background=True)
        pl.close()
        return img
    else:
        pl.show()

def plot_case_histogram(model_eval, params, test_data, constants):
    # plot histogram of errors across cases
    case_ids = [config['case_id'] for config in test_data]
    p_errors = []
    for config in test_data:
        pts = config['warped_geo']
        data = config['data']
        topology = config['topology']
        v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, pts)
        p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
        error = np.abs(data['p'] - p_pred)
        mean_error = np.mean(error)
        p_errors.append(mean_error)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    # plt.bar(case_ids, p_errors, color='skyblue')

    # color based on if test or train
    colors = ['skyblue' if config['train'] else 'salmon' for config in test_data]
    plt.bar(case_ids, p_errors, color=colors)

    plt.xlabel('Case ID')
    plt.ylabel('Mean Absolute Pressure Error')
    plt.title('Mean Absolute Pressure Error for Each Test Case')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.2)
    plt.tight_layout() 
    plt.show()

def relative_l2(pred, true, weights=None, eps=1e-12):
    """
    Relative L2 error:
        ||pred - true||_2 / ||true||_2

    Works for scalar or vector fields.
    If weights are provided (e.g. cell volumes), they are broadcast over trailing dims.
    """
    import jax
    import jax.numpy as jnp
    pred = np.asarray(pred)
    true = np.asarray(true)

    sq_err = (pred - true) ** 2
    sq_true = true ** 2

    if weights is not None:
        w = jnp.asarray(weights)
        while w.ndim < sq_err.ndim:
            w = w[..., None]
        sq_err = sq_err * w
        sq_true = sq_true * w

    num = np.sqrt(np.sum(sq_err))
    den = np.sqrt(np.sum(sq_true))

    return num / np.maximum(den, eps)

def _aero_plot_output(model_name, plots_dir):
    """Return the shared model-specific plot directory and filename slug."""
    import re
    from pathlib import Path

    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip()).strip("._")
    model_slug = model_slug or "unnamed_model"
    return Path(plots_dir) / model_slug, model_slug


@dataclass(frozen=True)
class XCWindow:
    """Geometry and framing options for one planar cross-section.

    ``plane`` is ``(axis_or_normal, point_on_plane)``. For example,
    ``('y', (0, 5, 0))`` selects the x-z plane at y=5 m. Set ``center`` and/or
    ``width`` to ``'auto'`` to frame the aircraft surface intersection. The
    padding is the fractional margin added on every side of that intersection.
    """

    plane: tuple
    center: object = "auto"
    width: object = "auto"
    aspect_ratio: float = 9 / 7
    padding: float = 0.15


def _xc_plane_geometry(plane):
    """Return a normalized plane normal and a point on the plane."""
    if not isinstance(plane, (tuple, list)) or len(plane) != 2:
        raise ValueError("XCWindow.plane must be (axis_or_normal, point_on_plane)")
    direction, origin = plane
    if isinstance(direction, str):
        axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        try:
            normal = axes[direction.lower()].copy()
        except KeyError as exc:
            raise ValueError("XCWindow plane axis must be 'x', 'y', or 'z'") from exc
    else:
        normal = np.asarray(direction, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("XCWindow plane normal must contain three finite components")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("XCWindow plane point must contain three finite coordinates")
    magnitude = np.linalg.norm(normal)
    if magnitude == 0.0:
        raise ValueError("XCWindow plane normal must be nonzero")
    return normal / magnitude, origin


def _xc_view_basis(normal):
    """Return horizontal and vertical unit vectors for a section camera."""
    vertical = np.array([0.0, 0.0, 1.0])
    vertical -= np.dot(vertical, normal) * normal
    if np.linalg.norm(vertical) < 1e-12:
        vertical = np.array([0.0, 1.0, 0.0])
        vertical -= np.dot(vertical, normal) * normal
    vertical /= np.linalg.norm(vertical)
    horizontal = np.cross(normal, vertical)
    horizontal /= np.linalg.norm(horizontal)
    return horizontal, vertical


def _cfd_wall_surface(config, values=None):
    """Build the existing CFD wall surface mesh without opening a plotter."""
    from .visualize_mesh import _visualize_surface

    class SurfaceCollector:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh

    collector = SurfaceCollector()
    if values is None:
        values = config["data"]["p"]
    _visualize_surface(
        collector,
        config["topology"],
        config["warped_geo"],
        np.asarray(values).reshape(-1),
    )
    return collector.mesh


def _resolve_xc_window(window, surface_mesh):
    """Resolve automatic XCWindow fields against a wall-surface intersection."""
    if not isinstance(window, XCWindow):
        raise TypeError("cross_section_stations entries must be XCWindow objects")
    normal, origin = _xc_plane_geometry(window.plane)
    horizontal, vertical = _xc_view_basis(normal)
    if not np.isfinite(window.aspect_ratio) or window.aspect_ratio <= 0.0:
        raise ValueError("XCWindow.aspect_ratio must be finite and greater than zero")
    if not np.isfinite(window.padding) or window.padding < 0.0:
        raise ValueError("XCWindow.padding must be finite and nonnegative")

    center_is_auto = isinstance(window.center, str) and window.center.lower() == "auto"
    width_is_auto = isinstance(window.width, str) and window.width.lower() == "auto"
    if isinstance(window.center, str) and not center_is_auto:
        raise ValueError("XCWindow.center must be a 3D point or 'auto'")
    if isinstance(window.width, str) and not width_is_auto:
        raise ValueError("XCWindow.width must be a positive number or 'auto'")

    intersection_points = None
    if center_is_auto or width_is_auto:
        intersection = surface_mesh.slice(normal=normal, origin=origin)
        if intersection.n_points == 0:
            raise ValueError(
                f"XCWindow plane through {tuple(origin)} does not intersect the aircraft surface"
            )
        intersection_points = np.asarray(intersection.points)
        offsets = intersection_points - origin
        horizontal_coords = offsets @ horizontal
        vertical_coords = offsets @ vertical

    if center_is_auto:
        center = (
            origin
            + horizontal * 0.5 * (horizontal_coords.min() + horizontal_coords.max())
            + vertical * 0.5 * (vertical_coords.min() + vertical_coords.max())
        )
    else:
        center = np.asarray(window.center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("XCWindow.center must contain three finite coordinates")
        plane_scale = max(float(np.linalg.norm(center - origin)), 1.0)
        if abs(float(np.dot(center - origin, normal))) > 1e-8 * plane_scale:
            raise ValueError("XCWindow.center must lie on XCWindow.plane")

    if width_is_auto:
        centered_offsets = intersection_points - center
        half_horizontal = float(np.max(np.abs(centered_offsets @ horizontal)))
        half_vertical = float(np.max(np.abs(centered_offsets @ vertical)))
        width = 2.0 * (1.0 + window.padding) * max(
            half_horizontal, half_vertical * window.aspect_ratio
        )
        if width <= 0.0:
            raise ValueError("Aircraft surface intersection has zero extent")
    else:
        width = float(window.width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("XCWindow.width must be finite and greater than zero")

    return replace(window, center=tuple(center), width=width)


def _xc_window_meshes(window, mirror_y=False):
    """Build translucent rectangle and border meshes for a resolved window."""
    import pyvista as pv

    normal, _ = _xc_plane_geometry(window.plane)
    horizontal, vertical = _xc_view_basis(normal)
    center = np.asarray(window.center, dtype=float)
    half_width = float(window.width) / 2.0
    half_height = half_width / window.aspect_ratio
    corners = np.asarray([
        center - horizontal * half_width - vertical * half_height,
        center + horizontal * half_width - vertical * half_height,
        center + horizontal * half_width + vertical * half_height,
        center - horizontal * half_width + vertical * half_height,
    ])
    if mirror_y:
        corners[:, 1] *= -1.0
    rectangle = pv.PolyData(corners, np.array([4, 0, 1, 2, 3]))
    border = pv.PolyData(corners, lines=np.array([5, 0, 1, 2, 3, 0]))
    return rectangle, border


def _mach_number(velocity, temperature, constants):
    """Return local Mach number from dimensional velocity and temperature."""
    velocity = np.asarray(velocity)
    temperature = np.asarray(temperature).reshape(-1)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("Mach number requires velocity values with shape (n_cells, 3)")
    if velocity.shape[0] != temperature.size:
        raise ValueError("Velocity and temperature must have the same cell count")

    gamma = float(constants["gamma"])
    gas_constant = float(constants["R"])
    sound_speed_squared = gamma * gas_constant * temperature
    if gamma <= 0.0 or gas_constant <= 0.0 or np.any(sound_speed_squared <= 0.0):
        raise ValueError(
            "Mach number requires positive gamma, gas constant, and temperature"
        )
    return np.linalg.norm(velocity, axis=1) / np.sqrt(sound_speed_squared)


def _field_values_and_primitives(
        model_eval, params, config, constants, state, model_type,
    ):
    """Resolve a physical field and any primitives needed for 2D resampling."""
    if state not in {"p", "Ma"}:
        raise NotImplementedError("supported plot states are 'p' and 'Ma'")

    data = config["data"]
    primitives = {}
    if state == "p":
        cfd_values = np.asarray(data["p"]).reshape(-1)
    else:
        cfd_velocity = np.asarray(data["U"])
        cfd_temperature = np.asarray(data["T"]).reshape(-1)
        primitives["CFD"] = (cfd_velocity, cfd_temperature)
        cfd_values = _mach_number(cfd_velocity, cfd_temperature, constants)

    if model_type == "CFD":
        return cfd_values, primitives

    prediction = model_eval(
        params, config["flow_condition"], config["warped_geo"]
    )
    if state == "p":
        predicted_values = (
            np.asarray(prediction[2]).reshape(-1) * constants["p"]["std"]
            + constants["p"]["mean"]
        )
    else:
        predicted_velocity = (
            np.asarray(prediction[0]) * constants["v"]["std"]
            + constants["v"]["mean"]
        )
        predicted_temperature = (
            np.asarray(prediction[1]).reshape(-1) * constants["T"]["std"]
            + constants["T"]["mean"]
        )
        predicted_values = _mach_number(
            predicted_velocity, predicted_temperature, constants
        )
        primitives["NN"] = (predicted_velocity, predicted_temperature)

    if predicted_values.size != cfd_values.size:
        raise ValueError(
            f"NN {state} has {predicted_values.size} values, but CFD {state} "
            f"has {cfd_values.size} values"
        )
    values = predicted_values - cfd_values if model_type == "error" else predicted_values
    return values, primitives


def _field_values(model_eval, params, config, constants, state, model_type):
    """Resolve one physical CFD, NN, or signed-error volume field."""
    values, _ = _field_values_and_primitives(
        model_eval, params, config, constants, state, model_type
    )
    return values


def _resolve_contour_levels(contour_levels, clim):
    """Normalize an isoline count or explicit values into physical levels."""
    if contour_levels is None:
        return None
    if isinstance(contour_levels, (int, np.integer)):
        if contour_levels <= 0:
            raise ValueError("contour_levels must be a positive integer")
        return np.linspace(clim[0], clim[1], int(contour_levels) + 2)[1:-1]

    levels = np.asarray(contour_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0 or not np.isfinite(levels).all():
        raise ValueError(
            "contour_levels must be a positive integer or a nonempty sequence "
            "of finite values"
        )
    return np.unique(levels)


def _contour_label_locations(contours, levels):
    """Return one arc-length midpoint on the longest line for each level."""
    if contours.n_lines == 0 or "field" not in contours.point_data:
        return np.empty((0, 3)), []

    stripped = contours.strip(join=True, max_length=10_000_000)
    lines = np.asarray(stripped.lines)
    field = np.asarray(stripped.point_data["field"])
    best_by_level = {}
    cursor = 0
    while cursor < lines.size:
        point_count = int(lines[cursor])
        point_ids = lines[cursor + 1:cursor + 1 + point_count]
        cursor += point_count + 1
        if point_count < 2:
            continue

        points = np.asarray(stripped.points[point_ids])
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length <= 0.0:
            continue

        line_value = float(np.nanmedian(field[point_ids]))
        level_index = int(np.argmin(np.abs(levels - line_value)))
        current = best_by_level.get(level_index)
        if current is not None and current[0] >= total_length:
            continue

        midpoint_distance = total_length / 2.0
        cumulative = np.cumsum(segment_lengths)
        segment_index = int(np.searchsorted(cumulative, midpoint_distance))
        previous_distance = 0.0 if segment_index == 0 else cumulative[segment_index - 1]
        fraction = (
            (midpoint_distance - previous_distance) / segment_lengths[segment_index]
        )
        midpoint = points[segment_index] + fraction * (
            points[segment_index + 1] - points[segment_index]
        )
        best_by_level[level_index] = (total_length, midpoint)

    selected_indices = sorted(best_by_level)
    locations = np.asarray([
        best_by_level[index][1] for index in selected_indices
    ])
    selected_levels = [float(levels[index]) for index in selected_indices]
    return locations, selected_levels


def _resolve_grid_2d_dimensions(grid_2d, aspect_ratio):
    """Return an isotropic regular-grid resolution, or ``None`` when disabled."""
    if grid_2d is None or grid_2d is False:
        return None
    if grid_2d is True:
        width_points = 600
        return width_points, max(2, round(width_points / aspect_ratio))
    if isinstance(grid_2d, (int, np.integer)):
        if grid_2d < 2:
            raise ValueError("grid_2d integer resolution must be at least 2")
        width_points = int(grid_2d)
        return width_points, max(2, round(width_points / aspect_ratio))
    if isinstance(grid_2d, (tuple, list)) and len(grid_2d) == 2:
        dimensions = tuple(int(value) for value in grid_2d)
        if min(dimensions) < 2:
            raise ValueError("grid_2d dimensions must each be at least 2")
        return dimensions
    raise TypeError(
        "grid_2d must be False, True, an integer width, or an (nx, ny) pair"
    )


def plot_surface_comparison(
        model_eval, params, config, constants, screenshot=False, model_name="",
        left="NN", right="CFD", plots_dir="post_processing_plots",
        panel_data=None, panel_topology=None, clim=(1e4, 4e4),
        window_size=(1800, 1400), smooth_scalars=False, cmap="jet",
        cross_section_stations=None,
    ):
    """Plot upper/lower planforms, nose up, with independent pressure sources.

    Sources are 'NN', 'low-fidelity', and 'CFD'. Both views use an orthographic
    camera and the same pressure scale (Pa). Symmetric geometry is mirrored
    about y=0; the lower view preserves the requested screen-left source.
    Native panel plotting requires panel_data (panel_mesh, pressure) and
    panel_topology (triangles, quads), optionally stored on config. Otherwise
    low-fidelity uses the interpolated inputs on the CFD wall mesh.
    Set smooth_scalars=True to interpolate face pressures onto mesh points,
    matching the smooth surface rendering used by visualize_paper.
    Use cmap to select the PyVista/Matplotlib colormap for both surfaces.
    ``cross_section_stations`` may contain :class:`XCWindow` objects. Each
    automatic window is fitted to the wall-surface intersection; its exact
    rectangle is drawn on the lower-surface view, and matching NN and CFD
    pressure sections are generated. Pass the final optimization case as
    config. Returns the saved PNG paths when screenshot=True, or displays the
    figures interactively otherwise.
    """
    import pyvista as pv

    allowed = {"NN", "low-fidelity", "CFD"}
    if left not in allowed or right not in allowed:
        raise ValueError(f"left and right must be one of {sorted(allowed)}")
    if len(clim) != 2 or not np.isfinite(clim).all() or clim[0] >= clim[1]:
        raise ValueError("clim must contain two finite, increasing pressures")
    geo, topology = config['warped_geo'], config['topology']
    panel_data = panel_data if panel_data is not None else config.get('low_fidelity_data')
    panel_topology = panel_topology if panel_topology is not None else config.get('panel_topology')

    def cfd_surface(values):
        return _cfd_wall_surface(config, values)

    volume_pressures = {
        'CFD': np.asarray(config['data']['p']).reshape(-1),
    }
    meshes = {}
    for source in {left, right}:
        if source == 'NN':
            if 'NN' not in volume_pressures:
                prediction = model_eval(params, config['flow_condition'], geo)[2]
                volume_pressures['NN'] = (
                    np.asarray(prediction).reshape(-1) * constants['p']['std']
                    + constants['p']['mean']
                )
            pressure = volume_pressures['NN']
            mesh = cfd_surface(pressure)
        elif source == 'CFD':
            mesh = cfd_surface(volume_pressures['CFD'])
        elif panel_data is not None and 'panel_mesh' in panel_data:
            if panel_topology is None:
                raise ValueError("Native panel pressure requires panel_topology")
            faces = []
            for key, size in [('triangles', 3), ('quads', 4)]:
                cells = np.asarray(panel_topology[key], dtype=np.int64).reshape(-1, size)
                faces.append(np.column_stack((np.full(len(cells), size), cells)).ravel())
            mesh = pv.PolyData(np.asarray(panel_data['panel_mesh']), np.concatenate(faces))
            mesh.cell_data['field'] = np.asarray(panel_data['pressure']).reshape(-1)
        else:
            lowfi = geo.get('low_fidelity_inputs')
            if lowfi is None:
                raise ValueError("No low-fidelity pressure data available")
            # Wall ghost nodes follow the same face order as the surface mesh.
            indices = np.asarray(topology['dual_graph']['dual_cell_types']) == 1
            mesh = cfd_surface(np.asarray(lowfi['p'])[indices])
        # Keep one physical half before mirroring; this also handles full panel meshes.
        mesh = mesh.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
        if mesh.n_cells == 0:
            raise ValueError(f"{source} mesh has no surface on positive y")
        if smooth_scalars:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
        meshes[source] = mesh

    resolved_windows = []
    if cross_section_stations is not None:
        reference_surface = cfd_surface(volume_pressures['CFD'])
        resolved_windows = [
            _resolve_xc_window(window, reference_surface)
            for window in cross_section_stations
        ]
        if resolved_windows and 'NN' not in volume_pressures:
            prediction = model_eval(params, config['flow_condition'], geo)[2]
            volume_pressures['NN'] = (
                np.asarray(prediction).reshape(-1) * constants['p']['std']
                + constants['p']['mean']
            )

    output_dir, slug = _aero_plot_output(model_name, plots_dir)
    if screenshot:
        output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for surface, camera_sign in [('upper', 1), ('lower', -1)]:
        plotter = pv.Plotter(off_screen=screenshot, window_size=window_size)
        try:
            plotter.set_background('white')
            for source, side in [(left, -1), (right, 1)]:
                mesh = meshes[source].copy(deep=True)
                mesh.points[:, 1] *= side * camera_sign
                plotter.add_mesh(mesh, scalars='field', cmap=cmap, clim=clim,
                                 lighting=False, show_edges=False, show_scalar_bar=False)
            if surface == 'lower':
                for window in resolved_windows:
                    _, plane_origin = _xc_plane_geometry(window.plane)
                    mirror_options = (False,) if abs(plane_origin[1]) < 1e-12 else (False, True)
                    for mirror_y in mirror_options:
                        rectangle, border = _xc_window_meshes(window, mirror_y=mirror_y)
                        plotter.add_mesh(
                            rectangle, color='grey', opacity=0.25,
                            show_edges=False, lighting=False, pickable=False,
                        )
                        plotter.add_mesh(
                            border, color='black', line_width=3,
                            lighting=False, pickable=False,
                            render_lines_as_tubes=True,
                        )
            plotter.add_scalar_bar(title=make_colorbar_label('p'), vertical=False,
                                  n_labels=4, fmt='%.0f', color='black',
                                  position_x=0.25, width=0.5, position_y=0.03, height=0.09)
            labels = {'NN': 'NN', 'CFD': 'CFD', 'low-fidelity': 'Low-fidelity'}
            plotter.add_text(labels[left], position='upper_left', color='black', font_size=18)
            plotter.add_text(labels[right], position='upper_right', color='black', font_size=18)
            plotter.add_text(f'{surface.capitalize()} surface', position='lower_left',
                             color='black', font_size=14)
            center = np.asarray(plotter.center)
            plotter.camera_position = [center + [0, 0, camera_sign * 100], center, (-1, 0, 0)]
            plotter.enable_parallel_projection()
            plotter.reset_camera()
            plotter.camera.zoom(0.85)
            if screenshot:
                path = output_dir / f'surface_comparison_{surface}_{slug}.png'
                plotter.screenshot(str(path))
                paths.append(path)
            else:
                plotter.show()
        finally:
            plotter.close()

    for window in resolved_windows:
        for state_model in ('NN', 'CFD'):
            path = plot_xc(
                model_eval, params, config, constants,
                xc_window=window, state='p', state_model=state_model,
                screenshot=screenshot, model_name=model_name,
                plots_dir=plots_dir, clim=clim, window_size=window_size,
                smooth_scalars=smooth_scalars, cmap=cmap,
                state_values=volume_pressures[state_model],
            )
            if path is not None:
                paths.append(path)
    return paths


def plot_field(
        model_eval, params, config, constants, screenshot=False, model_name="",
        model_type="NN", plots_dir="post_processing_plots",
        clim=None, window_size=(1800, 1400), smooth_scalars=False,
        cmap=None, cross_section_stations=None, state="p", camera="pos_2",
        pyvista_mesh_topology=None, parallel_projection=True, lighting=False,
        contour_levels=None, contour_line_width=2.0, filled=True,
        grid_2d=False, grid_2d_smoothing=2.0, contour_labels=False,
    ):
    """Plot one NN, CFD, or signed-error pressure or Mach field for one case.

    Two full-aircraft surface views are produced: a raw field and the same
    field with the supplied :class:`XCWindow` rectangles overlaid. One planar
    section is also produced for every window, using :func:`plot_xc` so its
    framing and rendering match :func:`plot_surface_comparison`.

    Screenshots are stored beneath::

        <plots_dir>/<model_name>/field_plots/case_<id>/<model_type>/

    A separate horizontal PDF colorbar is also stored directly in the
    ``field_plots`` directory as ``<cmap>_<state>_colorbar.pdf``. It has five
    ticks below the bar and its quantity label above it. Pressure colorbars are
    expressed as pressure coefficient, using the case freestream conditions;
    signed pressure error is converted by dividing by freestream dynamic
    pressure.

    Thus, two cross-section windows produce four images per model type. The
    saved paths are returned when ``screenshot=True``; otherwise the figures
    are displayed and an empty list is returned. ``parallel_projection`` and
    ``lighting`` apply to the two full-aircraft views; cross-section rendering
    intentionally retains :func:`plot_xc` behavior. Pressure defaults to
    10--40 kPa (or -3--3 kPa error), while Mach defaults to 0--1.5 (or
    -0.1--0.1 error). Signed error is NN minus CFD. Explicit ``cmap`` and
    ``clim`` values override these defaults. Set ``contour_levels`` to a
    positive isoline count or an explicit sequence of physical values to add
    matching black contours to the surfaces and cross-sections. Use ``filled``
    to toggle the filled-color actors and ``contour_line_width`` to control the
    black isolines. ``grid_2d`` optionally reconstructs each cross-section on a
    regular plane; ``True`` uses 600 samples across, an integer selects that
    width, and ``(nx, ny)`` supplies both dimensions. For Mach, velocity and
    temperature are independently reconstructed before Mach is calculated.
    ``grid_2d_smoothing`` is the optional Gaussian width in regular-grid
    samples; set it to zero for unsmoothed linear reconstruction. Set
    ``contour_labels=True`` to label each contour level on the regular-grid
    cross-sections; surface contours remain unlabeled.
    """
    import re

    import pyvista as pv

    if state not in {"p", "Ma"}:
        raise NotImplementedError("plot_field supports state='p' or state='Ma'")
    if model_type not in {"NN", "CFD", "error"}:
        raise ValueError("model_type must be 'NN', 'CFD', or 'error'")
    if clim is None:
        default_clims = {
            "p": {"field": (1e4, 4e4), "error": (-3e3, 3e3)},
            "Ma": {"field": (0.0, 1.5), "error": (-0.1, 0.1)},
        }
        clim = default_clims[state][
            "error" if model_type == "error" else "field"
        ]
    if cmap is None:
        cmap = "coolwarm" if model_type == "error" else "jet"
    if len(clim) != 2 or not np.isfinite(clim).all() or clim[0] >= clim[1]:
        raise ValueError("clim must contain two finite, increasing values")
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("window_size must contain two positive pixel dimensions")
    if not np.isfinite(contour_line_width) or contour_line_width <= 0.0:
        raise ValueError("contour_line_width must be finite and greater than zero")
    if not np.isfinite(grid_2d_smoothing) or grid_2d_smoothing < 0.0:
        raise ValueError("grid_2d_smoothing must be finite and nonnegative")
    resolved_contour_levels = _resolve_contour_levels(contour_levels, clim)
    if not filled and resolved_contour_levels is None:
        raise ValueError("filled=False requires contour_levels")
    if contour_labels and (grid_2d is None or grid_2d is False):
        raise ValueError("contour_labels=True requires grid_2d")
    if contour_labels and resolved_contour_levels is None:
        raise ValueError("contour_labels=True requires contour_levels")

    windows = [] if cross_section_stations is None else list(cross_section_stations)
    reference_surface = _cfd_wall_surface(config)
    resolved_windows = [
        _resolve_xc_window(window, reference_surface)
        for window in windows
    ]

    geo = config["warped_geo"]
    values, state_primitives = _field_values_and_primitives(
        model_eval, params, config, constants, state, model_type
    )

    output_root, model_slug = _aero_plot_output(model_name, plots_dir)
    case_id = str(config["case_id"])
    case_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "unknown"
    field_plots_dir = output_root / "field_plots"
    output_dir = field_plots_dir / f"case_{case_slug}" / model_type
    filename_suffix = f"{model_slug}_case_{case_slug}"
    if screenshot:
        output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if screenshot:
        colorbar_clim, colorbar_label = _standalone_field_colorbar_settings(
            state, model_type, clim, config, constants
        )
        cmap_name = getattr(
            cmap, "name", cmap if isinstance(cmap, str) else "custom"
        )
        cmap_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", cmap_name).strip("._")
        colorbar_path = field_plots_dir / (
            f"{cmap_slug or 'custom'}_{state}_colorbar.pdf"
        )
        save_standalone_colorbar(
            colorbar_path,
            clim=colorbar_clim,
            cmap=cmap,
            label=colorbar_label,
            orientation="horizontal",
            ticks=np.linspace(*colorbar_clim, 5),
            tick_format="%.2f",
            label_on_top=True,
            font_family="cmr10",
            label_fontsize=20,
            tick_fontsize=18,
        )
        paths.append(colorbar_path)

    topology = config["topology"]
    mirrored_points = np.asarray(geo["foam_mesh"]["points"]).copy()
    mirrored_points[:, 1] *= -1.0
    mirrored_geo = {"foam_mesh": {"points": mirrored_points}}

    surface_variants = (
        ("raw", False),
        ("xc_windows", True),
    )
    for variant, show_windows in surface_variants:
        plotter = pv.Plotter(off_screen=screenshot, window_size=window_size)
        try:
            plotter.set_background("white")
            scalar_bar_actor = None
            for surface_geo in (geo, mirrored_geo):
                surface_actor = visualize_surface(
                    topology, surface_geo, values,
                    title=None, cmap=cmap, clim=clim, camera=camera,
                    p=plotter, enable_picking=False, lighting=lighting,
                    show_scalar_bar=False, smooth_scalars=smooth_scalars,
                    contour_levels=resolved_contour_levels,
                    contour_line_width=contour_line_width, filled=filled,
                    add_axes=False, categories=False, return_actor=True,
                )
                if scalar_bar_actor is None:
                    scalar_bar_actor = surface_actor

            colorbar_quantity = "signed_error" if model_type == "error" else None
            plotter.add_scalar_bar(
                title=make_colorbar_label(state, quantity_name=colorbar_quantity),
                vertical=False, n_labels=5 if model_type == "error" else 4,
                fmt="%.2f" if state == "Ma" else "%.0f",
                color="black", position_x=0.25, width=0.5,
                position_y=0.03, height=0.09,
                mapper=scalar_bar_actor.mapper,
            )

            if show_windows:
                for window in resolved_windows:
                    rectangle, border = _xc_window_meshes(
                        window, mirror_y=True
                    )
                    plotter.add_mesh(
                        rectangle, color="grey", opacity=0.25,
                        show_edges=False, lighting=False, pickable=False,
                    )
                    plotter.add_mesh(
                        border, color="black", line_width=3,
                        lighting=False, pickable=False,
                        render_lines_as_tubes=True,
                    )
                    mirrored_surface = reference_surface.copy(deep=True)
                    mirrored_surface.points[:, 1] *= -1.0
                    normal, origin = _xc_plane_geometry(window.plane)
                    normal = normal.copy()
                    origin = origin.copy()
                    normal[1] *= -1.0
                    origin[1] *= -1.0
                    intersection = mirrored_surface.slice(
                        normal=normal, origin=origin
                    )
                    if intersection.n_points:
                        plotter.add_mesh(
                            intersection.tube(radius=0.06),
                            color="black", lighting=False, pickable=False,
                        )

            plotter.add_text(
                "Error (NN - CFD)" if model_type == "error" else model_type,
                position="upper_left", color="black", font_size=18,
            )
            if parallel_projection:
                plotter.enable_parallel_projection()
                plotter.reset_camera_clipping_range()

            if screenshot:
                output_path = output_dir / (
                    f"surface_{model_type}_{state}_{variant}_{filename_suffix}.png"
                )
                plotter.screenshot(str(output_path))
                paths.append(output_path)
            else:
                plotter.show()
        finally:
            plotter.close()

    for window in resolved_windows:
        path = plot_xc(
            model_eval, params, config, constants,
            xc_window=window, state=state, state_model=model_type,
            screenshot=screenshot, model_name=model_name,
            plots_dir=plots_dir, clim=clim, window_size=window_size,
            smooth_scalars=smooth_scalars, cmap=cmap,
            state_values=values,
            pyvista_mesh_topology=pyvista_mesh_topology,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            contour_levels=resolved_contour_levels,
            contour_line_width=contour_line_width,
            filled=filled,
            grid_2d=grid_2d,
            grid_2d_smoothing=grid_2d_smoothing,
            contour_labels=contour_labels,
            state_primitives=state_primitives,
        )
        if path is not None:
            paths.append(path)
    return paths


def plot_xc(
        model_eval, params, config, constants,
        plane=(0, 1, 0), center=(0.0, 0.0, 0.0), width=10.0,
        aspect_ratio=9 / 7, state="p", state_model="NN",
        screenshot=False, model_name="", plots_dir="post_processing_plots",
        clim=None, window_size=(1800, 1400),
        smooth_scalars=False, cmap=None, xc_window=None, state_values=None,
        pyvista_mesh_topology=None, output_dir=None, filename_suffix=None,
        contour_levels=None, contour_line_width=2.0, filled=True,
        grid_2d=False, grid_2d_smoothing=2.0, contour_labels=False,
        state_primitives=None,
    ):
    """Plot a planar cross-section of an NN, CFD, or signed-error field.

    ``plane`` is the plane normal and ``center`` is a point on the plane. The
    final view is centered at ``center`` and has physical width ``width`` in
    meters. ``aspect_ratio`` is image width divided by image height. Pass an
    ``XCWindow`` through ``xc_window`` to supply these settings together and,
    when requested, determine its framing from the aircraft wall surface.
    Whenever possible, the vertical image direction is aligned with global +z.
    ``state_values`` may provide an already evaluated physical field
    when generating several sections of the same case.

    Pressure and local Mach number are supported. Their colormap, limits,
    smoothing, contour, and output-directory arguments intentionally match
    :func:`plot_field`. When ``screenshot=True``, the saved PNG path is
    returned; otherwise the plot is displayed and ``None`` is returned.
    ``grid_2d`` reconstructs the section on a regular plane. For Mach, the
    velocity and temperature primitives are sampled before forming Mach.
    ``grid_2d_smoothing`` controls a mild, normalized Gaussian reconstruction
    in regular-grid samples and may be set to zero. ``contour_labels=True``
    adds one inline value label to the longest visible line at each contour
    level and is available only with ``grid_2d``.
    """
    from pathlib import Path

    import pyvista as pv

    if state not in {"p", "Ma"}:
        raise NotImplementedError("plot_xc supports state='p' or state='Ma'")
    if state_model not in {"NN", "CFD", "error"}:
        raise ValueError("state_model must be 'NN', 'CFD', or 'error'")
    if clim is None:
        default_clims = {
            "p": {"field": (1e4, 4e4), "error": (-3e3, 3e3)},
            "Ma": {"field": (0.0, 1.5), "error": (-0.1, 0.1)},
        }
        clim = default_clims[state][
            "error" if state_model == "error" else "field"
        ]
    if cmap is None:
        cmap = "coolwarm" if state_model == "error" else "jet"
    if len(clim) != 2 or not np.isfinite(clim).all() or clim[0] >= clim[1]:
        raise ValueError("clim must contain two finite, increasing values")
    if not np.isfinite(contour_line_width) or contour_line_width <= 0.0:
        raise ValueError("contour_line_width must be finite and greater than zero")
    if not np.isfinite(grid_2d_smoothing) or grid_2d_smoothing < 0.0:
        raise ValueError("grid_2d_smoothing must be finite and nonnegative")
    resolved_contour_levels = _resolve_contour_levels(contour_levels, clim)
    if not filled and resolved_contour_levels is None:
        raise ValueError("filled=False requires contour_levels")
    if contour_labels and (grid_2d is None or grid_2d is False):
        raise ValueError("contour_labels=True requires grid_2d")
    if contour_labels and resolved_contour_levels is None:
        raise ValueError("contour_labels=True requires contour_levels")

    if xc_window is not None:
        needs_surface = any(
            isinstance(value, str) and value.lower() == "auto"
            for value in (xc_window.center, xc_window.width)
        )
        resolved_window = _resolve_xc_window(
            xc_window, _cfd_wall_surface(config) if needs_surface else None
        )
        normal, _ = _xc_plane_geometry(resolved_window.plane)
        section_center = np.asarray(resolved_window.center, dtype=float)
        width = resolved_window.width
        aspect_ratio = resolved_window.aspect_ratio
    else:
        normal = np.asarray(plane, dtype=float)
        section_center = np.asarray(center, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("plane must contain three finite components")
    if section_center.shape != (3,) or not np.isfinite(section_center).all():
        raise ValueError("center must contain three finite coordinates")
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0.0:
        raise ValueError("plane must be a nonzero normal vector")
    normal /= normal_magnitude
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and greater than zero")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be finite and greater than zero")
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("window_size must contain two positive pixel dimensions")
    grid_2d_dimensions = _resolve_grid_2d_dimensions(grid_2d, aspect_ratio)

    geo, topology = config["warped_geo"], config["topology"]
    if state_values is not None:
        values = np.asarray(state_values).reshape(-1)
    else:
        values, resolved_primitives = _field_values_and_primitives(
            model_eval, params, config, constants, state, state_model
        )
        if state_primitives is None:
            state_primitives = resolved_primitives

    if pyvista_mesh_topology is None:
        cells_flat, celltypes = build_pyvista_mesh_topology(topology["foam_mesh"])
    else:
        cells_flat, celltypes = pyvista_mesh_topology
    grid = pv.UnstructuredGrid(
        cells_flat, celltypes, np.asarray(geo["foam_mesh"]["points"])
    )
    if values.size != grid.n_cells:
        raise ValueError(
            f"{state_model} {state} has {values.size} values, but the CFD mesh "
            f"has {grid.n_cells} cells"
        )

    if grid_2d_dimensions is None:
        grid.cell_data["field"] = values
        if smooth_scalars:
            grid = grid.cell_data_to_point_data(pass_cell_data=False)
        section = grid.slice(
            normal=normal, origin=section_center, generate_triangles=True
        )
    else:
        if state == "p":
            grid.cell_data["field"] = values
        else:
            if state_primitives is None:
                _, state_primitives = _field_values_and_primitives(
                    model_eval, params, config, constants, state, state_model
                )
            required_sources = (
                ("CFD", "NN") if state_model == "error" else (state_model,)
            )
            missing_sources = [
                source for source in required_sources
                if source not in state_primitives
            ]
            if missing_sources:
                raise ValueError(
                    "grid_2d Mach reconstruction is missing primitives for "
                    + ", ".join(missing_sources)
                )
            for source in required_sources:
                velocity, temperature = state_primitives[source]
                velocity = np.asarray(velocity)
                temperature = np.asarray(temperature).reshape(-1)
                if velocity.shape != (grid.n_cells, 3):
                    raise ValueError(
                        f"{source} velocity must have shape {(grid.n_cells, 3)}"
                    )
                if temperature.size != grid.n_cells:
                    raise ValueError(
                        f"{source} temperature must have {grid.n_cells} values"
                    )
                grid.cell_data[f"{source}_velocity"] = velocity
                grid.cell_data[f"{source}_temperature"] = temperature

        point_grid = grid.cell_data_to_point_data(pass_cell_data=False)
        native_section = point_grid.slice(
            normal=normal, origin=section_center, generate_triangles=True
        )
        if native_section.n_cells == 0:
            raise ValueError("The requested plane does not intersect the CFD mesh")
        horizontal, vertical = _xc_view_basis(normal)
        native_cell_offsets = (
            np.asarray(native_section.cell_centers().points) - section_center
        )
        horizontal_margin = 0.20 * width
        vertical_margin = 0.20 * width / aspect_ratio
        window_cells = (
            np.abs(native_cell_offsets @ horizontal)
            <= width / 2.0 + horizontal_margin
        ) & (
            np.abs(native_cell_offsets @ vertical)
            <= width / (2.0 * aspect_ratio) + vertical_margin
        )
        native_section = native_section.extract_cells(
            np.flatnonzero(window_cells)
        ).extract_surface(algorithm=None)
        native_section = native_section.clean(
            tolerance=1e-10, absolute=False, point_merging=True
        )
        if not native_section.is_all_triangles:
            native_section = native_section.triangulate()

        import matplotlib.tri as mtri
        from scipy.ndimage import gaussian_filter

        width_points, height_points = grid_2d_dimensions
        horizontal_offsets = np.linspace(-width / 2.0, width / 2.0, width_points)
        vertical_offsets = np.linspace(
            -width / (2.0 * aspect_ratio),
            width / (2.0 * aspect_ratio),
            height_points,
        )
        horizontal_grid, vertical_grid = np.meshgrid(
            horizontal_offsets, vertical_offsets, indexing="xy"
        )
        sample_points = (
            section_center
            + horizontal_grid[..., None] * horizontal
            + vertical_grid[..., None] * vertical
        )
        native_offsets = np.asarray(native_section.points) - section_center
        native_horizontal = native_offsets @ horizontal
        native_vertical = native_offsets @ vertical
        triangles = native_section.faces.reshape(-1, 4)[:, 1:]
        triangulation = mtri.Triangulation(
            native_horizontal, native_vertical, triangles=triangles
        )

        def reconstruct_regular_grid(point_values):
            interpolator = mtri.LinearTriInterpolator(
                triangulation, np.asarray(point_values)
            )
            reconstructed = interpolator(horizontal_grid, vertical_grid)
            reconstructed = np.asarray(
                np.ma.filled(reconstructed, np.nan), dtype=float
            )
            valid_values = np.isfinite(reconstructed)
            if grid_2d_smoothing > 0.0 and np.any(valid_values):
                weights = gaussian_filter(
                    valid_values.astype(float),
                    sigma=grid_2d_smoothing,
                    mode="constant",
                    cval=0.0,
                )
                numerator = gaussian_filter(
                    np.where(valid_values, reconstructed, 0.0),
                    sigma=grid_2d_smoothing,
                    mode="constant",
                    cval=0.0,
                )
                reconstructed[valid_values] = (
                    numerator[valid_values] / weights[valid_values]
                )
            return reconstructed

        sample_plane = pv.StructuredGrid()
        sample_plane.points = sample_points.reshape(-1, 3)
        sample_plane.dimensions = (width_points, height_points, 1)

        if state == "p":
            sampled_values = reconstruct_regular_grid(native_section["field"])
        else:
            sampled_mach = {}
            for source in required_sources:
                sampled_velocity = np.column_stack([
                    reconstruct_regular_grid(
                        np.asarray(native_section[f"{source}_velocity"])[..., i]
                    ).reshape(-1)
                    for i in range(3)
                ])
                sampled_temperature = reconstruct_regular_grid(
                    native_section[f"{source}_temperature"]
                ).reshape(-1)
                source_valid = (
                    np.isfinite(sampled_velocity).all(axis=1)
                    & np.isfinite(sampled_temperature)
                )
                source_mach = np.full(sample_plane.n_points, np.nan, dtype=float)
                source_mach[source_valid] = _mach_number(
                    sampled_velocity[source_valid],
                    sampled_temperature[source_valid],
                    constants,
                )
                sampled_mach[source] = source_mach
            if state_model == "error":
                sampled_values = sampled_mach["NN"] - sampled_mach["CFD"]
            else:
                sampled_values = sampled_mach[state_model]

        sampled_values = np.asarray(sampled_values).reshape(-1)
        valid = np.isfinite(sampled_values)
        sample_plane.point_data["field"] = sampled_values
        sample_plane.point_data["valid"] = valid.astype(np.uint8)
        section = sample_plane.threshold(
            0.5,
            scalars="valid",
            preference="point",
            all_scalars=True,
        )
    if section.n_cells == 0:
        raise ValueError("The requested plane does not intersect the CFD mesh")

    # Prefer global +z as image-up. For a horizontal section, use global +y.
    view_up = np.array([0.0, 0.0, 1.0])
    view_up -= np.dot(view_up, normal) * normal
    if np.linalg.norm(view_up) < 1e-12:
        view_up = np.array([0.0, 1.0, 0.0])
        view_up -= np.dot(view_up, normal) * normal
    view_up /= np.linalg.norm(view_up)

    # PyVista's parallel camera gets its horizontal extent from the pixel
    # aspect ratio. Preserve the requested pixel height and adjust the width so
    # the visible physical rectangle is exactly width x width/aspect_ratio.
    render_size = (max(1, round(window_size[1] * aspect_ratio)), window_size[1])
    plotter = pv.Plotter(off_screen=screenshot, window_size=render_size)
    output_path = None
    try:
        plotter.set_background("white")
        field_actor = plotter.add_mesh(
            section, scalars="field", cmap=cmap, clim=clim, lighting=False,
            show_edges=False, show_scalar_bar=False,
            opacity=1.0 if filled else 0.0,
            nan_opacity=0.0,
        )
        if resolved_contour_levels is not None:
            contour_section = section
            if "field" in contour_section.cell_data:
                contour_section = contour_section.cell_data_to_point_data(
                    pass_cell_data=False
                )
            contours = contour_section.contour(
                isosurfaces=resolved_contour_levels, scalars="field"
            )
            if contours.n_cells:
                plotter.add_mesh(
                    contours,
                    color="black",
                    line_width=contour_line_width,
                    lighting=False,
                    show_scalar_bar=False,
                    pickable=False,
                    render_lines_as_tubes=True,
                )
                if contour_labels:
                    label_points, label_levels = _contour_label_locations(
                        contours, resolved_contour_levels
                    )
                    if label_levels:
                        labels = [
                            f"{level:.2f}" if state == "Ma" else f"{level:.0f}"
                            for level in label_levels
                        ]
                        plotter.add_point_labels(
                            label_points,
                            labels,
                            bold=False,
                            font_size=14,
                            text_color="black",
                            show_points=False,
                            shape="rounded_rect",
                            shape_color="white",
                            shape_opacity=0.8,
                            margin=2,
                            pickable=False,
                            always_visible=True,
                        )
        if grid_2d_dimensions is not None or not filled:
            body_outline = _cfd_wall_surface(config).slice(
                normal=normal, origin=section_center
            )
            if body_outline.n_cells:
                plotter.add_mesh(
                    body_outline,
                    color="black",
                    line_width=contour_line_width,
                    lighting=False,
                    show_scalar_bar=False,
                    pickable=False,
                    render_lines_as_tubes=True,
                )
        plotter.add_scalar_bar(
            title=make_colorbar_label(
                state,
                quantity_name="signed_error" if state_model == "error" else None,
            ),
            vertical=False, n_labels=5 if state_model == "error" else 4,
            fmt="%.2f" if state == "Ma" else "%.0f",
            color="black", position_x=0.25, width=0.5,
            position_y=0.03, height=0.09,
            mapper=field_actor.mapper,
        )
        plotter.add_text(
            "Error (NN - CFD)" if state_model == "error" else state_model,
            position="upper_left", color="black", font_size=18,
        )

        camera_distance = max(float(grid.length), float(width), 1.0)
        plotter.camera_position = [
            section_center + normal * camera_distance,
            section_center,
            view_up,
        ]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = width / (2.0 * aspect_ratio)

        if screenshot:
            if output_dir is None:
                output_dir, slug = _aero_plot_output(model_name, plots_dir)
            else:
                output_dir = Path(output_dir)
                slug = filename_suffix
                if not slug:
                    _, slug = _aero_plot_output(model_name, plots_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            center_y = f"{section_center[1]:g}"
            output_path = output_dir / (
                f"xc_{state_model}_y{center_y}_{state}_{slug}.png"
            )
            plotter.screenshot(str(output_path))
        else:
            plotter.show()
    finally:
        plotter.close()
    return output_path


def _full_analysis_record(
        case,
        true_loads,
        predicted_loads,
        true_lift,
        true_drag,
        predicted_lift,
        predicted_drag,
        openfoam_lift,
        openfoam_drag,
        panel_lift,
        panel_drag,
        lift_error_percent,
        drag_error_percent,
    ):
    """Reduce one case's existing force results for optional post-processing."""
    import jax

    def as_float(value):
        array = np.asarray(jax.device_get(value), dtype=float).reshape(-1)
        return float(array[0])

    def force_components(surface_loads, lift, drag):
        pressure = surface_loads["pressure_forces"]
        viscous = surface_loads["viscous_forces"]
        return {
            "lift": as_float(lift),
            "drag": as_float(drag),
            "pressure_lift": as_float(-pressure[:, 2].sum()),
            "pressure_drag": as_float(-pressure[:, 0].sum()),
            "viscous_lift": as_float(-viscous[:, 2].sum()),
            "viscous_drag": as_float(-viscous[:, 0].sum()),
        }

    def mapping_scalar(mapping, *names):
        for name in names:
            if name not in mapping:
                continue
            array = np.asarray(mapping[name], dtype=float).reshape(-1)
            if array.size and np.isfinite(array[0]):
                return float(array[0])
        return np.nan

    flow = case.get("flow_condition", {})
    geometry = case.get("aircraft_geometry", {})
    pitch = mapping_scalar(flow, "pitch", "alpha")

    areas = np.asarray(jax.device_get(true_loads["surface_areas"]), dtype=float)
    normals = np.asarray(jax.device_get(true_loads["surface_normals"]), dtype=float)
    centroids = np.asarray(
        jax.device_get(true_loads["surface_centroids"]), dtype=float
    )
    theta = np.deg2rad(pitch if np.isfinite(pitch) else 0.0)
    body_normal_z = np.sin(theta) * normals[:, 0] + np.cos(theta) * normals[:, 2]
    planform_area = 0.5 * np.sum(areas * np.abs(body_normal_z))

    # A symmetry-plane CFD mesh contains only half the aircraft.
    y = centroids[:, 1]
    y_scale = max(float(np.ptp(y)), 1.0)
    if np.nanmin(y) >= -1e-6 * y_scale or np.nanmax(y) <= 1e-6 * y_scale:
        planform_area *= 2.0

    return {
        "case_id": str(case["case_id"]),
        "is_train": bool(case["train"]),
        "true": force_components(true_loads, true_lift, true_drag),
        "predicted": force_components(
            predicted_loads, predicted_lift, predicted_drag
        ),
        "openfoam_lift": as_float(openfoam_lift),
        "openfoam_drag": as_float(openfoam_drag),
        "panel_lift": as_float(panel_lift),
        "panel_drag": as_float(panel_drag),
        "lift_error_percent": as_float(lift_error_percent),
        "drag_error_percent": as_float(drag_error_percent),
        "pitch": pitch,
        "planform_area": float(planform_area),
        "wing_sweep": mapping_scalar(geometry, "wing_sweep", "sweep"),
        "wing_half_span": mapping_scalar(geometry, "wing_half_span", "span"),
        "transition_sweep": mapping_scalar(geometry, "transition_sweep"),
        "elevator_deflection": mapping_scalar(geometry, "elevator_deflection"),
    }

def _postprocess_full_aero_analysis(
        records,
        field_l2_summary,
        model_name: str = "",
        plots_dir: str = "post_processing_plots",
    ):
    """Create the optional diagnostic plots from already-computed case data.

    No model evaluation or force reconstruction happens here. The three
    additional figures are saved beneath ``plots_dir/<model_name>/``:

    * pressure-drag and viscous-drag parity, an L-D operating map, and an L/D
      ordering comparison;
    * lift/drag error versus the available flow and geometry parameters.
    * actual CFD and predicted lift/drag versus those parameters.

    Returns:
        A dictionary containing the numeric arrays, summary metrics, and paths
        of any saved plots.
    """
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("records must contain at least one aerodynamic case")

    def finite_ratio(numerator, denominator):
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        scale = max(float(np.nanmax(np.abs(denominator))), 1.0)
        valid = np.abs(denominator) > 1e-12 * scale
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=valid,
        )

    def average_ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        return ranks

    def spearman_rho(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if np.count_nonzero(valid) < 2:
            return np.nan
        x_rank = average_ranks(x_values[valid])
        y_rank = average_ranks(y_values[valid])
        if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
            return np.nan
        return float(np.corrcoef(x_rank, y_rank)[0, 1])

    def mae(y_true, y_pred):
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(valid):
            return np.nan
        return float(np.mean(np.abs(y_pred[valid] - y_true[valid])))

    def relative_error_percent(y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        finite_truth = np.abs(y_true[np.isfinite(y_true)])
        reference_floor = max(
            0.05 * float(np.max(finite_truth)) if finite_truth.size else 0.0,
            1e-12,
        )
        return 100.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), reference_floor)

    def column(path):
        if isinstance(path, tuple):
            parent, child = path
            return np.asarray([record[parent][child] for record in records], dtype=float)
        return np.asarray([record[path] for record in records], dtype=float)

    case_ids = np.asarray([record["case_id"] for record in records])
    is_train = np.asarray([record["is_train"] for record in records], dtype=bool)
    openfoam_lift = column("openfoam_lift")
    openfoam_drag = column("openfoam_drag")
    panel_lift = column("panel_lift")
    panel_drag = column("panel_drag")
    cfd_lift = column(("true", "lift"))
    cfd_drag = column(("true", "drag"))
    predicted_lift = column(("predicted", "lift"))
    predicted_drag = column(("predicted", "drag"))

    output_dir, model_slug = _aero_plot_output(model_name, plots_dir)
    saved_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    def parity_plot(ax, y_true, y_pred, title, quantity, unit):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        for mask, marker, label, size in (
            (valid & is_train, "o", "Train", 55),
            (valid & ~is_train, "s", "Test", 70),
        ):
            if np.any(mask):
                ax.scatter(
                    y_true[mask],
                    y_pred[mask],
                    s=size,
                    alpha=0.88,
                    marker=marker,
                    edgecolor="black",
                    linewidth=0.6,
                    label=label,
                )
        if np.any(valid):
            lo = min(float(np.min(y_true[valid])), float(np.min(y_pred[valid])))
            hi = max(float(np.max(y_true[valid])), float(np.max(y_pred[valid])))
            magnitude = max(abs(lo), abs(hi))
            pad = 0.05 * max(hi - lo, 0.1 * magnitude, 1e-6)
            lo, hi = lo - pad, hi + pad
            ax.axline((0.0, 0.0), slope=1.0, color="black", linestyle="--", linewidth=1.2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nMAE = {mae(y_true, y_pred):.3g} {unit}")
        ax.set_xlabel(f"CFD {quantity} [{unit}]")
        ax.set_ylabel(f"Predicted {quantity} [{unit}]")
        ax.grid(True, alpha=0.25)
        if np.any(valid):
            ax.legend(frameon=False)

    metrics = {
        "openfoam_lift_mae_kN": mae(openfoam_lift, predicted_lift) / 1e3,
        "openfoam_drag_mae_kN": mae(openfoam_drag, predicted_drag) / 1e3,
        "field_relative_l2": field_l2_summary,
        "train_field_relative_l2": field_l2_summary["train"],
        "test_field_relative_l2": field_l2_summary["test"],
    }

    def create_additional_plots():
        cfd_pressure_drag = column(("true", "pressure_drag"))
        predicted_pressure_drag = column(("predicted", "pressure_drag"))
        cfd_viscous_drag = column(("true", "viscous_drag"))
        predicted_viscous_drag = column(("predicted", "viscous_drag"))
        cfd_lift_to_drag = finite_ratio(cfd_lift, cfd_drag)
        predicted_lift_to_drag = finite_ratio(predicted_lift, predicted_drag)

        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        parity_plot(
            axes[0, 0],
            cfd_pressure_drag / 1e3,
            predicted_pressure_drag / 1e3,
            "Pressure-drag parity",
            "pressure drag",
            "kN",
        )
        parity_plot(
            axes[0, 1],
            cfd_viscous_drag / 1e3,
            predicted_viscous_drag / 1e3,
            "Viscous-drag parity",
            "viscous drag",
            "kN",
        )

        operating_ax = axes[1, 0]
        valid_operating = (
            np.isfinite(cfd_lift)
            & np.isfinite(cfd_drag)
            & np.isfinite(predicted_lift)
            & np.isfinite(predicted_drag)
            & (cfd_drag != 0.0)
            & (predicted_drag != 0.0)
        )
        cfd_lift_to_drag = finite_ratio(cfd_lift, cfd_drag)
        predicted_lift_to_drag = finite_ratio(predicted_lift, predicted_drag)
        valid_operating &= ~is_train & np.isfinite(cfd_lift_to_drag) & np.isfinite(predicted_lift_to_drag)
        ld_error = np.abs(predicted_lift_to_drag - cfd_lift_to_drag)
        error_norm = Normalize(vmin=0.0, vmax=2.0, clip=True)
        error_cmap = LinearSegmentedColormap.from_list(
            "ld_error", ["#1565c0", "#b71c1c"]
        )
        for index in np.flatnonzero(valid_operating):
            operating_ax.plot(
                [cfd_drag[index] / 1e3, predicted_drag[index] / 1e3],
                [cfd_lift[index] / 1e3, predicted_lift[index] / 1e3],
                color=error_cmap(error_norm(ld_error[index])),
                alpha=0.8,
                linewidth=1.2,
                zorder=1,
            )
        operating_ax.scatter(
            cfd_drag[valid_operating] / 1e3,
            cfd_lift[valid_operating] / 1e3,
            marker="o",
            s=3,
            color="#202020",
            label="Original CFD",
            zorder=2,
        )
        operating_ax.figure.colorbar(
            plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
            ax=operating_ax,
            label="Absolute L/D error",
        )
        operating_ax.set_title("Lift-drag operating map (test set)\nline color shows absolute L/D error")
        operating_ax.set_xlabel("Drag [kN]")
        operating_ax.set_ylabel("Lift [kN]")
        operating_ax.grid(True, alpha=0.25)
        operating_ax.legend(frameon=False)

        ordering_ax = axes[1, 1]
        valid_ratio = ~is_train & np.isfinite(cfd_lift_to_drag) & np.isfinite(predicted_lift_to_drag)
        cfd_ratio_valid = cfd_lift_to_drag[valid_ratio]
        predicted_ratio_valid = predicted_lift_to_drag[valid_ratio]
        ratio_order = np.argsort(cfd_ratio_valid)
        cfd_ratio_sorted = cfd_ratio_valid[ratio_order]
        predicted_ratio_sorted = predicted_ratio_valid[ratio_order]
        ranks = np.arange(1, ratio_order.size + 1)
        for rank, true_ratio, pred_ratio in zip(
            ranks, cfd_ratio_sorted, predicted_ratio_sorted
        ):
            ordering_ax.plot(
                [rank, rank], [true_ratio, pred_ratio], color="0.7", linewidth=0.8
            )
        ordering_ax.plot(
            ranks, cfd_ratio_sorted, "o-", markersize=4, label="Reconstructed CFD"
        )
        ordering_ax.plot(
            ranks, predicted_ratio_sorted, "s--", markersize=4, label="Prediction"
        )
        ratio_rho = spearman_rho(cfd_ratio_valid, predicted_ratio_valid)
        ratio_mae = mae(cfd_ratio_valid, predicted_ratio_valid)
        ordering_ax.set_title(
            f"Test L/D ordered by CFD value\nSpearman rho = {ratio_rho:.3f}, MAE = {ratio_mae:.3g}"
        )
        ordering_ax.set_xlabel("CFD L/D rank")
        ordering_ax.set_ylabel("Lift-to-drag ratio")
        ordering_ax.grid(True, alpha=0.25)
        ordering_ax.legend(frameon=False)

        path = output_dir / f"force_components_and_efficiency_{model_slug}.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths["force_components_and_efficiency"] = str(path)
        plt.close(figure)

        lift_error = relative_error_percent(predicted_lift, cfd_lift)
        drag_error = relative_error_percent(predicted_drag, cfd_drag)
        parameter_candidates = [
            ("pitch", "Pitch [deg]"),
            ("planform_area", "Estimated planform area [m$^2$]"),
            ("wing_sweep", "Wing sweep [deg]"),
            ("wing_half_span", "Wing half-span [m]"),
            ("transition_sweep", "Transition sweep [deg]"),
            ("elevator_deflection", "Elevator deflection [deg]"),
        ]
        parameter_columns = []
        for key, label in parameter_candidates:
            values = column(key)
            finite_values = values[np.isfinite(values)]
            if finite_values.size >= 2 and np.unique(finite_values).size >= 2:
                parameter_columns.append((values, label))

        if parameter_columns:
            num_columns = min(3, len(parameter_columns))
            num_rows = int(math.ceil(len(parameter_columns) / num_columns))
            figure, axes = plt.subplots(
                num_rows,
                num_columns,
                figsize=(5.0 * num_columns, 3.8 * num_rows),
                constrained_layout=True,
                squeeze=False,
            )
            for ax, (parameter, label) in zip(axes.flat, parameter_columns):
                valid = np.isfinite(parameter) & np.isfinite(lift_error) & np.isfinite(drag_error)
                train_valid = valid & is_train
                test_valid = valid & ~is_train
                ax.scatter(
                    parameter[train_valid],
                    lift_error[train_valid],
                    marker="o",
                    color="tab:blue",
                    alpha=0.78,
                    label="Lift error (train)",
                )
                ax.scatter(
                    parameter[test_valid],
                    lift_error[test_valid],
                    marker="o",
                    facecolors="none",
                    edgecolors="tab:blue",
                    linewidth=1.2,
                    label="Lift error (test)",
                )
                ax.scatter(
                    parameter[train_valid],
                    drag_error[train_valid],
                    marker="^",
                    color="tab:orange",
                    alpha=0.78,
                    label="Drag error (train)",
                )
                ax.scatter(
                    parameter[test_valid],
                    drag_error[test_valid],
                    marker="^",
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidth=1.2,
                    label="Drag error (test)",
                )
                lift_rho = spearman_rho(parameter[valid], lift_error[valid])
                drag_rho = spearman_rho(parameter[valid], drag_error[valid])
                ax.set_title(
                    f"Error vs {label.split(' [')[0]}\n"
                    f"Spearman rho: lift={lift_rho:.2f}, drag={drag_rho:.2f}"
                )
                ax.set_xlabel(label)
                ax.set_ylabel("Absolute relative error [%]")
                ax.grid(True, alpha=0.25)
            for ax in axes.flat[len(parameter_columns):]:
                ax.set_visible(False)
            axes.flat[0].legend(frameon=False, fontsize=8, ncol=2, loc="best")
            path = output_dir / f"error_vs_case_parameters_{model_slug}.png"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths["error_vs_case_parameters"] = str(path)
            plt.close(figure)

            figure, axes = plt.subplots(
                num_rows,
                num_columns,
                figsize=(5.0 * num_columns, 3.8 * num_rows),
                constrained_layout=True,
                squeeze=False,
            )
            for ax, (parameter, label) in zip(axes.flat, parameter_columns):
                drag_ax = ax.twinx()
                lift_valid = (
                    np.isfinite(parameter)
                    & np.isfinite(cfd_lift)
                    & np.isfinite(predicted_lift)
                )
                drag_valid = (
                    np.isfinite(parameter)
                    & np.isfinite(cfd_drag)
                    & np.isfinite(predicted_drag)
                )
                lift_handles = [
                    ax.scatter(
                        parameter[lift_valid],
                        cfd_lift[lift_valid] / 1e3,
                        color="tab:blue",
                        marker="o",
                        label="CFD lift",
                    ),
                    ax.scatter(
                        parameter[lift_valid],
                        predicted_lift[lift_valid] / 1e3,
                        color="tab:blue",
                        marker="x",
                        label="Predicted lift",
                    ),
                ]
                drag_handles = [
                    drag_ax.scatter(
                        parameter[drag_valid],
                        cfd_drag[drag_valid] / 1e3,
                        color="tab:orange",
                        marker="^",
                        label="CFD drag",
                    ),
                    drag_ax.scatter(
                        parameter[drag_valid],
                        predicted_drag[drag_valid] / 1e3,
                        color="tab:orange",
                        marker="x",
                        label="Predicted drag",
                    ),
                ]
                ax.set_title(f"Forces vs {label.split(' [')[0]}")
                ax.set_xlabel(label)
                ax.set_ylabel("Lift [kN]", color="tab:blue")
                drag_ax.set_ylabel("Drag [kN]", color="tab:orange")
                ax.tick_params(axis="y", labelcolor="tab:blue")
                drag_ax.tick_params(axis="y", labelcolor="tab:orange")
                ax.grid(True, alpha=0.25)
                ax.legend(
                    lift_handles + drag_handles,
                    [handle.get_label() for handle in lift_handles + drag_handles],
                    frameon=False,
                    fontsize=8,
                )
            for ax in axes.flat[len(parameter_columns):]:
                ax.set_visible(False)
            path = output_dir / f"forces_vs_case_parameters_{model_slug}.png"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            saved_paths["forces_vs_case_parameters"] = str(path)
            plt.close(figure)

        metrics.update(
            {
                "pressure_drag_mae_kN": mae(
                    cfd_pressure_drag, predicted_pressure_drag
                ) / 1e3,
                "viscous_drag_mae_kN": mae(
                    cfd_viscous_drag, predicted_viscous_drag
                ) / 1e3,
                "lift_to_drag_mae": ratio_mae,
                "lift_to_drag_spearman_rho": ratio_rho,
                "reconstructed_lift_mean_abs_error_percent": float(
                    np.nanmean(lift_error)
                ),
                "reconstructed_drag_mean_abs_error_percent": float(
                    np.nanmean(drag_error)
                ),
            }
        )

    create_additional_plots()

    return {
        "case_ids": case_ids,
        "is_train": is_train,
        "records": records,
        "metrics": metrics,
        "plot_paths": saved_paths,
    }

def _aero_coefficient_references(case, surface_loads, gas_constant=287.05):
    """Estimate q, modeled-domain planform area, and strip-based MAC in SI units.

    Use the same domain as the unscaled force integrals: do not double a half
    mesh's area. MAC is unchanged by mirroring. Strip integration is approximate
    because each face's projected area is assigned at its centroid.
    """
    flow = case['flow_condition']
    p_inf = float(np.asarray(flow['p_inf']).item())
    T_inf = float(np.asarray(flow['T_inf']).item())
    speed = float(np.linalg.norm(np.asarray(flow['v_inf'])))
    pitch = float(np.asarray(flow.get('pitch', 0.0)).item())
    if not all(np.isfinite(v) and v > 0 for v in (p_inf, T_inf, speed, gas_constant)):
        raise ValueError(f"Case {case['case_id']}: invalid freestream coefficient references")
    q = 0.5 * p_inf / (gas_constant * T_inf) * speed**2

    areas = np.asarray(surface_loads['surface_areas'], dtype=float)
    normals = np.asarray(surface_loads['surface_normals'], dtype=float)
    centroids = np.asarray(surface_loads['surface_centroids'], dtype=float)
    theta = np.deg2rad(pitch)
    body_normal_z = np.sin(theta) * normals[:, 0] + np.cos(theta) * normals[:, 2]
    projected_areas = 0.5 * areas * np.abs(body_normal_z)
    area = float(projected_areas.sum())
    wall_ids = case['topology']['foam_mesh']['wall_pt_ids']
    wall_y = np.asarray(case['warped_geo']['foam_mesh']['points'])[wall_ids, 1]
    y_min, y_max = float(wall_y.min()), float(wall_y.max())
    if not np.isfinite(area) or area <= 0 or not np.isfinite(y_max - y_min) or y_max <= y_min:
        raise ValueError(f"Case {case['case_id']}: invalid surface coefficient references")

    def strip_mac(num_strips):
        edges = np.linspace(y_min, y_max, num_strips + 1)
        strip_area, _ = np.histogram(centroids[:, 1], bins=edges, weights=projected_areas)
        return float(np.sum(strip_area**2 / np.diff(edges)) / area)

    mac = strip_mac(64)
    refined_mac = strip_mac(128)
    return q, area, mac, 100.0 * abs(refined_mac - mac) / mac

def analyze_aero_forces(
        model_eval,
        params,
        configs,
        constants,
        screenshot: bool = False,
        model_name: str = "",
        full_analysis: bool = False,
        plots_dir: str = "post_processing_plots",
    ):
    """Run the legacy force analysis, optionally adding diagnostic plots.

    Every case is evaluated exactly once. ``full_analysis`` only collects
    component-force and case metadata from that loop, then passes those values
    to the additional post-processing routine after the legacy plots.
    An NPZ alongside the text summary stores unrounded per-case values;
    ``values`` columns follow ``columns`` (including units), with rows in
    ``case_ids`` order and ``is_train`` identifying each data split.
    """
    import jax
    output_dir, model_slug = _aero_plot_output(model_name, plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # plot histogram of errors across cases
    case_ids = [config['case_id'] for config in configs]
    from .postprocessing import compute_forces, integrate_forces, integrate_moments
    compute_forces_jit = jax.jit(lambda p, U, nuTilda, T, warped_geo: compute_forces(p, U, nuTilda, T, warped_geo,  case['topology']), device=constants['device'])

    lifts_compute = []
    drags_compute = []
    lifts_openfoam = []
    drags_openfoam = []
    lifts_panel = []
    drags_panel = []
    lift_reconstructions = []
    drag_reconstructions = []
    full_analysis_records = []
    coefficient_references = []
    moments_compute = []
    moments_lowfi = []
    lowfi_moment_sources = []
    moment_reconstructions = []
    per_case_rows = []
    per_case_columns = (
            "p L2 [%]", "U L2 [%]", "nuTilda [%]", "T L2 [%]",
            "Surf p [%]", "Surf U [%]", "Surf nu [%]", "Surf T [%]",
            "L NN [kN]", "L Data [kN]", "L OF [kN]", "L LF [kN]", "L Err [%]",
            "D NN [kN]", "D Data [kN]", "D OF [kN]", "D LF [kN]", "D Err [%]",
            "My NN [kNm]", "My Data[kNm]", "My Err [%]",
            "My LF [kNm]", "My LF Err[%]",
    )
    per_case_header = f"{'Case':>6} " + " ".join(
        f"{label:>12}" for label in per_case_columns
    )
    per_case_lines = [
        "", "Per-case field errors and aerodynamic forces",
        "Field errors: relative L2 (%). Force errors: NN vs reconstructed Data (%).",
        "LF = low-fidelity; My LF error is signed relative to reconstructed CFD My.",
        per_case_header, "-" * len(per_case_header),
    ]

    # Per-case relative L2 errors for both data splits.
    field_l2_cases = {
        split: {field: [] for field in ("p", "U", "nuTilda", "T")}
        for split in ("train", "test")
    }
    surface_l2_cases = {
        split: {field: [] for field in ("p", "U", "nuTilda", "T")}
        for split in ("train", "test")
    }
    worst_field_cases = {field: [] for field in ("p", "u_mag")}

    # constants
    minimum_lift_ref = 2.5e5
    minimum_moment_ref = 5.0e6  # Nm (5,000 kNm)
    num = len(configs)
    for i, case in enumerate(configs):
        # print_content(case)
        # exit()
        # reconstruct forces from data
        p_true, U_true, nuTilda_true, T_true = case['data']['p'], case['data']['U'], case['data']['nuTilda'], case['data']['T']
        p_true = jax.device_put(p_true, device=constants['device'])
        U_true = jax.device_put(U_true, device=constants['device'])
        nuTilda_true = jax.device_put(nuTilda_true, device=constants['device'])
        T_true = jax.device_put(T_true, device=constants['device'])

        # T_true = np.full_like(T_true, case['flow_condition']['T_inf'])
        forces_reconstructed = compute_forces_jit(p_true, U_true, nuTilda_true, T_true, case['warped_geo'])
        L_reconstructed, D_d = integrate_forces(forces_reconstructed)
        lift_reconstructions.append(L_reconstructed)
        drag_reconstructions.append(D_d)
        M_reconstructed = float(integrate_moments(forces_reconstructed, axis="y"))
        moment_reconstructions.append(M_reconstructed)
        coefficient_references.append(_aero_coefficient_references(
            case, forces_reconstructed, constants.get('R', 287.05)
        ))

        # compute forces from model
        case_flow_condition = jax.device_put(case['flow_condition'], device=constants['device'])
        v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, case_flow_condition, case['warped_geo'])
        p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
        T_pred = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
        nuTilda_pred = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])
        p_pred = np.clip(p_pred, a_min=1000, a_max=np.inf)
        v_pred = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])

        # T_pred = np.full_like(T_pred, case['flow_condition']['T_inf'])
        forces_pred = compute_forces_jit(p_pred, v_pred, nuTilda_pred, T_pred, case['warped_geo'])
        L_pred, D_pred = integrate_forces(forces_pred)
        # L_pred = L_pred + 14000
        lifts_compute.append(L_pred)
        drags_compute.append(D_pred)
        M_pred = float(integrate_moments(forces_pred, axis="y"))
        moments_compute.append(M_pred)
        
        # get OpenFOAM forces
        L_of, D_of = case['data']['lift'][0], case['data']['drag'][0]
        lifts_openfoam.append(L_of)
        drags_openfoam.append(D_of)

        # Get low-fidelity forces (panel or Euler).
        if case['low_fidelity_info'] is not None:
            L_panel, D_panel = case['low_fidelity_info']['lift'].item(), case['low_fidelity_info']['drag'].item()
        else:
            L_panel, D_panel = 0.0, 0.0
        
        lowfi_info = case.get('low_fidelity_info') or {}
        lowfi_pressure = case['warped_geo'].get('low_fidelity_inputs', {}).get('p')
        if 'moment' in lowfi_info:
            saved_moment = np.asarray(lowfi_info['moment'], dtype=float).reshape(-1)
            # Accept scalar My or a Cartesian [Mx, My, Mz] moment vector, in Nm.
            if saved_moment.size not in (1, 3):
                raise ValueError(f"Case {case['case_id']}: expected scalar My or 3-component low-fidelity moment")
            M_lowfi = float(saved_moment[0 if saved_moment.size == 1 else 1])
            moment_source = "saved"
        elif lowfi_pressure is not None:
            surface_idx = np.asarray(case['topology']['foam_mesh']['faces']['surface_neighbors'])
            surface_pressure = np.asarray(lowfi_pressure).reshape(-1)[surface_idx]
            pressure_forces = (
                surface_pressure[:, None]
                * np.asarray(forces_reconstructed['surface_areas'])[:, None]
                * np.asarray(forces_reconstructed['surface_normals'])
            )
            M_lowfi = float(integrate_moments({
                'forces': pressure_forces,
                'surface_centroids': forces_reconstructed['surface_centroids'],
            }, axis="y"))
            moment_source = "interpolated-pressure"
        else:
            M_lowfi = np.nan
            moment_source = "unavailable"
        moments_lowfi.append(M_lowfi)
        lowfi_moment_sources.append(moment_source)

        # L_panel, D_panel = 0.0, 0.0 # divide by 2 because symmetry
        # p_panel = jax.device_put(case['warped_geo']['panel']['p'], device=constants['device'])
        # v_panel = jax.device_put(case['warped_geo']['panel']['v'], device=constants['device'])
        # T_panel = jax.device_put(case['warped_geo']['panel']['T'], device=constants['device'])
        # nuTilda_panel = jax.device_put(case['warped_geo']['panel']['nuTilda'], device=constants['device'])
        # forces_pred = compute_forces_jit(p_panel, v_panel, nuTilda_panel, T_panel, case['warped_geo'])
        # L_panel, D_panel = integrate_forces(forces_pred)
        
        lifts_panel.append(L_panel)
        drags_panel.append(D_panel)

        # Compute field error for the train/test summary table.
        is_train = case['train']
        split = "train" if is_train else "test"
        p_l2_error = float(relative_l2(p_pred, p_true))
        u_mag_l2_error = float(relative_l2(
            np.linalg.norm(v_pred, axis=1), np.linalg.norm(U_true, axis=1)
        ))
        field_l2_cases[split]["p"].append(p_l2_error)
        field_l2_cases[split]["U"].append(float(relative_l2(v_pred, U_true)))
        field_l2_cases[split]["nuTilda"].append(
            float(relative_l2(nuTilda_pred, nuTilda_true))
        )
        field_l2_cases[split]["T"].append(float(relative_l2(T_pred, T_true)))
        worst_field_cases["p"].append((case['case_id'], split, p_l2_error))
        worst_field_cases["u_mag"].append((case['case_id'], split, u_mag_l2_error))

        # Surface fields are the unique wall-adjacent subset of volume cells.
        surface_idx = np.unique(np.asarray(
            case['topology']['foam_mesh']['faces']['surface_neighbors']
        ))
        for field, predicted, truth in (
            ("p", p_pred, p_true), ("U", v_pred, U_true),
            ("nuTilda", nuTilda_pred, nuTilda_true), ("T", T_pred, T_true),
        ):
            surface_l2_cases[split][field].append(
                float(relative_l2(predicted[surface_idx], truth[surface_idx]))
                if surface_idx.size else np.nan
            )

        # print(f"Case {case['case_id']}: L_of={L_of:.2f}, D_of={D_of:.2f}, L_recon={L_reconstructed:.2f}, D_recon={D_d:.2f}, L_pred={L_pred:.2f}, D_pred={D_pred:.2f}")
        # print(f"Processed case {case['case_id']} ({i}/{num}): L={L_pred:.2f}/{L_reconstructed:.2f}/{L_of:.2f}, D={D_pred:.2f}/{D_d:.2f}/{D_of:.2f} \t(NN/Recon/OpenFOAM)")

        # For printing:
        L_nn_kN    = L_pred / 1e3
        L_rec_kN   = L_reconstructed / 1e3
        L_of_kN    = L_of / 1e3
        L_panel_kN = L_panel / 1e3

        D_nn_kN    = D_pred / 1e3
        D_rec_kN   = D_d / 1e3
        D_of_kN    = D_of / 1e3
        D_panel_kN = D_panel / 1e3

        L_ref = max(abs(L_reconstructed), minimum_lift_ref)
        L_err_pct = 100 * (L_pred - L_reconstructed) / L_ref
        D_err_pct = 100 * (D_pred - D_d) / D_d if D_d != 0 else float("nan")

        M_ref = max(abs(M_reconstructed), minimum_moment_ref)
        M_err_pct = 100.0 * (M_pred - M_reconstructed) / M_ref
        M_lowfi_err_pct = 100.0 * (M_lowfi - M_reconstructed) / M_ref

        if full_analysis:
            full_analysis_records.append(
                _full_analysis_record(
                    case,
                    forces_reconstructed,
                    forces_pred,
                    L_reconstructed,
                    D_d,
                    L_pred,
                    D_pred,
                    L_of,
                    D_of,
                    L_panel,
                    D_panel,
                    L_err_pct,
                    D_err_pct,
                )
            )

        # print(
        #     f"Case {case['case_id']:>4} ({i:>3}/{num}) | "
        #     f"L [kN] NN:{L_nn_kN:>8.3f}  Recon:{L_rec_kN:>8.3f}  OF:{L_of_kN:>8.3f}  Err:{L_err_pct:>7.2f}% | "
        #     f"D [kN] NN:{D_nn_kN:>8.3f}  Recon:{D_rec_kN:>8.3f}  OF:{D_of_kN:>8.3f}  Err:{D_err_pct:>7.2f}%"
        # )
        # print(f"L[kN]: {L_nn_kN} {L_rec_kN} {L_of_kN} {L_panel_kN} {L_err_pct}% |   ")
        # exit()

        # print(L_nn_kN, L_rec_kN, L_of_kN, L_panel_kN, L_err_pct)

        case_id = case['case_id']
        case_line = (
            f"{str(case_id)} | "
            f"L[kN]: {L_nn_kN:>8.3f} {L_rec_kN:>8.3f} {L_of_kN:>8.3f} {L_panel_kN:>8.3f} {L_err_pct:>8.2f}% |   "
            f"D[kN]: {D_nn_kN:>8.3f} {D_rec_kN:>8.3f} {D_of_kN:>8.3f} {D_panel_kN:>8.3f} {D_err_pct:>8.2f}%     (NN  Data  OpenFOAM  Low-fidelity  Error(NN-Data))"
        )
        print(case_line)
        per_case_values = [
            100.0 * field_l2_cases[split][field][-1]
            for field in ("p", "U", "nuTilda", "T")
        ] + [
            100.0 * surface_l2_cases[split][field][-1]
            for field in ("p", "U", "nuTilda", "T")
        ] + [
            L_nn_kN, L_rec_kN, L_of_kN, L_panel_kN, L_err_pct,
            D_nn_kN, D_rec_kN, D_of_kN, D_panel_kN, D_err_pct,
            M_pred / 1e3, M_reconstructed / 1e3, M_err_pct,
            M_lowfi / 1e3, M_lowfi_err_pct,
        ]
        per_case_rows.append(np.asarray(per_case_values, dtype=float))
        per_case_lines.append(
            f"{str(case_id):>6} " + " ".join(f"{value:>12.3f}" for value in per_case_values)
        )

    field_l2_summary = {}
    for split, split_values in field_l2_cases.items():
        field_l2_summary[split] = {}
        for field, values in split_values.items():
            values = np.asarray(values, dtype=float)
            field_l2_summary[split][field] = {
                "mean": float(np.mean(values)) if values.size else np.nan,
                "std": float(np.std(values)) if values.size else np.nan,
            }

    import matplotlib.pyplot as plt
    # lift_reconstructions_np = np.asarray(jax.device_get(lift_reconstructions), dtype=float)
    lifts_compute_np        = np.asarray(jax.device_get(lifts_compute), dtype=float)
    lifts_openfoam_np       = np.asarray(jax.device_get(lifts_openfoam), dtype=float)
    # lifts_openfoam_np       = np.asarray(jax.device_get(lift_reconstructions), dtype=float)
    lifts_panel_np        = np.asarray(jax.device_get(lifts_panel), dtype=float)

    # drag_reconstructions_np = np.asarray(jax.device_get(drag_reconstructions), dtype=float)
    drags_compute_np        = np.asarray(jax.device_get(drags_compute), dtype=float)
    drags_openfoam_np       = np.asarray(jax.device_get(drags_openfoam), dtype=float)
    # drags_openfoam_np       = np.asarray(jax.device_get(drag_reconstructions), dtype=float)
    drags_panel_np        = np.asarray(jax.device_get(drags_panel), dtype=float)

    # print train and test averages errors
    train_lift = [L for L, config in zip(lifts_compute_np, configs) if config['train']]
    train_drag = [D for D, config in zip(drags_compute_np, configs) if config['train']]
    test_lift = [L for L, config in zip(lifts_compute_np, configs) if not config['train']]
    test_drag = [D for D, config in zip(drags_compute_np, configs) if not config['train']]
    of_train_lift = [L for L, config in zip(lifts_openfoam_np, configs) if config['train']]
    of_train_drag = [D for D, config in zip(drags_openfoam_np, configs) if config['train']]
    of_test_lift = [L for L, config in zip(lifts_openfoam_np, configs) if not config['train']]
    of_test_drag = [D for D, config in zip(drags_openfoam_np, configs) if not config['train']]
    panel_train_lift = [L for L, config in zip(lifts_panel_np, configs) if config['train']]
    panel_train_drag = [D for D, config in zip(drags_panel_np, configs) if config['train']]
    panel_test_lift = [L for L, config in zip(lifts_panel_np, configs) if not config['train']]
    panel_test_drag = [D for D, config in zip(drags_panel_np, configs) if not config['train']]

    train_lift_error = np.mean(np.abs(np.asarray(train_lift) - np.asarray(of_train_lift)))
    train_drag_error = np.mean(np.abs(np.asarray(train_drag) - np.asarray(of_train_drag)))
    test_lift_error = np.mean(np.abs(np.asarray(test_lift) - np.asarray(of_test_lift)))
    test_drag_error = np.mean(np.abs(np.asarray(test_drag) - np.asarray(of_test_drag)))
    train_panel_lift_error = np.mean(np.abs(np.asarray(panel_train_lift) - np.asarray(of_train_lift)))
    train_panel_drag_error = np.mean(np.abs(np.asarray(panel_train_drag) - np.asarray(of_train_drag)))
    test_panel_lift_error = np.mean(np.abs(np.asarray(panel_test_lift) - np.asarray(of_test_lift)))
    test_panel_drag_error = np.mean(np.abs(np.asarray(panel_test_drag) - np.asarray(of_test_drag)))

    train_lift_l2 = relative_l2(np.asarray(train_lift), np.asarray(of_train_lift)) if train_lift else np.nan
    train_drag_l2 = relative_l2(np.asarray(train_drag), np.asarray(of_train_drag)) if train_drag else np.nan
    test_lift_l2 = relative_l2(np.asarray(test_lift), np.asarray(of_test_lift)) if test_lift else np.nan
    test_drag_l2 = relative_l2(np.asarray(test_drag), np.asarray(of_test_drag)) if test_drag else np.nan

    moments_compute = np.asarray(moments_compute)
    moment_reconstructions = np.asarray(moment_reconstructions)

    x = np.arange(len(case_ids))
    # width = 0.25
    width = 0.33

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # ======================================= PLOT 1A: Lift and Drag by Case =======================================
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(10, 11),
        sharex=True,
        constrained_layout=True
    )

    # --- Lift subplot ---
    ax1.bar(x,         lifts_openfoam_np,       width, label="OpenFOAM Force",      color=colors[0])
    ax1.bar(x - width, lifts_panel_np, width, label="Low-fidelity Force", color=colors[2])
    ax1.bar(x + width, lifts_compute_np,        width, label="Model Reconstructed Force",         color=colors[1])

    ax1.set_ylabel("Lift")
    ax1.set_title("Lift, Drag and Moment by Case")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(ncol=3)

    # --- Drag subplot ---
    ax2.bar(x,         drags_openfoam_np,       width, label="OpenFOAM Force",      color=colors[0])
    ax2.bar(x - width, drags_panel_np, width, label="Low-fidelity Force", color=colors[2])
    ax2.bar(x + width, drags_compute_np,        width, label="Model Reconstructed Force",         color=colors[1])

    ax2.set_ylabel("Drag")
    ax2.grid(axis="y", alpha=0.3)

    ax3.bar(x - width / 2, moment_reconstructions / 1e3, width,
            label="Reconstructed CFD", color=colors[0])
    ax3.bar(x + width / 2, moments_compute / 1e3, width,
            label="Model", color=colors[1])
    ax3.set_ylabel("My [kNm]")
    ax3.set_xlabel("Case ID")
    ax3.set_title("Y-axis moment about (0, 0, 0)")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()

    ax3.set_xticks(x)
    ax3.set_xticklabels(case_ids, rotation=45, ha="right")

    if screenshot:
        path = output_dir / f"lift_drag_histogram_{model_slug}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')

    # ======================================= PLOT 1B: Lift and Drag by Case =======================================


    # ======================================= PLOT 2: Lift and Drag ERRORS by Case =======================================
    is_train = np.array([config['train'] for config in configs], dtype=bool)
    from matplotlib.patches import Patch
    def relative_error_percent(pred, truth, min_ref=None):
        pred = np.asarray(pred)
        truth = np.asarray(truth)

        if min_ref is None:
            min_ref = max(0.05 * np.max(np.abs(truth)), 1e-12)

        denom = np.maximum(np.abs(truth), min_ref)
        err_pct = 100.0 * np.abs(pred - truth) / denom
        # err_pct = pred - truth
        # err_pct = np.abs(pred-truth)
        return err_pct, min_ref

    # error calculations
    lift_err_pct, lift_min_ref = relative_error_percent(
        lifts_compute_np, lifts_openfoam_np, minimum_lift_ref
    )
    drag_err_pct, drag_min_ref = relative_error_percent(
        drags_compute_np, drags_openfoam_np
    )

    lift_panel_err_pct, _ = relative_error_percent(
        lifts_panel_np, lifts_openfoam_np, minimum_lift_ref
    )
    drag_panel_err_pct, _ = relative_error_percent(
        drags_panel_np, drags_openfoam_np
    )

    # averages
    lift_avg_all   = np.mean(lift_err_pct)
    lift_avg_train = np.mean(lift_err_pct[is_train]) if np.any(is_train) else np.nan
    lift_avg_test  = np.mean(lift_err_pct[~is_train]) if np.any(~is_train) else np.nan
    lift_panel_avg_all   = np.mean(lift_panel_err_pct)
    lift_panel_avg_train = np.mean(lift_panel_err_pct[is_train]) if np.any(is_train) else np.nan
    lift_panel_avg_test  = np.mean(lift_panel_err_pct[~is_train]) if np.any(~is_train) else np.nan

    drag_avg_all   = np.mean(drag_err_pct)
    drag_avg_train = np.mean(drag_err_pct[is_train]) if np.any(is_train) else np.nan
    drag_avg_test  = np.mean(drag_err_pct[~is_train]) if np.any(~is_train) else np.nan
    drag_panel_avg_all   = np.mean(drag_panel_err_pct)
    drag_panel_avg_train = np.mean(drag_panel_err_pct[is_train]) if np.any(is_train) else np.nan
    drag_panel_avg_test  = np.mean(drag_panel_err_pct[~is_train]) if np.any(~is_train) else np.nan

    def field_l2_text(split, field):
        stats = field_l2_summary[split][field]
        if not np.isfinite(stats["mean"]):
            return "n/a"
        return f"{100.0 * stats['mean']:.3f} ± {100.0 * stats['std']:.3f}"

    summary_rows = [
        *[
            (f"Field relative L2: {field}", "%", field_l2_text("train", field), field_l2_text("test", field))
            for field in ("p", "U", "nuTilda", "T")
        ],
        ("Model lift MAE", "kN", f"{train_lift_error/1e3:.3f}", f"{test_lift_error/1e3:.3f}"),
        ("Model drag MAE", "kN", f"{train_drag_error/1e3:.3f}", f"{test_drag_error/1e3:.3f}"),
        ("Model lift relative L2", "%", f"{100.0 * train_lift_l2:.3f}", f"{100.0 * test_lift_l2:.3f}"),
        ("Model drag relative L2", "%", f"{100.0 * train_drag_l2:.3f}", f"{100.0 * test_drag_l2:.3f}"),
        ("Low-fidelity lift MAE", "kN", f"{train_panel_lift_error/1e3:.3f}", f"{test_panel_lift_error/1e3:.3f}"),
        ("Low-fidelity drag MAE", "kN", f"{train_panel_drag_error/1e3:.3f}", f"{test_panel_drag_error/1e3:.3f}"),
        ("Model lift relative error", "%", f"{lift_avg_train:.2f}", f"{lift_avg_test:.2f}"),
        ("Model drag relative error", "%", f"{drag_avg_train:.2f}", f"{drag_avg_test:.2f}"),
        ("Low-fidelity lift relative error", "%", f"{lift_panel_avg_train:.2f}", f"{lift_panel_avg_test:.2f}"),
        ("Low-fidelity drag relative error", "%", f"{drag_panel_avg_train:.2f}", f"{drag_panel_avg_test:.2f}"),
    ]
    for field in ("p", "U", "nuTilda", "T"):
        values_by_split = []
        for split in ("train", "test"):
            values = np.asarray(surface_l2_cases[split][field]) * 100.0
            values_by_split.append(
                f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values.size else "n/a"
            )
        summary_rows.append((f"Surface relative L2: {field}", "%", *values_by_split))

    moment_error_percent, _ = relative_error_percent(
        moments_compute, moment_reconstructions, minimum_moment_ref
    )
    moment_stats = []
    for mask in (is_train, ~is_train):
        pred, truth = moments_compute[mask], moment_reconstructions[mask]
        percentages = moment_error_percent[mask]
        percentages = percentages[np.isfinite(percentages)]
        moment_stats.append((
            np.mean(np.abs(pred - truth)) / 1e3 if pred.size else np.nan,
            100.0 * relative_l2(pred, truth) if pred.size else np.nan,
            np.mean(percentages) if percentages.size else np.nan,
        ))
    for index, (metric, unit) in enumerate((
        ("Model My MAE", "kNm"), ("Model My relative L2", "%"),
        ("Model My relative error", "%"),
    )):
        summary_rows.append((metric, unit, *[
            f"{stats[index]:.3f}" for stats in moment_stats
        ]))

    moments_lowfi = np.asarray(moments_lowfi)
    lowfi_moment_error_percent, _ = relative_error_percent(
        moments_lowfi, moment_reconstructions, minimum_moment_ref
    )
    lowfi_moment_stats = []
    for split_mask in (is_train, ~is_train):
        mask = split_mask & np.isfinite(moments_lowfi) & np.isfinite(moment_reconstructions)
        pred, truth = moments_lowfi[mask], moment_reconstructions[mask]
        lowfi_moment_stats.append((
            np.mean(np.abs(pred - truth)) / 1e3 if pred.size else np.nan,
            100.0 * relative_l2(pred, truth) if pred.size else np.nan,
            np.mean(lowfi_moment_error_percent[mask]) if pred.size else np.nan,
        ))
    for index, (metric, unit) in enumerate((
        ("Low-fidelity My MAE", "kNm"), ("Low-fidelity My relative L2", "%"),
        ("Low-fidelity My relative error", "%"),
    )):
        summary_rows.append((metric, unit, *[
            f"{stats[index]:.3f}" for stats in lowfi_moment_stats
        ]))

    coefficient_references = np.asarray(coefficient_references)
    q_values, reference_areas, reference_chords, mac_changes = coefficient_references.T
    force_reference = q_values * reference_areas
    coefficient_denominators = np.column_stack((
        force_reference, force_reference, force_reference * reference_chords,
    ))
    cfd_coefficients = np.column_stack((
        lifts_openfoam_np, drags_openfoam_np, moment_reconstructions,
    )) / coefficient_denominators
    model_coefficients = np.column_stack((
        lifts_compute_np, drags_compute_np, moments_compute,
    )) / coefficient_denominators
    lowfi_coefficients = np.column_stack((
        lifts_panel_np, drags_panel_np, moments_lowfi,
    )) / coefficient_denominators
    # Legacy L/D arrays use zero for absent low-fidelity data; exclude those
    # placeholders from the new coefficient metrics.
    lowfi_coefficients[[case.get('low_fidelity_info') is None for case in configs], :2] = np.nan
    coefficient_rows = []
    for label, predictions in (("Model", model_coefficients), ("Low-fidelity", lowfi_coefficients)):
        for index, coefficient in enumerate(("CL", "CD", "Cm")):
            errors = np.abs(predictions[:, index] - cfd_coefficients[:, index])
            values = []
            for split_mask in (is_train, ~is_train):
                valid = split_mask & np.isfinite(errors)
                values.append(f"{np.mean(errors[valid]):.6f}" if np.any(valid) else "n/a")
            coefficient_rows.append((f"{label} {coefficient} MAE", "-", *values))

    summary_lines = [
        "Aerodynamic error summary",
        "My: right-hand y-axis moment about (0, 0, 0), NN vs saved-CFD reconstructed loads.",
        "Model/CFD My includes pressure and viscous loads; no saved OpenFOAM moment is required.",
        "Low-fidelity My: saved moment (Nm, y-axis about origin) when available; otherwise pressure-only",
        "integration of interpolated low-fidelity pressures on the RANS surface (approximate).",
        "Unavailable/nonfinite low-fidelity moments are excluded from low-fidelity My summary metrics.",
        "Low-fidelity My sources: " + ", ".join(
            f"{source}={lowfi_moment_sources.count(source)}"
            for source in ("saved", "interpolated-pressure", "unavailable")
        ),
        f"My relative error: mean absolute percentage error with max(|CFD My|, {minimum_moment_ref / 1e3:,.0f} kNm) denominator.",
        "Per-case My error is signed and uses the same minimum reference.",
        "Surface fields: unique wall-adjacent volume cells, unweighted relative L2.",
        "Coefficient MAE: mean absolute coefficient difference (dimensionless, no reference floors).",
        "CL/CD reference: saved OpenFOAM L/D; Cm reference: reconstructed CFD My about origin.",
        "q = p_inf/(R*T_inf)*V_inf^2/2; S = half the absolute pitch-corrected surface projection.",
        "S uses the modeled domain without symmetry doubling, matching the unscaled load integrals.",
        "MAC = integral(c(y)^2 dy)/S from 64 spanwise strips; references are shared by all methods.",
        "MAC strip sensitivity (64 vs 128): maximum change = " + f"{np.max(mac_changes):.2f}%.",
    ]
    table_groups = [
        ("Volume field relative L2", [row for row in summary_rows if row[0].startswith("Field ")]),
        ("Surface field relative L2", [row for row in summary_rows if row[0].startswith("Surface ")]),
        ("Force and moment MAE", [row for row in summary_rows if row[0].endswith(" MAE")]),
        ("Force and moment relative L2", [row for row in summary_rows if row[0].endswith(" relative L2")]),
        ("Force and moment relative errors", [row for row in summary_rows if row[0].endswith(" relative error")]),
    ]
    table_groups.append(("Coefficient mean absolute errors", coefficient_rows))
    for title, rows in table_groups:
        # Keep model lift, drag, and moment together before low-fidelity metrics.
        rows = sorted(rows, key=lambda row: row[0].startswith("Low-fidelity "))
        summary_lines.extend([
            "", title,
            f"{'Metric':<32} {'Unit':>6} {'Train':>18} {'Test':>18}",
            "-" * 78,
        ])
        for metric, unit, train_value, test_value in rows:
            train_value = "n/a" if train_value == "nan" else train_value
            test_value = "n/a" if test_value == "nan" else test_value
            summary_lines.append(
                f"{metric:<32} {unit:>6} {train_value:>18} {test_value:>18}"
            )
    print("\n" + "\n".join(summary_lines))

    def top_force_cases(errors):
        errors = np.asarray(errors, dtype=float)
        valid = np.flatnonzero(np.isfinite(errors))
        order = valid[np.argsort(errors[valid])[::-1]][:3]
        return [
            (case_ids[index], "train" if is_train[index] else "test", errors[index])
            for index in order
        ]

    worst_case_lines = [
        "",
        "Top 3 worst cases",
        "Lift/Drag: absolute relative error vs OpenFOAM; My: vs reconstructed CFD",
        "Pressure/u_mag: per-case relative L2 error",
        f"{'Quantity':<12} {'Rank':>4} {'Case':>12} {'Split':>8} {'Error':>14}",
        "-" * 54,
    ]
    for quantity, errors in (("Lift", lift_err_pct), ("Drag", drag_err_pct), ("My", moment_error_percent)):
        for rank, (case_id, split, error) in enumerate(top_force_cases(errors), start=1):
            worst_case_lines.append(
                f"{quantity:<12} {rank:>4} {str(case_id):>12} {split:>8} {error:>13.3f}%"
            )
    for quantity, field in (("Pressure", "p"), ("u_mag", "u_mag")):
        rows = [row for row in worst_field_cases[field] if np.isfinite(row[2])]
        rows = sorted(rows, key=lambda row: row[2], reverse=True)[:3]
        for rank, (case_id, split, error) in enumerate(rows, start=1):
            worst_case_lines.append(
                f"{quantity:<12} {rank:>4} {str(case_id):>12} {split:>8} {100.0 * error:>13.3f}%"
            )

    per_case_lines.extend([
        "", "Low-fidelity moment source by case",
        *[f"{case_id}: {source}" for case_id, source in zip(case_ids, lowfi_moment_sources)],
    ])
    per_case_lines.extend([
        "", "Coefficient references by case (modeled-domain S; approximate MAC)",
        f"{'Case':>12} {'Pitch [deg]':>14} {'q [Pa]':>14} {'S [m^2]':>14} {'MAC [m]':>14} {'MAC change [%]':>16}",
        *[
            f"{str(case_id):>12} {float(np.asarray(case['flow_condition'].get('pitch', 0.0)).item()):>14.3f} "
            f"{q:>14.3f} {area:>14.3f} {mac:>14.5f} {change:>16.3f}"
            for case_id, case, (q, area, mac, change)
            in zip(case_ids, configs, coefficient_references)
        ],
    ])
    summary_path = output_dir / f"aerodynamic_error_summary_{model_slug}.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines + worst_case_lines + per_case_lines) + "\n")
    np.savez_compressed(
        summary_path.with_suffix(".npz"),
        case_ids=np.asarray(case_ids, dtype=str),
        is_train=np.asarray([case['train'] for case in configs], dtype=bool),
        columns=np.asarray(per_case_columns),
        values=np.asarray(per_case_rows, dtype=float).reshape(-1, len(per_case_columns)),
        lowfi_moment_sources=np.asarray(lowfi_moment_sources, dtype=str),
        coefficient_reference_columns=np.asarray([
            "q [Pa]", "S [m^2]", "MAC [m]", "MAC change [%]",
        ]),
        coefficient_references=np.asarray(coefficient_references, dtype=float).reshape(-1, 4),
        pitch_deg=np.asarray([
            float(np.asarray(case['flow_condition'].get('pitch', 0.0)).item())
            for case in configs
        ]),
    )

    # colors by split
    train_color = colors[0]
    test_color  = colors[1]

    lift_bar_colors = [train_color if t else test_color for t in is_train]
    drag_bar_colors = [train_color if t else test_color for t in is_train]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(10, 11),
        sharex=True,
        constrained_layout=True
    )

    legend_handles = [
        Patch(facecolor=train_color, label="Train case"),
        Patch(facecolor=test_color,  label="Test case"),
    ]

    # --- Lift ERROR subplot ---
    ax1.bar(x, lift_err_pct, width=width, color=lift_bar_colors)
    ax1.set_ylabel("Lift Error (%)")
    ax1.set_title(
        f"Lift Relative Error by Case "
        f"(Train: {lift_avg_train:.2f}%, Test: {lift_avg_test:.2f}%)"
    )
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(handles=legend_handles, loc="upper right")

    # --- Drag ERROR subplot ---
    ax2.bar(x, drag_err_pct, width=width, color=drag_bar_colors)
    ax2.set_ylabel("Drag Error (%)")
    ax2.set_title(
        f"Drag Relative Error by Case "
        f"(Train: {drag_avg_train:.2f}%, Test: {drag_avg_test:.2f}%)"
    )
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=legend_handles, loc="upper right")

    ax3.set_xticks(x)
    ax3.set_xticklabels(case_ids, rotation=45, ha="right")

    ax3.bar(x, moment_error_percent, width=width, color=lift_bar_colors)
    ax3.set_ylabel("My Error (%)")
    ax3.set_xlabel("Case ID")
    ax3.set_title(
        "Moment Relative Error vs Reconstructed CFD "
        f"(Train: {moment_stats[0][2]:.2f}%, Test: {moment_stats[1][2]:.2f}%)"
    )
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend(handles=legend_handles, loc="upper right")

    # ylim between 0 and 50 percent
    ax1.set_ylim(0, 50)
    ax2.set_ylim(0, 50)

    if screenshot:
        path = output_dir / f"lift_drag_error_histogram_{model_slug}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        
    # ======================================= PLOT 3: PARITY PLOTS =======================================
    is_train = np.array([config['train'] for config in configs], dtype=bool)

    def parity_plot(ax, y_true, y_pred, title, xlabel, ylabel):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        ax.scatter(
            y_true[is_train], y_pred[is_train],
            s=20, alpha=0.85, marker="o",
            edgecolor="black", linewidth=0.6,
            label="Train"
        )
        ax.scatter(
            y_true[~is_train], y_pred[~is_train],
            s=22, alpha=0.9, marker="s",
            edgecolor="black", linewidth=0.6,
            label="Test"
        )

        ax.axline(
            (0.0, 0.0),
            slope=1.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        # ax.set_xlim(lo, hi)
        # ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    def add_error_band(ax, minimum_reference):
        x_values = np.linspace(*ax.get_xlim(), 400)
        tolerance = 0.15 * np.maximum(np.abs(x_values), minimum_reference)
        lower_bound = x_values - tolerance
        upper_bound = x_values + tolerance
        ax.fill_between(
            x_values,
            lower_bound,
            upper_bound,
            color="tab:green",
            alpha=0.15,
            label="15%",
            zorder=0,
        )
        ax.legend(frameon=False)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(11, 3),
        constrained_layout=True
    )

    parity_plot(
        ax1,
        lifts_openfoam_np / 1e3,
        lifts_compute_np / 1e3,
        f"{model_name} Lift Parity",
        "CFD Lift [kN]",
        "Predicted Lift [kN]"
    )
    ax1.set_xlim(-500, 1200)
    ax1.set_ylim(-500, 1200)
    add_error_band(ax1, minimum_lift_ref / 1e3)

    parity_plot(
        ax2,
        drags_openfoam_np / 1e3,
        drags_compute_np / 1e3,
        f"{model_name} Drag Parity",
        "CFD Drag [kN]",
        "Predicted Drag [kN]"
    )
    ax2.set_xlim(0, 250)
    ax2.set_ylim(0, 250)
    add_error_band(ax2, drag_min_ref / 1e3)

    parity_plot(
        ax3,
        moment_reconstructions / 1e3,
        moments_compute / 1e3,
        f"{model_name} My Parity",
        "Reconstructed CFD My [kNm]",
        "Predicted My [kNm]",
    )
    # Match axis limits while allowing either sign and any moment magnitude.
    lo = min(ax3.get_xlim()[0], ax3.get_ylim()[0])
    hi = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)
    add_error_band(ax3, minimum_moment_ref / 1e3)
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)

    if screenshot:
        path = output_dir / f"lift_drag_parity_{model_slug}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    for index, (ax, coefficient) in enumerate(zip(axes, ("CL", "CD", "Cm"))):
        parity_plot(
            ax, cfd_coefficients[:, index], model_coefficients[:, index],
            f"{model_name} {coefficient} Parity",
            f"{'Reconstructed CFD' if coefficient == 'Cm' else 'CFD'} {coefficient}",
            f"Predicted {coefficient}",
        )
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        # A common raw-force floor would become a different coefficient floor
        # per case, so show an ordinary 15% band here without a floor.
        add_error_band(ax, 0.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.suptitle("Coefficient parity (15% band without reference floors)")
    if screenshot:
        path = output_dir / f"coefficient_parity_{model_slug}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    if full_analysis:
        return _postprocess_full_aero_analysis(
            full_analysis_records,
            field_l2_summary,
            model_name=model_name,
            plots_dir=plots_dir,
        )

import csv
import os
import matplotlib.pyplot as plt

def analyze_training(model_eval, res_func, params, train_data, test_data, constants, out_dir= "training_analysis"):
    import jax
    # evaluate residuals for all training data, and find out which points for which cases have the highest residuals
    topology = train_data[0]['topology']
    # residual_func = jax.jit(lambda data_config, graph_config: res_func(params, data_config, graph_config, topology, constants), device=constants['device'])
    # residual_func = lambda data_config, graph_config: res_func(params, data_config, graph_config, topology, constants)
    residual_func = res_func

    # Two things to do here:
    # Find top 1000 for each config and each residual component, and save their values, indices, and coordinates
    # coords = config['warped_geo']['foam_mesh']['cell']['centroids']
    # plot histogram of residuals for each component across the whole training set, to see if there are any weird outliers or if the distribution looks reasonable
    keys = ['res_vx', 'res_vy', 'res_vz', 'res_T', 'res_p', 'res_nuTilda']
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    top_k = 1000
    aggregate = {
        "train": {k: [] for k in keys},
        "test": {k: [] for k in keys},
    }
    top_rows = []
    summary_rows = []

    train_losses = []
    test_losses = []

    for split_name, configs in [("train", train_data), ("test", test_data)]:
        for config in configs:
            print(f"Analyzing config {config['case_id']} ({split_name})...")
            data = jax.device_put(config['data'], device=constants['device'])
            flow_condition = jax.device_put(config['flow_condition'], device=constants['device'])
            pts = jax.device_put(config['warped_geo'], device=constants['device'])
            se, res = residual_func(data, flow_condition, pts)
            loss = np.mean(jax.device_get(se))

            if split_name == "train":
                train_losses.append(loss)
            else:
                test_losses.append(loss)

            dual_cells = np.asarray(config['warped_geo']['dual_graph']['dual_cells'])
            dual_types = np.asarray(config['topology']['dual_graph']['dual_cell_types'])
            real_cells = dual_cells[dual_types == 0]

            if out_dir is not None:
                for key in keys:
                    vec = np.asarray(jax.device_get(res[key]), dtype=np.float32).reshape(-1)
                    n = vec.shape[0]
                    if n == 0:
                        continue

                    aggregate[split_name][key].append(vec)
                    coords = real_cells if real_cells.shape[0] == n else dual_cells[:n]
                    if coords.shape[0] != n:
                        raise ValueError(
                            f"Residual/coordinate length mismatch for case {config['case_id']}, key={key}: "
                            f"n_res={n}, n_coords={coords.shape[0]}"
                        )

                    k = min(top_k, n)
                    idx = np.argpartition(vec, -k)[-k:]
                    idx = idx[np.argsort(vec[idx])[::-1]]

                    summary_rows.append({
                        "split": split_name,
                        "case_id": config["case_id"],
                        "component": key,
                        "count": int(n),
                        "mean": float(np.mean(vec)),
                        "p95": float(np.percentile(vec, 95.0)),
                        "p99": float(np.percentile(vec, 99.0)),
                        "max": float(np.max(vec)),
                    })

                    for rank, i in enumerate(idx, start=1):
                        top_rows.append({
                            "split": split_name,
                            "case_id": config["case_id"],
                            "component": key,
                            "rank": rank,
                            "cell_index": int(i),
                            "residual": float(vec[i]),
                            "x": float(coords[i, 0]),
                            "y": float(coords[i, 1]),
                            "z": float(coords[i, 2]),
                    })

    # print average train and test losses
    avg_train_loss = np.mean(train_losses) if train_losses else float("nan")
    avg_test_loss = np.mean(test_losses) if test_losses else float("nan")
    print(f"Average train loss: {avg_train_loss:.6f}")
    print(f"Average test loss: {avg_test_loss:.6f}")

    if out_dir is not None:
        summary_path = os.path.join(out_dir, "residual_case_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "case_id", "component", "count", "mean", "p95", "p99", "max"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        topk_path = os.path.join(out_dir, "top_residual_points.csv")
        with open(topk_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "case_id", "component", "rank", "cell_index", "residual", "x", "y", "z"],
            )
            writer.writeheader()
            writer.writerows(top_rows)

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axs = axs.reshape(-1)
        for i, key in enumerate(keys):
            ax = axs[i]
            train_vals = np.concatenate(aggregate["train"][key]) if aggregate["train"][key] else np.array([])
            test_vals = np.concatenate(aggregate["test"][key]) if aggregate["test"][key] else np.array([])
            if train_vals.size:
                ax.hist(train_vals, bins=80, alpha=0.55, label="train", log=True)
            if test_vals.size:
                ax.hist(test_vals, bins=80, alpha=0.55, label="test", log=True)
            ax.set_title(key)
            ax.set_xlabel("|residual|")
            ax.set_ylabel("density")
            ax.legend()
            ax.grid(alpha=0.25)

        fig.suptitle("Residual distributions by component")
        hist_path = os.path.join(out_dir, "plots", "residual_histograms.png")
        fig.savefig(hist_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved summary to {summary_path}")
        print(f"Saved top-k points to {topk_path}")
        print(f"Saved histogram plot to {hist_path}")

    return

def analyze_pde_errors(model_eval, res_func, params, train_data, test_data, constants):
    import jax
    from train import compute_pinn_residuals

    all_configs = train_data + test_data
    topology = all_configs[0]['topology']
    residual_func = lambda flow, data_config, graph_config: res_func(params, data_config, flow, graph_config, topology, constants, compute_pinn_loss=True)
    residual_func_jit = jax.jit(residual_func, device=constants['device'])
    compute_pinn_residuals_jit = jax.jit(lambda u,t,p,nut,f,pts: compute_pinn_residuals(u,t,p,nut,f,pts,topology,constants), device=constants['device'])

    for config in all_configs:
        print(f"Analyzing PDE residuals for case {config['case_id']}...")

        # Configuration information:
        case_id = config['case_id']
        velocity = np.linalg.norm(config['flow_condition']['v_inf'])

        # Compute PDE residuals from model
        data = jax.device_put(config['data'], device=constants['device'])
        pts = jax.device_put(config['warped_geo'], device=constants['device'])
        freestream = jax.device_put(config['flow_condition'], device=constants['device'])
        freestream_eval = {**freestream}
        # import jax.numpy as jnp
        # freestream_eval['v_inf'] = jnp.array([197.04, 0.0, 0.0])
        _, aux = residual_func_jit(freestream_eval, data, pts)
        pde_residuals_predicted = {
            'res_continuity': jax.device_get(aux['res_continuity']),
            'res_mom_x': jax.device_get(aux['res_mom_x']),
            'res_mom_y': jax.device_get(aux['res_mom_y']),
            'res_mom_z': jax.device_get(aux['res_mom_z']),
            'res_energy': jax.device_get(aux['res_energy']),
        }
        pinn_loss_predicted = np.mean([res**2 for res in pde_residuals_predicted.values()])

        # Compute PDE residuals from data directly without using the model, to see what the "true" PDE residuals look like for the given data
        res_mom_x_data, res_mom_y_data, res_mom_z_data, res_energy_data, res_continuity_data, _ = compute_pinn_residuals_jit(
            data['U'], data['T'], data['p'], data['nuTilda'], freestream, pts, 
        )
        pde_residuals_data = {
            'res_continuity': jax.device_get(res_continuity_data),
            'res_mom_x': jax.device_get(res_mom_x_data),
            'res_mom_y': jax.device_get(res_mom_y_data),
            'res_mom_z': jax.device_get(res_mom_z_data),
            'res_energy': jax.device_get(res_energy_data),
        }
        pinn_loss_data = np.mean([res**2 for res in pde_residuals_data.values()])

        # ============================== Plot histograms ============================== 
        residual_names = pde_residuals_predicted.keys()
        # Fixed bounds for now
        x_min, x_max = 0.0, 0.1
        y_min, y_max = 10, 600000

        # Histogram bins over the visible x-range.
        # Values above x_max will be clipped into the last bin.
        n_bins = 100
        bins = np.linspace(x_min, x_max, n_bins + 1)

        n_res = len(residual_names)
        ncols = 3
        nrows = math.ceil(n_res / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4.5 * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, res_name in zip(axes, residual_names):
            # Pull residual vectors
            pred_vals = np.asarray(pde_residuals_predicted[res_name]).ravel()
            data_vals = np.asarray(pde_residuals_data[res_name]).ravel()
            pred_vals = np.abs(pred_vals)
            data_vals = np.abs(data_vals)

            # Anything above x_max gets squashed into the last bin.
            pred_plot = np.clip(np.abs(pred_vals), x_min, x_max)
            data_plot = np.clip(np.abs(data_vals), x_min, x_max)

            # Averages for titles (using abs residuals, matching what is plotted)
            pred_avg = np.mean(np.abs(pred_vals))
            data_avg = np.mean(np.abs(data_vals))

            # Overlay histograms
            ax.hist(
                data_plot,
                bins=bins,
                alpha=0.5,
                label='data',
                log=True,
            )
            ax.hist(
                pred_plot,
                bins=bins,
                alpha=0.5,
                label='pred',
                log=True,
            )

            ax.set_title(
                f"{res_name}\n"
                f"avg pred res = {pred_avg:.2e}, avg data res = {data_avg:.2e}"
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)

        # Hide any unused subplot axes
        for ax in axes[n_res:]:
            ax.axis('off')

        # Labels
        for ax in axes[-ncols:]:
            ax.set_xlabel("Residual magnitude")
        for row_start in range(0, len(axes), ncols):
            if row_start < len(axes):
                axes[row_start].set_ylabel("# Cells")

        # Put figure-level legend once
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        # Compact figure title with shortened scientific notation
        fig.suptitle(
            f"Case {case_id} | V_inf={velocity:.3f}\n"
            f"PINN loss: pred={pinn_loss_predicted:.2e} (data={pinn_loss_data:.2e})",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()


def analyze_panel_data(configs:list[dict]):
    
    return

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl


def get_state_label_and_units(state):
    labels = {
        "p": ("Pressure", "Pa"),
        "T": ("Temperature", "K"),
        "v": ("Velocity magnitude", "m/s"),
        "Ma": ("Mach number", None),
        "nuTilda": (r"$\tilde{\nu}$", r"m$^2$/s"),
        "error": ("Absolute error", None),
    }
    return labels.get(state, (state, None))


def make_colorbar_label(state, quantity_name=None):
    label, units = get_state_label_and_units(state)

    if quantity_name == "error":
        label = f"{label} absolute error"
    elif quantity_name == "signed_error":
        label = f"{label} error (NN - CFD)"

    if units:
        return f"{label} [{units}]"
    return label


def _standalone_field_colorbar_settings(state, model_type, clim, config, constants):
    """Return display limits and a math label for ``plot_field`` colorbars."""
    coefficient = r"C_p" if state == "p" else r"\mathrm{Ma}"
    label = rf"${coefficient}$"
    display_clim = np.asarray(clim, dtype=float)

    if state == "p":
        flow = config["flow_condition"]
        p_inf = float(np.asarray(flow["p_inf"]).item())
        T_inf = float(np.asarray(flow["T_inf"]).item())
        speed = float(np.linalg.norm(np.asarray(flow["v_inf"])))
        gas_constant = float(np.asarray(constants.get("R", 287.05)).item())
        dynamic_pressure = 0.5 * p_inf / (gas_constant * T_inf) * speed**2
        if not np.isfinite(dynamic_pressure) or dynamic_pressure <= 0.0:
            raise ValueError(
                "freestream dynamic pressure must be finite and positive"
            )
        pressure_offset = 0.0 if model_type == "error" else p_inf
        display_clim = (display_clim - pressure_offset) / dynamic_pressure

    if model_type == "error":
        label = (
            rf"${coefficient}^{{\mathrm{{pred}}}}"
            rf" - {coefficient}^{{\mathrm{{RANS}}}}$"
        )

    return tuple(display_clim), label


def save_standalone_colorbar(
    out_path,
    clim,
    cmap="viridis",
    label=None,
    orientation="vertical",
    dpi=300,
    ticks=None,
    tick_format=None,
    label_on_top=False,
    font_family=None,
    label_fontsize=None,
    tick_fontsize=None,
):
    font_settings = {}
    if font_family is not None:
        font_settings = {
            "font.family": font_family,
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        }

    with mpl.rc_context(font_settings):
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fixed_horizontal_canvas = orientation == "horizontal" and label_on_top
        if orientation == "vertical":
            fig, ax = plt.subplots(figsize=(1.2, 4.0), constrained_layout=True)
        elif fixed_horizontal_canvas:
            fig = plt.figure(figsize=(7.0, 1.0))
            ax = fig.add_axes((0.06, 0.34, 0.88, 0.22))
        else:
            fig, ax = plt.subplots(figsize=(7.0, 0.65), constrained_layout=True)

        cbar = fig.colorbar(sm, cax=ax, orientation=orientation)

        if ticks is not None:
            cbar.set_ticks(ticks)
        if tick_format is not None:
            tick_axis = (
                cbar.ax.xaxis if orientation == "horizontal" else cbar.ax.yaxis
            )
            tick_axis.set_major_formatter(
                mpl.ticker.FormatStrFormatter(tick_format)
            )
        if label is not None:
            label_kwargs = {"labelpad": 10} if label_on_top else {}
            if label_fontsize is not None:
                label_kwargs["fontsize"] = label_fontsize
            cbar.set_label(label, **label_kwargs)
        if tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=tick_fontsize)
        if orientation == "horizontal" and label_on_top:
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.tick_params(
                axis="x", top=False, labeltop=False, bottom=True, labelbottom=True
            )

        save_kwargs = {} if fixed_horizontal_canvas else {"bbox_inches": "tight"}
        fig.savefig(out_path, dpi=dpi, **save_kwargs)
        plt.close(fig)

from .visualize_mesh import add_sectional_planes

def pyvista_slice_to_mpl(
    slc,
    scalar_name,
    out_path,
    clim=None,
    xlim=None,
    ylim=None,
    cmap="viridis",
    shading="gouraud",
    dpi=300,
    show_axis_numbers=True,
):
    # For a y-normal slice, plot x-z coordinates
    x = slc.points[:, 0]
    z = slc.points[:, 2]

    triangles = slc.faces.reshape(-1, 4)[:, 1:]
    vals = slc.point_data[scalar_name]

    finite_pts = np.isfinite(vals)
    finite_tris = finite_pts[triangles].all(axis=1)

    triang = mtri.Triangulation(
        x,
        z,
        triangles=triangles[finite_tris],
    )

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    kwargs = dict(
        cmap=cmap,
        shading=shading,
    )

    if clim is not None:
        kwargs["vmin"], kwargs["vmax"] = clim

    ax.tripcolor(
        triang,
        vals,
        **kwargs,
    )

    ax.set_aspect("equal", adjustable="box")

    # No x/y labels, only numeric tick labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    if not show_axis_numbers:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # No colorbar here — colorbars are saved separately
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def visualize_paper(
        model_eval,
        params,
        config,
        constants,
        save_dir,
        state = 'p',
        camera = 'pos_2',
        window_size= None,
        enable_picking = True,
        pyvista_mesh_topology = None,
        return_image = False,
        print_l2 = False,
    ):
    title = "",
    freestream = config['flow_condition']
    pts = config['warped_geo']
    data = config['data']
    topology = config['topology']

    freestream_config = {**freestream}
    # import jax.numpy as jnp
    # freestream_config['v_inf'] = jnp.array([197.04, 0.0, 0.0])
    v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, freestream_config, pts)
    
    # unnormalize predictions
    # coords = data['centroid_coordinates']
    coords = pts['dual_graph']['dual_cells'][topology['dual_graph']['dual_cell_types']==0]
    v_pred = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])
    T_pred = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
    p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
    nuTilda_pred = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])

    # data = {
    #     'p': p_pred,
    #     'T': T_pred,
    # }

    cmap = "viridis"
    if state == 'p':
        pred_val, data_val, clim = p_pred, data['p']*1, (1e4, 4.e4)
        error_val = np.abs(p_pred-data['p'])
        clim_error = (0, 4e3)
        # pred_val, data_val, clim = np.abs(p_pred-data['p']), data['p'], (0, 1e4)
    elif state == 'T':
        pred_val, data_val, clim = T_pred, data['T'], (190, 270)
        error_val = np.abs(T_pred-data['T'])
        cmap = "coolwarm"
        clim_error = (0,20)

        # pred_val, data_val, clim = np.abs(T_pred-data['T']), data['T'], (0,30) # why this doesn work but above commented line works perfectly.....
        # cmap = 'viridis'
    elif state == 'nuTilda':
        pred_val, data_val, clim = nuTilda_pred, data['nuTilda'], (0, 0.1)
    elif state == 'v':
        pred_val, data_val, clim = np.linalg.norm(v_pred, axis=1), np.linalg.norm(data['U'], axis=1), (50, 340)
        error_val = np.abs(np.linalg.norm(v_pred, axis=1) - np.linalg.norm(data['U'], axis=1))
        clim_error = (0, 50)

    if print_l2:
        l2_error = np.linalg.norm(pred_val - data_val) / np.linalg.norm(data_val)
        print(f"Relative L2 error {state}: {l2_error:.4f}")

    import pyvista as pv

    # Build slice
    if pyvista_mesh_topology is None:
        cells_flat, celltypes = build_pyvista_mesh_topology(topology['foam_mesh'])
    else:
        cells_flat, celltypes = pyvista_mesh_topology
    grid = pv.UnstructuredGrid(cells_flat, celltypes, pts['foam_mesh']['points'] )
    grid.cell_data["data"] = data_val.astype(float)#-pred_val.astype(float)
    grid.cell_data["pred"] = pred_val.astype(float)
    grid.cell_data["error"] = error_val.astype(float)
    grid_smooth = grid.cell_data_to_point_data(pass_cell_data=False)

    def limits_from_center(center, height, aspect):
        cx, cy = center
        width = height * aspect
        return (cx - width / 2, cx + width / 2), (cy - height / 2, cy + height / 2)

    aspect = (33 - 15) / (7 - (-5))
    sectional_data = [
        (4,  *limits_from_center(center=(20, 1), height=20, aspect=aspect)),
        (12, *limits_from_center(center=(24, 1), height=12, aspect=aspect)),
    ]

    for y_plane, xlim, ylim in sectional_data:
        slc_data = grid_smooth.slice(
            normal="y",
            origin=(0, y_plane, 0),
            generate_triangles=True,
        )

        slc_pred = grid_smooth.slice(
            normal="y",
            origin=(0, y_plane, 0),
            generate_triangles=True,
        )

        slc_error = grid_smooth.slice(
            normal="y",
            origin=(0, y_plane, 0),
            generate_triangles=True,
        )
        # clim=None
        slice_cmap = cmap

        filetype = "pdf"
        filetype = "png"
        pyvista_slice_to_mpl(
            slc_data,
            "data",
            save_dir / f"{state}_{y_plane}_data{filetype}",
            clim=clim,
            xlim=xlim,
            ylim=ylim,
            cmap=slice_cmap,
        )

        pyvista_slice_to_mpl(
            slc_pred,
            "pred",
            save_dir / f"{state}_{y_plane}_pred.{filetype}",
            clim=clim,
            xlim=xlim,
            ylim=ylim,
            cmap=slice_cmap,
        )

        pyvista_slice_to_mpl(
            slc_error,
            "error",
            save_dir / f"{state}_{y_plane}_error.{filetype}",
            clim=clim_error,
            xlim=xlim,
            ylim=ylim,
            cmap="viridis",
        )

    # Separate colorbar for data/pred
    save_standalone_colorbar(
        save_dir / f"{state}_colorbar.{filetype}",
        clim=clim,
        cmap=slice_cmap,
        label=make_colorbar_label(state),
        orientation="horizontal",
    )

    # Separate colorbar for error
    save_standalone_colorbar(
        save_dir / f"{state}_error_colorbar.{filetype}",
        clim=clim_error,
        cmap="viridis",
        label=make_colorbar_label(state, quantity_name="error"),
        orientation="horizontal",
    )

    data_3d = [
        (pred_val, clim, "pred", cmap),
        (data_val, clim, "data", cmap),
        (error_val, clim_error, "error", "viridis"),
    ]

    for val, clim, name, cmap_type in data_3d:
        p = pv.Plotter(window_size=window_size, off_screen=True)
        # p = pv.Plotter(window_size=window_size, off_screen=False)
        # p = pv.Plotter(window_size=window_size, off_screen=False)
        p=visualize_surface(
            topology,
            pts,
            val,
            title=None,
            cmap=cmap_type,
            clim=clim,
            camera = camera,
            p=p,
            enable_picking=enable_picking,
            show_scalar_bar=False,
            smooth_scalars=True,
            add_axes=False,
        )
        y_inverted_pts = pts['foam_mesh']['points'].copy()
        y_inverted_pts[:, 1] *= -1
        pts2 = {'foam_mesh': {'points': y_inverted_pts}}
        p=visualize_surface(
            topology,
            pts2,
            val,
            title=None,
            cmap=cmap_type,
            clim=clim,
            camera = camera,
            p=p,
            enable_picking=enable_picking,
            show_scalar_bar=False,
            smooth_scalars=True,
            add_axes=False,
        )

        # if name == 'data':
        if 0:
            p = add_sectional_planes(
                p,
                topology,
                pts2,
                sectional_data,
            )

        # p.camera_position = [(-9.46278, 62.6017, 69.984), (17.8118, 9.04314, 7.81296), (0.475136, -0.552753, 0.684624)]
        # p.camera.view_angle = 30
        # p.camera.clipping_range = (51.9897, 132.623)
        # p.window_size = [1430, 896]
        p.camera_position = [(-9.9864, 53.1927, 76.8637), (17.9115, 6.19638, 9.85092), (0.292775, -0.721282, 0.627723)]
        p.camera.view_angle = 30
        p.camera.clipping_range = (46.2828, 160.848)
        p.window_size = [1430, 896]
        
        print('saved to', f'{save_dir}/3D{state}_{name}')
        # p.show()
        img = p.screenshot(f'{save_dir}/3D{state}_{name}', transparent_background=True)
        p.close()

    return

    off_screen = return_image
    if title is None:
        f_title = ""
    else:
        f_title = f"{title} (Predicted)"
    pl = pv.Plotter(shape=(2, 1), window_size=window_size, off_screen=off_screen)
    pl.subplot(0,0)
    p=visualize_surface(
        topology,
        pts,
        pred_val,
        title=f_title,
        cmap=cmap,
        clim=clim,
        camera = camera,
        p=pl,
        enable_picking=enable_picking,
        show_scalar_bar=False,
    )
    pl.add_mesh(slc_pred, scalars="pred", cmap=cmap, clim = clim, lighting = False, show_scalar_bar=False)
    
    if title is None:
        f_title = ""
    else:
        f_title = f"{title} (Data)"
    pl.subplot(1,0)
    p = visualize_surface(
        topology,
        pts,
        data_val,
        title=f_title,
        cmap=cmap,
        clim=clim,
        camera = camera,
        p=pl,
        enable_picking=False,
        lighting = None,
        show_scalar_bar=False,
    )
    pl.add_mesh(slc_data, scalars="data", cmap=cmap, clim = clim, lighting = False)
    # pl.subplot(0,2)
    # p = visualize_surface(
    #     topology,
    #     pts,
    #     np.abs(data['p'] - p_pred),
    #     title=f"{title} (Error P)",
    #     cmap="viridis",
    #     clim=(0, 1e4),
    #     camera = 'pos_2',
    #     p=pl,
    #     enable_picking=False
    # )
    # light = pv.Light(position=(0, 100, 500), focal_point=(0, 0, 0), light_type="scene light")
    # pl.add_light(light)

    pl.link_views()
    save_name = f'{save_dir}/case_{config["case_id"]}_{state}.png'
    if return_image:
        img = pl.screenshot(save_name, transparent_background=True)
        pl.close()
        return img
    else:
        pl.show()
