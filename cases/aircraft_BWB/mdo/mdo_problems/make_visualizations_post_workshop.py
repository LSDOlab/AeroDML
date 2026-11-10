"""Visualize BWB optimization CSDL histories, geometry, and GNN flow.

Use --histories-only for lightweight CPU plotting. Use --geometry-only for
geometry images/video without importing or evaluating ML or panel models. Full rendering reruns the
selected GNN and requires a GPU allocation and its model environment.
Use --model to match the optimization run being visualized.
Full flow rendering uses GNN predictions rather than saved panel Cp.
"""
import os
import re
import sys
import pickle
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))


def _evaluate_ml_frame(model, surf_x, pitch, panel_points=None):
    """Prepare full-field inference like the optimizer's ML forward wrapper."""
    import jax
    import jax.numpy as jnp

    previous_x64 = jax.config.read('jax_enable_x64')
    jax.config.update('jax_enable_x64', False)
    try:
        lowfi_data = None
        if panel_points is not None:
            panel_result = model.panel_functions.forward(panel_points, pitch)
            lowfi_data = {key: panel_result[key] for key in ('pressure', 'velocity')}
        new_surf_pts = surf_x[model.surf_25r_to_1r_map]
        new_vol_pts = model.compute_volume_pts(new_surf_pts, {'pitch': pitch})

        # ml_func_full is a raw JIT, unlike the optimizer's ml_func wrapper.
        # Panel/warper outputs can reside on CPU while the GNN runs on GPU.
        def to_gnn(value):
            return jax.device_put(jnp.asarray(value, dtype=jnp.float32),
                                  model.constants['gnn_device'])

        result = model.ml_func_full(
            to_gnn(new_vol_pts), to_gnn(pitch),
            lowfi_data=jax.tree.map(to_gnn, lowfi_data))
        # Finish inference and copy to NumPy for rendering before restoring x64.
        return jax.device_get(result)
    finally:
        jax.config.update('jax_enable_x64', previous_x64)


def _flatten_plot_items(plot_output):
    if isinstance(plot_output, list):
        items = []
        for item in plot_output:
            items.extend(_flatten_plot_items(item))
        return items
    return [plot_output]

def _to_pyvista_mesh_items(plot_output):
    """Convert geometry.plot() outputs to (mesh, kwargs) items for plotter.add_mesh()."""
    plot_items = _flatten_plot_items(plot_output)
    mesh_items = []

    def _with_normals(dataset):
        if hasattr(dataset, 'compute_normals'):
            try:
                return dataset.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    split_vertices=False,
                    inplace=False,
                )
            except Exception:
                return dataset
        return dataset

    for item in plot_items:
        mesh = None
        kwargs = {}

        if isinstance(item, dict) and 'mesh' in item:
            mesh = item['mesh']
            kwargs = dict(item.get('kwargs', {}))
        elif isinstance(item, tuple) and len(item) == 2:
            mesh, raw_kwargs = item
            kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        else:
            mesh = item.dataset if hasattr(item, 'dataset') else item

        if mesh is None:
            continue

        try:
            wrapped = pv.wrap(mesh)
        except Exception:
            continue

        if isinstance(wrapped, pv.MultiBlock):
            for block in wrapped:
                if block is not None and getattr(block, 'n_points', 0) > 0:
                    mesh_items.append((_with_normals(block), dict(kwargs)))
        elif getattr(wrapped, 'n_points', 0) > 0:
            mesh_items.append((_with_normals(wrapped), kwargs))

    return mesh_items


def _find_iteration_indices(hdf5_path):
    with h5py.File(hdf5_path, 'r') as handle:
        indices = []
        for key in handle.keys():
            match = re.fullmatch(r'iteration_(\d+)', key)
            if match:
                indices.append(int(match.group(1)))
    indices.sort()
    return indices


def _find_latest_summary_path(output_dir):
    summary_paths = list(Path(output_dir).rglob('modopt_summary.out'))
    if not summary_paths:
        return None
    return max(summary_paths, key=lambda path: path.stat().st_mtime)


def _load_modopt_summary(summary_path):
    rows = []
    with open(summary_path, 'r') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    'major': int(parts[1]),
                    'objective': float(parts[2]),
                    'optimality': float(parts[4]),
                    'feasibility': float(parts[5]),
                })
            except ValueError:
                continue
    return rows


def _plot_modopt_summary(summary_path, save_path):
    rows = _load_modopt_summary(summary_path)
    if not rows:
        print(f'No convergence rows found in {summary_path}')
        return

    iterations = np.array([row['major'] for row in rows], dtype=int)
    optimality = np.array([row['optimality'] for row in rows], dtype=float)
    feasibility = np.array([row['feasibility'] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.semilogy(iterations, optimality, marker='o', linewidth=2, label='optimality')
    ax.semilogy(iterations, feasibility, marker='s', linewidth=2, label='feasibility')
    ax.set_xlabel('major iteration')
    ax.set_ylabel('residual')
    ax.set_title('Optimization convergence history')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _extract_iteration_history(hdf5_path, iteration_indices, constraint_targets):
    history_iterations = []
    objective_history = []
    constraint_history = {key: [] for key in constraint_targets}

    with h5py.File(hdf5_path, 'r') as handle:
        for iteration_index in iteration_indices:
            group_name = f'iteration_{iteration_index}'
            if group_name not in handle:
                continue

            group = handle[group_name]
            history_iterations.append(iteration_index)
            if 'weighted_fuel_burn_objective' in group:
                objective_history.append(float(np.asarray(group['weighted_fuel_burn_objective'][()]).reshape(-1)[0]))
            else:
                objective_history.append(np.nan)

            for key in constraint_targets:
                if key in group:
                    constraint_history[key].append(np.asarray(group[key][()]).reshape(-1))
                else:
                    constraint_history[key].append(None)

    return np.asarray(history_iterations, dtype=int), np.asarray(objective_history, dtype=float), constraint_history


def _plot_objective_and_constraints(hdf5_path, iteration_indices, constraint_targets, save_path):
    history_iterations, objective_history, constraint_history = _extract_iteration_history(hdf5_path, iteration_indices, constraint_targets)
    if objective_history.size == 0:
        print(f'No iteration history found in {hdf5_path}')
        return

    nrows = (len(constraint_targets) + 5) // 5
    figure, axes = plt.subplots(nrows, 5, figsize=(30, 4*nrows), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(history_iterations, objective_history, marker='o', linewidth=2, color='black')
    axes[0].set_title('weighted_fuel_burn_objective')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('objective')
    axes[0].grid(True, alpha=0.25)

    axis_index = 1
    for key, target in constraint_targets.items():
        if axis_index >= len(axes):
            break

        ax = axes[axis_index]
        values = [entry for entry in constraint_history[key] if entry is not None]
        if not values:
            ax.set_visible(False)
            axis_index += 1
            continue

        stacked_values = np.vstack(values)
        local_iterations = np.asarray([history_iterations[i] for i, entry in enumerate(constraint_history[key]) if entry is not None], dtype=int)

        if stacked_values.ndim == 1 or stacked_values.shape[1] == 1:
            axis_values = stacked_values.reshape(-1)
            ax.plot(local_iterations, axis_values, marker='o', linewidth=2, label=key)
            target_value = np.asarray(target).reshape(-1)[0]
            ax.axhline(target_value, linestyle='--', color='tab:red', linewidth=1.5, label='target')
            ax.set_title(
                f'{key}\nvalue = {axis_values[-1]:.6g}, target = {target_value:.6g}',
                fontsize=10,
            )
        else:
            target_array = np.asarray(target).reshape(-1)
            title_lines = [key]
            for component_index in range(stacked_values.shape[1]):
                component_values = stacked_values[:, component_index]
                component_target = target_array[min(component_index, target_array.size - 1)]
                ax.plot(local_iterations, component_values, marker='o', linewidth=2, label=f'{key}[{component_index}]')
                ax.axhline(component_target, linestyle='--', linewidth=1.2, color=f'C{component_index}')
                title_lines.append(
                    f'[{component_index}] value = {component_values[-1]:.6g}, target = {component_target:.6g}'
                )
            ax.set_title('\n'.join(title_lines), fontsize=9)

        ax.set_xlabel('iteration')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        axis_index += 1

    for ax in axes[axis_index:]:
        ax.set_visible(False)

    figure.suptitle('Objective and constraint history', y=1.01)
    figure.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(figure)


def _plot_selected_variable_history(hdf5_path, iteration_indices, variable_specs, save_path):
    history = {}
    with h5py.File(hdf5_path, 'r') as handle:
        for spec in variable_specs:
            history[spec['key']] = []

        for iteration_index in iteration_indices:
            group_name = f'iteration_{iteration_index}'
            if group_name not in handle:
                continue

            group = handle[group_name]
            for spec in variable_specs:
                key = spec['key']
                if key not in group:
                    history[key].append(None)
                    continue
                value = np.asarray(group[key][()]).reshape(-1)
                reducer = spec.get('reducer')
                if reducer is None:
                    reduced_value = value
                else:
                    reduced_value = float(reducer(value))
                history[key].append(reduced_value)

    nrows = (len(variable_specs) + 4) // 5
    fig, axes = plt.subplots(nrows, 5, figsize=(30, 4*nrows), constrained_layout=True)
    axes = axes.ravel()

    axis_index = 0
    for spec in variable_specs:
        if axis_index >= len(axes):
            break

        values = history[spec['key']]
        valid_pairs = [(iteration, value) for iteration, value in zip(iteration_indices, values) if value is not None]
        if not valid_pairs:
            continue

        plot_iterations = np.asarray([pair[0] for pair in valid_pairs], dtype=int)
        plot_values = np.asarray([pair[1] for pair in valid_pairs], dtype=float)

        ax = axes[axis_index]
        lines = ax.plot(plot_iterations, plot_values, marker='o', linewidth=2)
        if spec['key'] in ('L', 'D_total'):
            # Optimizer order: cruise missions, one sizing case, stability case.
            count = len(lines)
            labels = ([f'Cruise {j}' for j in range(count - 2)] + ['Sizing', 'Stability']
                      if count >= 3 else [f'Condition {j}' for j in range(count)])
            ax.legend(lines, labels, frameon=False, fontsize=8)
        ax.set_title(
            f"{spec['label']}\nfinal value = {np.array2string(np.asarray(plot_values[-1]), precision=4)}",
            fontsize=10,
        )
        ax.set_xlabel('iteration')
        ax.grid(True, alpha=0.25)
        if 'ylabel' in spec:
            ax.set_ylabel(spec['ylabel'])
        axis_index += 1

    for ax in axes[axis_index:]:
        ax.set_visible(False)

    fig.suptitle('Selected variable history', y=1.01)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    final_ind = None

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='Optimization HDF5 history')
    parser.add_argument('--model', choices=('PG-RANS-GNN', 'Panel', 'RANS-GNN'),
                        help='Required for flow rendering; use the optimization model')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--frame-step', type=int, default=15)
    parser.add_argument('--mission-index', type=int, default=0, help='Cruise mission index (not sizing)')
    parser.add_argument('--state', choices=['p', 'U_mag'], default='p')
    parser.add_argument('--geometry-only', action='store_true', help='Render saved geometry without ML values, GNN imports, or panel solves')
    parser.add_argument('--histories-only', action='store_true', help='CPU plots only; skip geometry/GNN/video imports')
    parser.add_argument('--static-margin-bound', type=float, help='Override filename-derived stability target')
    args = parser.parse_args()
    if not args.histories_only and not args.geometry_only:
        if args.model is None:
            parser.error('--model is required for flow rendering')
        if args.model == 'Panel':
            parser.error('Panel histories support --histories-only or --geometry-only; GNN flow rendering requires a GNN run')
    USE_MULTIFIDELITY = args.model == 'PG-RANS-GNN'
    # Resolve user paths before switching to the local geometry/cache directory.
    hdf5_path = args.history.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    os.chdir(SCRIPT_DIR)
    if args.frame_step < 1 or args.mission_index < 0:
        parser.error('frame-step must be positive and mission-index nonnegative')
    fname = hdf5_path.stem
    frame_step = args.frame_step
    video_writer = None
    fps = 1

    plot_dir = args.output_dir or hdf5_path.parent / f'plots_{fname}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # sample every N frames while always keeping the first and last frame
    state_name = 'geometry' if args.geometry_only else args.state
    # state_name = 'U_mag'

    available_iterations = _find_iteration_indices(hdf5_path)
    if not available_iterations:
        raise RuntimeError(f'No iteration groups found in {hdf5_path}')
    if final_ind is None:
        final_ind = available_iterations[-1]
        print(f'auto-detected final iteration: {final_ind}')
    else:
        available_iterations = [iteration for iteration in available_iterations if iteration <= final_ind]
        if not available_iterations:
            raise RuntimeError(f'No iterations available up to final_ind={final_ind} in {hdf5_path}')
        final_ind = available_iterations[-1]

    summary_path = _find_latest_summary_path(hdf5_path.parent / f'{fname}_outputs')
    if summary_path is not None:
        _plot_modopt_summary(summary_path, plot_dir / f'{fname}_convergence_history.png')
    else:
        print(f'No modopt_summary.out found under {fname}_outputs')

    constraint_targets = {
        'constraint_cruise_trim': np.array([0.0]),
        'constraint_structural_sizing_trim': np.array([0.0]),
        'constraint_S1_max_stress': np.array([324e6]),
        'constraint_centerbody_volume': np.array([591.8 / 0.5]),
        'constraint_fuel_volume': np.array([0.0]),
        'constraint_takeoff_landing': np.array([10000.0, 10000.0]),
        'constraint_static_margin': np.array([args.static_margin_bound if args.static_margin_bound is not None else (0.03 if '_stab_relaxed_' in fname else 0.0 if '_stab_loose_' in fname else 0.05)]),
        'constraint_transition_le_monotonicity': np.array([0.0]),
        'constraint_cruise_cg_CM_nominal_trim': np.array([0.0]),
    }
    _plot_objective_and_constraints(
        hdf5_path,
        available_iterations,
        constraint_targets,
        plot_dir / f'{fname}_objective_constraint_history.png',
    )

    selected_variable_specs = [
        {'key': 'wing_position', 'label': 'Wing position', 'ylabel': 'm'},
        {'key': 'centerbody_chord_stretch_ratio', 'label': 'Centerbody chord ratios'},
        {'key': 'wing_chord_stretch', 'label': 'Wing chord stretch coefficients'},
        {'key': 'wing_twist', 'label': 'Wing twist coefficients'},
        {'key': 'weighted_fuel_burn_objective', 'label': 'Weighted fuel burn objective', 'ylabel': 'objective'},
        {'key': 'L', 'label': 'Lift (all saved conditions)', 'ylabel': 'N'},
        {'key': 'D_total', 'label': 'Total drag (all saved conditions)', 'ylabel': 'N'},
        {'key': 'L_D', 'label': 'Lift-to-drag ratio', 'ylabel': 'L/D'},
        {'key': 'structural_weight', 'label': 'Structural weight', 'ylabel': 'weight'},
        {'key': 'W_no_fuel', 'label': 'No-fuel weight', 'ylabel': 'weight'},
        {'key': 'fuel_volume', 'label': 'Fuel volume', 'ylabel': 'volume'},
        {'key': 'planform_area', 'label': 'Planform area', 'ylabel': 'area'},
        {'key': 'wing_half_span', 'label': 'Wing half span', 'ylabel': 'length'},
        {'key': 'center_half_span', 'label': 'Center half span', 'ylabel': 'length'},
        {'key': 'transition_half_span', 'label': 'Transition half span', 'ylabel': 'length'},
    ]
    _plot_selected_variable_history(
        hdf5_path,
        available_iterations,
        selected_variable_specs,
        plot_dir / f'{fname}_selected_variable_history.png',
    )

    if args.histories_only:
        print(f'History plots written to {plot_dir}')
        sys.exit(0)

    import jax
    jax.config.update('jax_enable_x64', True)
    import csdl_alpha as csdl
    import lsdo_function_spaces as lfs
    import lsdo_geo
    import pyvista as pv
    import imageio.v2 as imageio
    from geometry_functions_post_workshop import initial_refit

    lfs.num_workers = 1
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    geometry_dir = SCRIPT_DIR / 'geometry'
    imported = lfs.import_file_patched(
        file_name=str(geometry_dir / 'bwbv2_no_wingtip_coarse_refined_flat.stp'), parallelize=False)
    geometry = initial_refit(lsdo_geo.Geometry(
        functions=imported.functions, function_names=imported.function_names,
        name='imported_geometry', space=imported.space), plot=False)
    if not args.geometry_only and USE_MULTIFIDELITY:
        import meshio
        from VortexAD import find_cell_adjacency
        # Match the optimizer's node ordering and projection on the undeformed refit.
        mesh = meshio.read(str(geometry_dir / 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh'))
        points = find_cell_adjacency(points=mesh.points, cells=mesh.cells_dict)[0]
        projected_panel_mesh = geometry.project(
            points, grid_search_density_parameter=1, newton_tolerance=1.e-10,
            grid_search_density_cutoff=30, projection_tolerance=1.e-2,
            force_reprojection=False, plot=False)
    deformed_geometry = geometry.copy()
    expected_shape = deformed_geometry.stack_coefficients().shape
    # Saved coefficients already include every design-variable deformation.
    def load_frame(index):
        with h5py.File(hdf5_path, 'r') as handle:
            group = handle[f'iteration_{index}']
            coeff = group['geometry_coefficients'][()]
            if coeff.shape != expected_shape:
                raise ValueError(f'Geometry shape {coeff.shape} does not match post-workshop {expected_shape}')
            pitch = None
            if not args.geometry_only:
                pitch = np.asarray(group['pitch'][()]).reshape(-1)
                num_cruise = np.asarray(group['L_D'][()]).size
                if args.mission_index >= num_cruise:
                    raise ValueError(f'mission-index must be below {num_cruise}')
        # Preserve each tensor-product surface shape; the installed generic
        # unstack helper flattens it to (n, 3), which can break surface plotting.
        offset = 0
        for function in deformed_geometry.functions.values():
            shape = function.coefficients.shape
            count = int(np.prod(shape[:-1]))
            function.coefficients = csdl.Variable(
                value=coeff[offset:offset + count].reshape(shape))
            offset += count
        return None if pitch is None else pitch[args.mission_index:args.mission_index+1]

    if not args.geometry_only:
        if USE_MULTIFIDELITY:
            import pgnn_model_operation as ml_model
        else:
            import gnn_model_operation as ml_model
        topology = ml_model.topology
        print(f'Using {ml_model.__name__} (USE_MULTIFIDELITY={USE_MULTIFIDELITY})')
        with open(ml_model.surface_mesh_projection_file_path, 'rb') as handle:
            projected_surf_mesh_dafoam = pickle.load(handle)

    # initial geometry
    load_frame(available_iterations[0])

    plot_init = deformed_geometry.plot(opacity=0.25, color='yellow', show=False)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plot_init_geometry = _to_pyvista_mesh_items(plot_init)
    for mesh, mesh_kwargs in plot_init_geometry:
        render_kwargs = dict(mesh_kwargs)
        render_kwargs.setdefault('show_edges', False)
        render_kwargs.setdefault('smooth_shading', True)
        plotter.add_mesh(mesh, **render_kwargs)

    camera_obj = plotter.camera

    viewup = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get("z")

    camera_obj.SetViewUp(*viewup)

    camera_obj.roll = 0.
    plotter.camera = camera_obj

    camera = {
        'position': (-30, -25, 50),
        'focal_point': (15, -5, 5),
        'viewup': (0, 0, 1),
        'distance': 20,
    }
    plotter.camera_position = [camera['position'], camera['focal_point'], camera['viewup']]
    # plotter.camera = camera

    plotter.show(title='test', screenshot=str(plot_dir / 'bwb_initial_up.png'))
    # exit()

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background('white')

    camera = {
        'position': (-30, -25, 50),
        'focal_point': (15, -5, 5),
        'viewup': (0, 0, 1),
        'distance': 20,
    }

    sampled_frames = available_iterations[::frame_step]
    if sampled_frames[-1] != final_ind:
        sampled_frames.append(final_ind)
    sampled_frames = sorted(set(sampled_frames))

    video_path = plot_dir / f'bwb_video_{fname}_{state_name}.mp4'

    # Same bundled ImageIO/FFmpeg backend as sweep_geometry_jax.py.
    # 4:2:0 H.264 supports native players; faststart enables progressive playback.
    try:
        with imageio.get_writer(
            str(video_path),
            format='FFMPEG',
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',
            macro_block_size=1,
            ffmpeg_params=['-movflags', '+faststart'],
        ) as video_writer:
            for i in sampled_frames:
                print(f'working on frame {i} of {final_ind}')
                pitch = load_frame(i)
                plotter.clear()
                if args.geometry_only:
                    plot_items = deformed_geometry.plot(opacity=1.0, color='steelblue', show=False)
                    for mesh, mesh_kwargs in _to_pyvista_mesh_items(plot_items):
                        render_kwargs = dict(mesh_kwargs)
                        render_kwargs.setdefault('show_edges', False)
                        render_kwargs.setdefault('smooth_shading', True)
                        plotter.add_mesh(mesh, **render_kwargs)
                else:
                    surf_x = deformed_geometry.evaluate(projected_surf_mesh_dafoam, plot=False).value
                    panel_points = None
                    if USE_MULTIFIDELITY:
                        panel_points = deformed_geometry.evaluate(projected_panel_mesh, plot=False).value
                    L, D, M, surf_aux, aux = _evaluate_ml_frame(
                        ml_model, surf_x, pitch, panel_points)
                    warped_geo = aux['warped_geo']

                    from aerodml.visualize_mesh import visualize_surface

                    if state_name == 'p':
                        state = aux['p']
                        clim = (1e4, 4.e4)
                    elif state_name == 'U_mag':
                        state = np.linalg.norm(np.asarray(aux["v"]), axis=1)
                        clim = (40., 340.)
                    else:
                        raise ValueError(f"Invalid state: {state_name}")

                    visualize_surface(
                        topology,
                        {"foam_mesh": {"points": warped_geo['foam_mesh']['points']}},
                        state,
                        title=None,
                        # cmap=CMAP,
                        clim=clim,
                        camera=None,
                        p=plotter,
                        enable_picking=False,
                        show_scalar_bar=True,
                        smooth_scalars=False,
                    )
                    mirrored = np.asarray(warped_geo['foam_mesh']['points']).copy()
                    mirrored[:, 1] *= -1.0
                    visualize_surface(
                        topology,
                        {"foam_mesh": {"points": mirrored}},
                        state,
                        title=None,
                        # cmap=CMAP,
                        clim=clim,
                        camera=None,
                        p=plotter,
                        enable_picking=False,
                        show_scalar_bar=False,
                        smooth_scalars=False,
                        # add_axes=False,
                    )




                plotter.add_text(
                    f"iteration {i} ({state_name}) ",
                    position='lower_left',
                    font_size=16,
                    color='black',
                    shadow=False,
                )
                if not args.geometry_only:
                    L, D, M = 2*np.asarray(L).squeeze()/1000,2*np.asarray(D).squeeze()/1000, 2*np.asarray(M).squeeze()/1000
                    plotter.add_text(
                        f"L: {L:.2f} kN, D: {D:.2f} kN, M: {M:.2f} kNm",
                        position='upper_left',
                        font_size=16,
                        color='black',
                        shadow=False,
                    )

                plotter.camera_position = [camera['position'], camera['focal_point'], camera['viewup']]
                plotter.render()

                frame_rgb = plotter.screenshot(return_img=True)

                video_writer.append_data(np.asarray(frame_rgb[..., :3], dtype=np.uint8))
                if i in (available_iterations[0], final_ind):
                    imageio.imwrite(
                        str(plot_dir / f'iteration_{i}_{state_name}.png'),
                        np.asarray(frame_rgb[..., :3], dtype=np.uint8),
                    )

    finally:
        plotter.close()
        recorder.stop()
    print(f'Video written to {video_path}')