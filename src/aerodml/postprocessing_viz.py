from .utils import plot_states, normalize, unnormalize, stack_func, mask_set
stacked_normalize = stack_func(normalize, in_axes=(0, None, None))
stacked_unnormalize = stack_func(unnormalize, in_axes=(0, None, None))
import numpy as np
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
        error_val, error_clim = np.abs(data['p'] - p_pred), (0,5e3)
        # pred_val, data_val, clim = np.abs(p_pred-data['p']), data['p'], (0, 1e4)
        # clim = (-1e4,1e4)
    elif state == 'T':
        pred_val, data_val, clim = T_pred, data['T'], (180, 290)
        cmap = "coolwarm"

        # pred_val, data_val, clim = np.abs(T_pred-data['T']), data['T'], (0,30) # why this doesn work but above commented line works perfectly.....
        # cmap = 'viridis'
    elif state == 'nuTilda':
        pred_val, data_val, clim = nuTilda_pred, data['nuTilda'], (0, 0.1)
    elif state == 'v':
        pred_val, data_val, clim = np.linalg.norm(v_pred, axis=1), np.linalg.norm(data['U'], axis=1), (50, 340)

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
            np.abs(data['p'] - p_pred),
            title=f"{title} (Error)",
            cmap="viridis",
            clim=error_clim,
            camera = camera,
            p=pl,
            enable_picking=False,
            lighting = None,
            show_scalar_bar=False,
        )
        p.add_mesh(slc_pred, scalars="error", cmap="viridis", clim = error_clim, lighting = False)

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

def analyze_aero_forces(model_eval, params, configs, constants, screenshot:bool=False, model_name:str=""):
    import jax
    # plot histogram of errors across cases
    case_ids = [config['case_id'] for config in configs]
    from .postprocessing import compute_forces, integrate_forces, compute_surface_areas_and_normals
    compute_forces_jit = jax.jit(lambda p, U, nuTilda, T, warped_geo: compute_forces(p, U, nuTilda, T, warped_geo,  case['topology']), device=constants['device'])

    lifts_compute = []
    drags_compute = []
    lifts_openfoam = []
    drags_openfoam = []
    lifts_panel = []
    drags_panel = []
    lift_reconstructions = []
    drag_reconstructions = []

    # per-case relative L2 errors
    p_l2_cases = []
    U_l2_cases = []
    nuTilda_l2_cases = []
    T_l2_cases = []

    # constants
    minimum_lift_ref = 2.5e5
    num = len(configs)
    for i, case in enumerate(configs):
        # reconstruct forces from data
        p_true, U_true, nuTilda_true, T_true = case['data']['p'], case['data']['U'], case['data']['nuTilda'], case['data']['T']
        p_true = jax.device_put(p_true, device=constants['device'])
        U_true = jax.device_put(U_true, device=constants['device'])
        nuTilda_true = jax.device_put(nuTilda_true, device=constants['device'])
        T_true = jax.device_put(T_true, device=constants['device'])
        forces_reconstructed = compute_forces_jit(p_true, U_true, nuTilda_true, T_true, case['warped_geo'])
        L_reconstructed, D_d = integrate_forces(forces_reconstructed)
        lift_reconstructions.append(L_reconstructed)
        drag_reconstructions.append(D_d)

        # compute forces from model
        case_flow_condition = jax.device_put(case['flow_condition'], device=constants['device'])
        v_pred, T_pred, p_pred, nuTilda_pred = model_eval(params, case_flow_condition, case['warped_geo'])
        p_pred = stacked_unnormalize(p_pred, constants['p']['mean'], constants['p']['std'])
        T_pred = stacked_unnormalize(T_pred, constants['T']['mean'], constants['T']['std'])
        nuTilda_pred = stacked_unnormalize(nuTilda_pred, constants['nuTilda']['mean'], constants['nuTilda']['std'])
        p_pred = np.clip(p_pred, a_min=1000, a_max=np.inf)
        v_pred = stacked_unnormalize(v_pred, constants['v']['mean'], constants['v']['std'])
        forces_pred = compute_forces_jit(p_pred, v_pred, nuTilda_pred, T_pred, case['warped_geo'])
        L_pred, D_pred = integrate_forces(forces_pred)
        # L_pred = L_pred + 14000
        lifts_compute.append(L_pred)
        drags_compute.append(D_pred)
        
        # get OpenFOAM forces
        L_of, D_of = case['data']['lift'][0], case['data']['drag'][0]
        lifts_openfoam.append(L_of)
        drags_openfoam.append(D_of)

        # get panel method forces
        if case['low_fidelity_info'] is not None:
            L_panel, D_panel = case['low_fidelity_info']['lift'].item(), case['low_fidelity_info']['drag'].item()
        else:
            L_panel, D_panel = 0.0, 0.0
        
        # L_panel, D_panel = 0.0, 0.0 # divide by 2 because symmetry
        # p_panel = jax.device_put(case['warped_geo']['panel']['p'], device=constants['device'])
        # v_panel = jax.device_put(case['warped_geo']['panel']['v'], device=constants['device'])
        # T_panel = jax.device_put(case['warped_geo']['panel']['T'], device=constants['device'])
        # nuTilda_panel = jax.device_put(case['warped_geo']['panel']['nuTilda'], device=constants['device'])
        # forces_pred = compute_forces_jit(p_panel, v_panel, nuTilda_panel, T_panel, case['warped_geo'])
        # L_panel, D_panel = integrate_forces(forces_pred)
        
        lifts_panel.append(L_panel)
        drags_panel.append(D_panel)

        # Compute field error
        is_train = case['train']
        if not is_train:
            # per-case relative L2 errors
            p_l2 = relative_l2(p_pred, p_true)
            U_l2 = relative_l2(v_pred, U_true)
            nuTilda_l2 = relative_l2(nuTilda_pred, nuTilda_true)
            T_l2 = relative_l2(T_pred, T_true)

            p_l2_cases.append(float(p_l2))
            U_l2_cases.append(float(U_l2))
            nuTilda_l2_cases.append(float(nuTilda_l2))
            T_l2_cases.append(float(T_l2))

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

        if L_reconstructed > minimum_lift_ref:
            L_ref = L_reconstructed*1.0
        else:
            L_ref = minimum_lift_ref
        L_err_pct = 100 * (L_pred - L_reconstructed) / L_ref
        D_err_pct = 100 * (D_pred - D_d) / D_d if D_d != 0 else float("nan")

        # print(
        #     f"Case {case['case_id']:>4} ({i:>3}/{num}) | "
        #     f"L [kN] NN:{L_nn_kN:>8.3f}  Recon:{L_rec_kN:>8.3f}  OF:{L_of_kN:>8.3f}  Err:{L_err_pct:>7.2f}% | "
        #     f"D [kN] NN:{D_nn_kN:>8.3f}  Recon:{D_rec_kN:>8.3f}  OF:{D_of_kN:>8.3f}  Err:{D_err_pct:>7.2f}%"
        # )
        # print(f"L[kN]: {L_nn_kN} {L_rec_kN} {L_of_kN} {L_panel_kN} {L_err_pct}% |   ")
        # exit()

        # print(L_nn_kN, L_rec_kN, L_of_kN, L_panel_kN, L_err_pct)

        case_id = case['case_id']
        print(
            f"{str(case_id)} | "
            f"L[kN]: {L_nn_kN:>8.3f} {L_rec_kN:>8.3f} {L_of_kN:>8.3f} {L_panel_kN:>8.3f} {L_err_pct:>8.2f}% |   "
            f"D[kN]: {D_nn_kN:>8.3f} {D_rec_kN:>8.3f} {D_of_kN:>8.3f} {D_panel_kN:>8.3f} {D_err_pct:>8.2f}%     (NN  Data  OpenFOAM  Panel  Error(Data-Recon))"
        )

    p_l2_mean = np.mean(p_l2_cases)
    p_l2_std  = np.std(p_l2_cases)

    U_l2_mean = np.mean(U_l2_cases)
    U_l2_std  = np.std(U_l2_cases)

    nuTilda_l2_mean = np.mean(nuTilda_l2_cases)
    nuTilda_l2_std  = np.std(nuTilda_l2_cases)

    T_l2_mean = np.mean(T_l2_cases)
    T_l2_std  = np.std(T_l2_cases)

    print("\nMean relative L2 over test cases:")
    print(f"p        : {100*p_l2_mean:.3f}% ± {100*p_l2_std:.3f}%")
    print(f"U        : {100*U_l2_mean:.3f}% ± {100*U_l2_std:.3f}%")
    print(f"nuTilda  : {100*nuTilda_l2_mean:.3f}% ± {100*nuTilda_l2_std:.3f}%")
    print(f"T        : {100*T_l2_mean:.3f}% ± {100*T_l2_std:.3f}%")

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

    print(f'Lift train (test) error: {train_lift_error:.4f} ({test_lift_error:.4f})')
    print(f'Drag train (test) error: {train_drag_error:.4f} ({test_drag_error:.4f})')
    print(f'Panel Lift train (test) error: {train_panel_lift_error:.4f} ({test_panel_lift_error:.4f})')
    print(f'Panel Drag train (test) error: {train_panel_drag_error:.4f} ({test_panel_drag_error:.4f})')
    x = np.arange(len(case_ids))
    # width = 0.25
    width = 0.33

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # ======================================= PLOT 1A: Lift and Drag by Case =======================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True
    )

    # --- Lift subplot ---
    ax1.bar(x,         lifts_openfoam_np,       width, label="OpenFOAM Force",      color=colors[0])
    ax1.bar(x - width, lifts_panel_np, width, label="Panel Force", color=colors[2])
    ax1.bar(x + width, lifts_compute_np,        width, label="Model Reconstructed Force",         color=colors[1])

    ax1.set_ylabel("Lift")
    ax1.set_title("Lift and Drag by Case")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(ncol=3)

    # --- Drag subplot ---
    ax2.bar(x,         drags_openfoam_np,       width, label="OpenFOAM Force",      color=colors[0])
    ax2.bar(x - width, drags_panel_np, width, label="Panel Force", color=colors[2])
    ax2.bar(x + width, drags_compute_np,        width, label="Model Reconstructed Force",         color=colors[1])

    ax2.set_ylabel("Drag")
    ax2.set_xlabel("Case ID")
    ax2.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(case_ids, rotation=45, ha="right")

    if screenshot:
        os.makedirs('post_processing_plots', exist_ok=True)
        plt.savefig('post_processing_plots/lift_drag_histogram.png', dpi=300, bbox_inches='tight')

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

    # Print summary
    print(f"Lift Error:        {lift_avg_all:.2f}% (Train: {lift_avg_train:.2f}%, Test: {lift_avg_test:.2f}%)")
    print(f"Panel Lift Error:  {lift_panel_avg_all:.2f}% (Train: {lift_panel_avg_train:.2f}%, Test: {lift_panel_avg_test:.2f}%)")
    print(f"Drag Error:        {drag_avg_all:.2f}% (Train: {drag_avg_train:.2f}%, Test: {drag_avg_test:.2f}%)")
    print(f"Panel Drag Error:  {drag_panel_avg_all:.2f}% (Train: {drag_panel_avg_train:.2f}%, Test: {drag_panel_avg_test:.2f}%)")

    # colors by split
    train_color = colors[0]
    test_color  = colors[1]

    lift_bar_colors = [train_color if t else test_color for t in is_train]
    drag_bar_colors = [train_color if t else test_color for t in is_train]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
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
    ax2.set_xlabel("Case ID")
    ax2.set_title(
        f"Drag Relative Error by Case "
        f"(Train: {drag_avg_train:.2f}%, Test: {drag_avg_test:.2f}%)"
    )
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=legend_handles, loc="upper right")

    ax2.set_xticks(x)
    ax2.set_xticklabels(case_ids, rotation=45, ha="right")

    # ylim between 0 and 50 percent
    ax1.set_ylim(0, 50)
    ax2.set_ylim(0, 50)

    if screenshot:
        plt.savefig('post_processing_plots/lift_drag_error_histogram.png', dpi=300, bbox_inches='tight')
        
    # ======================================= PLOT 3: PARITY PLOTS =======================================
    is_train = np.array([config['train'] for config in configs], dtype=bool)

    def parity_plot(ax, y_true, y_pred, title, xlabel, ylabel):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        ax.scatter(
            y_true[is_train], y_pred[is_train],
            s=55, alpha=0.85, marker="o",
            edgecolor="black", linewidth=0.6,
            label="Train"
        )
        ax.scatter(
            y_true[~is_train], y_pred[~is_train],
            s=70, alpha=0.9, marker="s",
            edgecolor="black", linewidth=0.6,
            label="Test"
        )

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = 0.05 * (hi - lo + 1e-12)
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)

        # ax.set_xlim(lo, hi)
        # ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7, 3),
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

    parity_plot(
        ax2,
        drags_openfoam_np / 1e3,
        drags_compute_np / 1e3,
        f"{model_name} Drag Parity",
        "CFD Drag [kN]",
        "Predicted Drag [kN]"
    )
    ax2.set_xlim(19, 100)
    ax2.set_ylim(19, 100)

    # fig.suptitle("Force Prediction Parity Plot", fontsize=14)

    if screenshot:
        plt.savefig(f"post_processing_plots/lift_drag_parity_{model_name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

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
        "nuTilda": (r"$\tilde{\nu}$", r"m$^2$/s"),
        "error": ("Absolute error", None),
    }
    return labels.get(state, (state, None))


def make_colorbar_label(state, quantity_name=None):
    label, units = get_state_label_and_units(state)

    if quantity_name == "error":
        label = f"{label} absolute error"

    if units:
        return f"{label} [{units}]"
    return label


def save_standalone_colorbar(
    out_path,
    clim,
    cmap="viridis",
    label=None,
    orientation="vertical",
    dpi=300,
):
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(1.2, 4.0), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(4.0, 0.8), constrained_layout=True)

    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)

    if label is not None:
        cbar.set_label(label)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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