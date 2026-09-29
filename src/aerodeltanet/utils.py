import numpy as np
from matplotlib.tri import Triangulation
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable
import jax

def print_content(print_dict:dict, tabs:int=0)->str:
    tabs_tr = '  ' * tabs
    for k, v in print_dict.items():
        print(tabs_tr, f"Key: '{k}'")
        print(tabs_tr, f"  Type: {type(v)}")
        if isinstance(v, np.ndarray):
            print(tabs_tr, f"  Shape: {v.shape}, Dtype: {v.dtype}")
            if v.size < 10:
                print(tabs_tr, f"  Values: {v}")
        elif isinstance(v, list):
            print(tabs_tr, f"  Length: {len(v)}")
            if len(v) > 0:
                print(tabs_tr, f"  First element type: {type(v[0])}")
                print(tabs_tr, f"  First element: {v[0]}")
        elif isinstance(v, dict):
            print(tabs_tr, f"  Number of keys: {len(v)}")
            print_content(v, tabs + 1)
        # else:
            # print(tabs_tr, )

def stack_func(func:Callable, in_axes:tuple[int] = 0, loopify:bool = False)->Callable:
    if not loopify:
        return jax.vmap(func, in_axes=in_axes)
    else:
        raise NotImplementedError("Loopify option not implemented yet. This would involve manually batching inputs and iterating over them, which is more complex but can be useful for very large inputs that don't fit in memory when fully vectorized.")

def normalize(x, mean, std):
    assert x.shape == mean.shape == std.shape, f"Shapes of x, mean, and std must match, but got {x.shape}, {mean.shape}, {std.shape}"
    return (x - mean) / std

def unnormalize(x:jax.Array, mean:jax.Array, std:jax.Array) -> jax.Array:
    assert x.shape == mean.shape == std.shape, f"Shapes of x, mean, and std must match, but got {x.shape}, {mean.shape}, {std.shape}"
    x = x * std + mean
    return x

def rho_physical(p, T, R):
    return p / (R * T)

stacked_normalize = stack_func(normalize, in_axes=(0, None, None))
stacked_unnormalize = stack_func(unnormalize, in_axes=(0, None, None))

def mask_set(mask:np.ndarray , arrays:tuple[jax.Array])->tuple[jax.Array]:
    return_tuple = True
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
        return_tuple = False
    # check to make sure all arrays have the same first dimension
    len = arrays[0].shape[0]
    assert all(arr.shape[0] == len for arr in arrays), f"All input arrays must have the same first dimension. Got {[arr.shape[0] for arr in arrays]}"
    assert mask.shape == (len,), f"Mask shape {mask.shape} must match the first dimension of input arrays ({len},)"

    # apply mask to each array
    return tuple(arr[mask] for arr in arrays) if return_tuple else arrays[0][mask]

# ========================== Data saving ==========================
def convert(x):
    if hasattr(x, 'item'):  # np.array(0.5) or jax.Array
        return float(x)
    elif isinstance(x, dict):
        return {k: convert(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [convert(v) for v in x]
    return x

def serialize_for_json(data):
    return convert(data)

def reset_aux_data(suffix:str = None):
    import os
    if suffix is None:
        filename = 'training_aux_info.jsonl'
    else:
        filename = f'training_aux_info_{suffix}.jsonl'
    if os.path.exists(f'{filename}'):
        open(f'{filename}', 'w').close()

def save_aux_data(aux_info:dict, dir:str=None, suffix:str = ""):
    aux_info_last = {}
    for key, value in aux_info.items():
        aux_info_last[key] = value[-1]

    import json
    if dir is None:
        dir = "."
    with open(f"{dir}/training_aux_info_{suffix}.jsonl", "a") as f:
        f.write(json.dumps(serialize_for_json(aux_info_last)) + "\n")

def save_all_aux_data(aux_info:dict, name:str):
    import json
    reset_aux_data(name)
    aux_info_last = {}

    num_saves = max(len(value) for value in aux_info.values())
    for i in range(num_saves):
        for key, value in aux_info.items():
            aux_info_last[key] = value[i]

        with open(f"{name}.jsonl", "a") as f:
            f.write(json.dumps(serialize_for_json(aux_info_last)) + "\n")


def debug_data(array:np.ndarray, name:str = "Array", ordering ='largest', num_values:int = 5):
    # histogram of values
    print(f"Debugging array ==> shape: {array.shape}, dtype: {array.dtype}, min: {array.min()}, max: {array.max()}, mean: {array.mean()}, std: {array.std()}")
    if ordering == 'largest':
        flat_indices = np.argsort(array.flatten())[::-1][:num_values]
    elif ordering == 'smallest':
        flat_indices = np.argsort(array.flatten())[:num_values]
    else:
        raise ValueError(f"Invalid ordering: {ordering}. Must be 'largest' or 'smallest'.")
    unraveled_indices = np.unravel_index(flat_indices, array.shape)
    
    print(f"Top {num_values} values in {name} sorted by {ordering}:")
    for idx in zip(*unraveled_indices):
        print(f"  Index: {idx}, Value: {array[idx]}")

    plt.hist(array.flatten(), bins=100)
    plt.title(f"Histogram of {name}")

    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    plt.show()
# ========================================= OLD =========================================

def freestream_ic_from_physical(v_inf_star: float,
                                c_ref_star: float,
                                alpha_deg: float,
                                rho_inf_star: float,
                                scales: dict,
                                gamma: float = 5.0/3.0,
):
    """
    Uniform freestream ICs from physical inputs and returns nondimensional scalars:
      (rho, vx, vy, P), same as your original freestream_ic.

    Inputs (SI):
      v_inf_star   : freestream speed [m/s]
      c_ref_star   : freestream speed of sound [m/s]
      alpha_deg    : angle of attack [deg]
      rho_inf_star : freestream density [kg/m^3]
      gamma        : ratio of specific heats

    Returns (nondimensional):
      rho, vx, vy, P
    """
    import numpy as np


    # sanity: the caller's c_ref should match the provided c_ref_star for consistency
    assert np.isclose(scales['c_ref'], c_ref_star), "scales.c_ref must equal c_ref_star"

    # Physical freestream primitives
    alpha = np.deg2rad(alpha_deg)
    vx_star =  v_inf_star * np.cos(alpha)
    vy_star =  v_inf_star * np.sin(alpha)
    # From c^2 = gamma P / rho  ->  P = rho c^2 / gamma
    P_inf_star = rho_inf_star * c_ref_star**2 / gamma

    # Convert to nondimensional for the solver
    rho, vx, vy, P = phys_to_nondim_state(rho_inf_star, vx_star, vy_star, P_inf_star, scales)
    return rho, vx, vy, P

def phys_to_nondim_state(rho_star, vx_star, vy_star, P_star, scales: dict):
    """
    Convert SI → nondimensional.
    rho* [kg/m^3], v* [m/s], P* [Pa]
    """
    rho = rho_star / scales['rho_ref']
    vx  = vx_star  / scales['c_ref']
    vy  = vy_star  / scales['c_ref']
    P   = P_star   / (scales['rho_ref'] * scales['c_ref']**2)
    return rho, vx, vy, P

def nondim_to_phys_state(rho, vx, vy, P, scales: dict):
    """
    Convert nondimensional → SI.
    """
    rho_star = rho * scales['rho_ref']
    vx_star  = vx  * scales['c_ref']
    vy_star  = vy  * scales['c_ref']
    P_star   = P   * (scales['rho_ref'] * scales['c_ref']**2)
    return rho_star, vx_star, vy_star, P_star

def compute_forces(mesh, P_physical):
    """Compute lift and drag forces on the body from pressure on edges"""
    boundary_edges = np.where(mesh['edge_tag'] == 1)[0]
    fluid_cell_boundary = mesh['edge_cells'][boundary_edges, 0]  # fluid cell on left side
    surface_pressures = P_physical[fluid_cell_boundary]
    force_towards_surface = surface_pressures*mesh['edge_length'][boundary_edges]

    # get all angles on the boundary centroids. We know that center of cylinder is at (0,0).
    edge_pts_boundary = mesh['points'][mesh['edges'][boundary_edges]]
    mid_xy = edge_pts_boundary.mean(axis=1)                       # (251,2) midpoints
    x, y = mid_xy[:,0], mid_xy[:,1]
    # theta = np.arctan2(y, x)                           # θ: -π..π, θ=0 at +x stagnation
    # -x is stagnation, so shift by π
    theta = np.arctan2(y, x) + np.pi               # θ: 0..2π, θ=0 at -x stagnation
    # theta = (theta + 2*np.pi) % (2*np.pi)              # optional: 0..2π

    lift = np.sum(-force_towards_surface * mesh['edge_normal'][boundary_edges, 1])
    drag = np.sum(force_towards_surface * mesh['edge_normal'][boundary_edges, 0])
    return lift, drag, theta, surface_pressures

def plot_states(mesh, rho_phys, vx_phys, vy_phys, P_phys):
    triobj = Triangulation(mesh['points'][:, 0], mesh['points'][:, 1], triangles=mesh['triangles'])

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()

    # Data to plot: (array, title)
    fields = [
        ("rho", rho_phys, r"Density $\rho$ [kg/m$^3$]"),
        ("vx",  vx_phys,  r"Velocity $v_x$ [m/s]"),
        ("vy",  vy_phys,  r"Velocity $v_y$ [m/s]"),
        ("P",   P_phys,   r"Pressure $P$ [Pa]"),
    ]

    clims = {
        "rho": (1.1, 1.3),     # Density min,max
        "vx":  (0.0, 130.0),  # Velocity x min,max
        "vy":  (-60.0, 70.0),  # Velocity y min,max
        "P":   (75000, 90000) # Pressure min,max
    }

    # Draw each subplot
    for ax, (key, values, title) in zip(axes, fields):
        vmin, vmax = clims[key]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        norm = None
        tpc = ax.tripcolor(triobj, facecolors=values, shading='flat', norm=norm)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        fig.colorbar(tpc, ax=ax)
    return fig

def plot_progress(
        cons:tuple,
        flux:tuple,
        prim:tuple,
        prim_phys:tuple,
        freestream_phys:dict,
        mesh:dict,
        save_animation_path:str,
        dir:str,
        output_counter:int,
        n_iter:int,
        hist_plot:list,
    ):
    Mass, Momx, Momy, Energy = cons
    Mass_flux, Momx_flux, Momy_flux, Energy_flux = flux
    rho, vx, vy, P = prim
    rho_phys, vx_phys, vy_phys, P_phys = prim_phys
    q_inf, p_inf = freestream_phys['q_inf'], freestream_phys['p_inf']

    mass_mse, momx_mse, momy_mse, energy_mse = np.mean(np.abs(Mass_flux)), np.mean(np.abs(Momx_flux)), np.mean(np.abs(Momy_flux)), np.mean(np.abs(Energy_flux))
    flux_res = mass_mse + momx_mse + momy_mse + energy_mse

    lift, drag, surf_theta, surf_pressures = compute_forces(mesh, P_phys)
    Cl = lift / (q_inf*2.0)
    Cd = drag / (q_inf*2.0)
    # print("[it=" + str(n_iter) + " iter=" + f"{output_counter}/{n_iter}" + f"] Mean flux residual: {flux_res:.3} ({mass_mse:.3}, {momx_mse:.3}, {momy_mse:.3}, {energy_mse:.3})", end='')
    # print(f", \tLift: {lift:.3}N (Cl = {Cl:.3}), Drag: {drag:.3}N (Cd = {Cd:.3})")
    plot_str ="[it=" + str(n_iter) + " iter=" + f"{output_counter}/{n_iter}" + f"] Mean flux residual: {flux_res:.3} ({mass_mse:.3}, {momx_mse:.3}, {momy_mse:.3}, {energy_mse:.3})"
    plot_str += f", \tLift: {lift:.3}N (Cl = {Cl:.3}), Drag: {drag:.3}N (Cd = {Cd:.3})"

    # Tip: build the triangulation once outside the time loop for speed and reuse it here.
    triobj = Triangulation(mesh['points'][:, 0], mesh['points'][:, 1], triangles=mesh['triangles'])

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()

    # Data to plot: (array, title)
    fields = [
        ("rho", rho_phys, r"Density $\rho$ [kg/m$^3$]"),
        ("vx",  vx_phys,  r"Velocity $v_x$ [m/s]"),
        ("vy",  vy_phys,  r"Velocity $v_y$ [m/s]"),
        ("P",   P_phys,   r"Pressure $P$ [Pa]"),
    ]

    clims = {
        "rho": (1.1, 1.3),     # Density min,max
        "vx":  (0.0, 130.0),  # Velocity x min,max
        "vy":  (-60.0, 70.0),  # Velocity y min,max
        "P":   (75000, 90000) # Pressure min,max
    }

    # Draw each subplot
    for ax, (key, values, title) in zip(axes, fields):
        vmin, vmax = clims[key]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        norm = None
        tpc = ax.tripcolor(triobj, facecolors=values, shading='flat', norm=norm)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        fig.colorbar(tpc, ax=ax)

        # only plot in -2 to 2 window
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])

    # Save one combined image per frame (add an extension!)
    fname = f"{save_animation_path}/{dir}/phys_{output_counter}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    # Plot histograms of fluxes, primitives, and conserved variables
    fig, axes = plt.subplots(2,3, figsize=(14, 9), constrained_layout=True)
    bins = 50
    vals = [
        (Mass_flux, Momx_flux, Momy_flux, Energy_flux),
        (rho, vx, vy, P),
        (Mass, Momx, Momy, Energy),
    ]
    lbls = [
        (r"Mass flux", r"Mom$_x$ flux", r"Mom$_y$ flux", r"Energy flux"),
        (r"$\rho$", r"$v_x$", r"$v_y$", r"$P$"),
        (r"Mass", r"Mom$_x$", r"Mom$_y$", r"Energy"),
    ]
    titles = [
        "Fluxes per cell",
        "Primitive variables per cell",
        "Conserved variables per cell",
    ]
    for i in range(3):
        ax = axes[0,i]
        val_1, val_2, val_3, val_4 = vals[i]
        lbl_1, lbl_2, lbl_3, lbl_4 = lbls[i]
        title = titles[i]
        ax.hist(val_1, bins=bins, alpha=0.6, label=lbl_1)
        ax.hist(val_2,  bins=bins, alpha=0.6, label=lbl_2)
        ax.hist(val_3,  bins=bins, alpha=0.6, label=lbl_3)
        ax.hist(val_4,   bins=bins, alpha=0.6, label=lbl_4)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False)

    # plot Cp vs theta compared to theoretical
    ax = axes[1, 0]
    theta_theoretical = np.arange(0, 2*np.pi, 0.1)
    Cp_theoretical = 1 - 4 * (np.sin(theta_theoretical))**2
    Cp_simulation = (surf_pressures - p_inf) / q_inf
    ax.plot(surf_theta, Cp_simulation, 'o', label=r'Solver', markersize=2)
    ax.plot(theta_theoretical, Cp_theoretical, label=r'Theoretical', linewidth=1)
    ax.set_xlabel(r"Surface angle $\theta$ [degrees]")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Pressure Coefficient on Cylinder Surface")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    # Plot flux error history
    ax = axes[1, 1]
    flux_current = (mass_mse, momx_mse, momy_mse, energy_mse)
    hist_plot.append({'flux':flux_current, 'iter':n_iter})
    iters = [entry['iter'] for entry in hist_plot]
    for var_idx, var_name in enumerate(['Mass', 'Momx', 'Momy', 'Energy']):
        flux_values = [entry['flux'][var_idx] for entry in hist_plot]
        ax.plot(iters, flux_values, label=var_name)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Absolute Flux Error")
    ax.set_title("Flux Error History")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    fname = f"{save_animation_path}/{dir}/hist_{output_counter}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return plot_str


def plot_parameterization(
        params:dict,
        residual_fn_full_jit:Callable,
        meshes:list,
        test_meshes:list,
        residual_hist:list,
        ref_scales:dict,
        freestream_phys:dict,
        iteration:int,
        output_counter:int,
        save_path:str = None,
    ):
    fig, axes = plt.subplots(2,3, figsize=(14, 9), constrained_layout=True)
    # 6 plots:
    # 1) flux residual vs iterations for each shape + total loss
    # 2) Cd as function of R2 (ellipse y radius)
    # 3) Cl as a function of R2 (ellipse y radius)
    # 4) Theoretical surface pressure vs theta for each shape 
    # 5) Plot streamlines around a test shape (for example R2=0.1?) with velocity magnitude color map
    # 6) Plot streamlines around a test shape (for example R2=0.1?) with velocity magnitude color map up close

    loss, predictions_per_mesh = residual_fn_full_jit(
        params,
    )

    # sort meshes and predictions per mesh by R2 for consistent plotting
    meshes = meshes+test_meshes
    sorted_indices = np.argsort([mesh['chord'] for mesh in meshes])
    meshes = [meshes[i] for i in sorted_indices]
    predictions_per_mesh = [predictions_per_mesh[i] for i in sorted_indices]

    plot_data = []
    for mesh, predictions in zip(meshes, predictions_per_mesh):
        Mass, Momx, Momy, Energy = predictions['cons']
        Mass_flux, Momx_flux, Momy_flux, Energy_flux = predictions['flux']
        rho, vx, vy, P = predictions['prim']
        rho_phys, vx_phys, vy_phys, P_phys = nondim_to_phys_state(rho, vx, vy, P, ref_scales)

        # 1) Flux residuals
        mass_mse, momx_mse, momy_mse, energy_mse = np.mean(np.abs(Mass_flux)), np.mean(np.abs(Momx_flux)), np.mean(np.abs(Momy_flux)), np.mean(np.abs(Energy_flux))
        flux_res = mass_mse + momx_mse + momy_mse + energy_mse

        # 2), 3), 4): Forces
        a,b = mesh['R1'], mesh['R2']
        theta_theoretical = np.arange(0, 2*np.pi, 0.1)
        Cp_theoretical = 1 - (a+b)**2*(np.sin(theta_theoretical))**2 / (a**2*np.sin(theta_theoretical)**2 + b**2*np.cos(theta_theoretical)**2)
        lift, drag, surf_theta, surf_pressures = compute_forces(mesh, P_phys)
        q_inf, p_inf = freestream_phys['q_inf'], freestream_phys['p_inf']
        Cl = lift / (q_inf*2.0)
        Cd = drag / (q_inf*2.0)
        Cp_simulation = (surf_pressures - p_inf) / q_inf

        # aggregate for plotting later
        plot_data.append(
            {
                'mesh': mesh,
                'flux_res': flux_res,
                'Cd': Cd,
                'Cl': Cl,
                'theta_dist': surf_theta,
                'Cp_dist': Cp_simulation,
                'theta_theoretical': theta_theoretical,
                'Cp_theoretical': Cp_theoretical,
            }
        )

    # save history for later plotting
    residual_hist.append(
        {
            'iteration': iteration,
            'loss': loss,
            'fluxes': [data['flux_res'] for data in plot_data],
        }
    )
    if iteration == -1:
        residual_hist.append({key:val for key,val in residual_hist[-1].items()})  # duplicate first entry for better plotting
        residual_hist[-1]['iteration'] += 1 

    # 1) flux residual vs iterations for each shape + total loss
    ax = axes[0,0]
    iterations = [entry['iteration'] for entry in residual_hist]
    for shape_idx in range(len(meshes)):
        fluxes = [entry['fluxes'][shape_idx] for entry in residual_hist]
        mesh = plot_data[shape_idx]['mesh']
        label = mesh['R2']
        label = f'R2={label:.2f}'
        if mesh['test']:
            label += ' (test)'
            ax.plot(iterations, fluxes, label=label, linestyle='--')
        else:
            ax.plot(iterations, fluxes, label=label)
    ax.plot(iterations, [entry['loss'] for entry in residual_hist], 'k--', label='Total Loss')
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Flux Residual")
    ax.set_title("Flux Residual History")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    # 2) Cd as function of R2 (ellipse y radius)
    ax = axes[0,1]
    R2_values = [data['mesh']['R2'] for data in plot_data if not data['mesh']['test']]
    Cd_values = [data['Cd'] for data in plot_data if not data['mesh']['test']]
    ax.plot(R2_values, Cd_values, 'o-', label='Train')
    R2_values_test = [data['mesh']['R2'] for data in plot_data if data['mesh']['test']]
    Cd_values_test = [data['Cd'] for data in plot_data if data['mesh']['test']]
    ax.plot(R2_values_test, Cd_values_test, 'o--', label='Test')
    ax.set_xlabel("Ellipse Y Radius (R2)")
    ax.set_ylabel("Drag Coefficient Cd")
    ax.set_title("Drag Coefficient vs Ellipse Y Radius")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    # 3) Cl as a function of R2 (ellipse y radius)
    ax = axes[0,2]
    Cl_values = [data['Cl'] for data in plot_data if not data['mesh']['test']]
    ax.plot(R2_values, Cl_values, 'o-', label='Train')
    Cl_values_test = [data['Cl'] for data in plot_data if data['mesh']['test']]
    ax.plot(R2_values_test, Cl_values_test, 'o--', label='Test')
    ax.set_xlabel("Ellipse Y Radius (R2)")
    ax.set_ylabel("Lift Coefficient Cl")
    ax.set_title("Lift Coefficient vs Ellipse Y Radius")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    # 4) Theoretical surface pressure vs theta for each shape
    ax = axes[1,0]
    for i,data in enumerate(plot_data):
        mesh = data['mesh']
        color = plt.cm.viridis(i / len(plot_data))
        surf_theta = data['theta_dist']
        Cp_simulation = data['Cp_dist']
        if not mesh['test']:
            marker, markersize = 'o', 2
        else:
            marker, markersize = 'x', 4
        ax.plot(surf_theta, Cp_simulation, marker, markersize=markersize, label=f'R2={mesh["R2"]:.2f}',c=color)
        surf_theoretical = data['theta_theoretical']
        Cp_theoretical = data['Cp_theoretical']
        ax.plot(surf_theoretical, Cp_theoretical, c=color)
    ax.set_xlabel(r"Surface angle $\theta$ [degrees]")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Pressure Coefficient on Elipse Surface")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False)

    # 5) Plot streamlines around a test shape (for example R2=0.1?) with velocity magnitude color map
    if not save_path:
        plt.show()
    else:
        fname = f"{save_path}/param_{output_counter}.png"
        fig.savefig(fname, dpi=250)
        plt.close(fig)