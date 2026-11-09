"""Differentiable BWB panel solve, independent of geometry and dataset I/O.

The solver settings and field conversion match datasets/BWB/bwb_panel.py.
Mesh coordinates are body-fixed, in metres, and pitch is in degrees about the
origin's y axis. Connectivity is fixed; coordinates remain differentiable.
No CFD interpolation is performed here: pass pressure/velocity to the existing
``get_lowfi_panel_mesh`` mapping when assembling the panel-guided GNN.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import time

BWB_DIR = Path(__file__).resolve().parents[2] / "datasets" / "BWB"
DEFAULT_MESH = BWB_DIR / "panel" / (
    "bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh"
)
print_timings = True

@dataclass(frozen=True)
class PanelTopology:
    """VortexAD ordering after adjacency construction, before deformation."""

    cell_adjacency_data: tuple
    te_properties: tuple

    @property
    def reference_points(self):
        return self.cell_adjacency_data[0]

    @property
    def num_panels(self):
        return sum(len(cells) for cells in self.cell_adjacency_data[1].values())


@dataclass(frozen=True)
class PanelFunctions:
    """Host entry points plus a composable JAX function.

    ``forward(mesh, pitch)`` returns pressure (P,), velocity (P,3), Cp (P,),
    L (1,), Di (1,), M (3,), panel_forces (P,3), and Di_Trefftz (1,).
    Loads are the full panel aircraft's loads, with VortexAD conventions;
    these are diagnostics, not extra loads to add to the GNN prediction.

    ``vjp(mesh, pitch, seeds)`` returns (d_mesh, d_pitch). Seeds may contain
    any subset of output keys; omitted outputs receive zero cotangents.
    ``jax_forward`` is for JAX composition under an enabled x64 context.
    Host entry points temporarily enable x64 and restore the caller's setting,
    on successful completion. Do not call them from inside another JAX trace.
    """

    forward: Callable
    vjp: Callable
    jax_forward: Callable
    topology: PanelTopology
    output_shapes: dict
    device: object


def load_panel_topology(mesh_path=DEFAULT_MESH):
    """Read the mixed panel mesh and detect its fixed trailing-edge topology.

    Evaluate geometry projections in this reference-point ordering. Do not
    substitute raw meshio point ordering after adjacency deduplication.
    """
    import meshio
    from VortexAD import find_cell_adjacency, TE_detection

    mesh = meshio.read(Path(mesh_path))
    adjacency = find_cell_adjacency(points=mesh.points, cells=mesh.cells_dict)
    points, cells, _, edges2cells, points2cells = adjacency
    te_properties = TE_detection(
        points=points, cells=cells, edges2cells=edges2cells,
        points2cells=points2cells, threshold_theta=125.0,
    )
    return PanelTopology(tuple(adjacency), tuple(te_properties))


def build_panel_functions(topology, operating_reference, *, device, flow_condition=None):
    """Build once; evaluate many geometries without creating CSDL simulators.

    ``operating_reference`` has fixed positive scalars U0, p0, T0, matching
    bwb_panel.py. Optional ``flow_condition`` supplies fixed v_inf (scalar or
    streamwise vector), p_inf, T_inf for a particular case. As in the dataset,
    pressure conversion adds reference p0 even when case p_inf differs.
    ``device`` is an explicit JAX device (CPU or GPU), independent
    of the future warper/GNN devices. Both public input arrays must have shapes
    (N,3) and (1,); scalar pitch is also accepted by the host entry points.
    """
    import csdl_alpha as csdl
    import jax
    import jax.numpy as jnp
    from csdl_alpha.backends.jax import create_jax_function
    from VortexAD import PanelMethod
    from VortexAD.core.pm_class import default_input_dict

    reference_points = np.asarray(topology.reference_points, dtype=np.float64)
    if reference_points.ndim != 2 or reference_points.shape[1] != 3:
        raise ValueError("Panel reference points must have shape (N, 3).")
    if not np.isfinite(reference_points).all():
        raise ValueError("Panel reference points must be finite.")
    U0, p0, T0 = (float(operating_reference[key]) for key in ("U0", "p0", "T0"))
    if not all(np.isfinite(x) and x > 0 for x in (U0, p0, T0)):
        raise ValueError("U0, p0 and T0 must be finite positive scalars.")
    speed, pressure_inf, temperature_inf = U0, p0, T0
    if flow_condition is not None:
        velocity_inf = np.asarray(flow_condition["v_inf"], dtype=float).reshape(-1)
        if velocity_inf.size not in (1, 3) or np.any(velocity_inf[1:] != 0):
            raise ValueError("v_inf must be a positive speed or [speed, 0, 0]; use pitch for incidence.")
        speed = float(velocity_inf[0])
        pressure_inf = float(np.asarray(flow_condition["p_inf"]).item())
        temperature_inf = float(np.asarray(flow_condition["T_inf"]).item())
        if not all(np.isfinite(x) and x > 0 for x in (speed, pressure_inf, temperature_inf)):
            raise ValueError("Case speed, p_inf and T_inf must be finite positive scalars.")
    density = pressure_inf / (287.0 * temperature_inf)
    if device is None:
        raise ValueError("Select an explicit panel device.")

    names = ("Cp", "Vx", "Vy", "Vz", "L", "Di", "M", "panel_forces", "Di_Trefftz")
    recorder = csdl.Recorder(inline=False, debug=False)
    # tcp1 VortexAD mutates its module defaults in PanelMethod.__init__. Keep
    # that mutation local so separate panel builders/recorders cannot interfere.
    saved_defaults = dict(default_input_dict)
    previous_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    recorder.start()

    mesh = csdl.Variable(name="panel_mesh", value=reference_points)
    pitch = csdl.Variable(name="pitch", value=np.zeros(1))
    theta = pitch * np.pi / 180.0
    cosine, sine = csdl.cos(theta), csdl.sin(theta)
    x, y, z = mesh[:, 0], mesh[:, 1], mesh[:, 2]
    rotated_mesh = csdl.concatenate((
        (cosine * x + sine * z).reshape((-1, 1)),
        y.reshape((-1, 1)),
        (-sine * x + cosine * z).reshape((-1, 1)),
    ), axis=1)
    velocity = csdl.Variable(value=np.array([-speed, 0.0, 0.0]))
    solver = PanelMethod(solver_input_dict={
        "V_inf": csdl.expand(velocity, (1,) + mesh.shape, "i->aji"),
        "rho": csdl.Variable(value=density),
        "sos": csdl.Variable(value=np.sqrt(1.4 * 287.0 * T0)),
        "compressibility": False,
        "Cp cutoff": -5.0,
        "partition_size": 1,
        "reuse_AIC": True,
    }, skip_geometry=True)
    solver.options_dict = dict(solver.options_dict)
    solver.insert_grid_data(
        mesh=rotated_mesh,
        cell_adjacency_data=topology.cell_adjacency_data,
        TE_properties=topology.te_properties,
    )
    solver.declare_outputs(list(names))
    outputs = solver.evaluate()
    raw_function = create_jax_function(
        recorder.active_graph, outputs=[outputs[name] for name in names],
        inputs=[mesh, pitch],
    )

    default_input_dict.clear()
    default_input_dict.update(saved_defaults)
    recorder.stop()
    jax.config.update("jax_enable_x64", previous_x64)

    # These clips deliberately reproduce dataset preprocessing. In particular,
    # Vy is NOT clipped. Derivatives at clipping thresholds are nonsmooth.
    def jax_forward(panel_mesh, pitch_degrees):
        raw = dict(zip(names, raw_function(panel_mesh, pitch_degrees)))
        cp = raw["Cp"].reshape(-1)
        pressure = jnp.clip(p0 + 0.5 * density * speed**2 * cp, 100.0, 50000.0)
        velocity = jnp.stack((
            jnp.clip(raw["Vx"].reshape(-1), 0.0, 350.0),
            raw["Vy"].reshape(-1),
            jnp.clip(raw["Vz"].reshape(-1), -300.0, 300.0),
        ), axis=-1)
        return {
            "pressure": pressure, "velocity": velocity, "Cp": cp,
            "L": raw["L"].reshape(1), "Di": raw["Di"].reshape(1),
            "M": raw["M"].reshape(3),
            "panel_forces": raw["panel_forces"].reshape((-1, 3)),
            "Di_Trefftz": raw["Di_Trefftz"].reshape(1),
        }

    forward_jit = jax.jit(jax_forward, device=device)

    def reverse(panel_mesh, pitch_degrees, seeds):
        _, pullback = jax.vjp(jax_forward, panel_mesh, pitch_degrees)
        return pullback(seeds)

    reverse_jit = jax.jit(reverse, device=device)
    count = topology.num_panels
    shapes = {
        "pressure": (count,), "velocity": (count, 3), "Cp": (count,),
        "L": (1,), "Di": (1,), "M": (3,),
        "panel_forces": (count, 3), "Di_Trefftz": (1,),
    }

    def put_inputs(panel_mesh, pitch_degrees):
        points = jnp.asarray(panel_mesh, dtype=jnp.float64)
        angle = jnp.asarray(pitch_degrees, dtype=jnp.float64).reshape(1)
        if points.shape != reference_points.shape:
            raise ValueError(f"Expected panel mesh {reference_points.shape}, got {points.shape}.")
        return jax.device_put(points, device), jax.device_put(angle, device)

    def forward(panel_mesh, pitch_degrees):
        start = time.time()
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        result = forward_jit(*put_inputs(panel_mesh, pitch_degrees))
        # print("Panel L:", result["L"].devices(), result["L"].dtype)
        if print_timings:
            jax.block_until_ready(result)
            end = time.time()
            print(f'* evaluated PANEL (pitch: {pitch_degrees}, L: {result["L"].item()/1000:.3f}kN) ({end - start:.3f} sec)')
        jax.config.update("jax_enable_x64", previous)
        return result 

    def vjp(panel_mesh, pitch_degrees, seeds):
        start = time.time()
        unknown = set(seeds) - set(shapes)
        if unknown:
            raise ValueError(f"Unknown panel output seeds: {sorted(unknown)}")
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        cotangents = {}
        for name, shape in shapes.items():
            value = jnp.asarray(seeds.get(name, np.zeros(shape)), dtype=jnp.float64)
            if value.shape == () and shape == (1,):
                value = value.reshape(1)
            if value.shape != shape:
                raise ValueError(f"Seed {name!r} must have shape {shape}, got {value.shape}.")
            cotangents[name] = jax.device_put(value, device)
        result = reverse_jit(*put_inputs(panel_mesh, pitch_degrees), cotangents)
        if print_timings:
            jax.block_until_ready(result)
            end = time.time()
            print(f'* evaluated PANEL VJP (pitch: {pitch_degrees}) ({end - start:.3f} sec)')
        jax.config.update("jax_enable_x64", previous)
        return result

    return PanelFunctions(forward, vjp, jax_forward, topology, shapes, device)
