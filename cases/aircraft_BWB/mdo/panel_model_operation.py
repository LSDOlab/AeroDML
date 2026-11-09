"""Standalone CSDL panel operation, using the differentiable panel builder.

Inputs are body-fixed panel mesh points (in topology.reference_points ordering)
and pitch in degrees. Loads cover the FULL aircraft: do not multiply by two.
D is VortexAD's Di (induced drag), and M is its y-axis moment. Force outputs
retain VortexAD's conventions. No GNN parameters or CFD volume mesh are needed.
"""

import numpy as np
import jax
import csdl_alpha as csdl

from build_panel import build_panel_functions, load_panel_topology


def build_default_panel_functions(*, device=None):
    """Match the operating conditions in build_gnn.py without importing it.

    Pass an explicit JAX device to override the default device selection.
    Build before starting the outer CSDL recorder, then reuse across operations.
    """
    return build_panel_functions(
        load_panel_topology(),
        {"U0": 227.38, "p0": 30089.6, "T0": 228.714},
        device=jax.devices()[0] if device is None else device,
        flow_condition={
            "v_inf": np.array([227.3805]),
            "p_inf": np.array([30089.6]),
            "T_inf": np.array([228.714]),
        },
    )


class PANEL(csdl.CustomExplicitOperation):
    """Panel-only analogue of RANSGNN/PGRANSGNN, with reverse derivatives.

    Supply PanelFunctions built before the enclosing recorder starts. Outputs
    include L, D, M and surface_forces, plus pressure, velocity, Cp and
    Di_Trefftz. surface_forces are per-panel forces, not CFD wall-face forces.
    """

    def __init__(self, panel_functions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel_functions = panel_functions
        self._output_map = {
            'L': 'L', 'D': 'Di', 'M': 'M',
            'surface_forces': 'panel_forces',
            'pressure': 'pressure', 'velocity': 'velocity',
            'Cp': 'Cp', 'Di_Trefftz': 'Di_Trefftz',
        }

    def evaluate(self, panel_mesh, pitch):
        self.declare_input('panel_mesh', panel_mesh)
        self.declare_input('pitch', pitch)
        return {
            name: self.create_output(
                name, (1,) if name == 'M' else self.panel_functions.output_shapes[key]
            )
            for name, key in self._output_map.items()
        }

    def compute(self, inputs, outputs):
        result = self.panel_functions.forward(inputs['panel_mesh'], inputs['pitch'])
        for name, key in self._output_map.items():
            value = result[key][1:2] if name == 'M' else result[key]
            outputs[name] = np.asarray(value, dtype=np.float64)

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        if mode != 'rev':
            raise NotImplementedError('PANEL supports reverse-mode derivatives only')
        seeds = {}
        for name, key in self._output_map.items():
            if name not in d_outputs:
                continue
            if name == 'M':
                seeds[key] = np.zeros(3, dtype=np.float64)
                seeds[key][1] = np.asarray(d_outputs[name]).item()
            else:
                seeds[key] = np.asarray(d_outputs[name], dtype=np.float64)
        d_mesh, d_pitch = self.panel_functions.vjp(
            inputs['panel_mesh'], inputs['pitch'], seeds,
        )
        if 'panel_mesh' in d_inputs:
            d_inputs['panel_mesh'] += np.asarray(d_mesh, dtype=np.float64)
        if 'pitch' in d_inputs:
            d_inputs['pitch'] += np.asarray(d_pitch, dtype=np.float64)


if __name__ == '__main__':
    from pathlib import Path
    import pickle

    run_geometry = True
    # run_geometry = False
    sweep_vals = {'L': [], 'D': [], 'M': []}

    # Build the panel solver before starting the outer CSDL recorder.
    panel_functions = build_default_panel_functions()
    recorder = csdl.Recorder(inline=False, debug=True)
    recorder.start()

    scale_var = csdl.Variable(name='surf_x_scale', value=1.0)

    if run_geometry:
        from geometry import build_surface_csdl

        here = Path(__file__).resolve().parent
        geometry_directory = here.parents[1] / 'datasets' / 'BWB' / 'bwb_geometry'
        dafoam_directory = here.parent / 'model' / 'mesh_639k'
        x_surf_hash = 'NEWHASH'
        with open(dafoam_directory / f'panel_projections_{x_surf_hash}.pickle', 'rb') as f:
            panel_projections = pickle.load(f)

        recorder.inline = True
        csdl_surf_points, geometry_config_map, bwb_geometry = build_surface_csdl(
            recorder,
            geometry_directory / 'bwb_new_stored_refit.pickle',
            geometry_directory / 'bwbv2_no_wingtip_coarse_refined_flat.stp',
            dafoam_directory / f'projected_surface_mesh_{x_surf_hash}.pickle',
            dafoam_directory,
            x_surf_hash,
            index_unique=False,
            return_geometry=True,
        )
        panel_mesh_val = bwb_geometry.evaluate(panel_projections, plot=False)
        recorder.inline = False
    else:
        panel_mesh_val = panel_functions.topology.reference_points

    panel_surface_var = panel_mesh_val * scale_var
    pitch_dv_var = csdl.Variable(name='pitch_dv', value=np.array([1.0, -1.0]))
    pitch_var = pitch_dv_var[0]
    pitch_var1 = pitch_dv_var[1]

    # Panel outputs already cover the full aircraft.
    panel_outputs = PANEL(panel_functions).evaluate(panel_surface_var, pitch_var)
    L, D, My = (panel_outputs[name] for name in ('L', 'D', 'M'))
    surface_forces = panel_outputs['surface_forces']
    L.add_name('L')
    D.add_name('D')
    My.add_name('My')
    surface_forces.add_name('surface_forces')

    panel_outputs2 = PANEL(panel_functions).evaluate(panel_surface_var, pitch_var1)
    L2, D2, My2 = (panel_outputs2[name] for name in ('L', 'D', 'M'))
    surface_forces2 = panel_outputs2['surface_forces']
    L2.add_name('L2')
    D2.add_name('D2')
    My2.add_name('My2')
    surface_forces2.add_name('surface_forces2')

    L_tot = L + L2
    D_tot = D + D2
    My_tot = My + My2
    recorder.stop()

    js = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=(scale_var, pitch_dv_var),
        additional_outputs=[L, D, My, L2, D2, My2, L_tot, D_tot, My_tot],
        gpu=False,
    )
    # js.check_totals(step_size=1e-1)
    # js.check_totals(step_size=1e-2)
    # js.check_totals(step_size=1e-3)

    pitches = np.linspace(-6, 6, 13)
    scales = np.ones((13,))
    for scale, pitch in zip(scales, pitches):
        js[pitch_dv_var] = np.array([pitch, pitch])
        js[scale_var] = scale
        js.run()
        print(
            f"scale={scale}, pitch={pitch}, "
            f"L={js[L]}, D={js[D]}, My={js[My]}, "
            f"L_tot={js[L_tot]}, D_tot={js[D_tot]}, My_tot={js[My_tot]}"
        )
        sweep_vals['L'].append(js[L])
        sweep_vals['D'].append(js[D])
        sweep_vals['M'].append(js[My])

    for name in ('L', 'D', 'M'):
        values = np.asarray(jax.device_get(sweep_vals[name]), dtype=float).flatten()
        values_str = np.array2string(
            values, separator=", ", precision=8, threshold=np.inf, max_line_width=np.inf,
        )
        print(f"{name} = np.array({values_str})")
