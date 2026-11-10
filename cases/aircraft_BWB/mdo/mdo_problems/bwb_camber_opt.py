"""Cruise drag minimization with post-workshop wing twist and camber only.

Uses the beam script's geometry parameterization, reference-area convention,
aerodynamic interfaces, and CSDL/modopt recording workflow. Run in the GNN environment
inside an appropriate GPU allocation. Outputs are written beside this script in mdo_problems.
"""
import argparse
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    'USE_MULTIFIDELITY', choices=('PG-RANS-GNN', 'Panel', 'RANS-GNN'),
    help='Aerodynamic model used for cruise optimization',
)
USE_MULTIFIDELITY = parser.parse_args().USE_MULTIFIDELITY

_script_directory = Path(__file__).resolve().parent
# The geometry library resolves its internal caches against the working directory.
os.chdir(_script_directory)
sys.path.insert(0, str(_script_directory.parent))

import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo
from modopt import CSDLAlphaProblem, PySLSQP, InteriorPoint
from additional_solvers import atmos_model
from geometry_functions_post_workshop import (
    initial_refit, setup_and_evaluate_geometry_parameterization,
)

# ============================ run settings ============================
optimizer_name = 'InteriorPoint'  # 'SLSQP' or 'InteriorPoint'
cruise_pitch_deg = 0.0  # Fixed; twist and camber are the only design variables.
CL_target = 0.2
save_meshes = True
check_derivs = False
shutdown_inline = True

assert USE_MULTIFIDELITY in ('PG-RANS-GNN', 'Panel', 'RANS-GNN')
if optimizer_name not in ('SLSQP', 'InteriorPoint'):
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')

# Keep the requested name; suffix subsequent runs to preserve previous results.
problem_name = f'pw_camber_{USE_MULTIFIDELITY}_opt_{optimizer_name}'
fname = problem_name
attempt = 0
while Path(f'{fname}.hdf5').exists() or Path(f'{fname}_outputs').exists():
    attempt += 1
    fname = f'{problem_name}_{attempt}'
print(f'file name: {fname}')

lfs.num_workers = 1
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# ============================ geometry import ============================
geometry_directory = _script_directory / 'geometry'
imported_function_set = lfs.import_file_patched(
    file_name=str(geometry_directory / 'bwbv2_no_wingtip_coarse_refined_flat.stp'),
    parallelize=False,
)
imported_geometry = lsdo_geo.Geometry(
    functions=imported_function_set.functions,
    function_names=imported_function_set.function_names,
    name='imported_geometry', space=imported_function_set.space,
)
geometry = initial_refit(imported_geometry, plot=False)

# Panel and panel-guided models use the same ordered mesh as the beam script.
if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'Panel'):
    import meshio
    from VortexAD import find_cell_adjacency
    mesh = meshio.read(str(geometry_directory /
        'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh'))
    points_orig = find_cell_adjacency(points=mesh.points, cells=mesh.cells_dict)[0]
    projected_panel_mesh = geometry.project(
        points_orig, grid_search_density_parameter=1,
        newton_tolerance=1.e-10, grid_search_density_cutoff=30,
        projection_tolerance=1.e-2, force_reprojection=False, plot=False,
    )

# ============================ geometry parameterization ============================
num_wing_spanwise_design_dofs = 4
num_centerbody_spanwise_design_dofs = 4

lower_left_wing = geometry.functions[0]
lower_left_transition = geometry.functions[1]
lower_left_centerbody = geometry.functions[2]
lower_right_centerbody = geometry.functions[3]
lower_right_transition = geometry.functions[4]
lower_right_wing = geometry.functions[5]
upper_left_wing = geometry.functions[6]
upper_left_transition = geometry.functions[7]
upper_left_centerbody = geometry.functions[8]
upper_right_centerbody = geometry.functions[9]
upper_right_transition = geometry.functions[10]
upper_right_wing = geometry.functions[11]
left_wing_cap = geometry.functions[12]
right_wing_cap = geometry.functions[13]

upper_wing_root = upper_right_wing.evaluate(
    np.array([[0.667, 0.0]]), plot=False
).flatten()
lower_wing_root = lower_right_wing.evaluate(
    np.array([[0.667, 0.0]]), plot=False
).flatten()
wing_root = (upper_wing_root + lower_wing_root) / 2.0

upper_wing_tip = upper_right_wing.evaluate(
    np.array([[0.667, 1.0]]), plot=False
).flatten()
lower_wing_tip = lower_right_wing.evaluate(
    np.array([[0.667, 1.0]]), plot=False
).flatten()
wing_tip = (upper_wing_tip + lower_wing_tip) / 2.0

upper_centerbody_root = upper_right_centerbody.evaluate(
    np.array([[0.667, 0.0]]), plot=False
).flatten()
lower_centerbody_root = lower_right_centerbody.evaluate(
    np.array([[0.667, 0.0]]), plot=False
).flatten()
centerbody_root = (upper_centerbody_root + lower_centerbody_root) / 2.0

upper_centerbody_tip = upper_right_centerbody.evaluate(
    np.array([[0.667, 1.0]]), plot=False
).flatten()
lower_centerbody_tip = lower_right_centerbody.evaluate(
    np.array([[0.667, 1.0]]), plot=False
).flatten()
centerbody_tip = (upper_centerbody_tip + lower_centerbody_tip) / 2.0

wing_root_leading_edge = upper_right_wing.evaluate(
    np.array([[0.0, 0.0]]), plot=False
).flatten()
wing_tip_leading_edge = upper_right_wing.evaluate(
    np.array([[0.0, 1.0]]), plot=False
).flatten()

wing_root_to_tip_vector = wing_tip - wing_root
centerbody_root_to_tip_vector = centerbody_tip - centerbody_root
transition_root_to_tip_vector = wing_root - centerbody_tip
wing_leading_edge_root_to_tip_vector = (
    wing_tip_leading_edge - wing_root_leading_edge
)

wing_sweep_computed = csdl.arctan2(
    wing_leading_edge_root_to_tip_vector[0],
    wing_leading_edge_root_to_tip_vector[1],
)
wing_half_span_computed = wing_root_to_tip_vector[1]
centerbody_half_span_computed = centerbody_root_to_tip_vector[1]
transition_half_span_computed = transition_root_to_tip_vector[1]


# These are the active geometry variables from bwb_mdo.py.
wing_sweep = csdl.Variable(value=wing_sweep_computed.value * 180.0 / np.pi)
wing_position = csdl.Variable(value=wing_root.value[0])
wing_half_span = csdl.Variable(value=wing_half_span_computed.value)
centerbody_half_span = csdl.Variable(value=centerbody_half_span_computed.value)
transition_half_span = csdl.Variable(value=transition_half_span_computed.value)

# Keep the wing chord stretch coefficients as direct inputs, as in the original.
wing_chord_stretch_coefficients = csdl.Variable(
    shape=(num_wing_spanwise_design_dofs,), value=0.0
)

# Pure explicit preprocessing from recursive centerbody chord ratios to the
# additive centerbody stretches expected by the geometry parameterization.
centerbody_chord_dist_0 = np.array([30.0, 27.796, 24.233, 20.187])
baseline_centerbody_chord_stretch_ratio = np.concatenate(
    (
        np.array([1.0]),
        centerbody_chord_dist_0[1:] / centerbody_chord_dist_0[:-1],
    )
)
centerbody_chord_stretch_ratio = csdl.Variable(
    shape=(num_centerbody_spanwise_design_dofs,),
    value=baseline_centerbody_chord_stretch_ratio,
)

centerbody_chord_dist = csdl.Variable(
    shape=(num_centerbody_spanwise_design_dofs,), value=0.0
)
centerbody_chord_dist = centerbody_chord_dist.set(
    csdl.slice[0],
    centerbody_chord_stretch_ratio[0] * centerbody_chord_dist_0[0],
)
for chord_index in range(1, num_centerbody_spanwise_design_dofs):
    centerbody_chord_dist = centerbody_chord_dist.set(
        csdl.slice[chord_index],
        centerbody_chord_stretch_ratio[chord_index]
        * centerbody_chord_dist[chord_index - 1],
    )
centerbody_chord_stretch_coefficients = (
    centerbody_chord_dist - centerbody_chord_dist_0
)
wing_twist_coefficients = csdl.Variable(
    shape=(num_wing_spanwise_design_dofs,), value=0.0
)
airfoil_thickness_coefficients = csdl.Variable(shape=(8, 8), value=0.0)
airfoil_camber_coefficients = csdl.Variable(shape=(6, 8), value=0.0)
cruise_trim_elevator_deflection = csdl.Variable(value=0.0)


# Keep the names aligned with bwb_mdo.py.
wing_sweep.add_name("wing_sweep")
wing_position.add_name("wing_position")
wing_half_span.add_name("wing_half_span")
centerbody_half_span.add_name("center_half_span")
transition_half_span.add_name("transition_half_span")
wing_chord_stretch_coefficients.add_name("wing_chord_stretch")
centerbody_chord_stretch_ratio.add_name("centerbody_chord_stretch_ratio")
centerbody_chord_dist.add_name("centerbody_chord_dist")
centerbody_chord_stretch_coefficients.add_name("centerbody_chord_stretch")
wing_twist_coefficients.add_name("wing_twist")
airfoil_thickness_coefficients.add_name("airfoil_thickness_coeff")
airfoil_camber_coefficients.add_name("airfoil_camber_coeff")
cruise_trim_elevator_deflection.add_name("elevator_deflection")


# Independent inputs exposed by JSIM and by the sweep utilities.
geometry_inputs = {
    "wing_sweep": wing_sweep,
    "wing_position": wing_position,
    "wing_half_span": wing_half_span,
    "centerbody_half_span": centerbody_half_span,
    "transition_half_span": transition_half_span,
    "wing_chord_stretch_coefficients": wing_chord_stretch_coefficients,
    "centerbody_chord_stretch_ratio": centerbody_chord_stretch_ratio,
    "wing_twist_coefficients": wing_twist_coefficients,
    "airfoil_thickness_coefficients": airfoil_thickness_coefficients,
    "airfoil_camber_coefficients": airfoil_camber_coefficients,
    "elevator_deflection": cruise_trim_elevator_deflection,
}


# Internal dictionary consumed by the post-workshop geometry parameterization.
# centerbody_chord_stretch_coefficients is derived above and is not a JSIM input.
geometric_variables = {
    "wing_sweep": wing_sweep,
    "wing_position": wing_position,
    "wing_half_span": wing_half_span,
    "centerbody_half_span": centerbody_half_span,
    "transition_half_span": transition_half_span,
    "wing_chord_stretch_coefficients": wing_chord_stretch_coefficients,
    "centerbody_chord_stretch_coefficients": (
        centerbody_chord_stretch_coefficients
    ),
    "wing_twist_coefficients": wing_twist_coefficients,
    "airfoil_thickness_coefficients": airfoil_thickness_coefficients,
    "airfoil_camber_coefficients": airfoil_camber_coefficients,
    "elevator_deflection": cruise_trim_elevator_deflection,
}


deformed_geometry = setup_and_evaluate_geometry_parameterization(
    geometry, geometric_variables, plot=False,
)

# Sample LE/TE on the new right-side patches (u: LE -> TE, v: root -> tip).
# Derive strip dimensions from the actual geometry rather than treating spline
# stretch coefficients as pointwise chord changes. Preserve the original
# eight-station trapezoidal area and MAC integration convention.
reference_stations = [(9, v) for v in np.linspace(0., 1., 4)] + [
    (11, v) for v in np.linspace(0., 1., 4)
]
reference_le = deformed_geometry.evaluate([
    (surface, np.array([[0., v]])) for surface, v in reference_stations
]).reshape((8, 3))
reference_te = deformed_geometry.evaluate([
    (surface, np.array([[1., v]])) for surface, v in reference_stations
]).reshape((8, 3))
ffd_sectional_chord_dist = reference_te[:, 0] - reference_le[:, 0]
wing_chord_dist = ffd_sectional_chord_dist[4:]
wing_chord_dist.add_name('wing_chord_dist')
ffd_sectional_avg_chord_dist = (
    ffd_sectional_chord_dist[:-1] + ffd_sectional_chord_dist[1:]
) / 2.
ffd_sectional_span_dist = reference_le[1:, 1] - reference_le[:-1, 1]
planform_area = 2 * csdl.sum(ffd_sectional_avg_chord_dist * ffd_sectional_span_dist)
planform_area.add_name('planform_area')
wingspan = 2 * reference_le[-1, 1]
AR = wingspan**2 / planform_area
AR.add_name('AR')
MAC = 2 * csdl.sum(ffd_sectional_span_dist * (
    ffd_sectional_chord_dist[:-1]**2 + ffd_sectional_chord_dist[1:]**2
) / 2.) / planform_area
MAC.add_name('MAC')


geometry_coefficients = deformed_geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')

# ============================ single cruise condition ============================
# Same atmosphere and coefficient normalization as the beam script. The GNN
# helper itself uses its fixed M=0.75 / 30,000 ft training flow condition.
cruise_mach = 0.75
cruise_h = 30000.0  # feet
altitude = csdl.Variable(value=np.array([cruise_h * 0.3048 / 1000.]), name='altitude')
mach = csdl.Variable(value=np.array([cruise_mach]), name='mach')
rho_array, sos_array, mu_array = atmos_model(altitude)
V_inf_array = sos_array * mach
pitch = csdl.Variable(value=np.array([cruise_pitch_deg]), name='pitch')

recorder.inline = not shutdown_inline
from gnn_function import compute_rans_gnn, evaluate_cfd_surface, compute_panel
panel_surf_gnn = None
if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'Panel'):
    panel_surf_gnn = deformed_geometry.evaluate(projected_panel_mesh, plot=False)
    panel_mesh = panel_surf_gnn.expand((1,) + panel_surf_gnn.shape, 'ij->aij')
    panel_mesh.add_name('panel_mesh')

if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'RANS-GNN'):
    cfd_surface = evaluate_cfd_surface(deformed_geometry)
    aero_outputs = compute_rans_gnn(
        cfd_surface, pitch.flatten(), panel_surface_var=panel_surf_gnn,
    )
else:
    # Same cruise Panel branch as bwb_mdo.py:
    # D is induced drag only, with no viscous or wave-drag additions.
    aero_outputs = compute_panel(pitch.flatten(), panel_surf_gnn)
# Both helpers return full-aircraft loads; do not multiply by two again.
L = aero_outputs['L']
D_total = aero_outputs['D']
M = aero_outputs['M']
qS = 0.5 * rho_array * V_inf_array**2 * planform_area
CL = L / qS
CD = D_total / qS
CM = M / (qS * MAC)
L_D = L / D_total
for name, variable in {'L': L, 'D_total': D_total, 'M': M,
                       'CL': CL, 'CD': CD, 'CM': CM, 'L_D': L_D}.items():
    variable.add_name(name)

# ============================ optimization definition ============================
# Degrees relative to the geometry helper's built-in baseline wing twist.
wing_twist_coefficients.set_as_design_variable(lower=-5.0, upper=5.0)
# Camber coefficients are fractions of local chord; shape is (6, 8).
airfoil_camber_coefficients.set_as_design_variable(lower=-0.03, upper=0.03, scaler=100.)
CL.set_as_constraint(equals=CL_target)
# Minimize dimensional drag; save CD as well for comparison with ref_camber_opt.
D_total.set_as_objective(scaler=1.e-5)

# ============================ saved variables ============================
# Preserve applicable beam-script names, including fixed geometry inputs needed
# to reconstruct the shape. Structural/mission-only diagnostics do not apply.
for variable in [*geometry_inputs.values(), centerbody_chord_dist,
                 centerbody_chord_stretch_coefficients, wing_chord_dist,
                 geometry_coefficients, planform_area, AR, MAC, pitch,
                 altitude, mach, L, D_total, M, CL, CD, CM, L_D]:
    variable.save()
if save_meshes:
    if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'RANS-GNN'):
        aero_outputs['surface_points'].save()
    aero_outputs['surface_forces'].save()
    if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'Panel'):
        panel_mesh.save()
if USE_MULTIFIDELITY == 'Panel':
    # Retain the beam script's batched Cp convention: row 0 is cruise here.
    Cp = aero_outputs['Cp'].reshape((1, -1))
    Cp.add_name('Cp')
    Cp.save()
csdl.save_optimization_variables()

# Like the beam script, keep the outer CSDL simulator on CPU; the custom GNN
# operation manages its own JAX/GPU evaluation.
sim = csdl.experimental.JaxSimulator(
    recorder, gpu=False, save_on_update=True, filename=fname, output_saved=True,
)
start = time.time()
sim.run()
print(f'Forward compile/run time: {time.time() - start:.1f} seconds')
print('Initial CL:', sim[CL], 'CD:', sim[CD], 'drag (N):', sim[D_total])
if check_derivs:
    sim.check_optimization_derivatives()

start = time.time()
prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)
print(f'Problem setup time: {time.time() - start:.1f} seconds')
if optimizer_name == 'SLSQP':
    optimizer = PySLSQP(prob, solver_options={'maxiter': 300, 'acc': 1.e-6})
else:
    optimizer = InteriorPoint(prob, maxiter=500, opt_tol=1.e-5, feas_tol=5.e-6)

start = time.time()
optimizer.solve()
print('Optimization time:', time.time() - start)
print('Success:', optimizer.results['success'])
print('Optimized objective:', optimizer.results['objective'])
print(f'History: {fname}.hdf5')
