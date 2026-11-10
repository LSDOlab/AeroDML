"""Beam/mission model migrated to the post-workshop geometry.

Parametric u=0.667 is the reference used by geometry_functions_post_workshop,
not a chord fraction. Outputs and geometry caches are stored beside this script.
Physical attachment seeds below are projected onto the initial refit and their
resulting surface coordinates follow the deformed geometry.
"""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('USE_MULTIFIDELITY', choices=('PG-RANS-GNN', 'Panel', 'RANS-GNN'),
                    help='Aerodynamic model used for cruise optimization')
USE_MULTIFIDELITY = parser.parse_args().USE_MULTIFIDELITY

_script_directory = Path(__file__).resolve().parent
# The geometry library resolves its internal caches against the working directory.
os.chdir(_script_directory)
sys.path.insert(0, str(_script_directory.parent))

# Importing all packages 
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

from modopt import CSDLAlphaProblem
from modopt import PySLSQP #, SNOPT
from modopt import InteriorPoint

import lsdo_geo
from VortexAD import find_cell_adjacency, TE_detection
from VortexAD import plot_pressure_distribution
import meshio

import aframe as af
from aeroelastic_coupling_utils import NodalMap

from geometry_functions_new import compute_volume, project_centerbody_volume_points
from geometry_functions_new import project_transition_volume_points
from geometry_functions_post_workshop import initial_refit, setup_and_evaluate_geometry_parameterization

import time
import pickle
from datetime import datetime

from additional_solvers import compute_static_margin, estimate_CDw, estimate_Cf, estimate_fuel_volume
from additional_solvers import estimate_fuel_burn_w_reserve, atmos_model, takeoff, landing

from dataclasses import dataclass

import resource

# Preserve the original run identifier and optimizer settings.
attempt = 2030
optimizer_name = 'InteriorPoint'  # 'SLSQP' or 'InteriorPoint'

print('requested attempt', attempt)
lfs.num_workers=1
save_meshes = True
shutdown_inline = True

requested_attempt = attempt

recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# region ============================ geometry import and setup ============================
file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
file_path = str(_script_directory / 'geometry') + os.sep 

# importing geometry
# geometry = lsdo_geo.import_geometry(
#     file_path + file_name,
#     parallelize=False,
# )

imported_function_set = lfs.import_file_patched(file_name=file_path + file_name, parallelize=False)
imported_geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)
# exit()
geometry= initial_refit(imported_geometry, plot=False)

# oml_indices = [key for key in geometry.functions.keys()]
# wing_c_indices = [0,1,8,9]
# wing_r_transition_indices = [2,3]
# wing_r_indices = [4,5,6,7]
# wing_l_transition_indices = [10,11]
# wing_l_indices = [12,13,14,15] 

# left_wing_transition = geometry.declare_component(wing_l_transition_indices)
# left_wing = geometry.declare_component(wing_l_indices)
# right_wing_transition = geometry.declare_component(wing_r_transition_indices)
# right_wing = geometry.declare_component(wing_r_indices)
# center_wing = geometry.declare_component(wing_c_indices)
# oml = geometry.declare_component(oml_indices)

# wing_parameterization=15
# num_v = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
# wing_refit_bspline = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))
# left_wing_function_set = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))
# right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))
# for i, function in left_wing_function_set.functions.items():
#     geometry.functions[i] = function
#     left_wing.functions[i] = function

# for i, function in right_wing_function_set.functions.items():
#     geometry.functions[i] = function
#     right_wing.functions[i] = function
# endregion

# region ============================ aero mesh setup ============================
grid_type = 'mixed'
if grid_type == 'mixed':
    file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh'
elif grid_type == 'tri':
    file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch.stl'

mesh = meshio.read(
    file_path + file_name,
)

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

cell_adjacency_data = find_cell_adjacency(points=points_orig, cells=cells_dict)

points_orig = cell_adjacency_data[0] 
cells_dict = cell_adjacency_data[1] 
cell_adjacency = cell_adjacency_data[2] 
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]

TE_properties = TE_detection(
    points=points_orig,
    cells=cells_dict,
    edges2cells=edges2cells,
    points2cells=points2cells,
    threshold_theta=125.
)

upper_TE_cells = TE_properties[0] 
lower_TE_cells = TE_properties[1] 
TE_edges = TE_properties[2] 
TE_node_indices = TE_properties[3]

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

if False:
    TE_coloring = np.zeros(shape=(len(combined_cells),))
    TE_coloring[upper_TE_cells] = 1
    TE_coloring[lower_TE_cells] = -1

    plot_pressure_distribution(points_orig, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')
    exit()

start = time.time()

# suggested by Andrew
projected_panel_mesh = geometry.project(points_orig,
                                        grid_search_density_parameter=1,
                                        newton_tolerance=1.e-10,
                                        grid_search_density_cutoff=30,
                                        projection_tolerance=1.e-2,
                                        force_reprojection=False,
                                        plot=False,
                                        )
# exit()
end_mesh_projection = time.time()
print(f'panel mesh projection time: {end_mesh_projection-start} seconds')
# exit()
# project panel centers

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()


panel_centers = np.zeros((len(combined_cells), 3))
for i, cell in enumerate(combined_cells):
    panel_centers[i] = np.mean(points_orig[cell], axis=0)

# Reproject using the post-workshop surface numbering and UV convention.
panel_center_projection_cache = _script_directory / 'mesh_projections/panel_centers_sep10.pkl'
panel_center_projection_cache.parent.mkdir(parents=True, exist_ok=True)
if not os.path.exists(panel_center_projection_cache):
    print('projecting panel centers')

    projected_panel_centers = geometry.project(
            panel_centers, 
            grid_search_density_parameter=1,
            newton_tolerance=1.e-10,
            grid_search_density_cutoff=15,
            projection_tolerance=1.e-2,
            force_reprojection=False, 
            plot=False
        )
    print('done projecting panel centers')
    with open(panel_center_projection_cache, 'wb') as f:
        pickle.dump(projected_panel_centers, f)

with open(panel_center_projection_cache, 'rb') as f:
    projected_panel_centers = pickle.load(f)

if 0:
    # Projection diagnostics (also runs when the projection was loaded from cache).
    # Distances use the same units as the geometry and the original panel centers.
    projected_panel_xyz = np.asarray(
        geometry.evaluate(projected_panel_centers, non_csdl=True)
    ).reshape((-1, 3))
    if projected_panel_xyz.shape != panel_centers.shape:
        raise ValueError("Cached projection count does not match the panel centers.")
    panel_projection_residual = panel_centers - projected_panel_xyz
    panel_projection_error = np.linalg.norm(panel_projection_residual, axis=1)

    # The local surface normal is the cross product of its parametric tangents.
    projection_du = np.asarray(geometry.evaluate(
        projected_panel_centers, parametric_derivative_orders=(1, 0), non_csdl=True
    )).reshape((-1, 3))
    projection_dv = np.asarray(geometry.evaluate(
        projected_panel_centers, parametric_derivative_orders=(0, 1), non_csdl=True
    )).reshape((-1, 3))
    projection_normals = np.cross(projection_du, projection_dv)
    projection_normal_lengths = np.linalg.norm(projection_normals, axis=1)
    projection_tangent_scale = (
        np.linalg.norm(projection_du, axis=1) * np.linalg.norm(projection_dv, axis=1)
    )
    valid_projection_normals = (
        np.isfinite(projection_normal_lengths)
        & (projection_normal_lengths > 1.e-12 * projection_tangent_scale)
    )
    projection_unit_normals = np.full_like(projection_normals, np.nan)
    projection_unit_normals[valid_projection_normals] = (
        projection_normals[valid_projection_normals]
        / projection_normal_lengths[valid_projection_normals, None]
    )
    projection_normal_signed = np.einsum(
        'ij,ij->i', panel_projection_residual, projection_unit_normals
    )
    panel_normal_error = np.abs(projection_normal_signed)
    panel_tangent_error = np.linalg.norm(
        panel_projection_residual - projection_normal_signed[:, None] * projection_unit_normals,
        axis=1,
    )

    print("\nPanel-center projection errors (geometry units):")
    print(f"  Panels: {len(panel_projection_error)}; "
        f"undefined surface normals: {np.count_nonzero(~valid_projection_normals)}")
    for error_label, error_values in (
        ("Total", panel_projection_error),
        ("Normal magnitude", panel_normal_error),
        ("Tangential magnitude", panel_tangent_error),
    ):
        finite_errors = error_values[np.isfinite(error_values)]
        if finite_errors.size:
            print(f"  {error_label:20s}: mean={finite_errors.mean():.6e}, "
                f"RMS={np.sqrt(np.mean(finite_errors**2)):.6e}, "
                f"median={np.median(finite_errors):.6e}, "
                f"p95={np.percentile(finite_errors, 95):.6e}, "
                f"max={finite_errors.max():.6e} "
                f"(nonfinite={error_values.size - finite_errors.size})")
        else:
            print(f"  {error_label}: no finite values")

    print("  Worst 10 panels by total distance (zero-based panel indices):")
    print("    panel   surface       u          v          total        normal       tangent")
    worst_projection_indices = np.argsort(
        np.where(np.isfinite(panel_projection_error), panel_projection_error, np.inf)
    )[::-1][:10]
    for panel_index in worst_projection_indices:
        surface_index, uv = projected_panel_centers[panel_index]
        uv = np.asarray(uv).ravel()
        print(f"    {panel_index:6d}  {surface_index!s:>8s}  "
            f"{uv[0]:9.5f}  {uv[1]:9.5f}  "
            f"{panel_projection_error[panel_index]:.6e}  "
            f"{panel_normal_error[panel_index]:.6e}  "
            f"{panel_tangent_error[panel_index]:.6e}")
    # Tangential residual need not vanish for projections constrained to patch edges.
    # Normal magnitudes avoid depending on each patch's normal orientation.

# endregion
# region ============================ structures mesh setup ============================
num_nonwing_nodes = 4
num_wing_nodes = 7

# LE and TE points
LE_nonwing_pts = np.array([
    [0., 0., 0.],
    [3.899, 2.5, 0.],
    [9.813, 5.0, 0.],
    [15.541, 7.934, 0.659],
])

TE_nonwing_pts = np.array([
    [30., 0., 0.],
    [30., 2.5, 0.],
    [30., 5.0, 0.],
    [24.88, 7.934, 0.691],
])

wing_root_LE_TE = [
    np.array([17.815, 9.891, 1.040]), # LE
    np.array([23.815, 9.891, 1.040]), # TE
]

wing_tip_LE_TE = [
    np.array([29.019, 25.852, 2.123]), # LE
    np.array([32.016, 25.852, 2.254]), # TE
]

LE_wing_pts = np.linspace(wing_root_LE_TE[0], wing_tip_LE_TE[0], num_wing_nodes)
TE_wing_pts = np.linspace(wing_root_LE_TE[1], wing_tip_LE_TE[1], num_wing_nodes)

LE_points = np.concatenate((LE_nonwing_pts, LE_wing_pts)) # we don't want the center line
TE_points = np.concatenate((TE_nonwing_pts, TE_wing_pts)) # we don't want the center line

LE_points_projected = geometry.project(LE_points, plot=False)
TE_points_projected = geometry.project(TE_points, plot=False)
# exit()
# airfoil top and bottom points
top_nonwing_pts = np.array([
    [8.973, 0, 2.251],
    [11.706, 2.5, 1.958],
    [15.851, 5., 1.514],
    [18.352, 7.934, 1.415],
])

bottom_nonwing_pts = np.array([
    [8.973, 0, -2.251],
    [11.706, 2.5, -1.958],
    [15.851, 5., -1.514],
    [18.398, 7.934, 0.078],
])

wing_root_top_bottom = [
    np.array([20.001, 9.891, 1.537]), # top
    np.array([20.040, 9.891, 0.699]), # bottom
]

tip_top_bottom = [
    np.array([30.102, 25.852, 2.384]), # top
    np.array([30.137, 25.852, 2.026]), # bottom
]

top_wing_pts = np.linspace(wing_root_top_bottom[0], tip_top_bottom[0], num_wing_nodes)
bottom_wing_pts = np.linspace(wing_root_top_bottom[1], tip_top_bottom[1], num_wing_nodes)

top_pts = np.concatenate((top_nonwing_pts, top_wing_pts))[1:,:] # we don't want the center line
bot_pts = np.concatenate((bottom_nonwing_pts, bottom_wing_pts))[1:,:] # we don't want the center line

top_pts_projected = geometry.declare_component([9, 10, 11]).project(top_pts, plot=False)
bot_pts_projected = geometry.declare_component([3, 4, 5]).project(bot_pts, plot=False)
# endregion

# region ============================ engine parameter setup ============================
# engine cg
# engine_weight_lbs = 8760 # lbs
# engine_weight_N = engine_weight_lbs*lbf_to_N
# engine_diam = 2.67 # meters
# engine_length = 4.24 # meters

'''
PW2000 engine parameters (used on C-17)
- D = 1.994 m
- L = 3.729 m
- dry mass = 3221 kg
- max thrust: 194.54e3 N
'''

# parameters for GE90-115, engine on the current Boeing 777s
engine_diam_0 = 3.43 # meters
engine_length_0 = 7.28 # meters
engine_mass_0 = 8762 # baseline engine mass
engine_T_max_0 = 3.86102524e+05 # baseline engine max thrust

engine_mass_frac = csdl.Variable(
    value=3221/engine_mass_0 # ratio between PW2000 andd GE90-115B
) # engine scaling parameter (scale between 0.1 and 1.5)
engine_mass_frac.add_name('engine_mass_frac')

engine_mass = engine_mass_frac*engine_mass_0 # scaled engine mass
engine_weight_N = engine_mass*9.81 # engine weight (N)
engine_max_thrust = engine_mass_frac*engine_T_max_0
engine_length = engine_length_0 * engine_mass_frac
engine_diam = engine_diam_0 * engine_mass_frac**0.5

engine_mass.add_name('engine_mass')
engine_max_thrust.add_name('engine_max_thrust')
engine_length.add_name('engine_length')
engine_diam.add_name('engine_diam')

TSFC_PW2000 = 0.582 # lb/lbf/hr at cruise
# At sea level, this number is around 0.33

# endregion

# region ============================ mass properties setup and point projections ============================
lbf_to_N = 4.44822
# mass properties setup
empty_weight_lbf = 126636 #lbf
empty_weight_N = empty_weight_lbf*lbf_to_N

# a guess on initial wing mass (acts as a deficit )
wing_mass_0 = 5000 # kg, reasonable initial guess
wing_weight_0 = wing_mass_0*9.81

empty_weight_N_nowing = empty_weight_N - wing_weight_0

# mass point projections
engine_sec_LE_0 = np.array([7.347, -4., 0.])
engine_sec_TE_0 = np.array([30., -4., 0.])
engine_loc_0 = np.array([24.702, -4., 0.824]) # 80% of center span, around 80% of chord
ecf_0 = (engine_loc_0[0]-engine_sec_LE_0[0])/(engine_sec_TE_0[0]-engine_sec_LE_0[0]) # engine chord fraction
# ecf_0 here is around 76.6% of the chord (just a starting point)
# ecf_0 = 0.8 # engine chord fraction


engine_loc_parametric = geometry.project(engine_loc_0, plot=False)
engine_sec_LE_parametric = geometry.project(engine_sec_LE_0, plot=False)
engine_sec_TE_parametric = geometry.project(engine_sec_TE_0, plot=False)


transition_volume_LE_points = np.array([
    [9.813, 5.0, 0.],
    [12.311, 6.223, 0.158],
    [14.707, 7.445, 0.510],
    [16.604, 8.668, 0.853],
    [17.815, 9.891, 1.040],
])

transition_volume_TE_points = np.array([
    [30., 5.0, 0.],
    [28.627, 6.223, 0.167],
    [25.813, 7.445, 0.545],
    [23.963, 8.668, 0.870],
    [23.815, 9.891, 1.040],
])

transition_volume_LE_points_proj = geometry.project(transition_volume_LE_points)
transition_volume_TE_points_proj = geometry.project(transition_volume_TE_points)

transition_volume_projection_points = project_transition_volume_points(
    geometry, 
    transition_volume_LE_points, 
    transition_volume_TE_points
)
# exit()
# endregion

# region ============================ mission parameters ===========================

# problem type and mission flags
problem_type = 'stability'
do_planform = False
do_stability = False
do_runway = False

# additional flags to control
# moment trim flags 
do_moment_trim = True # NOTE: MAKE SURE TO TURN ON AND OFF 
include_elevator = True

# stability flags
# stability_level = 'tight'
stability_level = 'relaxed'
do_airfoil_shape = False
do_airfoil_shape = True

if problem_type == 'planform':
    do_planform = True
elif problem_type == 'stability':
    do_planform = True
    do_stability = True
    do_runway = True

'''
NOTE: The problem types that have been tested so far 
are baseline and planform. We are still working on stability,
so it may not work out of the box.

Variable ordering:
cruise missions (9)
structural sizing missions (2)
stability condition (1)

EACH VARIABLE HAS THIS SHAPE
- this is so that it's easy to assess the full behavior
- for the smaller-scale optimization problems, we can just use
    the above flags to turn DVs and constraints on or off

'''
nmi_to_m = 1852.

# 9 missions ====
payload_weight_lbf = np.array([[160000]*3+[100000]*3+[50000]*3]).flatten() # lbf
cruise_range_nmi = np.array([2500, 2000, 1500]*3).flatten() # nmi
payload_weighting = np.array([0.25, 0.5, 0.25])
range_weighting = np.array([0.25, 0.5, 0.25])
nominal_ind = 4 # NOMINAL MISSION INDEX --> medium payload, medium range

if False: # testing number of missions
    ncm = 81
    payload_weight_lbf = np.array([160000]*ncm).flatten() # lbf
    cruise_range_nmi = np.array([2500]*ncm).flatten() # nmi
    payload_weighting = np.array([1.]*ncm)
    range_weighting = np.array([1.])
    nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

# 1 mission ====
payload_weight_lbf = np.array([100000]) # lbf
cruise_range_nmi = np.array([2000]) # nmi
payload_weighting = np.array([1.])
range_weighting = np.array([1.])
nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

# payload_weight_lbf = np.array([120000]) # lbf
# cruise_range_nmi = np.array([2200]) # nmi
# payload_weighting = np.array([1.])
# range_weighting = np.array([1.])
# nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

# payload_weight_lbf = np.array([140000]) # lbf
# cruise_range_nmi = np.array([2300]) # nmi
# payload_weighting = np.array([1.])
# range_weighting = np.array([1.])
# nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

# payload_weight_lbf = np.array([160000]) # lbf
# cruise_range_nmi = np.array([2400]) # nmi
# payload_weighting = np.array([1.])
# range_weighting = np.array([1.])
# nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

payload_weight_lbf = np.array([210000]) # lbf
cruise_range_nmi = np.array([2000]) # nmi
payload_weighting = np.array([1.])
range_weighting = np.array([1.])
nominal_ind = 0 # NOMINAL MISSION INDEX --> medium payload, medium range

payload_weight_N = payload_weight_lbf*lbf_to_N
cruise_range_m = cruise_range_nmi*nmi_to_m
weighting_grid = np.einsum('i,j->ij', payload_weighting, range_weighting)
weighting_array = weighting_grid.flatten()

# cruise mission parameters
cruise_mach = 0.75
cruise_h = 30000 # altitude in feet
cruise_h_km = cruise_h*0.3048/1000

# structural sizing mission parameters (SEA LEVEL)
SS_mach = 0.75
V_inf_SS = 340.3*SS_mach
rho_SS = 1.225

# number of missions
num_cruise = len(weighting_array)
num_sizing = 1
num_nodes = num_cruise+num_sizing

# per mission type counter
up2_cruise  = num_cruise
up2_struct  = up2_cruise+num_sizing

# mach_array = np.concatenate(
    # (
        # np.array([cruise_mach]*num_cruise), 
        # np.array([SS_mach]*num_sizing)
    # )
# )
mach_array = np.array([cruise_mach]*num_cruise + [SS_mach]*num_sizing)

# stability condition
# if do_stability:
dalpha_stab = 0.1 # for stability analysis
stab_ind = num_cruise+num_sizing
num_stab = 1
num_nodes += num_stab
up2_stab    = up2_struct+num_stab

mach_array = np.concatenate((mach_array, np.array([cruise_mach])*num_stab))

mach_array_np = mach_array


'''
NOTE:
- assume cruise uses 75% of fuel fraction (to be conservative)
- original fuel weight
'''

# endregion

# region ============================ geometry parameterization ===========================

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


# These are the active geometry variables from tcp0_beam_post_workshop.py.
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


# Keep the names aligned with tcp0_beam_post_workshop.py.
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




centerbody_volume_LE_points = np.array([
    [0., 0., 0.],
    [1.433, 1.25, 0.],
    [3.899, 2.5, 0.],
    [6.748, 3.75, 0.],
    [9.813, 5., 0.],
])
centerbody_volume_TE_points = np.array([
    [30., 0., 0.],
    [30., 1.25, 0.],
    [30., 2.5, 0.],
    [30., 3.75, 0.],
    [30., 5., 0.],
])

volume_projection_points, _ = project_centerbody_volume_points(
    geometry, 
    centerbody_volume_LE_points, 
    centerbody_volume_TE_points
) # don't need this second output


deformed_geometry = setup_and_evaluate_geometry_parameterization(
    geometry, geometric_variables, plot=False,
)

# Prevent a forward-turning transition LE between centerbody and wing root.
# Post-workshop right transition patches: lower=4, upper=10; u=0 is LE,
# v runs outward from centerbody tip to wing root. Average the two skins.
# This is a sampled monotonicity condition, not a continuous-curve guarantee.
transition_le_num_samples = 5
transition_le_uv = np.column_stack((
    np.zeros(transition_le_num_samples),
    np.linspace(0., 1., transition_le_num_samples),
))
transition_le_upper = deformed_geometry.functions[10].evaluate(transition_le_uv)
transition_le_lower = deformed_geometry.functions[4].evaluate(transition_le_uv)
transition_le_x = (transition_le_upper[:, 0] + transition_le_lower[:, 0]) / 2.
transition_le_dx = transition_le_x[1:] - transition_le_x[:-1]
transition_le_dx.add_name('constraint_transition_le_monotonicity')

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

# endregion

centerbody_volume = 2*compute_volume(deformed_geometry, volume_projection_points)
centerbody_volume_0 = centerbody_volume.value
print('initial centerbody volume (m^3): ', centerbody_volume_0)

geometry_coefficients = deformed_geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')
# geometry_coefficients.save()

# endregion

# region ============================ aero solver setup ============================
@dataclass
class BWBPanelSolverParameters:
    panel_mesh: csdl.Variable = None
    panel_center: csdl.Variable = None
    panel_centers_p: list = None
    upper_TE_cells: np.ndarray = None
    lower_TE_cells: np.ndarray = None
    TE_edges: np.ndarray = None
    TE_node_indices: np.ndarray = None
    triangles: np.ndarray = None
    cell_adjacency: np.ndarray = None
    points2cells: np.ndarray = None
    edges2cells: np.ndarray = None
    pressure_fs: lfs.FunctionSetSpace = None
    left_panel_inds: np.ndarray = None

params = BWBPanelSolverParameters()

# storing mesh and connectivity info
params.triangles = combined_cells
params.cell_adjacency = cell_adjacency
params.points2cells = points2cells
params.edges2cells = edges2cells

# storing TE information
params.upper_TE_cells = upper_TE_cells
params.lower_TE_cells = lower_TE_cells
params.TE_edges = TE_edges
params.TE_node_indices = TE_node_indices

left_panel_inds = np.where(panel_centers[:,1] < 0)[0]
params.left_panel_inds = left_panel_inds

# aero mesh
panel_mesh_deformed = deformed_geometry.evaluate(projected_panel_mesh, plot=False)
# exit()
panel_mesh = panel_mesh_deformed.expand((1,) + panel_mesh_deformed.shape, 'ij->aij')
panel_mesh.add_name('panel_mesh')
# panel_mesh.save()
panel_center = deformed_geometry.evaluate(projected_panel_centers, plot=False)

params.panel_mesh = panel_mesh
params.panel_center = panel_center
params.panel_centers_p = projected_panel_centers

# plotting mesh for debugging the geometry --> need recorder.inline = True
if False:
    TE_coloring = np.zeros(shape=(len(combined_cells),))
    TE_coloring[upper_TE_cells] = 1
    TE_coloring[lower_TE_cells] = -1

    plot_pressure_distribution(panel_mesh.value, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')
    exit()

nominal_zero_pitch = False # says that the pitch angle at nominal cruise is 0
if nominal_zero_pitch:
    # pitch = csdl.Variable(shape=(num_cruise+num_sizing,), value=0.) # DV
    num_pitch_dv = num_cruise+num_sizing-1 # removing nominal here bc we set nominal to 0
    pitch_dv = csdl.Variable(shape=(num_pitch_dv,), value=0.) # DV

    # inserting nominal value
    pitch = csdl.Variable(shape=(num_pitch_dv+1,), value=0.)
    pitch = pitch.set(csdl.slice[:nominal_ind], pitch_dv[:nominal_ind])
    pitch = pitch.set(csdl.slice[nominal_ind+1:], pitch_dv[nominal_ind:])
    pitch.add_name('pitch')
else:
    num_pitch_dv = num_cruise+num_sizing

    pitch_cruise_dv = csdl.Variable(shape=(num_cruise,), value=0., name='pitch_cruise_dv')
    pitch_sizing_dv = csdl.Variable(shape=(num_sizing,), value=0., name='pitch_sizing_dv')
    # Preserve the original ordering for downstream calculations and saved outputs.
    pitch_dv = csdl.concatenate((pitch_cruise_dv, pitch_sizing_dv))

    pitch = pitch_dv
    pitch.add_name('pitch')

# atmosphere model

altitude = csdl.Variable(value=np.zeros((num_nodes,)))
altitude = altitude.set(csdl.slice[:up2_cruise], value=cruise_h_km) # cruise altitude
altitude = altitude.set(csdl.slice[up2_cruise:up2_struct], value=0.) # sizing @ sea level

# if do_stability:
altitude = altitude.set(csdl.slice[up2_struct:up2_stab], value=cruise_h_km) # stability
    
mach = csdl.Variable(value=mach_array_np)

rho_array, sos_array, mu_array = atmos_model(altitude)
# print(f'density (kg/m^3): {rho_array.value}')
# print(f'speed of sound (m/s): {sos_array.value}')
# print(f'dynamic viscosity (Pa/s): {mu_array.value}')
V_inf_array = sos_array*mach

# ============================================================
# Structural sizing condition:
# Match dynamic pressure to nominal cruise while remaining
# at sea level.
# ============================================================

# Nominal cruise dynamic pressure
q_cruise = (
    0.5
    * rho_array[nominal_ind]
    * V_inf_array[nominal_ind]**2
)

# Sea-level sizing velocity that gives the same dynamic pressure
V_SS_same_q = csdl.sqrt(
    2.0 * q_cruise
    / rho_array[up2_cruise:up2_struct]
)

# Corresponding sea-level Mach number
M_SS_same_q = (
    V_SS_same_q
    / sos_array[up2_cruise:up2_struct]
)

# Replace only the structural-sizing Mach
mach = mach.set(
    csdl.slice[up2_cruise:up2_struct],
    M_SS_same_q
)

# Recompute all velocities consistently with the updated Mach
V_inf_array = sos_array * mach
# ============================================================
# Structural sizing condition:
# Match dynamic pressure to nominal cruise while remaining
# at sea level.
# ============================================================

V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
V_vec = V_vec.set(csdl.slice[:,0], value=-V_inf_array)

# aoa_array = pitch - gamma
aoa_array = csdl.Variable(value=np.zeros(rho_array.shape))
aoa_array = aoa_array.set(csdl.slice[:up2_struct], pitch[:up2_struct])
# if do_stability:
aoa_array = aoa_array.set(csdl.slice[stab_ind], pitch[nominal_ind]+dalpha_stab)



aoa_rad = aoa_array * np.pi/180.
# TO-DO: REMOVE AND INSERT V_INF AND PITCH INTO PANEL METHOD
V_rot_mat = csdl.Variable(value=0., shape=(num_nodes,3,3))
V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(aoa_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(aoa_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(aoa_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(aoa_rad))

V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
point_velocities = csdl.expand(V_vec_rot, (num_nodes,) + panel_mesh.shape[1:], 'ij->iaj')

# endregion

# region ============================ structures solver setup ============================
# structural mesh elements and nodes
num_half_nodes = num_nonwing_nodes + num_wing_nodes
num_half_el = num_half_nodes-1
num_beam_nodes = num_half_nodes*2 - 1
num_beam_el = num_beam_nodes-1

half_beam_mesh = (deformed_geometry.evaluate(LE_points_projected) + (deformed_geometry.evaluate(TE_points_projected)))/2.
width_fraction = 0.45
height_fraction = 0.65
projected_height = height_fraction*csdl.norm(
    deformed_geometry.evaluate(top_pts_projected) - deformed_geometry.evaluate(bot_pts_projected),
    axes=(1,)
)
projected_width = width_fraction*csdl.norm(
    deformed_geometry.evaluate(LE_points_projected)[1:,:] - deformed_geometry.evaluate(TE_points_projected)[1:,:],
    axes=(1,)
)

# beam mesh
beam_mesh = csdl.Variable(value=np.zeros((num_beam_nodes,3)))
beam_mesh = beam_mesh.set(csdl.slice[num_half_nodes-1:,:], value=half_beam_mesh)
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,0::2], value=half_beam_mesh[1:,0::2][::-1,:])
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,1], value=-half_beam_mesh[1:,1][::-1])

# beam thicknesses
thickness_0 = 0.001
half_ttop = np.ones((num_half_el,)) * thickness_0
half_ttop = csdl.Variable(value=half_ttop)
half_ttop.add_name('half_ttop')

ttop = csdl.Variable(value=np.zeros(num_beam_el)) # we assume symmetry on the top and bottom
ttop = ttop.set(csdl.slice[:num_half_el], value=half_ttop[::-1])
ttop = ttop.set(csdl.slice[num_half_el:], value=half_ttop)

half_tweb = np.ones((num_half_el,)) * thickness_0
half_tweb = csdl.Variable(value=half_tweb)
half_tweb.add_name('half_tweb')

tweb = csdl.Variable(value=np.zeros(num_beam_el))
tweb = tweb.set(csdl.slice[:num_half_el], value=half_tweb[::-1])
tweb = tweb.set(csdl.slice[num_half_el:], value=half_tweb)

# beam height
half_height = np.zeros((num_half_el,))
half_height = csdl.Variable(value=half_height)

height = csdl.Variable(value=np.zeros(num_beam_el))
height = height.set(csdl.slice[:num_half_el], value=projected_height[::-1])
height = height.set(csdl.slice[num_half_el:], value=projected_height)

height.add_name('beam_height')
# height.save()

# beam width
half_width = np.zeros((num_half_el,))
half_width = csdl.Variable(value=half_width)

width = csdl.Variable(value=np.zeros(num_beam_el))
width = width.set(csdl.slice[:num_half_el], value=projected_width[::-1])
width = width.set(csdl.slice[num_half_el:], value=projected_width)

width.add_name('beam_width')
# width.save()

# beam cross section
beam_CS = af.CSBox(ttop=ttop, tbot=ttop, tweb=tweb, height=height, width=width)

# beam material (NO LONGER USED)
# aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# beam OBJECT
BWB_beam = af.Beam(
    name='BWB_beam',
    mesh=beam_mesh,
    E=69E9,
    G=26E9,
    density=2700,
    # material=aluminum,
    cs=beam_CS
)
BWB_beam_mass = BWB_beam.mass # property of beam
BWB_beam_mass.add_name('structural_mass')
# BWB_beam_mass.save()
structural_weight = BWB_beam_mass*9.81
structural_weight.add_name('structural_weight')

cg_beam = BWB_beam.cg
cg_beam.add_name('wing_cg')

# force map from aero to beam
mapper = NodalMap(normalization_eps=1.e-6)
force_map = mapper.evaluate(panel_center, beam_mesh.reshape((-1,3)))

# endregion

# region ============================ mass properties ============================
# payload cg (same as centerbody centroid)
LE_points_CB_TR = deformed_geometry.evaluate(LE_points_projected[:5])

LE_centerline = LE_points_CB_TR[0]
TE_centerline = deformed_geometry.evaluate(TE_points_projected[0])
payload_cg = (LE_centerline+TE_centerline)/2

payload_cg.add_name('payload_cg')
# payload_cg.save()

engine_cg_evaluated = deformed_geometry.evaluate(engine_loc_parametric)
engine_sec_LE = deformed_geometry.evaluate(engine_sec_LE_parametric)
engine_sec_TE = deformed_geometry.evaluate(engine_sec_TE_parametric)

ecf = csdl.Variable(value=np.array([ecf_0])) # engine position chord fraction DV
ecf.add_name('engine_chord_fraction')
engine_x_pos = engine_sec_TE[0]*ecf + engine_sec_LE[0]*(1-ecf)

engine_cg = csdl.Variable(value=engine_loc_0)
engine_cg = engine_cg.set(csdl.slice[0], engine_x_pos)
engine_cg = engine_cg.set(csdl.slice[1:], engine_cg_evaluated[1:])

engine_cg.add_name('engine_cg_neg')
# engine_cg.save()

engine_cg_mirror = csdl.Variable(value=engine_loc_0)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[0], engine_x_pos)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[1], -engine_cg_evaluated[1])
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[2], engine_cg_evaluated[2])

engine_cg_mirror.add_name('engine_cg_pos')
# engine_cg_mirror.save()

# fuel cg (same as transition centroid)
# transition_volume = 2*compute_volume(geometry, transition_volume_projection_points)
transition_volume = 2*compute_volume(deformed_geometry, transition_volume_projection_points)
transition_volume.add_name('transition_volume')
# transition_volume.save()

transition_LE = deformed_geometry.evaluate(transition_volume_LE_points_proj)
transition_TE = deformed_geometry.evaluate(transition_volume_TE_points_proj)

transition_avg = (transition_LE + transition_TE)/2
transition_cg = csdl.average(transition_avg, axes=(0,))

transition_cg.add_name('fuel_cg_pos')
# transition_cg.save()

transition_cg_mirror = csdl.Variable(value=np.zeros(transition_cg.shape))
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[:], transition_cg)
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[1], -transition_cg[1])

transition_cg_mirror.add_name('fuel_cg_neg')
# transition_cg_mirror.save()

fuel_cg = (transition_cg + transition_cg_mirror)/2. # dumb to do it this way but it's exact

# endregion

recorder.inline = not shutdown_inline

# region ============================ mission analysis ============================

'''
Computations for conditions:
- cruise conditions need drag approximations to measure L/D
- stability one only needs static margin
- structural conditions need none of these
'''
# panel_method

# Pass ONLY structural sizing conditions to the panel solver and drag helper.
# With num_sizing=1 this is one panel solve. The shared geometry has one copy.
pm_solver_inputs = {
    'V_inf': point_velocities[up2_cruise:up2_struct],
    'rho': rho_array[up2_cruise:up2_struct],
    'sos': sos_array[up2_cruise:up2_struct],
    'compressibility': True,
    'Cp cutoff': -5.,
    'partition_size': 1,
    'reuse_AIC': True,
    # 'mesh_path': file_path+file_name, # already done externally
    'ref_area': planform_area, # does not matter bc we don't use the coefficients,
    'ref_chord': MAC,
}


# panel_forces, D_total, L, M = panel()

# panel_forces = (1, )
# D_total, L, M = (3, )

# vectorization:
# num_cruise, num_sizing, num_stab,

panel_method_inputs = {
    'pm_solver_inputs': pm_solver_inputs,
    'mesh_data': {
        'panel_mesh': panel_mesh,
        'cell_adjacency_data': cell_adjacency_data,
        'TE_properties': TE_properties,
    },
    'flight_conditions': {
        'rho': rho_array[up2_cruise:up2_struct],
        'mu': mu_array[up2_cruise:up2_struct],
        'V_inf': V_inf_array[up2_cruise:up2_struct],
        'mach': mach[up2_cruise:up2_struct],
    },
    'reference_geometry': {
        'planform_area': planform_area,
        'MAC': MAC,
        'wing_sweep': wing_sweep,
        'LE_points_CB_TR': LE_points_CB_TR,
    },
    'strip_data': {
        'sectional_avg_chord': ffd_sectional_avg_chord_dist,
        'sectional_span': ffd_sectional_span_dist,
        'num_centerbody_stations':
            num_centerbody_spanwise_design_dofs,
        'num_wing_spanwise_design_dofs':num_wing_spanwise_design_dofs,
    },
    'mission_data': {
        'num_cruise': 0,  # local panel batch contains only sizing conditions
        'num_sizing': num_sizing,
    },
}

from panel_method_function import compute_panel_method
panel_outputs = compute_panel_method(panel_method_inputs)
panel_forces = panel_outputs['panel_forces']
# Keep global mission indexing for the GNN and downstream mission analysis.
# Only the sizing slots come from the panel method; the GNN fills the others.
D_total = csdl.Variable(shape=(num_nodes,), value=0.).set(
    csdl.slice[up2_cruise:up2_struct], panel_outputs['D_total'])
L = csdl.Variable(shape=(num_nodes,), value=0.).set(
    csdl.slice[up2_cruise:up2_struct], panel_outputs['L'])
M = csdl.Variable(shape=(num_nodes,), value=0.).set(
    csdl.slice[up2_cruise:up2_struct], panel_outputs['M'])

print(panel_forces.shape)
print(D_total.shape)
print(L.shape)
print(M.shape)

# panel_forces, D_total, L, M = panel()

# panel_forces = (1, )
# D_total, L, M = (3, )

# vectorization:
# num_cruise, num_sizing, num_stab,

# def get_condition(array, condition):
#     if condition 
#     return []

# ==================================== indices ====================================
CRUISE_INDICES = list(range(0, num_cruise))
SIZING_INDICES = list(range(num_cruise, num_cruise + num_sizing))
STABILITY_INDICES = list(range(num_cruise + num_sizing, num_cruise + num_sizing + num_stab))
# .............................................................................
# ==================================== GNN ====================================
from gnn_function import compute_rans_gnn, evaluate_cfd_surface, compute_panel
if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'RANS-GNN'):
    cfd_surface = evaluate_cfd_surface(deformed_geometry)
if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'Panel'):
    panel_surf_gnn = panel_mesh_deformed
else:
    panel_surf_gnn = None

gnn_pitches = csdl.concatenate((pitch_dv[CRUISE_INDICES], pitch_dv[CRUISE_INDICES] + dalpha_stab))
gnn_condition_indices = CRUISE_INDICES + STABILITY_INDICES
# print(gnn_pitches.shape)
# exit()
for i, index in zip(range(len(gnn_condition_indices)), gnn_condition_indices):
    pitch_gnn = gnn_pitches[i]
    if USE_MULTIFIDELITY in ('PG-RANS-GNN', 'RANS-GNN'):
        gnn_outputs = compute_rans_gnn(cfd_surface, pitch_gnn.flatten(), panel_surface_var = panel_surf_gnn)
    elif USE_MULTIFIDELITY in ('Panel',):
        gnn_outputs = compute_panel(pitch_gnn.flatten(), panel_surf_gnn)

    L = L.set(csdl.slice[index], gnn_outputs['L'])
    D_total = D_total.set(csdl.slice[index], gnn_outputs['D'])
    M = M.set(csdl.slice[index], gnn_outputs['M'])
# ==================================== GNN ====================================
# .............................................................................

CM = M/(0.5*rho_array*V_inf_array**2*planform_area*MAC)
CL = L/(0.5*rho_array*V_inf_array**2*planform_area)
L.add_name('L')
D_total.add_name('D_total')
CL.add_name('CL')
CM.add_name('CM')

# structural sizing missions (beam)
beam_forces = csdl.Variable(value=np.zeros((num_sizing,num_beam_nodes,3)))
beam_forces = beam_forces.set(csdl.slice[0,:], force_map.T() @ panel_forces[0,:])
# beam_forces = beam_forces.set(csdl.slice[1,:], force_map.T() @ panel_forces[1,:])

beam_loads = csdl.Variable(value=np.zeros((num_sizing,num_beam_nodes, 6)))
beam_loads = beam_loads.set(csdl.slice[:,:,:3], beam_forces)
BWB_beam.fix(node=num_half_nodes-1)

# S1
safety_factor = 1.5
BWB_beam.add_load(beam_loads[0,:])
frame_S1 = af.Frame(beams=[BWB_beam])
frame_S1.solve()
stress_dict_S1 = frame_S1.compute_stress()
beam_stress_S1 = stress_dict_S1[BWB_beam.name]
beam_stress_S1_MPa = beam_stress_S1/1.e6
beam_max_stress_S1_MPa = csdl.maximum(beam_stress_S1_MPa, rho=1e5) # NOTE: TUNE (seems to work)
beam_max_stress_S1 = beam_max_stress_S1_MPa*1.e6
beam_max_stress_S1_SF = beam_max_stress_S1*safety_factor



# fuel burn + takeoff weight calculations
V_cruise_array = V_inf_array[:num_cruise]

L_cruise = L[:num_cruise]
D_cruise = D_total[:num_cruise]
L_D = L_cruise/D_cruise
L_D.add_name('L_D')
# L_D.save()
# TSFC = 0.355 # lb/lbf/hr
TSFC = TSFC_PW2000 # lb/lbf/hr
# W2 = empty_weight_no_wing_engine_N+wing_weight_N+payload_weight_N
# Wf, TOGW = estimate_fuel_burn(cruise_range_m, TSFC, V_cruise_array, L_D, W2)
W_empty = empty_weight_N_nowing + structural_weight # 2x factor on structural weight accounts for others
W_empty.add_name('W_empty')
# W_bar = empty_weight_no_wing_engine_N+structural_weight+payload_weight_N+engine_weight_N*2 # weight w/o fuel
W_bar = W_empty + payload_weight_N # weight w/o fuel
W_bar.add_name('W_no_fuel')
# W_bar.save()
Wf, TOGW, W2 = estimate_fuel_burn_w_reserve(cruise_range_m, TSFC, V_cruise_array, L_D, W_bar)
Wf.add_name('Wf')
# Wf.save()
TOGW.add_name('TOGW')
# TOGW.save()
W2.add_name('W2')
# W2.save()

# reserve fuel of 20%
# fuel_volume = estimate_fuel_volume(Wf)
cruise_fuel_volume = estimate_fuel_volume(Wf)
fuel_volume = 1.2*cruise_fuel_volume # 1.2 scaler for the 20% margin in fuel burn
fuel_volume.add_name('fuel_volume')
# fuel_volume.save()
max_fuel_volume = csdl.maximum(fuel_volume, rho=1000)
max_fuel_volume.add_name('max_fuel_volume')
# max_fuel_volume.save()

weighted_fuel_burn_array = Wf * weighting_array
weighted_fuel_burn = csdl.sum(weighted_fuel_burn_array)


# cg computation (accounting for fuel weight)
TOGW_nn = csdl.Variable(shape=(num_nodes,), value=0.)
TOGW_nn = TOGW_nn.set(csdl.slice[:num_cruise], value=TOGW)
TOGW_nn = TOGW_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=TOGW[0])
# if do_stability:
TOGW_nn = TOGW_nn.set(csdl.slice[stab_ind], value=TOGW[nominal_ind])

Wf_nn = csdl.Variable(shape=(num_nodes,), value=0.)
Wf_nn = Wf_nn.set(csdl.slice[:num_cruise], value=Wf)
Wf_nn = Wf_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=Wf[0])
# if do_stability:
Wf_nn = Wf_nn.set(csdl.slice[stab_ind], value=Wf[nominal_ind])

payload_weight_nn = csdl.Variable(shape=(num_nodes,), value=0.)
payload_weight_nn = payload_weight_nn.set(csdl.slice[:num_cruise], value=payload_weight_N)
payload_weight_nn = payload_weight_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=payload_weight_N[0])
# if do_stability:
payload_weight_nn = payload_weight_nn.set(csdl.slice[stab_ind], value=payload_weight_N[nominal_ind])

add_weight = TOGW_nn - structural_weight - 2*engine_weight_N - 1.2*Wf_nn - payload_weight_nn # computing remaining weight at CG

target_shape = (num_nodes, 3)

cg_W_prod_wing = csdl.expand(structural_weight*cg_beam, target_shape, 'i->ai')
cg_W_prod_engine = csdl.expand(
    engine_weight_N*(engine_cg+engine_cg_mirror),
    target_shape,
    'i->ai'
)
cg_W_prod_fuel = csdl.expand(Wf_nn, target_shape, 'i->ia')/2. * csdl.expand(
    transition_cg+transition_cg_mirror,
    target_shape,
    'i->ai'
)
cg_W_prod_payload = csdl.expand(payload_weight_nn, target_shape, 'i->ia') * csdl.expand(
    payload_cg,
    target_shape,
    'i->ai'
)
cg_W_prod = cg_W_prod_wing+cg_W_prod_engine+cg_W_prod_fuel+cg_W_prod_payload

# cg_W_prod = wing_weight_N*cg_beam + engine_weight_N*(engine_cg+engine_cg_mirror) + \
#             Wf/2*(transition_cg+transition_cg_mirror) + payload_weight_N*payload_cg

TOGW_exp = csdl.expand(TOGW_nn, target_shape, 'i->ia')
add_weight_exp = csdl.expand(add_weight, target_shape, 'i->ia')

cg = (cg_W_prod) / (TOGW_exp-add_weight_exp) # add_weight assumed to be at CG
cg.add_name('cg')
# cg.save()
cg_x = cg[:,0]

# force trim:
cruise_trim = L_cruise-TOGW

MTOW = TOGW[0]
L_SS = L[up2_cruise:up2_struct]
load_factors = csdl.Variable(value=np.array([2.5]))
SS_trim = L_SS - load_factors*MTOW


# moment trim
aero_ref_pt = 0.
# CM_cg_cruise = CM[nominal_ind] + CL[nominal_ind]*(aero_ref_pt-cg_beam[0])/MAC
# CM_cg = CM + CL*(aero_ref_pt-cg_x)/MAC # WRONG
CM_cg = CM + CL*(cg_x-aero_ref_pt)/MAC
CM_cg.add_name('CM_cg')
# CM_cg.save()


CM_cg_cruise_nominal = CM_cg[nominal_ind]

# takeoff and landing (constrain both to be less than 10000 feet)
# max_thrust_lbf = 48000*2 # lb
# max_thrust_N = max_thrust_lbf*lbf_to_N
max_thrust_N = engine_max_thrust*2 # from the engine scaling model
max_weight_TO = TOGW[0] # using largest weight for takeoff and landing
max_weight_LA = W2[0] # using largest weight for takeoff and landing
takeoff_length = takeoff(max_weight_TO, max_thrust_N, planform_area) # m # used W2[0] during optimizations on accident
landing_length = landing(max_weight_LA, planform_area) # m

takeoff_length_ft = takeoff_length/.3048
landing_length_ft = landing_length/.3048

takeoff_landing_ft = csdl.Variable(value=0., shape=(2,))
takeoff_landing_ft = takeoff_landing_ft.set(csdl.slice[0], takeoff_length_ft)
takeoff_landing_ft = takeoff_landing_ft.set(csdl.slice[1], landing_length_ft)

# static margin
# if do_stability: # NOTE: UPDATE FOR NOMINAL INDEX
alpha_list = [aoa_array[nominal_ind], aoa_array[stab_ind]] # pitch array is different from pitch dv
CL_list = [CL[nominal_ind], CL[stab_ind]]
CM_list = [CM_cg[nominal_ind], CM_cg[stab_ind]] # y-component taken above

static_margin = compute_static_margin(alpha_list, CL_list, CM_list)

neutral_point = static_margin*MAC + cg[nominal_ind,0] # neutral point based on nominal condition
neutral_point.add_name('neutral_point')
# neutral_point.save()
SM_missions = (neutral_point-cg[:,0])/MAC
SM_missions.add_name('static_margin_missions')
# SM_missions.save()
# endregion

# region ============================ DVs ============================
if nominal_zero_pitch:
    pitch_dv.set_as_design_variable(lower=-5, upper=4.5)
else:
    pitch_cruise_dv.set_as_design_variable(lower=-5, upper=4.5)
    pitch_sizing_dv.set_as_design_variable(lower=-5, upper=10.0)
pitch_dv.add_name('pitch_dv')

half_ttop.set_as_design_variable(lower=0.001, upper=0.1, scaler=100)
half_tweb.set_as_design_variable(lower=0.001, upper=0.1, scaler=100)

# if do_moment_trim:
if include_elevator:
    cruise_trim_elevator_deflection.set_as_design_variable(lower=0.0, upper=7.0)

if do_planform:
    # centerbody_twist_dist.set_as_design_variable(lower=-5, upper=5)
    # centerbody_twist_dist.set_as_design_variable(lower=-1, upper=1)
    # wing_twist_dist.set_as_design_variable(lower=-15, upper=5)
    wing_twist_coefficients.set_as_design_variable(lower=-5, upper=5)

    centerbody_half_span.set_as_design_variable(lower=3, upper=7) # orig: 5 --> 04/13, changed upper limit from 8 to 9 before running SM=10% case
    transition_half_span.set_as_design_variable(lower=3.5, upper=7) # orig: 5
    wing_half_span.set_as_design_variable(lower=8, upper=20)# orig: 16

    # Same bounds as combined_sweep_ranges in sweep_geometry_jax.py.
    centerbody_chord_stretch_ratio.set_as_design_variable(
        lower=np.array([0.9, 0.9, 0.8, 0.8]),
        # upper=np.array([1.3, 1.10, 0.98, 0.98]),
        upper=np.array([1.10, 0.97, 0.97, 0.94])-0.01,
    )
    wing_chord_stretch_coefficients.set_as_design_variable(
        lower=np.array([-3.0000005, -2.49952453, -1.99904855, -1.49857258]),
        upper=np.array([3.0000005, 2.49952453, 1.99904855, 1.49857258]),
    )
    wing_sweep.set_as_design_variable(lower=15., upper=42., scaler=0.1)
    wing_position.set_as_design_variable(
        lower=20.186592500296157 - 7., upper=20.186592500296157 + 2.,
    )

if do_stability:
    # ecf.set_as_design_variable(lower=0.6, upper=0.85) # engine chord fraction
    
    if do_airfoil_shape:
        # thickness_percent_change.set_as_design_variable(lower=-10, upper=15)
        # camber_percent_change.set_as_design_variable(lower=-20, upper=10)

        # airfoil_thickness_coefficients.set_as_design_variable(lower=-0.05, upper=0.05, scaler=1e1)
        # airfoil_camber_coefficients.set_as_design_variable(lower=-0.03, upper=0.03, scaler=1e2)

        airfoil_thickness_coefficients.set_as_design_variable(lower=-0.04, upper=0.04, scaler=1e1)
        airfoil_camber_coefficients.set_as_design_variable(lower=-0.02, upper=0.02, scaler=1e2)

# endregion

# region ============================ constraints ============================

# X increases aft: prevent the transition LE from turning forward toward
# the wing root. Coupled through the evaluated geometry to chord ratios,
# wing_position, and the other geometry inputs.
if do_planform:
    transition_le_dx.set_as_constraint(lower=0.)
    transition_le_dx.save()

# ==== trim constraints ====
# cruise trim
cruise_trim.set_as_constraint(equals=0., scaler=1.e-6)
cruise_trim.add_name('constraint_cruise_trim')

CM_cg_cruise_nominal.add_name('constraint_cruise_cg_CM_nominal_trim')
if do_moment_trim:
    # CM_cg_cruise_nominal.set_as_constraint(equals=0.)
    # CM_constraint = -0.2
    CM_constraint = 0.
    CM_cg_cruise_nominal.set_as_constraint(equals=CM_constraint)

# structural sizing trim
# use MTOW of heaviest payload and largest range
SS_trim.set_as_constraint(equals=0., scaler=1.e-6)
SS_trim.add_name('constraint_structural_sizing_trim')

# mass stress constraints
max_allowable_stress = 324e6 # 
beam_max_stress_S1_SF.set_as_constraint(upper=max_allowable_stress, scaler=1.e-9)
beam_max_stress_S1_SF.add_name('constraint_S1_max_stress')

# MSC_S1 = SCV_max_S1*1.
# MSC_S1.set_as_constraint(upper=0., scaler=1.e3)
# MSC_S1.add_name('MSC_S1')
# 
# MSC_S2 = SCV_max_S2*1.
# MSC_S2.set_as_constraint(upper=0., scaler=1.e3)
# MSC_S2.add_name('MSC_S2')


if problem_type != 'baseline':
    # # ==== volume constraints ====
    # C_17_cargo_volume = 558.36 # m^3
    C_17_cargo_volume = 591.8 # m^3
    C_17_fuselage_volume = 1958.9 # m^2
    volume_frac = 0.5
    min_external_volume = C_17_cargo_volume/volume_frac

    # centerbody_volume.set_as_constraint(lower=C_17_cargo_volume, scaler=1e-2)
    centerbody_volume.set_as_constraint(lower=min_external_volume, scaler=1e-3)
    centerbody_volume.add_name('constraint_centerbody_volume')

    tvSF = 0.5 # fuel volume excess factor (fuel tank volume is a fraction of the transition volume)
    fuel_storage_volume = transition_volume*tvSF
    fuel_volume_constraint = fuel_storage_volume - max_fuel_volume
    fuel_volume_constraint.set_as_constraint(lower=0., scaler=1.e-1)
    fuel_volume_constraint.add_name('constraint_fuel_volume')

    # # ==== takeoff and landing length constraint ====
    takeoff_landing_ft.set_as_constraint(upper=10000, scaler=1.e-4)
    takeoff_landing_ft.add_name('constraint_takeoff_landing')
    # takeoff_length_ft.set_as_constraint(upper=10000, scaler=1.e-4)
    # takeoff_length_ft.add_name('takeoff_length_constraint')
    # landing_length_ft.set_as_constraint(upper=10000, scaler=1.e-4)
    # landing_length_ft.add_name('landing_length_constraint')

    # ==== engine collision constraint ====
    # this enforces that the engines are separated by at least 1 diameter
    # NOTE: THIS DOES NOT PROVIDE ANYTHING. REMOVED
    # engine_y_pos = engine_cg_mirror[1]
    # engine_gap_scaler = 1.
    # engine_min_gap = (engine_y_pos - engine_diam*engine_gap_scaler)/engine_diam # normalized by actual engine diameter
    # engine_min_gap.set_as_constraint(lower=0)
    # engine_min_gap.add_name('engine_spacing_constraint')

if do_stability:
    # ==== static margin constraint ====
    if stability_level == 'loose':
        static_margin_bound = 0.0
        # static_margin_bound = 0.0109
    if stability_level == 'relaxed':
        static_margin_bound = 0.02
        # static_margin_bound = 0.0588
    elif stability_level == 'tight':
        static_margin_bound = 0.05
        # static_margin_bound = 0.10
    # static_margin.set_as_constraint(lower=static_margin_bound, scaler=1e0)
    static_margin.set_as_constraint(lower=static_margin_bound, scaler=1e2)
    static_margin.add_name('constraint_static_margin')

# if do_climb:
#     climb_trim_z.set_as_constraint(equals=0., scaler=1e-6)
#     climb_trim_z.add_name('climb_trim_z_constraint')
#     climb_trim_x.set_as_constraint(equals=0., scaler=1e-6)
#     climb_trim_x.add_name('climb_trim_x_constraint')

# if do_OEI:
#     OEI_trim_z = csdl.Variable(value=np.zeros(2,))
#     OEI_trim_z = OEI_trim_z.set(csdl.slice[0], OEI_cruise_trim_z)
#     OEI_trim_z = OEI_trim_z.set(csdl.slice[1], OEI_climb_trim_z)
#     OEI_trim_z.set_as_constraint(equals=0, scaler=1e-6)
#     OEI_trim_z.add_name('OEI_trim_z')

#     OEI_trim_x = csdl.Variable(value=np.zeros(2,))
#     OEI_trim_x = OEI_trim_x.set(csdl.slice[0], OEI_cruise_trim_x)
#     OEI_trim_x = OEI_trim_x.set(csdl.slice[1], OEI_climb_trim_x)
#     OEI_trim_x.set_as_constraint(equals=0, scaler=1e-6)
#     OEI_trim_x.add_name('OEI_trim_x')
    
#     max_thrust_constraint.set_as_constraint(lower=0., scaler=1e-3)
#     max_thrust_constraint.add_name('max_thrust_constraint')
# endregion

# ============================ objective ============================
weighted_fuel_burn.set_as_objective(scaler=1e-5)
weighted_fuel_burn.add_name('weighted_fuel_burn_objective')

csdl.save_optimization_variables()

testing = False
# testing = True
check_derivs = False

# If checking derivatives
# testing = True
# check_derivs = True

if not check_derivs:

    pitch_dv.save()
    pitch.save()
    planform_area.save()
    AR.save()
    MAC.save()
    geometry_coefficients.save()
    # panel_mesh.save()
    payload_cg.save()
    engine_cg.save()
    engine_cg_mirror.save()
    transition_volume.save()
    transition_cg.save()
    transition_cg_mirror.save()
    # Panel diagnostics below have sizing-only indexing (row 0 = sizing).
    panel_outputs['Cp'].save()
    L.save()
    panel_outputs['Di'].save()
    M.save()
    CL.save()
    panel_outputs['CDi'].save()
    CM.save()
    panel_outputs['Cf'].save()
    # CB_LE_sweep.save()
    # TR_LE_sweep.save()
    # FF.save()
    # CD0.save()
    panel_outputs['Df'].save()
    panel_outputs['CDw'].save()
    panel_outputs['Dw'].save()
    D_total.save()
    cg_beam.save()
    structural_weight.save()
    W_empty.save()


    L_D.save()
    W_bar.save()
    Wf.save()
    TOGW.save()
    W2.save()
    fuel_volume.save()
    max_fuel_volume.save()

    CM_cg.save()
    neutral_point.save()
    SM_missions.save()

    cg.save()

    # engine_mass.save()
    # engine_max_thrust.save()
    # engine_length.save()
    # engine_diam.save()

    # beam specifics
    height.save()
    width.save()
    BWB_beam_mass.save()

    # inputs/DVs from planform and on
    # centerbody_twist_dist.save()
    wing_twist_coefficients.save()
    centerbody_half_span.save()
    transition_half_span.save()
    wing_half_span.save()
    centerbody_chord_dist.save()
    wing_chord_dist.save()
    wing_sweep.save()
    wing_position.save()
    centerbody_chord_stretch_ratio.save()
    cruise_trim_elevator_deflection.save()
    airfoil_thickness_coefficients.save()
    airfoil_camber_coefficients.save()
    ecf.save()
    engine_mass_frac.save()

    if not do_moment_trim:
        CM_cg_cruise_nominal.save()

problem_name = 'pw_opt' + problem_type

# if not do_moment_trim:
#     problem_name += '_no_moment_trim'

# if problem_type == 'stability' or problem_type == 'full':
#     if not do_airfoil_shape:
#         problem_name += '_no_airfoil_shape'
#     problem_name += f'_stab_{stability_level}'

if USE_MULTIFIDELITY == 'PG-RANS-GNN':
    problem_name += '_pgnn'
elif USE_MULTIFIDELITY == 'Panel':
    problem_name += '_panel'
else:
    problem_name += '_gnn'

attempt = requested_attempt
while True:
    fname = f'{problem_name}_opt_{optimizer_name}_{attempt}'
    hdf5_name = f'{fname}.hdf5'
    output_dir = f'{fname}_outputs'
    if not os.path.exists(hdf5_name) and not os.path.exists(output_dir):
        break
    print(f'Attempt {attempt} already exists; trying {attempt + 1}')
    attempt += 1

print(f'using attempt {attempt}')
print(f'file name: {fname}')

sim = csdl.experimental.JaxSimulator(recorder, 
                                     gpu=False, 
                                     save_on_update=True, 
                                     filename=fname, 
                                     output_saved=True)

def warm_start_function(sim, file_name_ws, opt_iter):
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    var_names = {}
    if nominal_zero_pitch:
        var_names['pitch_dv'] = pitch_dv
    var_names['half_ttop'] = half_ttop
    var_names['half_tweb'] = half_tweb
    var_names_rem = {
        'elevator_deflection': cruise_trim_elevator_deflection,
        # 'centerbody_twist': centerbody_twist_dist,
        'wing_twist': wing_twist_coefficients,
        'center_half_span': centerbody_half_span,
        'transition_half_span': transition_half_span,
        'wing_half_span': wing_half_span,
        # 'centerbody_chord': centerbody_chord_stretch_coefficients,
        # 'wing_chord': wing_chord_stretch_coefficients,
        'centerbody_chord_stretch_ratio': centerbody_chord_stretch_ratio,
        'wing_chord_stretch': wing_chord_stretch_coefficients,
        'wing_sweep': wing_sweep,
        'wing_position': wing_position,
        # 'engine_chord_fraction': ecf,
        # 'thickness_percent_change': airfoil_thickness_coefficients,
        # 'camber_percent_change': airfoil_camber_coefficients,
    }
    for name, value in var_names_rem.items():
        var_names[name] = value
    import_vars = csdl.inline_import(file_name_ws, f'iteration_{opt_iter}')
    if not nominal_zero_pitch:
        saved_pitch = import_vars['pitch'].value.reshape(-1)
        sim[pitch_cruise_dv] = saved_pitch[:num_cruise]
        sim[pitch_sizing_dv] = saved_pitch[num_cruise:]
    for name, value in var_names.items():
    # for name, value in import_vars.items():
        sim[value] = import_vars[name].value
    # print(import_vars['MSC_violation'].value)
    # print(np.max(import_vars['stress_S1'].value))
    # exit()
    return sim

# warm starting options
warm_start_test = False

if warm_start_test:
    print('updating simulator values based on user-provided hdf5 file')
    file_name_warm_start = 'stability_no_airfoil_shape_stab_tight_opt_SLSQP_1_missions_4_nicholas.hdf5'
    opt_iter = 0
    sim = warm_start_function(sim, file_name_warm_start, opt_iter)

if testing:
    print('starting compile of sample forward run')
    start_time = time.time()
    sim.run()
    end_time = time.time()
    print(f'run+compile time: {end_time-start_time} seconds')


    # print('starting second forward run')
    # start_time = time.time()
    # sim.run()
    # end_time = time.time()
    # print(f'run time: {end_time-start_time} seconds')

    if check_derivs:
        print('starting compile of check totals')
        start_time = time.time()
        asdf = sim.check_totals(step_size=1.e-4)
        end_time = time.time()
        print(f'check totals run+compile time: {end_time-start_time} seconds')    
        # exit()

    exit()


print('starting compile of sample forward run')
start_time = time.time()
sim.run()
end_time = time.time()
print(f'compile+run time: {end_time-start_time} seconds')

def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux, ru_maxrss is in kilobytes
    return usage.ru_maxrss / 1024  # convert to MB

print(f"Peak memory so far (after sim.run()): {get_peak_memory_mb():.2f} MB")

print(f'============ problem type: {problem_type} ============')
print(f'============ number of cruise missions: {num_cruise} ============')

print('setting up Problem')

now = datetime.now()
prob_start_date = now.date()
prob_start_time = now.time()
print('================')
print(f'prob start date: {prob_start_date}')
print(f'prob start time: {prob_start_time}')
print('================')

start_problem_time = time.time()
prob = CSDLAlphaProblem(problem_name=fname, simulator=sim)
end_problem_time = time.time()
print(f'problem compile+run time: {end_problem_time-start_problem_time} seconds')

now = datetime.now()
prob_end_date = now.date()
prob_end_time = now.time()
print('================')
print(f'prob end date: {prob_end_date}')
print(f'prob end time: {prob_end_time}')
print('================')

print(f"Peak memory so far (after problem compile): {get_peak_memory_mb():.2f} MB")

print('setting up optimizer')
start_opt_setup = time.time()
if optimizer_name == 'SLSQP':
    # optimizer = PySLSQP(prob, solver_options={'maxiter':300, 'acc':1e-5})
    save_vars_list = ['x', 'objective', 'optimality', 'feasibility', 
                        'step', 'mode', 'iter', 'majiter', 'ismajor', 
                        'constraints', 'gradient', 'multipliers', 'jacobian']
    SLSQP_solver_options={
        'maxiter':300, 
        'acc':1e-3,
        # 'save_itr': 'all',
        # 'save_vars': save_vars_list,
        # 'save_filename': fname + '_optimizer_data.hdf5',
    }

    optimizer = PySLSQP(
        prob, 
        solver_options=SLSQP_solver_options
    )
elif optimizer_name == 'InteriorPoint':
    optimizer = InteriorPoint(
        prob,
        maxiter=200,
        opt_tol=1.e-3,
        feas_tol=1.e-3,
    )
# elif optimizer_name == 'SNOPT':
#     snopt_options = {
#         'Major iterations': 200,
#         'Major optimality': 1.e-6,
#         'Major feasibility': 1.e-6,
#         'Print frequency': 1,
#         'Summary frequency': 1,
#     }
#     optimizer = SNOPT(prob, solver_options=snopt_options)
else:
    raise ValueError('Invalid optimizer selected. Please try again.')
end_opt_setup = time.time()
print(f'optimizer setup time: {end_opt_setup-start_opt_setup} seconds')

now = datetime.now()
opt_start_date = now.date()
opt_start_time = now.time()
print('================')
print(f'optimizer start date: {opt_start_date}')
print(f'optimizer start time: {opt_start_time}')
print('================')

print('running optimizer')
opt_start = time.time()
optimizer.solve()
opt_end = time.time()

now = datetime.now()
opt_end_date = now.date()
opt_end_time = now.time()
print('================')
print(f'optimizer end date: {opt_end_date}')
print(f'optimizer end time: {opt_end_time}')
print('================')

success = optimizer.results['success']
print('\tTime:', opt_end - opt_start)
print('\tSuccess:', success)
# print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['objective'])





exit()

cell_types = cells_dict.keys()
ncpt = []
Cp_per_cell_type = []
mu_per_cell_type = []
Cp_output = sim[Cp][0]
mu_output = sim[mu][0]
counter = 0
for key in cell_types:
    num_cells_key = cells_dict[key].shape[0]
    Cp_per_cell_type.append(
        Cp_output[counter:counter+num_cells_key]
    )
    mu_per_cell_type.append(
        mu_output[counter:counter+num_cells_key]
    )
    counter += num_cells_key

mesh_data =  meshio.Mesh(
    points=panel_mesh.value[0],
    # points=points_orig,
    cells=cells_dict,
    # cell_data = {'Cp': Cp_per_cell_type, 'mu': mu_per_cell_type}
)

# meshio.write("output_mesh.vtk", mesh_data)
meshio.write("bad_mesh.vtk", mesh_data)
# meshio.write("orig_mesh.vtk", mesh_data)