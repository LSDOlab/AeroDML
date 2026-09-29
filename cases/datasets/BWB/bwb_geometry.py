# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import sys
import time
from pathlib import Path
import shutil

# MPI
from mpi4py import MPI

# CSDL packages
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

# project/ directory
ROOT = Path(__file__).resolve().parents[1]

# allow imports from project/A
sys.path.insert(0, str(ROOT))
print(ROOT)

from csdl_dafoam_helper_functions import Timer, hash_array_tol, quiet_barrier, compute_vertex_normals, average_normals_at_duplicate_points
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

def build_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled:bool=True):
    if comm is not None:
        rank = comm.Get_rank()
    else:
        rank = 0

    #region Geometry setup I
    if geometry_pickle_file_path.is_file():
        with Timer(f'reading geometry', rank, timing_enabled):
            geometry = read_geometry_pickle(geometry_pickle_file_path)
    else:
        if rank == 0:
            print('No geometry pickle file found.')
            with Timer('importing geometry', rank, timing_enabled):
                geometry = lsdo_geo.import_geometry(stp_file_path,
                                                    parallelize=False)

            # These are hardcoded indices for geometry
            oml_indices                 = [key for key in geometry.functions.keys()]
            wing_c_indices              = [0,1,8,9]
            wing_r_transition_indices   = [2,3]
            wing_r_indices              = [4,5,6,7]
            wing_l_transition_indices   = [10,11]
            wing_l_indices              = [12,13,14,15] 

            with Timer('declaring geometry components', rank, timing_enabled):
                left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
                left_wing               = geometry.declare_component(wing_l_indices)
                right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
                right_wing              = geometry.declare_component(wing_r_indices)
                center_wing             = geometry.declare_component(wing_c_indices)
                oml = geometry.declare_component(oml_indices)

            wing_parameterization   = 15
            num_v                   = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
            
            with Timer('BSplineSpace', rank, timing_enabled):
                wing_refit_bspline      = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))

            with Timer('left wing refit', rank, timing_enabled):
                left_wing_function_set  = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

            with Timer('right wing refit', rank, timing_enabled):
                right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

            with Timer('allocating left wing functions', rank, timing_enabled):
                for i, function in left_wing_function_set.functions.items():
                    geometry.functions[i]   = function
                    left_wing.functions[i]  = function

            with Timer('allocating right wing functions', rank, timing_enabled):
                for i, function in right_wing_function_set.functions.items():
                    geometry.functions[i]   = function
                    right_wing.functions[i] = function

            with Timer('pickling geometry', rank, timing_enabled):
                write_geometry_pickle(geometry, geometry_pickle_file_path)
        
        # Wait for root rank to finish writing
        quiet_barrier(comm)
        if rank != 0:
            with Timer(f'reading geometry', rank, timing_enabled):
                geometry = read_geometry_pickle(geometry_pickle_file_path)   

    # region Design variables
    # ============================ Design variables ===========================
    # Not recommended
    percent_change_in_thickness_dof_wing        = csdl.Variable(shape=(8,4), value=0.)
    percent_change_in_thickness_dof_body        = csdl.Variable(shape=(8,4), value=0.)
    percent_change_in_thickness_dof             = csdl.concatenate(
                                                    (percent_change_in_thickness_dof_wing,
                                                    percent_change_in_thickness_dof_body), axis=1)

    # Not recommended
    normalized_percent_camber_change_dof_wing   = csdl.Variable(shape=(6,4), value=0.)
    normalized_percent_camber_change_dof_body   = csdl.Variable(shape=(6,4), value=0.)
    normalized_percent_camber_change_dof        = csdl.concatenate(
                                                    (normalized_percent_camber_change_dof_wing,
                                                    normalized_percent_camber_change_dof_body), axis=1)

    transition_span                             = csdl.Variable(value=4.891)

    # np.array([30., 30-3.899, 30-9.813]) -> Reference
    centerbody_chord_stretches                  = csdl.Variable(shape=(3,), value=0.) # meters: changes during optimizaiton alot

    centerbody_twists_dv                           = csdl.Variable(shape=(2,), value=np.array([0., 0.])) # radians
    centerbody_twist_root                           = csdl.Variable(shape=(1,), value=np.array([0.])) # radians
    centerbody_twists = csdl.concatenate((centerbody_twist_root, centerbody_twists_dv)) # radians

    centerbody_dihedral_translations            = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])) # meters
    centerbody_span                             = csdl.Variable(value=10.) # meters

    wing_chord_stretching_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.])) # delta chord in meters (linear)
    wing_span                                   = csdl.Variable(value=25.852 - 9.891) # meters
    wing_sweep_translation                      = csdl.Variable(value=0.) # how far the tip moves back in meters
    wing_dihedral_translation_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.])) # in meters
    root_twist  = csdl.Variable(shape=(1,), value=np.array([0]))
    tip_twist   = csdl.Variable(shape=(1,), value=np.array([0.]))
    mid_twist   = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
    wing_twists = csdl.concatenate((root_twist, mid_twist, tip_twist)) # radians
    wing_twists.flatten()

    # wing_twists                                 = csdl.Variable(shape=(4,), value=np.array([0., 0., 0., 0.]))
    # percent_change_in_thickness_dof             = csdl.Variable(shape=(8,8), value=0.)
    # normalized_percent_camber_change_dof        = csdl.Variable(shape=(6,8), value=0.)

    geometry_values_dict = {
        'centerbody_chord_stretches': centerbody_chord_stretches,
        'wing_chord_stretching_b_spline_coefficients': wing_chord_stretching_b_spline_coefficients,
        'centerbody_span': centerbody_span,
        'transition_span': transition_span,
        'wing_span': wing_span,
        'wing_sweep_translation': wing_sweep_translation,
        'centerbody_dihedral_translations': centerbody_dihedral_translations,
        'wing_dihedral_translation_b_spline_coefficients': wing_dihedral_translation_b_spline_coefficients,
        'centerbody_twists': centerbody_twists,
        'wing_twists': wing_twists,
        'percent_change_in_thickness_dof': percent_change_in_thickness_dof,
        'normalized_percent_camber_change_dof': normalized_percent_camber_change_dof,
    }

    # region Geometry setup II
    if comm is not None:
        comm_size = comm.Get_size()
        with Timer(f'setting up geometry', rank, 1):
            # Had to "serialize" this because I was getting race conditions in cache I/O
            for r in range(comm_size):
                quiet_barrier(comm)
                if rank == r:
                    geometry = setup_geometry(geometry, geometry_values_dict)
                quiet_barrier(comm)
    else:
        geometry = setup_geometry(geometry, geometry_values_dict)

    geometry_config_map = {
        'span': wing_span,
        'sweep': wing_sweep_translation,
        'wing_chord': wing_chord_stretching_b_spline_coefficients,
        'root_twist': root_twist,
        'mid_twist': mid_twist,
        'tip_twist': tip_twist,
        '%_thickness_change_wing': percent_change_in_thickness_dof_wing,
        '%_camber_change_wing': normalized_percent_camber_change_dof_wing,
        'center_chord': centerbody_chord_stretches,
        'center_span': centerbody_span,
        'center_twist': centerbody_twists_dv,
        'center_twist_root': centerbody_twist_root,
        '%_thickness_change_body': percent_change_in_thickness_dof_body,
        '%_camber_change_body': normalized_percent_camber_change_dof_body,
        'transition_span': transition_span,
    }

    return geometry, geometry_config_map, None