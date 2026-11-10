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
# from mpi4py import MPI

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
from bwb_helper_functions import (
    setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle,
    write_simple_pickle, initial_refit, setup_and_evaluate_geometry_parameterization)

def build_initial_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled:bool=True):
    imported_function_set = lfs.import_file_patched(file_name=stp_file_path, parallelize=False)
    imported_geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                        function_names=imported_function_set.function_names,
                                        name='imported_geometry',
                                        space=imported_function_set.space)

    start = time.time()
    new_geometry = initial_refit(imported_geometry, plot=False)

    initial_refit_outputs = (imported_geometry, new_geometry)
    end = time.time()
    print('initial refit', end - start)
    return initial_refit_outputs, imported_function_set

def build_bwb_geometry(comm, geo_refit, geometry_pickle_file_path, timing_enabled:bool=True):
    end = time.time()

    geometry = geo_refit

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


    deformed_geometry = setup_and_evaluate_geometry_parameterization(
        geometry,
        geometric_variables,
        plot=False,
    )

    # geometric_variables = {
    #     'wing_sweep': wing_sweep,
    #     'transition_sweep': transition_sweep,
    #     'wing_half_span': wing_half_span,
    #     'center_wing_half_span': center_wing_half_span,
    #     'transition_half_span': transition_half_span,
    #     'wing_chord_stretch_coefficients': wing_chord_stretch_coefficients,
    #     'center_wing_chord_stretch_coefficients': center_wing_chord_stretch_coefficients,
    #     'wing_twist_coefficients': wing_twist_coefficients,
    #     'airfoil_thickness_coefficients': airfoil_thickness_coefficients,
    #     'airfoil_camber_coefficients': airfoil_camber_coefficients,
    #     'elevator_deflection': cruise_trim_elevator_deflection
    #     }
    
    end2 = time.time()
    # print(end2 - end)
    print('geometry setup', end2 - end)
    geometry_config_map = geometry_inputs
    return deformed_geometry, geometry_config_map, None

from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle, rotate_pitch

def compute_geometry_csdl(
        comm, 
        geometry_pickle_file_path, 
        stp_file_path, 
        surface_mesh_projection_file_path, 
        x_surf_dafoam_initial,
        normals,
        recorder, 
        timing_enabled=True,
        return_baseline_projections = False,
    ):
    # print('sdfs')
    initial_refit_outputs, _ = build_initial_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled=timing_enabled)
    geometry_project = initial_refit_outputs[1]
    # np.savez(
    #     "x_surf_dafoam_initial.npz",
    #     x_surf_dafoam_initial=x_surf_dafoam_initial,
    # )
    # exit()

    # region Surface mesh projection
    # Now do we do the same check for the surface mesh projection
    if surface_mesh_projection_file_path.is_file():
        print('Found surface mesh projection pickle!')
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)
    else:
        print('No projected surface mesh file found.')
        if x_surf_dafoam_initial is None or normals is None:
            raise TypeError('Need x_surf_dafoam_initial and normals for projections')
        try:
            # # ORIGINAL CODE
            # print(normals.shape,x_surf_dafoam_initial.shape)
            # projected_surf_mesh_dafoam = geometry_project.project(
            #     x_surf_dafoam_initial, 
            #     grid_search_density_parameter = 1,      # 1 
            #     projection_tolerance          = 1e-3,   # 1.e-3m 
            #     grid_search_density_cutoff    = 12,    # 150
            #     force_reprojection            = False,
            #     direction                     = normals,
            #     # plot                          = show_plots and not is_headless(),
            #     # interactive                   = interactive_plots,                     
            #     num_workers                   = 1
            # )
            projected_surf_mesh_dafoam = geometry_project.project(
                x_surf_dafoam_initial,
                grid_search_density_parameter=1.,
                newton_tolerance=1.e-10,
                use_line_search=True,
                force_reprojection=True, # Want this False normally
                plot=True
            )
            print('Writing surface mesh projection pickle...')
            write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
            print('Done!')

        # Added this exception because I was getting an ungraceful MPI termination
        except Exception as e:
            import traceback
            print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
            comm.Abort(1) # Abort MPI processes instead of letting them hang

    if return_baseline_projections:
        x_surf_dafoam_full_geo_baseline = geometry_project.evaluate(projected_surf_mesh_dafoam, plot=False)

    geometry, geometry_config_map, other = build_bwb_geometry(comm, geometry_project, geometry_pickle_file_path, timing_enabled=timing_enabled)
    # geometry = geometry_project
    # geometry_config_map = {'temp': csdl.Variable(value=1.0)}
    # geometry.functions[0].coefficients = geometry.functions[0].coefficients*geometry_config_map['temp']

    # rank = comm.rank
    print(f'Done reading projected surface mesh!')
    # print(projected_surf_mesh_dafoam)
    # projected_surf_mesh_dafoam = [(5, np.array([0.58831883, 0.15947931]))]
    x_surf_dafoam_full_geo = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)
    recorder.inline = False
    # print(geometry)
    # x_surf_dafoam_full_geo = 1.0
    if return_baseline_projections:
        return geometry, geometry_config_map, x_surf_dafoam_full_geo, x_surf_dafoam_full_geo_baseline
    else:
        return geometry, geometry_config_map, x_surf_dafoam_full_geo


if __name__ == '__main__':
    from mpi4py import MPI
    comm           = MPI.COMM_WORLD

    geometry_directory = 'bwb_geometry'
    geometry_pickle_file_name = 'bwb_new_stored_refit_new.pickle'
    stp_file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'

    stp_file_path                     = Path(geometry_directory)/stp_file_name
    geometry_pickle_file_path = Path(geometry_directory)/geometry_pickle_file_name

    timing_enabled = 1
    # test:
    # recorder = csdl.Recorder(inline=False, debug=False)
    if comm.rank == 0:
        recorder = csdl.Recorder(inline=True, debug=False)
        recorder.start()
        print('running:', comm.rank)
        initial_refit_outputs = build_initial_bwb_geometry(comm, geometry_pickle_file_path, stp_file_path, timing_enabled=timing_enabled)
        geometry = initial_refit_outputs[0]

        # recorder.inline = False
        print('running 2:', comm.rank)
        geometry, geometry_config_map, other = build_bwb_geometry(comm, initial_refit_outputs, geometry_pickle_file_path, timing_enabled=timing_enabled)
