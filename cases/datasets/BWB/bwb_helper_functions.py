import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
import scipy.sparse.linalg as spsl


from pathlib import Path
import pickle
def write_geometry_pickle(geometry, geometry_file_path):

    with open(geometry_file_path, 'wb+') as handle:
        geometry_copy = geometry.copy()
        for i, function in geometry.functions.items():
            function_copy = function.copy()
            function_copy.coefficients = function.coefficients.value.copy()
            geometry_copy.functions[i] = function_copy

        pickle.dump(geometry_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

def read_geometry_pickle(geometry_file_path):
    
    with open(geometry_file_path, 'rb') as handle:
        function_set = pickle.load(handle)
        for function in function_set.functions.values():
            function.coefficients = csdl.Variable(value=function.coefficients)

    return lg.Geometry(functions=function_set.functions, function_names=function_set.function_names, 
                                name=function_set.name, space=function_set.space)


def read_simple_pickle(file_path):
    with open(file_path, 'rb') as handle:
        contents = pickle.load(handle)

    return contents


def write_simple_pickle(var_to_write, file_path):
    with open(file_path, 'wb+') as handle:
        var_to_write_copy = var_to_write.copy()
        pickle.dump(var_to_write_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)



def gather_array_to_rank0(x_local: np.ndarray, comm):
    """
    Gather local arrays to rank 0.
    
    Returns on rank 0:
        - x_full: (N_total, dim) array
        - sizes: list of local row counts from each rank
        - index_ranges: list of (start, stop) tuples for each rank's slice
    
    On other ranks:
        - x_full is None
        - sizes and index_ranges are available for consistency
    """
    from mpi4py import MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_npts = np.array([x_local.shape[0]], dtype=np.int32)
    sizes = np.zeros(size, dtype=np.int32)
    comm.Allgather([local_npts, MPI.INT], [sizes, MPI.INT])

    if x_local.ndim > 1:
        dim = x_local.shape[1]
        counts = sizes * dim
    else:
        dim = -1
        counts = sizes
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

    # Compute start/stop indices for slicing back full array
    starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    stops = starts + sizes
    index_ranges = list(zip(starts, stops))

    sendbuf = x_local.flatten()
    recvbuf = None
    if rank == 0:
        total_count = np.sum(counts)
        recvbuf = np.empty(total_count, dtype=np.float64)

    comm.Gatherv(sendbuf, (recvbuf, counts, displacements, MPI.DOUBLE), root=0)

    if rank == 0:
        if dim == -1:
            x_full = recvbuf.reshape((-1))
        else:
            x_full = recvbuf.reshape((-1, dim))
        return x_full, sizes, index_ranges
    else:
        return None, sizes, index_ranges

# x_vol_dafoam = rotate_pitch(x_vol_dafoam, flow_conditions_group.pitch_inf)
def rotate_pitch(xyz_dafoam, pitch, center = (0,0,0), use_csdl = True):
    "rotates about y axis with pitch in degrees"
    n_points = xyz_dafoam.shape[0] // 3
    xyz = xyz_dafoam.reshape((n_points,3))

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xc, yc, zc = center

    # Freestream is positive, nose points in negative dir.
    theta = pitch * np.pi / 180.0

    # Rotate:
    if use_csdl == True:
        c = csdl.cos(theta)
        s = csdl.sin(theta)
    elif use_csdl == 'jax':
        import jax.numpy as jnp
        c = jnp.cos(theta)
        s = jnp.sin(theta)
    else:
        c = np.cos(theta)
        s = np.sin(theta)

    x_rel = x - xc
    y_rel = y - yc
    z_rel = z - zc

    x_rot = xc + c * x_rel + s * z_rel
    y_rot = yc + y_rel
    z_rot = zc - s * x_rel + c * z_rel

    if use_csdl == 'jax':
        import jax.numpy as jnp
        return jnp.stack((x_rot, y_rot, z_rot), axis=1).reshape(xyz_dafoam.shape)
    if not use_csdl:
        return np.stack((x_rot, y_rot, z_rot), axis=1).reshape(xyz_dafoam.shape)
    xyz_rot = csdl.concatenate(
        (
            x_rot.reshape((n_points, 1)),
            y_rot.reshape((n_points, 1)),
            z_rot.reshape((n_points, 1)),
        ),
        axis=1,
    )
    xyz_rot = xyz_rot.reshape(xyz_dafoam.shape)
    assert xyz_dafoam.shape == xyz_rot.shape
    return xyz_rot

import lsdo_function_spaces as lfs
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities,construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import time
def setup_geometry(geometry: lg.Geometry, geometry_values_dict):

    centerbody_chord_stretches                  = geometry_values_dict['centerbody_chord_stretches']
    wing_chord_stretching_b_spline_coefficients = geometry_values_dict['wing_chord_stretching_b_spline_coefficients']
    centerbody_span                             = geometry_values_dict['centerbody_span']
    transition_span                             = geometry_values_dict['transition_span']
    wing_span                                   = geometry_values_dict['wing_span']
    wing_sweep_translation                      = geometry_values_dict['wing_sweep_translation']
    centerbody_dihedral_translations            = geometry_values_dict['centerbody_dihedral_translations']
    wing_dihedral_translation_b_spline_coefficients = geometry_values_dict['wing_dihedral_translation_b_spline_coefficients']
    centerbody_twists                           = geometry_values_dict['centerbody_twists']
    wing_twists                                 = geometry_values_dict['wing_twists']
    percent_change_in_thickness_dof             = geometry_values_dict['percent_change_in_thickness_dof']
    normalized_percent_camber_change_dof        = geometry_values_dict['normalized_percent_camber_change_dof']

    centerline_LE       = np.array([-1.1, 0, 0])
    centerline_TE       = np.array([30.5, 0, 0])
    centerbody_mid_LE_R = np.array([3.899, 2.5, 0])
    centerbody_mid_TE_R = np.array([30.5, 2.5, 0.])
    C_t_joint_LE_R      = np.array([9.813, 5, 0])
    C_t_joint_TE_R      = np.array([30.5, 5, 0])
    transition_mid_LE_R = np.array([16.539, 8.617, 0.841])
    transition_mid_TE_R = np.array([24.006, 8.617, 0.860])
    wing_root_LE_R      = np.array([17.815, 9.891, 1.040])
    wing_root_TE_R      = np.array([23.815, 9.891, 1.040])
    wing_sect1_LE_R     = np.array([20.931, 14.33, 1.341])
    wing_sect1_TE_R     = np.array([26.096, 14.33, 1.377])
    wing_sect2_LE_R     = np.array([25.903, 21.413, 1.822])
    wing_sect2_TE_R     = np.array([29.735, 21.413, 1.916])
    wing_tip_LE_R       = np.array([29.019, 25.852, 2.123])
    wing_tip_TE_R       = np.array([32.016, 25.852, 2.254])

    dz_cl   = np.array([0, 0, 2.25])
    dz_cm   = np.array([0, 0, 2.25])
    dz_ctj  = np.array([0, 0, 1.514025])
    dz_tm   = np.array([0, 0, 1.514025])
    dz_wr   = np.array([0, 0, 0.5])
    dz_ws1  = np.array([0, 0, 0.5])
    dz_ws2  = np.array([0, 0, 0.5])
    dz_wt   = np.array([0, 0, 0.3])

    x_val_half = np.array([
        [centerline_LE[0], centerline_TE[0]+1.],
        [centerbody_mid_LE_R[0]-1, centerbody_mid_TE_R[0]+1],
        [C_t_joint_LE_R[0]-0.5, C_t_joint_TE_R[0]+1.],
        [transition_mid_LE_R[0]-0.5, transition_mid_TE_R[0]+0.5],
        [wing_root_LE_R[0]-0.5, wing_root_TE_R[0]+0.5],
        [wing_sect1_LE_R[0]-0.5, wing_sect1_TE_R[0]+0.5],
        [wing_sect2_LE_R[0]-0.5, wing_sect2_TE_R[0]+0.5],
        [wing_tip_LE_R[0]-0.5, wing_tip_TE_R[0]+0.5],
    ])

    x_val_full = np.concatenate(
        (x_val_half[1:,:][::-1,:], x_val_half)
    )

    z_val_half = np.array([
        [centerline_LE[2], centerline_TE[2]],
        [centerbody_mid_LE_R[2], centerbody_mid_TE_R[2]],
        [C_t_joint_LE_R[2], C_t_joint_TE_R[2]],
        [transition_mid_LE_R[2], transition_mid_TE_R[2]],
        [wing_root_LE_R[2], wing_root_TE_R[2]],
        [wing_sect1_LE_R[2], wing_sect1_TE_R[2]],
        [wing_sect2_LE_R[2], wing_sect2_TE_R[2]],
        [wing_tip_LE_R[2], wing_tip_TE_R[2]],
    ])
    
    z_delta_half = np.array([
        [-dz_cl[2]-0.5,dz_cl[2]+0.5],  
        [-dz_cm[2]-0.5,dz_cm[2]+0.5],  
        [-dz_ctj[2]-0.2,dz_ctj[2]+0.2],  
        [-dz_tm[2]-0.2,dz_tm[2]+0.2],  
        [-dz_wr[2]-0.2,dz_wr[2]+0.2],  
        [-dz_ws1[2]-0.2,dz_ws1[2]+0.2],  
        [-dz_ws2[2]-0.2,dz_ws2[2]+0.2],  
        [-dz_wt[2]-0.2,dz_wt[2]+0.2], 
    ])
    
    z_val_full = np.concatenate(
        (z_val_half[1:,:][::-1,:] + z_delta_half[1:,:][::-1,:], z_val_half + z_delta_half)
    )

    # y_half = np.array([0, 5, 9.891, 25.852 + 0.25])
    y_half = np.array([0, 2.5, 5, 8.617, 9.891, 14.33, 21.413, 25.852 + 0.25])
    # y_half = np.array([0, 25.852])
    y_full = np.concatenate((-y_half[1:][::-1], y_half))
    num_ffd_sections = x_val_full.shape[0]
    FFD_block_points = np.zeros((2,num_ffd_sections,2,3))
    for i in range(num_ffd_sections):
        x_val = x_val_full[i]
        z_val = z_val_full[i]
        x_grid, z_grid = np.meshgrid(x_val, z_val, indexing='ij')
        y_val = y_full[i]

        cross_section = np.zeros((2,2,3))
        cross_section[:,:,1] = y_val
        cross_section[:,:,0] = x_grid
        cross_section[:,:,2] = z_grid

        FFD_block_points[:,i,:] = cross_section

    # dummy custom FFD block
    if False:
        x_val = np.array([-10, 35])
        z_val = np.array([-5, 5])
        x_grid, z_grid = np.meshgrid(x_val, z_val, indexing='ij')
        wingtip_cross_section = np.zeros((2,2,3))
        wingtip_cross_section[:,:,1] = 30
        wingtip_cross_section[:,:,0] = x_grid
        wingtip_cross_section[:,:,2] = z_grid
        FFD_block_points = np.zeros((2,2,2,3))
        FFD_block_points[:,1,:] = wingtip_cross_section
        wingtip_cross_section[:,:,1] *= -1
        FFD_block_points[:,0,:] = wingtip_cross_section

    num_ffd_coefficients_chordwise = 8

    BWB_ffd_block = construct_ffd_block_from_corners(
        entities=geometry,
        corners=FFD_block_points,
        num_coefficients=(num_ffd_coefficients_chordwise,2,2),
        degree=(2,2,1)
    )
    # BWB_ffd_block.plot()
    center_section_index = num_ffd_sections // 2
    num_centerbody_ffd_sections = 3
    num_wing_ffd_sections = 4

    ffd_sectional_parameterization = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=BWB_ffd_block.coefficients,
        principal_parametric_dimension=1,
    )
    # ffd_sectional_parameterization.plot()

    space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    wing_chord_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=wing_chord_stretching_b_spline_coefficients, name='wing_chord_stretching_b_spline')

    centerbody_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='centerbody_span_stretching_b_spline')

    wing_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_span_stretching_b_spline')

    wing_sweep_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_sweep_translation_b_spline')

    wing_dihedral_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=wing_dihedral_translation_b_spline_coefficients, name='dihedral_translation_b_spline')

    # twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
    #                                 coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline')

    # Extract DOFs
    # centerbody_chord_stretches = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_chord_stretches')

    # centerbody_span = csdl.Variable(value=10., name='centerbody_span')
    # transition_span = csdl.Variable(value=4.891, name='transition_span')
    # wing_span = csdl.Variable(value=25.852 - 9.891, name='wing_span')

    initial_centerbody_span = 10.
    initial_transition_span = 4.891
    initial_wing_span = 25.852 - 9.891
    centerbody_outer_span_translation = (centerbody_span - initial_centerbody_span)/2   # Divide by 2 because halfspan
    centerbody_span_stretching_b_spline.coefficients = centerbody_span_stretching_b_spline.coefficients.set(
        csdl.slice[1], centerbody_outer_span_translation
    )
    transition_outer_span_translation = centerbody_outer_span_translation + (transition_span - initial_transition_span)
    wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
        csdl.slice[0], transition_outer_span_translation
    )
    wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
        csdl.slice[1], transition_outer_span_translation + (wing_span - initial_wing_span)
    )

    # wing_sweep_translation = csdl.Variable(value=0., name='wing_sweep_translation')
    wing_sweep_translation_b_spline.coefficients = wing_sweep_translation_b_spline.coefficients.set(
        csdl.slice[1], wing_sweep_translation
    )

    # centerbody_dihedral_translations = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_dihedral_translations')

    # wing_twists = csdl.Variable(shape=(num_wing_ffd_sections,), value=0., name='wing_twists')
    # centerbody_twists = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_twists')

    # endregion Create Parameterization Objects

    # region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
    parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
    # chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
    # wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
    # sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
    # twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)
    chord_stretches = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    wing_chord_stretches = wing_chord_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    wing_translation_stretch = 0.9*wing_chord_stretches[0] + 0.1*centerbody_chord_stretches[-1]
    chord_stretches = chord_stretches.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                            centerbody_chord_stretches)
    chord_stretches = chord_stretches.set(csdl.slice[-num_wing_ffd_sections:], wing_chord_stretches)
    chord_stretches = chord_stretches.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                            centerbody_chord_stretches[::-1])
    chord_stretches = chord_stretches.set(csdl.slice[:num_wing_ffd_sections],
                                            wing_chord_stretches[::-1])
    chord_stretches = chord_stretches.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                            wing_translation_stretch)
    chord_stretches = chord_stretches.set(csdl.slice[num_wing_ffd_sections],
                                            wing_translation_stretch)

    centerbody_span_translations = centerbody_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_centerbody_ffd_sections))
    wing_span_translations = wing_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    wing_transition_section_translation = 1.1*wing_span_translations[0] + (-0.1)*wing_span_translations[-1]

    span_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    span_translations = span_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                              centerbody_span_translations)
    span_translations = span_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_span_translations)

    span_translations = span_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                              -centerbody_span_translations[::-1])
    span_translations = span_translations.set(csdl.slice[:num_wing_ffd_sections],
                                              -wing_span_translations[::-1])
    
    span_translations = span_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                              wing_transition_section_translation)
    span_translations = span_translations.set(csdl.slice[(num_wing_ffd_sections)],
                                                -wing_transition_section_translation)
    
    wing_sweep_translations = wing_sweep_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    transition_sweep_translation = 0.9*wing_sweep_translations[0]
    
    sweep_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    sweep_translations = sweep_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_sweep_translations)
    sweep_translations = sweep_translations.set(csdl.slice[:num_wing_ffd_sections], wing_sweep_translations[::-1])

    sweep_translations = sweep_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                                transition_sweep_translation)
    sweep_translations = sweep_translations.set(csdl.slice[num_wing_ffd_sections],
                                                transition_sweep_translation)
    
    wing_dihedral_translations = wing_dihedral_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    transition_dihedral = 0.9*wing_dihedral_translations[0] + 0.1*centerbody_dihedral_translations[-1]
    dihedral_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                                      centerbody_dihedral_translations)
    dihedral_translations = dihedral_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_dihedral_translations)
    dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                                      centerbody_dihedral_translations[::-1])
    dihedral_translations = dihedral_translations.set(csdl.slice[:num_wing_ffd_sections], wing_dihedral_translations[::-1])
    dihedral_translations = dihedral_translations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_dihedral)
    dihedral_translations = dihedral_translations.set(csdl.slice[num_wing_ffd_sections], transition_dihedral)

    transition_twist = 0.9*wing_twists[0] + 0.1*centerbody_twists[-1]
    
    twist_rotations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    twist_rotations = twist_rotations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections], centerbody_twists)
    twist_rotations = twist_rotations.set(csdl.slice[-num_wing_ffd_sections:], wing_twists)
    twist_rotations = twist_rotations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1], centerbody_twists[::-1])
    twist_rotations = twist_rotations.set(csdl.slice[:num_wing_ffd_sections], wing_twists[::-1])
    twist_rotations = twist_rotations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_twist)
    twist_rotations = twist_rotations.set(csdl.slice[num_wing_ffd_sections], transition_twist)

    sectional_parameters = VolumeSectionalParameterizationInputs()
    sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretches)
    sectional_parameters.add_sectional_translation(axis=csdl.Variable(value=np.array([0., 1., 0.])), translation=span_translations)
    sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translations)
    sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translations)
    sectional_parameters.add_sectional_rotation(axis=csdl.Variable(value=np.array([0., 1., 0.])), rotation=twist_rotations)

    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

    # Apply shape variables
    original_block_thickness = BWB_ffd_block.coefficients.value[:, :, 1, 2] - BWB_ffd_block.coefficients.value[:, :, 0, 2]

    percent_change_in_thickness = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
    # percent_change_in_thickness_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections//2+1), value=0.,
    #                                                 name='percent_change_in_thickness')
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,:num_ffd_sections//2+1], percent_change_in_thickness_dof)
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,num_ffd_sections//2+1:], percent_change_in_thickness_dof[:,-2::-1])
    delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
    thickness_upper_translation = 1/2 * delta_block_thickness
    thickness_lower_translation = -thickness_upper_translation
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)

    # Parameterize camber change as normalized by the original block (kind of like chord) length
    block_length = BWB_ffd_block.coefficients.value[1, :, 0, 0] - BWB_ffd_block.coefficients.value[0, :, 0, 0]
    block_length = csdl.expand(block_length, (num_ffd_coefficients_chordwise, num_ffd_sections), 'j->ij')

    normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
    # normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2, num_ffd_sections//2+1), value=0.,
    #                                                      name='normalized_percent_camber_change')
    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,:num_ffd_sections//2+1],
                                                                            normalized_percent_camber_change_dof)
    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,num_ffd_sections//2+1:], 
                                                                            normalized_percent_camber_change_dof[:,-2::-1])
    camber_change = (normalized_percent_camber_change / 100) * block_length
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], 
                                            ffd_coefficients[:,:,:,2] + 
                                            csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

    geometry_coefficients = BWB_ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients) # type: ignore
    # geometry.plot()

    return geometry
    # jax_inputs = [centerbody_chord_stretches, wing_chord_stretching_b_spline.coefficients, centerbody_span, transition_span, wing_span, 
    #               wing_sweep_translation, centerbody_dihedral_translations, wing_dihedral_translation_b_spline.coefficients,
    #               centerbody_twists, wing_twists, percent_change_in_thickness_dof, normalized_percent_camber_change_dof]
    # # jax outputs is a list containing all the geometry coefficients (geometry.functions[:].coefficients)
    # jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] \
    #             + [BWB_ffd_block.coefficients]

    # recorder = csdl.get_current_recorder()
    # jax_sim = csdl.experimental.JaxSimulator(
    #     recorder=recorder,
    #     additional_inputs=jax_inputs,
    #     additional_outputs=jax_outputs,
    #     gpu=False
    # )

    # jax_sim.run()

    # # Geometry Variables Video
    # import vedo

    # video = vedo.Video(name="Y1_demo/BWB_Geometry_Variables.mp4", fps=11, backend='opencv')
    # # video = vedo.Video(name="Y1_demo/BWB_Geometry_Variables.mp4", duration=20, backend='opencv')

    # wing_chord_stretching_b_spline.coefficients.add_name('wing_chord_stretching_b_spline_coefficients')
    # wing_dihedral_translation_b_spline.coefficients.add_name('wing_dihedral_translation_b_spline_coefficients')

    # parameters_with_offsets = [
    #     (centerbody_chord_stretches, 10.),
    #     (wing_chord_stretching_b_spline.coefficients, 3.),
    #     (centerbody_span, centerbody_span.value), 
    #     (transition_span, transition_span.value), 
    #     (wing_span, wing_span.value*0.3),
    #     (wing_sweep_translation, 10.),
    #     (centerbody_dihedral_translations, 3.),
    #     (wing_dihedral_translation_b_spline.coefficients, 3.),
    #     (centerbody_twists, 30*np.pi/180),
    #     (wing_twists, 30*np.pi/180), # 21 dv not including shape variables
    #     (percent_change_in_thickness_dof, 100.), # 64 of these
    #     (normalized_percent_camber_change_dof, 100.)    # 48 of these
    # ]

    # camera = {
    #     # 'pos': (4*num_bodies + 1, 0, -1*num_bodies/2),
    #     'pos': (-50, -50, 50),
    #     # 'pos': (5*num_bodies + 1, 0, 0),
    #     'focalPoint': (10, -10, 0),
    #     # 'focalPoint': (0, 0, 0),
    #     'viewup': (0, 0, 1),
    # }

    # for parameter, offset in parameters_with_offsets:
    #     if len(parameter.value.shape) == 1:
    #         for i in range(len(parameter.value)):
    #             values = np.hstack((np.linspace(parameter.value[i], parameter.value[i] + offset, 11).flatten(),
    #                             np.linspace(parameter.value[i] + offset, parameter.value[i], 11).flatten()))
    #             for value in values:
    #                 parameter_value = parameter.value
    #                 parameter_value[i] = value
    #                 jax_sim[parameter] = parameter_value
    #                 jax_sim.run()
    #                 frame = geometry.plot(show=False)
    #                 ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
    #                 if len(parameter.value) == 1:
    #                     text = f"Parameter: {parameter.name}, Value: {value:.2f}"
    #                 else:
    #                     text = f"Parameter: {parameter.name}[{i}], Value: {value:.2f}"
    #                 vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
    #                 video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
    #                                              size=(1920, 1200))
    #                 # video_plotter.show(frame, viewup='z')
    #                 video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
    #                 video.add_frame()
    #     elif len(parameter.value.shape) == 2:
    #         for i in range(parameter.value.shape[0]):
    #             for j in range(parameter.value.shape[1]):
    #                 values = np.hstack((np.linspace(parameter.value[i, j], parameter.value[i, j] + offset, 11).flatten(),
    #                                 np.linspace(parameter.value[i, j] + offset, parameter.value[i, j], 11).flatten()))
    #                 for value in values:
    #                     parameter_value = parameter.value
    #                     parameter_value[i, j] = value
    #                     jax_sim[parameter] = parameter_value
    #                     jax_sim.run()
    #                     frame = geometry.plot(show=False)
    #                     ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
    #                     text = f"Parameter: {parameter.name}[{i,j}], Value: {value:.2f}"
    #                     vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
    #                     video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
    #                                                  size=(1920, 1200))
    #                     # video_plotter.show(frame, viewup='z')
    #                     video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
    #                     video.add_frame()
    
    # video.close()
    # return BWB_ffd_block
import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import numpy as np
import time
from typing import Optional

enforce_no_centerbody_trailing_edge_sweep = True
enforce_no_wing_leading_edge_sweep = True
enforce_thickness_to_chord_ratio = True
use_baseline_twist = True


# CENTER_REFIT_GRID_RESOLUTION = (200, 11)
CENTER_REFIT_GRID_RESOLUTION = (200, 35)
# CENTER_REFIT_GRID_RESOLUTION = (200, 45)    # I think it got worse?
# CENTER_REFIT_GRID_RESOLUTION = (200, 22)
CENTER_REFIT_SPANWISE_SMOOTHING = 0
# CENTER_REFIT_SPANWISE_SMOOTHING = 5
# CENTER_REFIT_SPANWISE_SMOOTHING = 500
CENTER_REFIT_BOUNDARY_WEIGHT = 1.e6
# CENTER_REFIT_BOUNDARY_WEIGHT = 1.e10


def _center_refit_sqrt_weights(parametric_coordinates):
    spanwise_coordinates = parametric_coordinates[:, 1]
    boundary_rows = np.isclose(spanwise_coordinates, 0.) | np.isclose(
        spanwise_coordinates,
        1.,
    )
    weights = np.ones(parametric_coordinates.shape[0])
    weights[boundary_rows] = np.sqrt(CENTER_REFIT_BOUNDARY_WEIGHT)
    return weights


def _build_center_refit_rhs(basis_matrix, parametric_coordinates, points):
    sqrt_weights = _center_refit_sqrt_weights(parametric_coordinates)
    weighted_basis_matrix = basis_matrix.multiply(sqrt_weights[:, None])
    point_weights = np.repeat(sqrt_weights[:, None], points.shape[1], axis=1)
    return weighted_basis_matrix.T @ (points * point_weights)


def _build_center_refit_inverse(
    basis_matrix,
    spanwise_curvature_basis_matrix,
    parametric_coordinates,
):
    """Build a dimensionlessly scaled span-curvature-regularized fit inverse."""
    unweighted_fitting_matrix = (basis_matrix.T @ basis_matrix).tocsc()
    sqrt_weights = _center_refit_sqrt_weights(parametric_coordinates)
    weighted_basis_matrix = basis_matrix.multiply(sqrt_weights[:, None])
    fitting_matrix = (weighted_basis_matrix.T @ weighted_basis_matrix).tocsc()
    if CENTER_REFIT_SPANWISE_SMOOTHING:
        spanwise_curvature_penalty = (
            spanwise_curvature_basis_matrix.T @ spanwise_curvature_basis_matrix
        ).tocsc()
        penalty_scale = (
            np.mean(unweighted_fitting_matrix.diagonal())
            / np.mean(spanwise_curvature_penalty.diagonal())
        )
        fitting_matrix = fitting_matrix + (
            CENTER_REFIT_SPANWISE_SMOOTHING
            * penalty_scale
            * spanwise_curvature_penalty
        )
    return np.linalg.inv(fitting_matrix.toarray())


def _build_symmetric_spanwise_knot_vector(
    num_center_wing_dofs: int,
    num_wing_dofs: int,
    degree: int = 3,
    section_lengths=None,
):
    """Build a symmetric open knot vector with simple interior knots."""
    if section_lengths is None:
        center_length = float(num_center_wing_dofs)
        transition_length = float(degree)
        wing_length = float(num_wing_dofs)
    else:
        center_length, transition_length, wing_length = map(float, section_lengths)
    total_length = 2. * (center_length + transition_length + wing_length)
    left_wing_end = wing_length / total_length
    left_transition_end = (wing_length + transition_length) / total_length
    right_center_end = 1. - left_transition_end
    right_transition_end = 1. - left_wing_end

    full_center_coefficients = 2 * num_center_wing_dofs - 1
    center_intervals = full_center_coefficients - degree
    wing_intervals = num_wing_dofs - degree
    if center_intervals < 1 or wing_intervals < 1:
        raise ValueError(
            f'At least {degree + 1} coefficients are required in each '
            f'center/wing region for degree {degree}.'
        )

    # Adjacent center and wing basis functions overlap across p knot spans.
    # Those spans form the transition without adding transition coefficients.
    transition_intervals = degree

    section_edges = (
        0.,
        left_wing_end,
        left_transition_end,
        right_center_end,
        right_transition_end,
        1.,
    )
    section_interval_counts = (
        wing_intervals,
        transition_intervals,
        center_intervals,
        transition_intervals,
        wing_intervals,
    )
    simple_breakpoints = [np.array([section_edges[0]])]
    for start, stop, num_intervals in zip(
        section_edges[:-1], section_edges[1:], section_interval_counts
    ):
        simple_breakpoints.append(
            np.linspace(start, stop, num_intervals + 1)[1:]
        )
    simple_breakpoints = np.concatenate(simple_breakpoints)
    knots = np.concatenate((
        np.zeros(degree),
        simple_breakpoints,
        np.ones(degree),
    ))
    section_bounds = {
        'center_wing': (0.5, right_center_end),
        'transition': (right_center_end, right_transition_end),
        'wing': (right_transition_end, 1.),
    }
    return knots, section_bounds


def _map_spanwise_coordinates(parametric_coordinates, section_bounds):
    """Map section-local root-to-tip coordinates into the global spline."""
    start, stop = section_bounds
    coordinates = np.asarray(parametric_coordinates)
    return start + coordinates * (stop - start)


def _evaluate_symmetric_spanwise_parameterization(
    center_wing_coefficients,
    wing_coefficients,
    center_wing_parametric_coordinates,
    transition_parametric_coordinates,
    wing_parametric_coordinates,
    section_lengths=None,
):
    """Evaluate one symmetric B-spline over center wing, transition, and wing.

    The supplied center-wing and wing coefficients retain their root-to-tip
    ordering. Their cubic basis functions overlap across the transition; no
    transition coefficients are added. All interior knots are simple, so the
    cubic spline is C2 throughout.
    """
    degree = 3
    if section_lengths is None:
        center_length = float(center_wing_coefficients.shape[0])
        transition_length = float(degree)
        wing_length = float(wing_coefficients.shape[0])
    else:
        center_length, transition_length, wing_length = map(float, section_lengths)

    full_center_coefficients = csdl.concatenate((
        center_wing_coefficients[::-1],
        center_wing_coefficients[1:],
    ), axis=0)

    symmetric_coefficients = csdl.concatenate((
        wing_coefficients[::-1],
        full_center_coefficients,
        wing_coefficients,
    ), axis=0)

    knots, section_bounds = _build_symmetric_spanwise_knot_vector(
        num_center_wing_dofs=center_wing_coefficients.shape[0],
        num_wing_dofs=wing_coefficients.shape[0],
        degree=3,
        section_lengths=(center_length, transition_length, wing_length),
    )
    space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(symmetric_coefficients.shape[0],),
        knots=(knots,),
    )
    function = lfs.Function(
        space=space,
        coefficients=symmetric_coefficients.reshape((-1, 1)),
    )

    def evaluate(section, parametric_coordinates):
        global_coordinates = _map_spanwise_coordinates(
            parametric_coordinates,
            section_bounds[section],
        )
        return function.evaluate(global_coordinates, plot=False).flatten()

    values = {
        'center_wing': evaluate('center_wing', center_wing_parametric_coordinates),
        'transition': evaluate('transition', transition_parametric_coordinates),
        'wing': evaluate('wing', wing_parametric_coordinates),
    }
    return function, values


def _make_oml_refit_space(
    chordwise_resolution,
    base_spanwise_coefficients,
    interface_locations,
    interface_knot_multiplicity,
):
    """Create the OML refit space with optional knots at patch interfaces."""
    if interface_knot_multiplicity not in (0, 1, 2, 3):
        raise ValueError('interface_knot_multiplicity must be 0, 1, 2, or 3.')

    degree = 3
    chordwise_knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0., 1., chordwise_resolution - degree + 1),
        np.ones(degree),
    ))
    spanwise_knots = np.concatenate((
        np.zeros(degree),
        np.linspace(0., 1., base_spanwise_coefficients - degree + 1),
        np.ones(degree),
    ))
    if interface_knot_multiplicity:
        inserted_knots = np.repeat(
            np.asarray(interface_locations),
            interface_knot_multiplicity,
        )
        spanwise_knots = np.sort(np.concatenate((spanwise_knots, inserted_knots)))

    spanwise_coefficients = (
        base_spanwise_coefficients
        + len(interface_locations) * interface_knot_multiplicity
    )
    return lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(degree, degree),
        coefficients_shape=(chordwise_resolution, spanwise_coefficients),
        knots=(chordwise_knots, spanwise_knots),
    )

def initial_refit(
    geometry: lsdo_geo.Geometry,
    plot: bool = False,
    interface_knot_multiplicity: int = 1,
):
    imported_geometry = geometry.copy()
    
    wing_c_indices = [0,1,8,9]  # lower right, upper right, lower left, upper left
    # wing_r_transition_indices = [2,3]   # lower, upper
    # wing_r_indices = [4,5,6,7]  # lower, upper, lower tip, upper tip
    wing_l_transition_indices = [10,11] # lower, upper
    wing_l_indices = [12,13,14,15] # lower, upper, lower tip, upper tip

    left_wing_transition = geometry.declare_component(wing_l_transition_indices)
    left_wing = geometry.declare_component(wing_l_indices)
    # right_wing_transition = geometry.declare_component(wing_r_transition_indices)
    # right_wing = geometry.declare_component(wing_r_indices)
    centerbody = geometry.declare_component(wing_c_indices)
    # oml = geometry.declare_component(oml_indices)

    spanwise_resolution_wing = 4
    chordwise_resolution_wing = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
    wing_surface_b_spline_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(3,3),
        coefficients_shape=(chordwise_resolution_wing, spanwise_resolution_wing)
    )
    lower_indices_center_transition = [10, 8, 0, 2] # left transition, left centerbody, right centerbody, right transition
    upper_indices_center_transition = [11, 9, 1, 3] # left transition, left centerbody, right centerbody, right transition

    lower_left_wing_index = 12
    lower_right_wing_index = 4
    upper_left_wing_index = 13
    upper_right_wing_index = 5
    # lower_indices_wing = [12, 4]
    # upper_indices_wing = [13, 5]

    # Left-side patches need opposite span direction compared to right-side patches
    # for a globally consistent left->right parameter direction when stitching.
    left_lower_indices = set(lower_indices_center_transition[:2])
    left_upper_indices = set(upper_indices_center_transition[:2])

    # discretize each surface
    grid_resolution_wing = (chordwise_resolution_wing*3, 4)

    lower_center_transition_functions = []
    for i, idx in enumerate(lower_indices_center_transition):
        function = geometry.functions[idx]
        function_coefficients = function.coefficients.value
        function_coefficients = np.swapaxes(function_coefficients, 0, 1)  # swap axes to get (chordwise, spanwise, 3)
        if idx in left_lower_indices:
            function_coefficients = function_coefficients[:,::-1,:]  # reverse spanwise direction to match global left->right convention
        else:
            function_coefficients = function_coefficients[::-1,:,:]  # reverse chordwise direction to match global LE->TE convention

        # function.coefficients = csdl.Variable(value=function_coefficients, name=f'function_{idx}_coefficients')
        new_function_space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=(3,3),
            coefficients_shape=function_coefficients.shape[:-1],
            knots=(function.space.knots[1], function.space.knots[0]),
        )
        new_function = lfs.Function(space=new_function_space, coefficients=function_coefficients)
        lower_center_transition_functions.append(new_function)
    
    lower_left_transition = lower_center_transition_functions[0]
    lower_left_centerbody = lower_center_transition_functions[1]
    lower_right_centerbody = lower_center_transition_functions[2]
    lower_right_transition = lower_center_transition_functions[3]


    upper_center_transition_functions = []
    for i, idx in enumerate(upper_indices_center_transition):
        function = geometry.functions[idx]
        function_coefficients = function.coefficients.value
        function_coefficients = np.swapaxes(function_coefficients, 0, 1)  # swap axes to get (chordwise, spanwise, 3)
        if idx in left_upper_indices:
            function_coefficients = function_coefficients[:,::-1,:]  # reverse spanwise direction to match global left->right convention
            function_coefficients = function_coefficients[::-1,:,:]  # reverse chordwise direction to match global LE->TE convention
        else:
            pass    # It's already oriented correctly on the right upper side
 
        new_function_space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=(3,3),
            coefficients_shape=function_coefficients.shape[:-1],
            knots=(function.space.knots[1], function.space.knots[0]),
            # knot_indices=function.space.knot_indices
        )
        new_function = lfs.Function(space=new_function_space, coefficients=function_coefficients)
        upper_center_transition_functions.append(new_function)

    upper_left_transition = upper_center_transition_functions[0]
    upper_left_centerbody = upper_center_transition_functions[1]
    upper_right_centerbody = upper_center_transition_functions[2]
    upper_right_transition = upper_center_transition_functions[3]

    # Refit wing surfaces
    def flip_dimensions_of_parametric_coordinates(parametric_coordinates):
        flipped = parametric_coordinates.copy()
        flipped[:, 0] = parametric_coordinates[:, 1]
        flipped[:, 1] = parametric_coordinates[:, 0]
        return flipped

    lower_left_wing_parametric_coordinates_evaluating_old = geometry.functions[lower_left_wing_index].space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    parametric_coordinates_flipped_uv = flip_dimensions_of_parametric_coordinates(lower_left_wing_parametric_coordinates_evaluating_old)
    u_orig = parametric_coordinates_flipped_uv[:, 0].copy()  # spanwise
    v_orig = parametric_coordinates_flipped_uv[:, 1].copy()  # chordwise
    # Left side: reverse span so it increases left->right in global space.
    lower_left_wing_parametric_coordinates_evaluating_old[:, 0] = 1.0 - u_orig
    # Left side already has LE->TE for chordwise parameter.
    lower_left_wing_parametric_coordinates_evaluating_old[:, 1] = v_orig
    lower_left_wing_discretization = geometry.functions[lower_left_wing_index].evaluate(lower_left_wing_parametric_coordinates_evaluating_old).value.reshape((grid_resolution_wing[0], grid_resolution_wing[1], 3))
    lower_left_wing_parametric_coordinates = wing_surface_b_spline_space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    lower_left_basis_matrix = wing_surface_b_spline_space.compute_basis_matrix(lower_left_wing_parametric_coordinates)
    fitting_matrix = lower_left_basis_matrix.T @ lower_left_basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    rhs = lower_left_basis_matrix.T @ lower_left_wing_discretization.reshape((-1, 3))
    lower_left_wing_coefficients = spsl.spsolve(fitting_matrix, rhs).reshape((chordwise_resolution_wing, spanwise_resolution_wing, 3))

    upper_left_wing_parametric_coordinates_evaluating_old = geometry.functions[upper_left_wing_index].space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    parametric_coordinates_flipped_uv = flip_dimensions_of_parametric_coordinates(upper_left_wing_parametric_coordinates_evaluating_old)
    u_orig = parametric_coordinates_flipped_uv[:, 0].copy()  # spanwise
    v_orig = parametric_coordinates_flipped_uv[:, 1].copy()  # chordwise
    # Left side: reverse span so it increases left->right in global space.
    upper_left_wing_parametric_coordinates_evaluating_old[:, 0] = 1.0 - u_orig
    # Left side chordwise is TE->LE, so flip to LE->TE.
    upper_left_wing_parametric_coordinates_evaluating_old[:, 1] = 1.0 - v_orig
    upper_left_wing_discretization = geometry.functions[upper_left_wing_index].evaluate(upper_left_wing_parametric_coordinates_evaluating_old).value.reshape((grid_resolution_wing[0], grid_resolution_wing[1], 3))
    upper_left_wing_parametric_coordinates = wing_surface_b_spline_space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    upper_left_basis_matrix = wing_surface_b_spline_space.compute_basis_matrix(upper_left_wing_parametric_coordinates)
    fitting_matrix = upper_left_basis_matrix.T @ upper_left_basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    rhs = upper_left_basis_matrix.T @ upper_left_wing_discretization.reshape((-1, 3))
    upper_left_wing_coefficients = spsl.spsolve(fitting_matrix, rhs).reshape((chordwise_resolution_wing, spanwise_resolution_wing, 3))

    lower_right_wing_parametric_coordinates_evaluating_old = geometry.functions[lower_right_wing_index].space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    parametric_coordinates_flipped_uv = flip_dimensions_of_parametric_coordinates(lower_right_wing_parametric_coordinates_evaluating_old)
    u_orig = parametric_coordinates_flipped_uv[:, 0].copy()  # spanwise
    v_orig = parametric_coordinates_flipped_uv[:, 1].copy()  # chordwise
    # Right side: keep span as-is (root->tip is left->right globally).
    lower_right_wing_parametric_coordinates_evaluating_old[:, 0] = u_orig
    # Right side chordwise is TE->LE, so flip to LE->TE.
    lower_right_wing_parametric_coordinates_evaluating_old[:, 1] = 1.0 - v_orig
    lower_right_wing_discretization = geometry.functions[lower_right_wing_index].evaluate(lower_right_wing_parametric_coordinates_evaluating_old).value.reshape((grid_resolution_wing[0], grid_resolution_wing[1], 3))
    lower_right_wing_parametric_coordinates = wing_surface_b_spline_space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    lower_right_basis_matrix = wing_surface_b_spline_space.compute_basis_matrix(lower_right_wing_parametric_coordinates)
    fitting_matrix = lower_right_basis_matrix.T @ lower_right_basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    rhs = lower_right_basis_matrix.T @ lower_right_wing_discretization.reshape((-1, 3))
    lower_right_wing_coefficients = spsl.spsolve(fitting_matrix, rhs).reshape((chordwise_resolution_wing, spanwise_resolution_wing, 3))

    upper_right_wing_parametric_coordinates_evaluating_old = geometry.functions[upper_right_wing_index].space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    parametric_coordinates_flipped_uv = flip_dimensions_of_parametric_coordinates(upper_right_wing_parametric_coordinates_evaluating_old)
    u_orig = parametric_coordinates_flipped_uv[:, 0].copy()  # spanwise
    v_orig = parametric_coordinates_flipped_uv[:, 1].copy()  # chordwise
    # Right side: keep span as-is (root->tip is left->right globally).
    upper_right_wing_parametric_coordinates_evaluating_old[:, 0] = u_orig
    # Right side already has LE->TE for chordwise parameter.
    upper_right_wing_parametric_coordinates_evaluating_old[:, 1] = v_orig
    upper_right_wing_discretization = geometry.functions[upper_right_wing_index].evaluate(upper_right_wing_parametric_coordinates_evaluating_old).value.reshape((grid_resolution_wing[0], grid_resolution_wing[1], 3))
    upper_right_wing_parametric_coordinates = wing_surface_b_spline_space.generate_parametric_grid(grid_resolution=(grid_resolution_wing[0], grid_resolution_wing[1])).copy()
    upper_right_basis_matrix = wing_surface_b_spline_space.compute_basis_matrix(upper_right_wing_parametric_coordinates)
    fitting_matrix = upper_right_basis_matrix.T @ upper_right_basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    rhs = upper_right_basis_matrix.T @ upper_right_wing_discretization.reshape((-1, 3))
    upper_right_wing_coefficients = spsl.spsolve(fitting_matrix, rhs).reshape((chordwise_resolution_wing, spanwise_resolution_wing, 3))

    lower_left_wing = lfs.Function(space=wing_surface_b_spline_space, coefficients=lower_left_wing_coefficients)
    upper_left_wing = lfs.Function(space=wing_surface_b_spline_space, coefficients=upper_left_wing_coefficients)
    lower_right_wing = lfs.Function(space=wing_surface_b_spline_space, coefficients=lower_right_wing_coefficients)
    upper_right_wing = lfs.Function(space=wing_surface_b_spline_space, coefficients=upper_right_wing_coefficients)

    # Create wing caps using coefficients at each tip
    wing_cap_b_spline_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(3,1),
        coefficients_shape=(chordwise_resolution_wing, 2)
    )
    left_wing_cap_coefficients_lower = lower_left_wing_coefficients[:, 0]
    left_wing_cap_coefficients_upper = upper_left_wing_coefficients[:, 0]
    left_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution_wing, 2, 3), value=0.)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 0, :], left_wing_cap_coefficients_lower)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 1, :], left_wing_cap_coefficients_upper)
    left_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=left_wing_cap_coefficients)
    right_wing_cap_coefficients_lower = lower_right_wing_coefficients[:, -1]
    right_wing_cap_coefficients_upper = upper_right_wing_coefficients[:, -1]
    right_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution_wing, 2, 3), value=0.)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 0, :], right_wing_cap_coefficients_lower)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 1, :], right_wing_cap_coefficients_upper)
    right_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=right_wing_cap_coefficients)


    new_geometry = lsdo_geo.Geometry(functions={0: lower_left_wing, 1: lower_left_transition, 2: lower_left_centerbody,
                                                    3: lower_right_centerbody, 4: lower_right_transition, 5: lower_right_wing,
                                                    6: upper_left_wing, 7: upper_left_transition, 8: upper_left_centerbody,
                                                    9: upper_right_centerbody, 10: upper_right_transition, 11: upper_right_wing,
                                                    12: left_wing_cap, 13: right_wing_cap})


    if plot:
        # geometry_plot = geometry.plot(show=False, opacity=0.5)
        geometry_plot = geometry.plot(
            # plot_types=['point_cloud'], point_types=['coefficients'],
            show=False, opacity=0.5)
        new_geometry.plot(
            # plot_types=['point_cloud'], point_types=['coefficients'],
            additional_plotting_elements=geometry_plot, color='red', opacity=0.8)

    return new_geometry


def parameterization_forward_model(geometry:lsdo_geo.Geometry, variables:dict[str,csdl.Variable], states:dict[str,csdl.Variable], 
                                   apply_post_solve_deformations:bool = True, plot:bool=False):

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

    num_centerbody_spanwise_design_dofs = 4
    num_wing_spanwise_design_dofs = 4
    spanwise_parameter_degree = 3

    right_centerbody_coefficients = csdl.Variable(shape=lower_right_centerbody.coefficients.shape[:2] + (2,3), value=0.)
    right_centerbody_coefficients = right_centerbody_coefficients.set(csdl.slice[:, :, 0], lower_right_centerbody.coefficients)
    right_centerbody_coefficients = right_centerbody_coefficients.set(csdl.slice[:, :, 1], upper_right_centerbody.coefficients)
    right_wing_coefficients = csdl.Variable(shape=lower_right_wing.coefficients.shape[:2] + (2,3), value=0.)
    right_wing_coefficients = right_wing_coefficients.set(csdl.slice[:, :, 0], lower_right_wing.coefficients)
    right_wing_coefficients = right_wing_coefficients.set(csdl.slice[:, :, 1], upper_right_wing.coefficients)
    right_transition_coefficients = csdl.Variable(shape=lower_right_transition.coefficients.shape[:2] + (2,3), value=0.)
    right_transition_coefficients = right_transition_coefficients.set(csdl.slice[:, :, 0], lower_right_transition.coefficients)
    right_transition_coefficients = right_transition_coefficients.set(csdl.slice[:, :, 1], upper_right_transition.coefficients)
    
    right_centerbody_sectional_parameterization = lsdo_geo.SectionalParameterization(right_centerbody_coefficients, principal_parametric_dimension=1)
    right_wing_sectional_parameterization = lsdo_geo.SectionalParameterization(right_wing_coefficients, principal_parametric_dimension=1)
    right_transition_sectional_parameterization = lsdo_geo.SectionalParameterization(right_transition_coefficients, principal_parametric_dimension=1)

    # # TODO: PROBABLY WANT TO REDO THIS SINCE I WANT TO USE ONE B-SPLINE FOR THE WHOLE BWB (for each parameter individually)
    reference_centerbody_half_span = right_centerbody_coefficients.value[0,-1,0][1] - right_centerbody_coefficients.value[0,0,0][1]

    reference_wing_half_span = right_wing_coefficients.value[0,-1,0][1] - right_wing_coefficients.value[0,0,0][1]

    reference_transition_half_span = right_transition_coefficients.value[0,-1,0][1] - right_transition_coefficients.value[0,0,0][1]
    # transition_half_span = reference_transition_half_span
    # transition_linear_interpolation_coefficients = 1 - (right_transition_coefficients.value[0,:,0][1] - right_transition_coefficients.value[0,0,0][1]) / transition_half_span
    
    # wing_spanwise_coordinates = 1. - wing_linear_interpolation_coefficients
    # centerbody_spanwise_coordinates = 1. - centerbody_linear_interpolation_coefficients
    # transition_spanwise_coordinates = 1. - transition_linear_interpolation_coefficients
    # reference_section_lengths = (
    #     reference_centerbody_half_span,
    #     reference_transition_half_span,
    #     reference_wing_half_span,
    # )

    reference_half_span = reference_centerbody_half_span + reference_transition_half_span + reference_wing_half_span
    centerbody_half_span_parametric_space_width = reference_centerbody_half_span / reference_half_span / 2
    transition_half_span_parametric_space_width = reference_transition_half_span / reference_half_span / 2
    wing_half_span_parametric_space_width = reference_wing_half_span / reference_half_span / 2
    # knots = []
    # knots.append(np.zeros(spanwise_parameter_degree,))  # No +1 because it's included in the arange for the next section
    # knots.append(np.append(np.linspace(0., wing_half_span_parametric_space_width, num_wing_spanwise_design_dofs - spanwise_parameter_degree, endpoint=False), wing_half_span_parametric_space_width))
    # knots.append(np.append(np.linspace(wing_half_span_parametric_space_width, wing_half_span_parametric_space_width + transition_half_span_parametric_space_width, spanwise_parameter_degree, endpoint=False), wing_half_span_parametric_space_width + transition_half_span_parametric_space_width))
    # knots.append(np.linspace(wing_half_span_parametric_space_width + transition_half_span_parametric_space_width, 0.5, num_centerbody_spanwise_design_dofs - spanwise_parameter_degree + 2, endpoint=True)) 
    knots = []
    knots.append(np.zeros(spanwise_parameter_degree,))  # No +1 because it's included in the arange for the next section
    knots.append(np.linspace(0., wing_half_span_parametric_space_width, num_wing_spanwise_design_dofs - spanwise_parameter_degree, endpoint=False))
    knots.append(np.linspace(wing_half_span_parametric_space_width, wing_half_span_parametric_space_width + transition_half_span_parametric_space_width, spanwise_parameter_degree, endpoint=False))
    knots.append(np.linspace(wing_half_span_parametric_space_width + transition_half_span_parametric_space_width, 0.5, num_centerbody_spanwise_design_dofs - spanwise_parameter_degree + 2, endpoint=True)) 
    # ^ I think +2 cause extra centerline knot and also including both start and endpoint?
    half_knots = np.concatenate(knots)
    knots = np.concatenate((half_knots, 1. - half_knots[::-1][1:]))  # Don't include the first knot of the mirrored half

    parameters_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1,
                                                 degree=spanwise_parameter_degree,
                                                #  coefficients_shape=(2*(num_centerbody_spanwise_design_dofs+num_wing_spanwise_design_dofs+2)-1,), # -1 since centerline is shared
                                                 coefficients_shape=(2*(num_centerbody_spanwise_design_dofs+num_wing_spanwise_design_dofs)-1,), # -1 since centerline is shared
                                                 knots=(knots,))

    centerbody_section_parametric_coordinates = (right_centerbody_coefficients.value[0,:,0,1] / reference_half_span) / 2 + 0.5
    transition_section_parametric_coordinates = ((right_transition_coefficients.value[0,:,0,1] - right_transition_coefficients.value[0,0,0,1]) / reference_half_span) / 2 \
                                        + (0.5 + centerbody_half_span_parametric_space_width)
    wing_section_parametric_coordinates = ((right_wing_coefficients.value[0,:,0,1] - right_wing_coefficients.value[0,0,0,1]) / reference_half_span) / 2 \
                                        + (0.5 + centerbody_half_span_parametric_space_width + transition_half_span_parametric_space_width)

    centerbody_section_parametric_coordinates = centerbody_section_parametric_coordinates.reshape((-1,1))
    transition_section_parametric_coordinates = transition_section_parametric_coordinates.reshape((-1,1))
    wing_section_parametric_coordinates = wing_section_parametric_coordinates.reshape((-1,1))

    def mirror_coefficients(coefficients):
        mirrored_coefficients = csdl.Variable(shape=(2*coefficients.shape[0]-1,), value=0.)
        mirrored_coefficients = mirrored_coefficients.set(csdl.slice[:coefficients.shape[0]-1], coefficients[::-1][:-1])
        mirrored_coefficients = mirrored_coefficients.set(csdl.slice[coefficients.shape[0]-1:], coefficients)
        return mirrored_coefficients
    
    def create_and_evaluate_b_spline_parameterization(coefficients) -> dict[str, csdl.Variable]:
        parameter_b_spline_function = lfs.Function(space=parameters_b_spline_space, coefficients=coefficients)
        # evaluate centerbody, transition, and wing parameterization at the spanwise coordinates of each section
        sectional_values = {}
        sectional_values['centerbody'] = parameter_b_spline_function.evaluate(centerbody_section_parametric_coordinates)
        sectional_values['transition'] = parameter_b_spline_function.evaluate(transition_section_parametric_coordinates)
        sectional_values['wing'] = parameter_b_spline_function.evaluate(wing_section_parametric_coordinates)
        return sectional_values

    def map_coefficients_to_sectional_values(centerbody_coefficients:csdl.Variable, wing_coefficients:csdl.Variable):
        # half_coefficients = csdl.concatenate((centerbody_coefficients, centerbody_coefficients[-1], wing_coefficients[0], wing_coefficients))
        half_coefficients = csdl.concatenate((centerbody_coefficients, wing_coefficients))
        coefficients = mirror_coefficients(half_coefficients)
        sectional_values = create_and_evaluate_b_spline_parameterization(coefficients)
        return sectional_values

    # centerbody_sweep_translation_dofs = states['centerbody_sweep_translation_dofs']
    # wing_tip_sweep_translation = states['wing_tip_sweep_translation']
    # transition_tip_sweep_translation = states['transition_tip_sweep_translation']

    use_centerbody_sweep_translation = 'centerbody_sweep_translation_coefficients' in states
    use_wing_sweep_translation = 'wing_sweep_translation_coefficients' in states
    use_sweep_translation = (use_centerbody_sweep_translation or use_wing_sweep_translation)
    if use_sweep_translation:
        if use_centerbody_sweep_translation:
            centerbody_sweep_translation_coefficients = states['centerbody_sweep_translation_coefficients']
        else:
            centerbody_sweep_translation_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)
        if use_wing_sweep_translation:
            wing_sweep_translation_coefficients = states['wing_sweep_translation_coefficients']
        else:
            wing_sweep_translation_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)

        sweep_translations = map_coefficients_to_sectional_values(centerbody_sweep_translation_coefficients, wing_sweep_translation_coefficients)
        centerbody_sweep_translations = sweep_translations['centerbody']
        transition_sweep_translations = sweep_translations['transition']
        wing_sweep_translations = sweep_translations['wing']

    use_centerbody_span_translation = 'centerbody_half_span' in variables
    use_transition_span_translation = (use_centerbody_span_translation or 'transition_half_span' in variables
                                       or 'wing_half_span' in variables)    # Needed if there's wing span cause smoothing is needed
    use_wing_span_translation = (use_transition_span_translation or 'wing_half_span' in variables)
    use_span_translation = (use_centerbody_span_translation or use_transition_span_translation or use_wing_span_translation)
    if use_span_translation:
        if 'centerbody_half_span' in variables:
            centerbody_half_span = variables['centerbody_half_span']
        else:
            centerbody_half_span = csdl.Variable(value=reference_centerbody_half_span)
        if 'transition_half_span' in variables:
            transition_half_span = variables['transition_half_span']
        else:
            transition_half_span = csdl.Variable(value=reference_transition_half_span)
        if 'wing_half_span' in variables:
            wing_half_span = variables['wing_half_span']
        else:
            wing_half_span = csdl.Variable(value=reference_wing_half_span)

        centerbody_tip_span_stretch = centerbody_half_span - reference_centerbody_half_span
        transition_tip_span_stretch = transition_half_span - reference_transition_half_span + centerbody_tip_span_stretch
        wing_tip_span_stretch = wing_half_span - reference_wing_half_span + centerbody_tip_span_stretch + transition_tip_span_stretch

        centerbody_span_translation_coefficients = csdl.linear_combination(0., centerbody_tip_span_stretch, num_steps=num_centerbody_spanwise_design_dofs).flatten()
        wing_span_translation_coefficients = csdl.linear_combination(transition_tip_span_stretch, wing_tip_span_stretch, num_steps=num_wing_spanwise_design_dofs).flatten()

        span_translations = map_coefficients_to_sectional_values(centerbody_span_translation_coefficients, wing_span_translation_coefficients)
        # centerbody_span_translations = span_translations['centerbody']
        # transition_span_translations = span_translations['transition']
        # wing_span_translations = span_translations['wing']
        # Centerbody absolutely needs to go through 0 at the centerline, so linearly interpolate from 0 to transition root translation
        centerbody_linear_interpolation_coefficients = 1 - (right_centerbody_coefficients.value[0,:,0,1] / reference_centerbody_half_span)
        centerbody_span_translations = csdl.linear_combination(0., centerbody_tip_span_stretch, 
                                                               start_weights=centerbody_linear_interpolation_coefficients,
                                                               stop_weights=1-centerbody_linear_interpolation_coefficients,
                                                               num_steps=len(centerbody_linear_interpolation_coefficients)).flatten()
        # NOTE: Evaluating the B-spline to get the translations on the left side would be incorrect because the sign isn't mirrored
        # I got a weird result when I tried optimization so I'm going to try linearly interpolating for all of the span translations
        transition_linear_interpolation_coefficients = 1 - (right_transition_coefficients.value[0,:,0,1] - right_transition_coefficients.value[0,0,0,1]) / reference_transition_half_span
        transition_span_translations = csdl.linear_combination(centerbody_tip_span_stretch, transition_tip_span_stretch, 
                                                               start_weights=transition_linear_interpolation_coefficients,
                                                                stop_weights=1-transition_linear_interpolation_coefficients,
                                                                num_steps=len(transition_linear_interpolation_coefficients)).flatten()
        wing_linear_interpolation_coefficients = 1 - (right_wing_coefficients.value[0,:,0,1] - right_wing_coefficients.value[0,0,0,1]) / reference_wing_half_span
        wing_span_translations = csdl.linear_combination(transition_tip_span_stretch, wing_tip_span_stretch, 
                                                         start_weights=wing_linear_interpolation_coefficients,
                                                            stop_weights=1-wing_linear_interpolation_coefficients,
                                                            num_steps=len(wing_linear_interpolation_coefficients)).flatten()

    use_centerbody_dihedral_translation = ('centerbody_dihedral_translation_coefficients' in states)
    use_wing_dihedral_translation = ('wing_tip_dihedral_translation' in states)
    use_dihedral_translation = (use_centerbody_dihedral_translation or use_wing_dihedral_translation)
    if use_dihedral_translation:
        if 'wing_tip_dihedral_translation' in states:
            wing_tip_dihedral_translation = states['wing_tip_dihedral_translation']
        else:
            wing_tip_dihedral_translation = csdl.Variable(value=0.)

        if 'centerbody_dihedral_translation_coefficients' in states:
            raise NotImplementedError("Centerbody dihedral translation coefficients are not implemented yet.")

        centerbody_dihedral_translation_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)
        wing_dihedral_translation_coefficients = csdl.linear_combination(0., wing_tip_dihedral_translation, num_steps=num_wing_spanwise_design_dofs).flatten()

        dihedral_translations = map_coefficients_to_sectional_values(centerbody_dihedral_translation_coefficients, wing_dihedral_translation_coefficients)
        centerbody_dihedral_translations = dihedral_translations['centerbody']
        transition_dihedral_translations = dihedral_translations['transition']
        wing_dihedral_translations = dihedral_translations['wing']

    use_centerbody_chord_stretch = ('centerbody_chord_stretch_coefficients' in variables)
    use_wing_chord_stretch = ('wing_chord_stretch_coefficients' in variables)
    use_chord_stretch = (use_centerbody_chord_stretch or use_wing_chord_stretch)
    if use_chord_stretch:
        if 'centerbody_chord_stretch_coefficients' in variables:
            centerbody_chord_coefficients = variables['centerbody_chord_stretch_coefficients']
        else:
            centerbody_chord_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)
        if 'wing_chord_stretch_coefficients' in variables:
            wing_chord_stretch_coefficients = variables['wing_chord_stretch_coefficients']
        else:
            wing_chord_stretch_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)
        
        chord_stretches = map_coefficients_to_sectional_values(centerbody_chord_coefficients, wing_chord_stretch_coefficients)
        centerbody_chord_stretches = chord_stretches['centerbody']
        transition_chord_stretches = chord_stretches['transition']
        wing_chord_stretches = chord_stretches['wing']

        # right_centerbody_initial_chords = right_centerbody_coefficients.value[-1,:,0,0] - right_centerbody_coefficients.value[0,:,0,0]
        # right_transition_initial_chords = right_transition_coefficients.value[-1,:,0,0] - right_transition_coefficients.value[0,:,0,0]
        # right_wing_initial_chords = right_wing_coefficients.value[-1,:,0,0] - right_wing_coefficients.value[0,:,0,0]
        # centerbody_chord_stretches = centerbody_chord_stretches * right_centerbody_initial_chords
        # transition_chord_stretches = transition_chord_stretches * right_transition_initial_chords
        # wing_chord_stretches = wing_chord_stretches * right_wing_initial_chords

        

    use_centerbody_thickness_stretch = ('centerbody_thickness_stretch_coefficients' in variables or
                                       'centerbody_thickness_stretch_coefficients' in states)
    use_wing_thickness_stretch = ('wing_thickness_stretch_coefficients' in variables or
                                 'wing_thickness_stretch_coefficients' in states)
    use_thickness_stretch = (use_centerbody_thickness_stretch or use_wing_thickness_stretch)
    if use_thickness_stretch:
        if 'centerbody_thickness_stretch_coefficients' in variables:
            centerbody_thickness_stretch_coefficients = variables['centerbody_thickness_stretch_coefficients']
        elif 'centerbody_thickness_stretch_coefficients' in states:
            centerbody_thickness_stretch_coefficients = states['centerbody_thickness_stretch_coefficients']
        else:
            centerbody_thickness_stretch_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)

        if 'wing_thickness_stretch_coefficients' in variables:
            wing_thickness_stretch_coefficients = variables['wing_thickness_stretch_coefficients']
        elif 'wing_thickness_stretch_coefficients' in states:
            wing_thickness_stretch_coefficients = states['wing_thickness_stretch_coefficients']
        else:
            wing_thickness_stretch_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)

        thickness_stretches = map_coefficients_to_sectional_values(centerbody_thickness_stretch_coefficients, wing_thickness_stretch_coefficients)
        centerbody_thickness_stretches = thickness_stretches['centerbody']
        transition_thickness_stretches = thickness_stretches['transition']
        wing_thickness_stretches = thickness_stretches['wing']

    baseline_wing_twist_coefficients = np.linspace(2.5, 0., num_wing_spanwise_design_dofs)
    use_centerbody_twist = ('centerbody_twist_coefficients' in variables)
    use_wing_twist = ('wing_twist_coefficients' in variables or use_baseline_twist)
    use_twist = (use_centerbody_twist or use_wing_twist)
    if use_twist:
        if 'centerbody_twist_coefficients' in variables:
            centerbody_twist_coefficients = variables['centerbody_twist_coefficients']*np.pi/180
        else:
            centerbody_twist_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)    

        if 'wing_twist_coefficients' in variables and apply_post_solve_deformations:
            wing_twist_coefficients = variables['wing_twist_coefficients']*np.pi/180
        else:
            wing_twist_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)
        if use_baseline_twist:
            wing_twist_coefficients = wing_twist_coefficients + baseline_wing_twist_coefficients*np.pi/180

        twists = map_coefficients_to_sectional_values(centerbody_twist_coefficients, wing_twist_coefficients)
        centerbody_twists = twists['centerbody']
        transition_twists = twists['transition']
        wing_twists = twists['wing']

    # if enforce_thickness_to_chord_ratio:
    #     wing_thickness_stretch_active = wing_chord_stretch_active
    #     centerbody_thickness_stretch_active = centerbody_chord_stretch_active
    # else:
    #     wing_thickness_stretch_active = ('wing_thickness_stretch_coefficients' in variables)
    #     centerbody_thickness_stretch_active = ('centerbody_thickness_stretch_coefficients' in variables)
    # transition_thickness_stretch_active = wing_thickness_stretch_active or centerbody_thickness_stretch_active


    right_centerbody_sectional_parameterization_inputs = lsdo_geo.SectionalParameters()
    if use_centerbody_sweep_translation:
        right_centerbody_sectional_parameterization_inputs.add_translation(np.array([1., 0., 0.]), centerbody_sweep_translations)
    if use_centerbody_span_translation:
        right_centerbody_sectional_parameterization_inputs.add_translation(np.array([0., 1., 0.]), centerbody_span_translations)
    if use_centerbody_dihedral_translation:
        right_centerbody_sectional_parameterization_inputs.add_translation(np.array([0., 0., 1.]), centerbody_dihedral_translations)
    if use_centerbody_chord_stretch:
        right_centerbody_sectional_parameterization_inputs.add_stretch(np.array([1., 0., 0.]), centerbody_chord_stretches)
    if use_centerbody_thickness_stretch:
        right_centerbody_sectional_parameterization_inputs.add_stretch(np.array([0., 0., 1.]), centerbody_thickness_stretches)
    if use_centerbody_twist:
        right_centerbody_sectional_parameterization_inputs.add_rotation(np.array([0., 1., 0.]), centerbody_twists)


    right_wing_sectional_parameterization_inputs = lsdo_geo.SectionalParameters()
    if use_wing_sweep_translation:
        right_wing_sectional_parameterization_inputs.add_translation(np.array([1., 0., 0.]), wing_sweep_translations)
    if use_wing_span_translation:
        right_wing_sectional_parameterization_inputs.add_translation(np.array([0., 1., 0.]), wing_span_translations)
    if use_wing_dihedral_translation:
        right_wing_sectional_parameterization_inputs.add_translation(np.array([0., 0., 1.]), wing_dihedral_translations)
    if use_wing_chord_stretch:
        # if enforce_no_wing_leading_edge_sweep:
        #     right_wing_sectional_parameterization_inputs.add_stretch(np.array([1., 0., 0.]), wing_chord_stretches, parametric_coordinate=np.array([0., 0.5]))
        # else:
        right_wing_sectional_parameterization_inputs.add_stretch(np.array([1., 0., 0.]), wing_chord_stretches)
    if use_wing_thickness_stretch:
        right_wing_sectional_parameterization_inputs.add_stretch(np.array([0., 0., 1.]), wing_thickness_stretches)
    # Always applying this now since we always need to at least apply the baseline twist distribution
    if use_wing_twist:
        right_wing_sectional_parameterization_inputs.add_rotation(np.array([0., 1., 0.]), wing_twists)


    right_transition_sectional_parameterization_inputs = lsdo_geo.SectionalParameters()
    if use_sweep_translation:
        right_transition_sectional_parameterization_inputs.add_translation(np.array([1., 0., 0.]), transition_sweep_translations)
    if use_transition_span_translation:
        right_transition_sectional_parameterization_inputs.add_translation(np.array([0., 1., 0.]), transition_span_translations)
    if use_dihedral_translation:
        right_transition_sectional_parameterization_inputs.add_translation(np.array([0., 0., 1.]), transition_dihedral_translations)
    if use_chord_stretch:
        # stretch_origin_parametric_coordinates = []
        # if enforce_no_centerbody_trailing_edge_sweep:
        #     # stretch_origin_parametric_coordinates.append(np.array([1., 0.5]))
        #     # stretch_origin_parametric_coordinates.append(np.array([1., 0.5]))
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        # else:
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        # if enforce_no_wing_leading_edge_sweep:
        #     stretch_origin_parametric_coordinates.append(np.array([0., 0.5]))
        #     stretch_origin_parametric_coordinates.append(np.array([0., 0.5]))
        # else:
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        #     stretch_origin_parametric_coordinates.append(np.array([0.5, 0.5]))
        # origins_b_spline = lfs.Function(space=transition_parameters_b_spline_space, coefficients=np.array(stretch_origin_parametric_coordinates))
        # stretch_origin_parametric_coordinates = origins_b_spline.evaluate(1-transition_linear_interpolation_coefficients, non_csdl=True)
        # right_transition_sectional_parameterization_inputs.add_stretch(np.array([1., 0., 0.]), transition_chord_stretches,
        #                                                                 parametric_coordinate=stretch_origin_parametric_coordinates)
        right_transition_sectional_parameterization_inputs.add_stretch(np.array([1., 0., 0.]), transition_chord_stretches)
    if use_thickness_stretch:
        right_transition_sectional_parameterization_inputs.add_stretch(np.array([0., 0., 1.]), transition_thickness_stretches)
    # There is always wing twist now since we always need to at least apply the baseline twist distribution
    if use_twist:
        right_transition_sectional_parameterization_inputs.add_rotation(np.array([0., 1., 0.]), transition_twists)

    updated_right_wing_coefficients = right_wing_sectional_parameterization.evaluate(right_wing_sectional_parameterization_inputs, plot=False)
    updated_right_centerbody_coefficients = right_centerbody_sectional_parameterization.evaluate(right_centerbody_sectional_parameterization_inputs, plot=False)
    updated_right_transition_coefficients = right_transition_sectional_parameterization.evaluate(right_transition_sectional_parameterization_inputs, plot=False)

    if apply_post_solve_deformations:
        # Apply explicit deformations after the parameterization solver to keep the parameterization solver as cheap as possible
            
        if 'airfoil_thickness_coefficients' in variables.keys() or 'airfoil_camber_coefficients' in variables.keys():
            airfoil_parameters_b_spline_space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=(3,3),
            # coefficients_shape=(8,8)
            coefficients_shape=(8,15)
            )
    
            center_tip_y = updated_right_centerbody_coefficients[0,-1,1,1].value
            transition_tip_y = updated_right_transition_coefficients[0,-1,1,1].value
            wing_tip_y = updated_right_wing_coefficients[0,-1,1,1].value
    
            chord_lengths_right_center = csdl.expand(updated_right_centerbody_coefficients.value[-1,:,0,0].flatten() - updated_right_centerbody_coefficients.value[0,:,0,0].flatten(),
                updated_right_centerbody_coefficients.shape[:2], 'j->ij').value
            chord_lengths_right_transition = csdl.expand(updated_right_transition_coefficients.value[-1,:,0,0].flatten() - updated_right_transition_coefficients.value[0,:,0,0].flatten(),
                updated_right_transition_coefficients.shape[:2], 'j->ij').value
            chord_lengths_right_wing = csdl.expand(updated_right_wing_coefficients.value[-1,:,0,0].flatten() - updated_right_wing_coefficients.value[0,:,0,0].flatten(),
                updated_right_wing_coefficients.shape[:2], 'j->ij').value
            leading_edge_x_center = csdl.expand(updated_right_centerbody_coefficients.value[0,:,0,0].flatten(), updated_right_centerbody_coefficients.shape[:2], 'j->ij').value
            leading_edge_x_transition = csdl.expand(updated_right_transition_coefficients.value[0,:,0,0].flatten(), updated_right_transition_coefficients.shape[:2], 'j->ij').value
            leading_edge_x_wing = csdl.expand(updated_right_wing_coefficients.value[0,:,0,0].flatten(), updated_right_wing_coefficients.shape[:2], 'j->ij').value
    
            scaled_lower_right_center_parametric_coordinates = np.zeros(shape=lower_right_centerbody.coefficients.shape[:2]+(2,))
            scaled_lower_right_center_parametric_coordinates[:,:,0] = \
                ((updated_right_centerbody_coefficients.value[:,:,0,0] - leading_edge_x_center)/chord_lengths_right_center)
            scaled_lower_right_center_parametric_coordinates[:,:,1] = updated_right_centerbody_coefficients.value[:,:,0,1]/wing_tip_y
            scaled_lower_right_center_parametric_coordinates[:,:,1] = scaled_lower_right_center_parametric_coordinates[:,:,1]*0.5 + 0.5
            scaled_upper_right_center_parametric_coordinates = np.zeros(shape=upper_right_centerbody.coefficients.shape[:2]+(2,))
            scaled_upper_right_center_parametric_coordinates[:,:,0] = \
                ((updated_right_centerbody_coefficients.value[:,:,1,0] - leading_edge_x_center)/chord_lengths_right_center)
            scaled_upper_right_center_parametric_coordinates[:,:,1] = updated_right_centerbody_coefficients.value[:,:,1,1]/wing_tip_y
            scaled_upper_right_center_parametric_coordinates[:,:,1] = scaled_upper_right_center_parametric_coordinates[:,:,1]*0.5 + 0.5
    

            scaled_lower_right_transition_parametric_coordinates = np.zeros(shape=lower_right_transition.coefficients.shape[:2]+(2,))
            scaled_lower_right_transition_parametric_coordinates[:,:,0] = \
                ((updated_right_transition_coefficients.value[:,:,0,0] - leading_edge_x_transition)/chord_lengths_right_transition)
            scaled_lower_right_transition_parametric_coordinates[:,:,1] = updated_right_transition_coefficients.value[:,:,0,1]/wing_tip_y
            scaled_lower_right_transition_parametric_coordinates[:,:,1] = scaled_lower_right_transition_parametric_coordinates[:,:,1]*0.5 + 0.5
            scaled_upper_right_transition_parametric_coordinates = np.zeros(shape=upper_right_transition.coefficients.shape[:2]+(2,))
            scaled_upper_right_transition_parametric_coordinates[:,:,0] = \
                ((updated_right_transition_coefficients.value[:,:,1,0] - leading_edge_x_transition)/chord_lengths_right_transition)
            scaled_upper_right_transition_parametric_coordinates[:,:,1] = updated_right_transition_coefficients.value[:,:,1,1]/wing_tip_y
            scaled_upper_right_transition_parametric_coordinates[:,:,1] = scaled_upper_right_transition_parametric_coordinates[:,:,1]*0.5 + 0.5
    

            scaled_lower_right_wing_parametric_coordinates = np.zeros(shape=lower_right_wing.coefficients.shape[:2]+(2,))
            scaled_lower_right_wing_parametric_coordinates[:,:,0] = \
                ((updated_right_wing_coefficients.value[:,:,0,0] - leading_edge_x_wing)/chord_lengths_right_wing)
            scaled_lower_right_wing_parametric_coordinates[:,:,1] = updated_right_wing_coefficients.value[:,:,0,1]/wing_tip_y
            scaled_lower_right_wing_parametric_coordinates[:,:,1] = scaled_lower_right_wing_parametric_coordinates[:,:,1]*0.5 + 0.5
            scaled_upper_right_wing_parametric_coordinates = np.zeros(shape=upper_right_wing.coefficients.shape[:2]+(2,))
            scaled_upper_right_wing_parametric_coordinates[:,:,0] = \
                ((updated_right_wing_coefficients.value[:,:,1,0] - leading_edge_x_wing)/chord_lengths_right_wing)
            scaled_upper_right_wing_parametric_coordinates[:,:,1] = updated_right_wing_coefficients.value[:,:,1,1]/wing_tip_y
            scaled_upper_right_wing_parametric_coordinates[:,:,1] = scaled_upper_right_wing_parametric_coordinates[:,:,1]*0.5 + 0.5
    
            chords_right_center_mesh = updated_right_centerbody_coefficients[-1,:,1,0] - updated_right_centerbody_coefficients[0,:,1,0]
            chords_right_transition_mesh = updated_right_transition_coefficients[-1,:,1,0] - updated_right_transition_coefficients[0,:,1,0]
            chords_right_wing_mesh = updated_right_wing_coefficients[-1,:,1,0] - updated_right_wing_coefficients[0,:,1,0]
            chords_right_center_mesh = csdl.expand(chords_right_center_mesh, out_shape=updated_right_centerbody_coefficients[:,:,0,0].shape, action='j->ij')
            chords_right_transition_mesh = csdl.expand(chords_right_transition_mesh, out_shape=updated_right_transition_coefficients[:,:,0,0].shape, action='j->ij')
            chords_right_wing_mesh = csdl.expand(chords_right_wing_mesh, out_shape=updated_right_wing_coefficients[:,:,0,0].shape, action='j->ij')
    
            thickness_right_center_mesh = updated_right_centerbody_coefficients[:,:,1,2] - updated_right_centerbody_coefficients[:,:,0,2]
            thickness_right_transition_mesh = updated_right_transition_coefficients[:,:,1,2] - updated_right_transition_coefficients[:,:,0,2]
            thickness_right_wing_mesh = updated_right_wing_coefficients[:,:,1,2] - updated_right_wing_coefficients[:,:,0,2]
    
    
        def split_stitched_airfoil_parameter_values(airfoil_parameter_values, discretizations):
            airfoil_parameter_sections = []
            start_index = 0
            for discretization in discretizations:
                point_count = discretization[1].reshape((-1, 3)).shape[0]
                stop_index = start_index + point_count
                airfoil_parameter_sections.append(airfoil_parameter_values[start_index:stop_index])
                start_index = stop_index
            return airfoil_parameter_sections
    
        def apply_thickness_to_volume_surface(volume_mesh, surface_index, thickness_values, is_lower_surface):
            surface_points = volume_mesh[:,:,surface_index,:]
            # thickness_values = thickness_values.reshape(surface_points.shape[0], surface_points.shape[1])
            if is_lower_surface:
                surface_points = surface_points.set(csdl.slice[:,:,2], surface_points[:,:,2] - thickness_values/2)
            else:
                surface_points = surface_points.set(csdl.slice[:,:,2], surface_points[:,:,2] + thickness_values/2)
            return volume_mesh.set(csdl.slice[:,:,surface_index,:], surface_points)
        
        def apply_camber_to_volume_surface(volume_mesh, surface_index, camber_values):
            surface_points = volume_mesh[:,:,surface_index,:]
            camber_values = camber_values.reshape(surface_points.shape[0], surface_points.shape[1])
            surface_points = surface_points.set(csdl.slice[:,:,2], surface_points[:,:,2] + camber_values)
            return volume_mesh.set(csdl.slice[:,:,surface_index,:], surface_points)
    
        if 'airfoil_thickness_coefficients' in variables.keys():
            airfoil_thickness_coefficients = variables['airfoil_thickness_coefficients']
            airfoil_thickness_coefficients_mirrored = csdl.concatenate((airfoil_thickness_coefficients[:,:0:-1], airfoil_thickness_coefficients), axis=1)
            airfoil_thickness_function = lfs.Function(space=airfoil_parameters_b_spline_space, coefficients=airfoil_thickness_coefficients_mirrored)
            
            # evaluate airfoil thickness function at discretization points and apply thickness deformation to the discretization
            airfoil_percent_thickness_to_chord_change_lower_right_center = \
                airfoil_thickness_function.evaluate(scaled_lower_right_center_parametric_coordinates, plot=False).reshape(updated_right_centerbody_coefficients[:,:,0,0].shape)
            airfoil_percent_thickness_to_chord_change_upper_right_center = \
                airfoil_thickness_function.evaluate(scaled_upper_right_center_parametric_coordinates, plot=False).reshape(updated_right_centerbody_coefficients[:,:,0,0].shape)
            airfoil_percent_thickness_to_chord_change_lower_right_transition = \
                airfoil_thickness_function.evaluate(scaled_lower_right_transition_parametric_coordinates, plot=False).reshape(updated_right_transition_coefficients[:,:,0,0].shape)
            airfoil_percent_thickness_to_chord_change_upper_right_transition = \
                airfoil_thickness_function.evaluate(scaled_upper_right_transition_parametric_coordinates, plot=False).reshape(updated_right_transition_coefficients[:,:,0,0].shape)
            airfoil_percent_thickness_to_chord_change_lower_right_wing = \
                airfoil_thickness_function.evaluate(scaled_lower_right_wing_parametric_coordinates, plot=False).reshape(updated_right_wing_coefficients[:,:,0,0].shape)
            airfoil_percent_thickness_to_chord_change_upper_right_wing = \
                airfoil_thickness_function.evaluate(scaled_upper_right_wing_parametric_coordinates, plot=False).reshape(updated_right_wing_coefficients[:,:,0,0].shape)
            # airfoil_thickness_lower_left_wing = airfoil_thickness_function.evaluate(scaled_lower_left_wing_parametric_coordinates, plot=False)
            # airfoil_thickness_upper_left_wing = airfoil_thickness_function.evaluate(scaled_upper_left_wing_parametric_coordinates, plot=False)
    
            airfoil_thickness_lower_right_center = airfoil_percent_thickness_to_chord_change_lower_right_center * thickness_right_center_mesh
            airfoil_thickness_upper_right_center = airfoil_percent_thickness_to_chord_change_upper_right_center * thickness_right_center_mesh
            airfoil_thickness_lower_right_transition = airfoil_percent_thickness_to_chord_change_lower_right_transition * thickness_right_transition_mesh
            airfoil_thickness_upper_right_transition = airfoil_percent_thickness_to_chord_change_upper_right_transition * thickness_right_transition_mesh
            airfoil_thickness_lower_right_wing = airfoil_percent_thickness_to_chord_change_lower_right_wing * thickness_right_wing_mesh
            airfoil_thickness_upper_right_wing = airfoil_percent_thickness_to_chord_change_upper_right_wing * thickness_right_wing_mesh

    
            updated_right_centerbody_coefficients = apply_thickness_to_volume_surface(updated_right_centerbody_coefficients, 0, airfoil_thickness_lower_right_center, True)
            updated_right_centerbody_coefficients = apply_thickness_to_volume_surface(updated_right_centerbody_coefficients, 1, airfoil_thickness_upper_right_center, False)
            updated_right_transition_coefficients = apply_thickness_to_volume_surface(updated_right_transition_coefficients, 0, airfoil_thickness_lower_right_transition, True)
            updated_right_transition_coefficients = apply_thickness_to_volume_surface(updated_right_transition_coefficients, 1, airfoil_thickness_upper_right_transition, False)
            # updated_left_centerbody_coefficients = apply_thickness_to_volume_surface(updated_left_centerbody_coefficients, 0, lower_center_thickness_sections[-3], True)
            # updated_left_centerbody_coefficients = apply_thickness_to_volume_surface(updated_left_centerbody_coefficients, 1, upper_center_thickness_sections[-3], False)
            # updated_left_transition_coefficients = apply_thickness_to_volume_surface(updated_left_transition_coefficients, 0, lower_center_thickness_sections[-4], True)
            # updated_left_transition_coefficients = apply_thickness_to_volume_surface(updated_left_transition_coefficients, 1, upper_center_thickness_sections[-4], False)
            updated_right_wing_coefficients = apply_thickness_to_volume_surface(updated_right_wing_coefficients, 0, airfoil_thickness_lower_right_wing, True)
            updated_right_wing_coefficients = apply_thickness_to_volume_surface(updated_right_wing_coefficients, 1, airfoil_thickness_upper_right_wing, False)
            # updated_left_wing_coefficients = apply_thickness_to_volume_surface(updated_left_wing_coefficients, 0, airfoil_thickness_lower_left_wing, True)
            # updated_left_wing_coefficients = apply_thickness_to_volume_surface(updated_left_wing_coefficients, 1, airfoil_thickness_upper_left_wing, False)
    
    
        if 'airfoil_camber_coefficients' in variables.keys():
            airfoil_camber_coefficients_excluding_leading_and_trailing = variables['airfoil_camber_coefficients']
            airfoil_camber_coefficients = csdl.Variable(shape=(airfoil_camber_coefficients_excluding_leading_and_trailing.shape[0]+2,)+ 
                                                            airfoil_camber_coefficients_excluding_leading_and_trailing.shape[1:],
                                                            value=0.)
            airfoil_camber_coefficients = airfoil_camber_coefficients.set(csdl.slice[1:-1,:], airfoil_camber_coefficients_excluding_leading_and_trailing)
            airfoil_camber_coefficients_mirrored = csdl.concatenate((airfoil_camber_coefficients[:,:0:-1], airfoil_camber_coefficients), axis=1)
            airfoil_camber_function = lfs.Function(space=airfoil_parameters_b_spline_space, coefficients=airfoil_camber_coefficients_mirrored)
    
            # evaluate airfoil camber function at discretization points and apply camber deformation to the discretization
            airfoil_camber_lower_right_center = \
                airfoil_camber_function.evaluate(scaled_lower_right_center_parametric_coordinates, plot=False).reshape(updated_right_centerbody_coefficients[:,:,0,0].shape)
            airfoil_camber_upper_right_center = \
                airfoil_camber_function.evaluate(scaled_upper_right_center_parametric_coordinates, plot=False).reshape(updated_right_centerbody_coefficients[:,:,0,0].shape)
            airfoil_camber_lower_right_transition = \
                airfoil_camber_function.evaluate(scaled_lower_right_transition_parametric_coordinates, plot=False).reshape(updated_right_transition_coefficients[:,:,0,0].shape)
            airfoil_camber_upper_right_transition = \
                airfoil_camber_function.evaluate(scaled_upper_right_transition_parametric_coordinates, plot=False).reshape(updated_right_transition_coefficients[:,:,0,0].shape)
            # airfoil_camber_lower_left_center = airfoil
            airfoil_camber_lower_right_wing = \
                airfoil_camber_function.evaluate(scaled_lower_right_wing_parametric_coordinates, plot=False).reshape(updated_right_wing_coefficients[:,:,0,0].shape)
            airfoil_camber_upper_right_wing = \
                airfoil_camber_function.evaluate(scaled_upper_right_wing_parametric_coordinates, plot=False).reshape(updated_right_wing_coefficients[:,:,0,0].shape)
            # airfoil_camber_lower_left_wing = airfoil_camber_function.evaluate(scaled_lower_left_wing_parametric_coordinates, plot=False)
            # airfoil_camber_upper_left_wing = airfoil_camber_function.evaluate(scaled_upper_left_wing_parametric_coordinates, plot=False)
    
            # Apply camber deformation to the discretization points
            # lower_center_camber_sections = split_stitched_airfoil_parameter_values(airfoil_camber_lower_center, lower_center_discretizations)
            # upper_center_camber_sections = split_stitched_airfoil_parameter_values(airfoil_camber_upper_center, upper_center_discretizations)
    
            airfoil_camber_lower_right_center = airfoil_camber_lower_right_center * chords_right_center_mesh
            airfoil_camber_upper_right_center = airfoil_camber_upper_right_center * chords_right_center_mesh
            airfoil_camber_lower_right_transition = airfoil_camber_lower_right_transition * chords_right_transition_mesh
            airfoil_camber_upper_right_transition = airfoil_camber_upper_right_transition * chords_right_transition_mesh
            airfoil_camber_lower_right_wing = airfoil_camber_lower_right_wing * chords_right_wing_mesh
            airfoil_camber_upper_right_wing = airfoil_camber_upper_right_wing * chords_right_wing_mesh
    
            updated_right_centerbody_coefficients = apply_camber_to_volume_surface(updated_right_centerbody_coefficients, 0, airfoil_camber_lower_right_center)
            updated_right_centerbody_coefficients = apply_camber_to_volume_surface(updated_right_centerbody_coefficients, 1, airfoil_camber_upper_right_center)
            updated_right_transition_coefficients = apply_camber_to_volume_surface(updated_right_transition_coefficients, 0, airfoil_camber_lower_right_transition)
            updated_right_transition_coefficients = apply_camber_to_volume_surface(updated_right_transition_coefficients, 1, airfoil_camber_upper_right_transition)
            # updated_left_centerbody_coefficients = apply_camber_to_volume_surface(updated_left_centerbody_coefficients, 0, lower_center_camber_sections[-3])
            # updated_left_centerbody_coefficients = apply_camber_to_volume_surface(updated_left_centerbody_coefficients, 1, upper_center_camber_sections[-3])
            # updated_left_transition_coefficients = apply_camber_to_volume_surface(updated_left_transition_coefficients, 0, lower_center_camber_sections[-4])
            # updated_left_transition_coefficients = apply_camber_to_volume_surface(updated_left_transition_coefficients, 1, upper_center_camber_sections[-4])
            updated_right_wing_coefficients = apply_camber_to_volume_surface(updated_right_wing_coefficients, 0, airfoil_camber_lower_right_wing)
            updated_right_wing_coefficients = apply_camber_to_volume_surface(updated_right_wing_coefficients, 1, airfoil_camber_upper_right_wing)
            # updated_left_wing_coefficients = apply_camber_to_volume_surface(updated_left_wing_coefficients, 0, airfoil_camber_lower_left_wing)
            # updated_left_wing_coefficients = apply_camber_to_volume_surface(updated_left_wing_coefficients, 1, airfoil_camber_upper_left_wing)
    
    
        # add elevator deflection as a local rotation on the last elevator_num_rows of discretization points for the center body
        elevator_num_rows = 5
        if 'elevator_deflection' in variables:
            elevator_deflection = variables['elevator_deflection']*np.pi/180
            elevator_rotation_axis = np.array([0., -1., 0.])
            elevator_rotation_point_upper = updated_right_centerbody_coefficients[-elevator_num_rows, 0, 1]
            elevator_rotation_point_lower = updated_right_centerbody_coefficients[-elevator_num_rows, 0, 0]
            elevator_rotation_point = (elevator_rotation_point_upper + elevator_rotation_point_lower) / 2
            elevator_points = updated_right_centerbody_coefficients[-elevator_num_rows:,:,:,:]
            elevator_points_shape = elevator_points.shape
            elevator_points_rotated = lsdo_geo.rotate(elevator_points, rotation_origin=elevator_rotation_point, axis_vector=elevator_rotation_axis, angles=elevator_deflection)
            updated_right_centerbody_coefficients = updated_right_centerbody_coefficients.set(csdl.slice[-elevator_num_rows:,:,:,:], elevator_points_rotated.reshape(elevator_points_shape))

            # Identify the coefficients in the transition that (initially) have an x coordinate >= the x coordinate of the elevator rotation point, and apply the same rotation to those points
            # indices_for_transition_points_to_rotate = right_transition_coefficients.value[:,:,:,0] >= elevator_rotation_point.value[0]
            # transition_point_indices = tuple(index.tolist() for index in np.where(indices_for_transition_points_to_rotate))
            # transition_points_slice = csdl.slice[
            #     transition_point_indices[0], transition_point_indices[1], transition_point_indices[2], :]
            # transition_points_to_rotate = updated_right_transition_coefficients[transition_points_slice]
            elevator_num_transition_columns = 6
            transition_points_to_rotate = updated_right_transition_coefficients[-elevator_num_rows:,:elevator_num_transition_columns,:,:]
            transition_points_shape = transition_points_to_rotate.shape
            transition_points_rotated = lsdo_geo.rotate(transition_points_to_rotate, rotation_origin=elevator_rotation_point, axis_vector=elevator_rotation_axis, angles=elevator_deflection)
            # updated_right_transition_coefficients = updated_right_transition_coefficients.set(transition_points_slice, transition_points_rotated.reshape(transition_points_shape))
            updated_right_transition_coefficients = updated_right_transition_coefficients.set(csdl.slice[-elevator_num_rows:,:elevator_num_transition_columns,:,:], transition_points_rotated.reshape(transition_points_shape))

    
    # region Construct geometry after explicit changes
    updated_left_wing_coefficients = updated_right_wing_coefficients[:,::-1,:,:]
    updated_left_wing_coefficients = updated_left_wing_coefficients.set(csdl.slice[:,:,:,1], -updated_left_wing_coefficients[:,:,:,1]) # flip y axis
    updated_left_centerbody_coefficients = updated_right_centerbody_coefficients[:,::-1,:,:]
    updated_left_centerbody_coefficients = updated_left_centerbody_coefficients.set(csdl.slice[:,:,:,1], -updated_left_centerbody_coefficients[:,:,:,1]) # flip y axis
    updated_left_transition_coefficients = updated_right_transition_coefficients[:,::-1,:,:]
    updated_left_transition_coefficients = updated_left_transition_coefficients.set(csdl.slice[:,:,:,1], -updated_left_transition_coefficients[:,:,:,1]) # flip y axis

    lower_right_centerbody = lower_right_centerbody.copy()
    upper_right_centerbody = upper_right_centerbody.copy()
    lower_right_wing = lower_right_wing.copy()
    upper_right_wing = upper_right_wing.copy()
    lower_right_transition = lower_right_transition.copy()
    upper_right_transition = upper_right_transition.copy()
    lower_left_centerbody = lower_left_centerbody.copy()
    upper_left_centerbody = upper_left_centerbody.copy()
    lower_left_wing = lower_left_wing.copy()
    upper_left_wing = upper_left_wing.copy()
    lower_left_transition = lower_left_transition.copy()
    upper_left_transition = upper_left_transition.copy()

    lower_right_centerbody.coefficients = updated_right_centerbody_coefficients[:,:,0,:]
    upper_right_centerbody.coefficients = updated_right_centerbody_coefficients[:,:,1,:]
    lower_right_wing.coefficients = updated_right_wing_coefficients[:,:,0,:]
    upper_right_wing.coefficients = updated_right_wing_coefficients[:,:,1,:]
    lower_right_transition.coefficients = updated_right_transition_coefficients[:,:,0,:]
    upper_right_transition.coefficients = updated_right_transition_coefficients[:,:,1,:]
    lower_left_centerbody.coefficients = updated_left_centerbody_coefficients[:,:,0,:]
    upper_left_centerbody.coefficients = updated_left_centerbody_coefficients[:,:,1,:]
    lower_left_wing.coefficients = updated_left_wing_coefficients[:,:,0,:]
    upper_left_wing.coefficients = updated_left_wing_coefficients[:,:,1,:]
    lower_left_transition.coefficients = updated_left_transition_coefficients[:,:,0,:]
    upper_left_transition.coefficients = updated_left_transition_coefficients[:,:,1,:]
    

    # Create wing caps using coefficients at each tip
    wing_cap_b_spline_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(3,1),
        coefficients_shape=(lower_right_wing.coefficients.shape[0], 2)
    )
    left_wing_cap_coefficients_lower = lower_left_wing.coefficients[:, 0]
    left_wing_cap_coefficients_upper = upper_left_wing.coefficients[:, 0]
    left_wing_cap_coefficients = csdl.Variable(shape=(lower_right_wing.coefficients.shape[0], 2, 3), value=0.)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 0, :], left_wing_cap_coefficients_lower)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 1, :], left_wing_cap_coefficients_upper)
    left_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=left_wing_cap_coefficients)
    right_wing_cap_coefficients_lower = lower_right_wing.coefficients[:, -1]
    right_wing_cap_coefficients_upper = upper_right_wing.coefficients[:, -1]
    right_wing_cap_coefficients = csdl.Variable(shape=(lower_right_wing.coefficients.shape[0], 2, 3), value=0.)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 0, :], right_wing_cap_coefficients_lower)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 1, :], right_wing_cap_coefficients_upper)
    right_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=right_wing_cap_coefficients)


    new_geometry = lsdo_geo.Geometry(functions={0: lower_left_wing, 1: lower_left_transition, 2: lower_left_centerbody,
                                                    3: lower_right_centerbody, 4: lower_right_transition, 5: lower_right_wing,
                                                    6: upper_left_wing, 7: upper_left_transition, 8: upper_left_centerbody,
                                                    9: upper_right_centerbody, 10: upper_right_transition, 11: upper_right_wing,
                                                    12: left_wing_cap, 13: right_wing_cap})

    return new_geometry


def setup_and_evaluate_geometry_parameterization(geometry: lsdo_geo.Geometry, variables: dict[str, csdl.Variable],
                                                 plot:bool=False) -> lsdo_geo.Geometry:

    initial_refit_geometry = geometry

    # enforce_no_centerbody_trailing_edge_sweep = False
    # enforce_no_wing_leading_edge_sweep = True
    # enforce_thickness_to_chord_ratio = False

    num_centerbody_spanwise_design_dofs = 4
    num_wing_spanwise_design_dofs = 4

    # wing_tip_sweep_translation = csdl.Variable(value=0.)
    # wing_tip_sweep_translation.add_name('wing_tip_sweep_translation')
    wing_sweep_translation_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)
    wing_sweep_translation_coefficients.add_name('wing_sweep_translation_coefficients')
    centerbody_sweep_translation_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)
    centerbody_sweep_translation_coefficients.add_name('centerbody_sweep_translation_coefficients')
    # transition_tip_sweep_translation = csdl.Variable(value=0.)
    # transition_tip_sweep_translation.add_name('transition_tip_sweep_translation')

    wing_thickness_stretch_coefficients = csdl.Variable(shape=(num_wing_spanwise_design_dofs,), value=0.)
    wing_thickness_stretch_coefficients.add_name('wing_thickness_stretch_coefficients')
    centerbody_thickness_stretch_coefficients = csdl.Variable(shape=(num_centerbody_spanwise_design_dofs,), value=0.)
    centerbody_thickness_stretch_coefficients.add_name('centerbody_thickness_stretch_coefficients')

    wing_tip_dihedral_translation = csdl.Variable(value=0.)
    wing_tip_dihedral_translation.add_name('wing_tip_dihedral_translation')

    states : dict[str, csdl.Variable] = {}
    # states['wing_tip_sweep_translation'] = wing_tip_sweep_translation
    states['wing_sweep_translation_coefficients'] = wing_sweep_translation_coefficients
    if enforce_no_centerbody_trailing_edge_sweep:
        states['centerbody_sweep_translation_coefficients'] = centerbody_sweep_translation_coefficients
    # states['transition_tip_sweep_translation'] = transition_tip_sweep_translation
    if enforce_thickness_to_chord_ratio:
        states['wing_thickness_stretch_coefficients'] = wing_thickness_stretch_coefficients
    if enforce_thickness_to_chord_ratio:
        states['centerbody_thickness_stretch_coefficients'] = centerbody_thickness_stretch_coefficients
    # states['wing_tip_dihedral_translation'] = wing_tip_dihedral_translation

    # This forward model evluation is for the setup of the implicit system
    geometry = parameterization_forward_model(initial_refit_geometry, variables, states,
                                              plot=plot, apply_post_solve_deformations=False)

    parameterization_solver = lsdo_geo.ParameterizationSolver()
    
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


    # cumulative_sum_of_spanwise_resolutions = np.cumsum(spanwise_resolutions[:3]+[1]+spanwise_resolutions[3:]) # add 1 for the center section extra spanwise section
    # spanwise_section_boundaries = cumulative_sum_of_spanwise_resolutions / num_coefficients_spanwise
    upper_wing_root = upper_right_wing.evaluate(np.array([[0.667, 0.]]), plot=False).flatten()
    lower_wing_root = lower_right_wing.evaluate(np.array([[0.667, 0.]]), plot=False).flatten()
    wing_root = (upper_wing_root + lower_wing_root) / 2
    upper_wing_tip = upper_right_wing.evaluate(np.array([[0.667, 1.]]), plot=False).flatten()
    lower_wing_tip = lower_right_wing.evaluate(np.array([[0.667, 1.]]), plot=False).flatten()
    wing_tip = (upper_wing_tip + lower_wing_tip) / 2
    upper_centerbody_root = upper_right_centerbody.evaluate(np.array([[0.667, 0.]]), plot=False).flatten()
    lower_centerbody_root = lower_right_centerbody.evaluate(np.array([[0.667, 0.]]), plot=False).flatten()
    centerbody_root = (upper_centerbody_root + lower_centerbody_root) / 2
    upper_centerbody_tip = upper_right_centerbody.evaluate(np.array([[0.667, 1.]]), plot=False).flatten()
    lower_centerbody_tip = lower_right_centerbody.evaluate(np.array([[0.667, 1.]]), plot=False).flatten()
    centerbody_tip = (upper_centerbody_tip + lower_centerbody_tip) / 2

    wing_root_leading_edge = upper_right_wing.evaluate(np.array([[0.0, 0.]]), plot=False).flatten()
    wing_root_trailing_edge = upper_right_wing.evaluate(np.array([[1.0, 0.]]), plot=False).flatten()
    wing_tip_leading_edge = upper_right_wing.evaluate(np.array([[0.0, 1.]]), plot=False).flatten()
    wing_tip_trailing_edge = upper_right_wing.evaluate(np.array([[1.0, 1.]]), plot=False).flatten()
    wing_station_2_leading_edge = upper_right_wing.evaluate(np.array([[0.0, 1/3]]), plot=False).flatten()
    wing_station_2_trailing_edge = upper_right_wing.evaluate(np.array([[1.0, 1/3]]), plot=False).flatten()
    wing_station_3_leading_edge = upper_right_wing.evaluate(np.array([[0.0, 2/3]]), plot=False).flatten()
    wing_station_3_trailing_edge = upper_right_wing.evaluate(np.array([[1.0, 2/3]]), plot=False).flatten()
    wing_root_quarter_chord_upper = upper_right_wing.evaluate(np.array([0.667, 0.]), plot=False).flatten()
    wing_root_quarter_chord_lower = lower_right_wing.evaluate(np.array([0.667, 0.]), plot=False).flatten()
    wing_tip_quarter_chord_upper = upper_right_wing.evaluate(np.array([0.667, 1.]), plot=False).flatten()
    wing_tip_quarter_chord_lower = lower_right_wing.evaluate(np.array([0.667, 1.]), plot=False).flatten()
    wing_station_2_quarter_chord_upper = upper_right_wing.evaluate(np.array([0.667, 1/3]), plot=False).flatten()
    wing_station_2_quarter_chord_lower = lower_right_wing.evaluate(np.array([0.667, 1/3]), plot=False).flatten()
    wing_station_3_quarter_chord_upper = upper_right_wing.evaluate(np.array([0.667, 2/3]), plot=False).flatten()
    wing_station_3_quarter_chord_lower = lower_right_wing.evaluate(np.array([0.667, 2/3]), plot=False).flatten()

    centerbody_root_leading_edge = upper_right_centerbody.evaluate(np.array([[0.0, 0.0]]), plot=False).flatten()
    centerbody_root_trailing_edge = upper_right_centerbody.evaluate(np.array([[1.0, 0.0]]), plot=False).flatten()
    centerbody_tip_leading_edge = upper_right_centerbody.evaluate(np.array([[0.0, 1.0]]), plot=False).flatten()
    centerbody_tip_trailing_edge = upper_right_centerbody.evaluate(np.array([[1.0, 1.0]]), plot=False).flatten()
    centerbody_station_2_leading_edge = upper_right_centerbody.evaluate(np.array([[0.0, 1/3]]), plot=False).flatten()
    centerbody_station_2_trailing_edge = upper_right_centerbody.evaluate(np.array([[1.0, 1/3]]), plot=False).flatten()
    centerbody_station_3_leading_edge = upper_right_centerbody.evaluate(np.array([[0.0, 2/3]]), plot=False).flatten()
    centerbody_station_3_trailing_edge = upper_right_centerbody.evaluate(np.array([[1.0, 2/3]]), plot=False).flatten()
    centerbody_root_quarter_chord_upper = upper_right_centerbody.evaluate(np.array([0.667, 0.]), plot=False).flatten()
    centerbody_root_quarter_chord_lower = lower_right_centerbody.evaluate(np.array([0.667, 0.]), plot=False).flatten()
    centerbody_tip_quarter_chord_upper = upper_right_centerbody.evaluate(np.array([0.667, 1.]), plot=False).flatten()
    centerbody_tip_quarter_chord_lower = lower_right_centerbody.evaluate(np.array([0.667, 1.]), plot=False).flatten()
    centerbody_station_2_quarter_chord_upper = upper_right_centerbody.evaluate(np.array([0.667, 1/3]), plot=False).flatten()
    centerbody_station_2_quarter_chord_lower = lower_right_centerbody.evaluate(np.array([0.667, 1/3]), plot=False).flatten()
    centerbody_station_3_quarter_chord_upper = upper_right_centerbody.evaluate(np.array([0.667, 2/3]), plot=False).flatten()
    centerbody_station_3_quarter_chord_lower = lower_right_centerbody.evaluate(np.array([0.667, 2/3]), plot=False).flatten()

    # wing_root_to_tip_vector = wing_tip - wing_root
    wing_leading_edge_root_to_tip_vector = wing_tip_leading_edge - wing_root_leading_edge
    centerbody_root_to_tip_vector = centerbody_tip - centerbody_root
    transition_root_to_tip_vector = wing_root - centerbody_tip

    # wing_sweep_computed = csdl.arctan2(wing_root_to_tip_vector[0], wing_root_to_tip_vector[1])
    # centerbody_sweep_computed = csdl.arctan2(centerbody_root_to_tip_vector[0], centerbody_root_to_tip_vector[1])
    # centerbody_trailing_edge_sweep_1 = csdl.arctan2(centerbody_station_2_trailing_edge[0]-centerbody_root_trailing_edge[0], centerbody_station_2_trailing_edge[1]-centerbody_root_trailing_edge[1])
    # centerbody_trailing_edge_sweep_2 = csdl.arctan2(centerbody_station_3_trailing_edge[0]-centerbody_root_trailing_edge[0], centerbody_station_3_trailing_edge[1]-centerbody_root_trailing_edge[1])
    # centerbody_trailing_edge_sweep_3 = csdl.arctan2(centerbody_tip_trailing_edge[0]-centerbody_root_trailing_edge[0], centerbody_tip_trailing_edge[1]-centerbody_root_trailing_edge[1])
    # transition_sweep_computed = csdl.arctan2(transition_root_to_tip_vector[0], transition_root_to_tip_vector[1])

    # wing_half_span_computed = wing_root_to_tip_vector[1]
    wing_half_span_computed = wing_leading_edge_root_to_tip_vector[1]
    centerbody_half_span_computed = centerbody_root_to_tip_vector[1]
    transition_half_span_computed = transition_root_to_tip_vector[1]

    # wing_dihedral_computed = csdl.arctan2(wing_root_to_tip_vector[2], wing_root_to_tip_vector[1])
    # centerbody_dihedral_computed = csdl.arctan2(centerbody_root_to_tip_vector[2], centerbody_root_to_tip_vector[1])
    # transition_dihedral_computed = csdl.arctan2(transition_root_to_tip_vector[2], transition_root_to_tip_vector[1])

    wing_root_chord_computed = wing_root_trailing_edge[0] - wing_root_leading_edge[0]
    wing_tip_chord_computed = wing_tip_trailing_edge[0] - wing_tip_leading_edge[0]
    wing_station_2_chord_computed = wing_station_2_trailing_edge[0] - wing_station_2_leading_edge[0]
    wing_station_3_chord_computed = wing_station_3_trailing_edge[0] - wing_station_3_leading_edge[0]
    wing_chords_computed = csdl.concatenate((wing_root_chord_computed.reshape(1,), 
                                        wing_station_2_chord_computed.reshape(1,),
                                        wing_station_3_chord_computed.reshape(1,),
                                        wing_tip_chord_computed.reshape(1,)), axis=0)
    centerbody_root_chord_computed = centerbody_root_trailing_edge[0] - centerbody_root_leading_edge[0]
    centerbody_tip_chord_computed = centerbody_tip_trailing_edge[0] - centerbody_tip_leading_edge[0]
    centerbody_station_2_chord_computed = centerbody_station_2_trailing_edge[0] - centerbody_station_2_leading_edge[0]
    centerbody_station_3_chord_computed = centerbody_station_3_trailing_edge[0] - centerbody_station_3_leading_edge[0]
    centerbody_chords_computed = csdl.concatenate((centerbody_root_chord_computed.reshape(1,), 
                                                    centerbody_station_2_chord_computed.reshape(1,), 
                                                    centerbody_station_3_chord_computed.reshape(1,), 
                                                    centerbody_tip_chord_computed.reshape(1,)), axis=0)

    wing_root_thickness_computed = upper_wing_root[2] - lower_wing_root[2]
    wing_tip_thickness_computed = upper_wing_tip[2] - lower_wing_tip[2]
    # centerbody_root_thickness_computed = upper_centerbody_root[2] - lower_centerbody_root[2]
    # centerbody_tip_thickness_computed = upper_centerbody_tip[2] - lower_centerbody_tip[2]
    wing_root_thickness_to_chord_ratio_computed = wing_root_thickness_computed / wing_root_chord_computed
    wing_tip_thickness_to_chord_ratio_computed = wing_tip_thickness_computed / wing_tip_chord_computed
    wing_station_2_thickness_computed = wing_station_2_quarter_chord_upper[2] - wing_station_2_quarter_chord_lower[2]
    wing_station_3_thickness_computed = wing_station_3_quarter_chord_upper[2] - wing_station_3_quarter_chord_lower[2]
    wing_station_2_thickness_to_chord_ratio_computed = wing_station_2_thickness_computed / wing_station_2_chord_computed
    wing_station_3_thickness_to_chord_ratio_computed = wing_station_3_thickness_computed / wing_station_3_chord_computed
    wing_thickness_to_chord_ratios_computed = csdl.concatenate((wing_root_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                wing_station_2_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                wing_station_3_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                wing_tip_thickness_to_chord_ratio_computed.reshape(1,)), axis=0)
    wing_thicknesses_computed = csdl.concatenate((wing_root_thickness_computed.reshape(1,),
                                                wing_station_2_thickness_computed.reshape(1,),
                                                wing_station_3_thickness_computed.reshape(1,),
                                                wing_tip_thickness_computed.reshape(1,)), axis=0)

    centerbody_root_thickness_computed = centerbody_root_quarter_chord_upper[2] - centerbody_root_quarter_chord_lower[2]
    centerbody_tip_thickness_computed = centerbody_tip_quarter_chord_upper[2] - centerbody_tip_quarter_chord_lower[2]
    centerbody_station_2_thickness_computed = centerbody_station_2_quarter_chord_upper[2] - centerbody_station_2_quarter_chord_lower[2]
    centerbody_station_3_thickness_computed = centerbody_station_3_quarter_chord_upper[2] - centerbody_station_3_quarter_chord_lower[2]
    centerbody_root_thickness_to_chord_ratio_computed = centerbody_root_thickness_computed / centerbody_root_chord_computed
    centerbody_tip_thickness_to_chord_ratio_computed = centerbody_tip_thickness_computed / centerbody_tip_chord_computed
    centerbody_station_2_thickness_to_chord_ratio_computed = centerbody_station_2_thickness_computed / centerbody_station_2_chord_computed
    centerbody_station_3_thickness_to_chord_ratio_computed = centerbody_station_3_thickness_computed / centerbody_station_3_chord_computed
    centerbody_thickness_to_chord_ratios_computed = csdl.concatenate((centerbody_root_thickness_to_chord_ratio_computed.reshape(1,),
                                                                    centerbody_station_2_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                    centerbody_station_3_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                    centerbody_tip_thickness_to_chord_ratio_computed.reshape(1,)), axis=0)
    centerbody_thicknesses_computed = csdl.concatenate((centerbody_root_thickness_computed.reshape(1,),
                                                centerbody_station_2_thickness_computed.reshape(1,),
                                                centerbody_station_3_thickness_computed.reshape(1,),
                                                centerbody_tip_thickness_computed.reshape(1,)), axis=0)

    # Enforce geometric constraints and add geometric variables to the solver.
    if enforce_no_centerbody_trailing_edge_sweep and 'centerbody_chord_stretch_coefficients' in variables.keys():
        # NOTE: Although the above computations are correct, it seems more efficient to just enforce that the change in x must be 0.
        centerbody_trailing_edge_sweep_1_constraint = centerbody_station_2_trailing_edge[0] - centerbody_root_trailing_edge[0]
        centerbody_trailing_edge_sweep_2_constraint = centerbody_station_3_trailing_edge[0] - centerbody_root_trailing_edge[0]
        centerbody_trailing_edge_sweep_3_constraint = centerbody_tip_trailing_edge[0] - centerbody_root_trailing_edge[0]
        centerbody_trailing_edge_constraints = csdl.concatenate((centerbody_trailing_edge_sweep_1_constraint, centerbody_trailing_edge_sweep_2_constraint, centerbody_trailing_edge_sweep_3_constraint), axis=0)

        parameterization_solver.add_equality_constraint(centerbody_trailing_edge_constraints, 0.)
        parameterization_solver.add_state(centerbody_sweep_translation_coefficients, cost=1.e-2)
        # added 4 dofs and only 3 constraints so add an x location anchor to prevent drifting in x direction
        # anchor_point = upper_right_centerbody.evaluate(np.array([[0.667, 0.]]), plot=False).flatten()
        anchor_point = upper_right_centerbody.evaluate(np.array([[1., 0.]]), plot=False).flatten()
        parameterization_solver.add_equality_constraint(anchor_point[0], anchor_point[0].value)

    if enforce_no_wing_leading_edge_sweep and \
        ('wing_sweep' in variables.keys() or 'wing_chord_stretch_coefficients' in variables.keys()):
        if 'wing_sweep' in variables.keys():
            wing_sweep = variables['wing_sweep']*np.pi/180
        else:
            wing_sweep = np.arctan2(wing_leading_edge_root_to_tip_vector.value[1], wing_leading_edge_root_to_tip_vector.value[0])
        wing_root_to_station_2_leading_edge_vector = wing_station_2_leading_edge - wing_root_leading_edge
        wing_station_2_to_station_3_leading_edge_vector = wing_station_3_leading_edge - wing_station_2_leading_edge
        wing_station_3_to_tip_leading_edge_vector = wing_tip_leading_edge - wing_station_3_leading_edge

        tan_sweep = csdl.tan(wing_sweep)

        parameterization_solver.add_equality_constraint(wing_root_to_station_2_leading_edge_vector[1]*tan_sweep, wing_root_to_station_2_leading_edge_vector[0])
        parameterization_solver.add_equality_constraint(wing_station_2_to_station_3_leading_edge_vector[1]*tan_sweep, wing_station_2_to_station_3_leading_edge_vector[0])
        # parameterization_solver.add_equality_constraint(wing_station_3_to_tip_leading_edge_vector[1]*tan_sweep, wing_station_3_to_tip_leading_edge_vector[0])

        if 'wing_sweep' not in variables.keys():    # If wing sweep in problem, coefficients will be added as states then
            parameterization_solver.add_state(wing_sweep_translation_coefficients, cost=1.e-2)

    if enforce_thickness_to_chord_ratio:
        # if 'wing_root_chord' in variables.keys():
        #     wing_root_thickness_to_chord_ratio = wing_root_thickness_to_chord_ratio_computed.value
        #     parameterization_solver.add_equality_constraint(wing_root_thickness_to_chord_ratio_computed, wing_root_thickness_to_chord_ratio)
        #     parameterization_solver.add_state(wing_root_thickness_stretch)
        # if 'wing_tip_chord' in variables.keys():
        #     wing_tip_thickness_to_chord_ratio = wing_tip_thickness_to_chord_ratio_computed.value
        #     parameterization_solver.add_equality_constraint(wing_tip_thickness_to_chord_ratio_computed, wing_tip_thickness_to_chord_ratio)
        #     parameterization_solver.add_state(wing_tip_thickness_stretch)
        if 'wing_chord_stretch_coefficients' in variables.keys():
            wing_thickness_to_chord_ratios = wing_thickness_to_chord_ratios_computed.value
            # parameterization_solver.add_equality_constraint(wing_thickness_to_chord_ratios_computed, wing_thickness_to_chord_ratios)
            # I believe the below formulation keeps the problem linear by multiplying both sides by the chord lengths
            parameterization_solver.add_equality_constraint(wing_thicknesses_computed, wing_thickness_to_chord_ratios*wing_chords_computed)
            parameterization_solver.add_state(wing_thickness_stretch_coefficients)
        if 'centerbody_chord_stretch_coefficients' in variables.keys():
            centerbody_thickness_to_chord_ratios = centerbody_thickness_to_chord_ratios_computed.value
            # parameterization_solver.add_equality_constraint(centerbody_thickness_to_chord_ratios_computed, centerbody_thickness_to_chord_ratios)
            # I believe the below formulation keeps the problem linear by multiplying both sides by the chord lengths
            parameterization_solver.add_equality_constraint(centerbody_thicknesses_computed, centerbody_thickness_to_chord_ratios*centerbody_chords_computed)
            parameterization_solver.add_state(centerbody_thickness_stretch_coefficients)

    
    geometric_variables = lsdo_geo.GeometricVariables()
    if 'wing_sweep' in variables.keys():
        wing_sweep = variables['wing_sweep']*np.pi/180
        tan_sweep = csdl.tan(wing_sweep)
        geometric_variables.add_variable(computed_value=wing_leading_edge_root_to_tip_vector[0], desired_value=wing_leading_edge_root_to_tip_vector[1]*tan_sweep)
        # parameterization_solver.add_state(wing_tip_sweep_translation)
        parameterization_solver.add_state(wing_sweep_translation_coefficients)
    # if 'centerbody_sweep' in variables.keys():
    #     centerbody_sweep = variables['centerbody_sweep']*np.pi/180
    #     geometric_variables.add_variable(computed_value=centerbody_root_to_tip_vector[0], desired_value=centerbody_root_to_tip_vector[1]*csdl.tan(centerbody_sweep))
    #     parameterization_solver.add_state(centerbody_tip_sweep_translation)
    # if 'transition_sweep' in variables.keys():
    #     transition_sweep = variables['transition_sweep']*np.pi/180
    #     geometric_variables.add_variable(computed_value=transition_root_to_tip_vector[0], desired_value=transition_root_to_tip_vector[1]*csdl.tan(transition_sweep))
    #     if 'wing_sweep' not in variables.keys():    # If wing sweep in problem, coefficients have already been added as states
    #         parameterization_solver.add_state(wing_sweep_translation_coefficients)
    if 'wing_position' in variables.keys():
        wing_position = variables['wing_position']
        geometric_variables.add_variable(computed_value=wing_root[0], desired_value=wing_position)
        if 'wing_sweep' not in variables.keys():    # If wing sweep in problem, coefficients have already been added as states
            parameterization_solver.add_state(wing_sweep_translation_coefficients)
    # if 'wing_half_span' in variables.keys():
    #     wing_half_span = variables['wing_half_span']
    #     geometric_variables.add_variable(computed_value=wing_half_span_computed, desired_value=wing_half_span)
    #     parameterization_solver.add_state(wing_tip_wingspan_stretch)
    # if 'centerbody_half_span' in variables.keys():
    #     centerbody_half_span = variables['centerbody_half_span']
    #     geometric_variables.add_variable(computed_value=centerbody_half_span_computed, desired_value=centerbody_half_span)
    #     parameterization_solver.add_state(centerbody_tip_wingspan_stretch)
    # if 'transition_half_span' in variables.keys():
    #     transition_half_span = variables['transition_half_span']
    #     geometric_variables.add_variable(computed_value=transition_half_span_computed, desired_value=transition_half_span)
    #     parameterization_solver.add_state(transition_tip_wingspan_stretch)
    if 'wing_dihedral' in variables.keys():
        wing_dihedral = variables['wing_dihedral']*np.pi/180
        geometric_variables.add_variable(computed_value=wing_leading_edge_root_to_tip_vector[2], desired_value=wing_leading_edge_root_to_tip_vector[1]*csdl.tan(wing_dihedral))
        parameterization_solver.add_state(wing_tip_dihedral_translation)
    # if 'centerbody_dihedral' in variables.keys():
    #     centerbody_dihedral = variables['centerbody_dihedral']*np.pi/180
    #     geometric_variables.add_variable(computed_value=centerbody_root_to_tip_vector[2], desired_value=centerbody_root_to_tip_vector[1]*csdl.tan(centerbody_dihedral))
    #     parameterization_solver.add_state(centerbody_tip_dihedral_translation)
    # if 'transition_dihedral' in variables.keys():
    #     transition_dihedral = variables['transition_dihedral']*np.pi/180
    #     geometric_variables.add_variable(computed_value=transition_root_to_tip_vector[2], desired_value=transition_root_to_tip_vector[1]*csdl.tan(transition_dihedral))
    #     parameterization_solver.add_state(transition_tip_dihedral_translation)
    # if 'wing_root_chord' in variables.keys():
    #     wing_root_chord = variables['wing_root_chord']
    #     geometric_variables.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
    #     parameterization_solver.add_state(wing_root_chord_stretch)
    # if 'wing_tip_chord' in variables.keys():
    #     wing_tip_chord = variables['wing_tip_chord']
    #     geometric_variables.add_variable(computed_value=wing_tip_chord_computed, desired_value=wing_tip_chord)
    #     parameterization_solver.add_state(wing_tip_chord_stretch)
    # if 'centerbody_root_chord' in variables.keys():
    #     centerbody_root_chord = variables['centerbody_root_chord']
    #     geometric_variables.add_variable(computed_value=centerbody_root_chord_computed, desired_value=centerbody_root_chord)
    #     parameterization_solver.add_state(centerbody_root_chord_stretch)
    # if 'centerbody_tip_chord' in variables.keys():
    #     centerbody_tip_chord = variables['centerbody_tip_chord']
    #     geometric_variables.add_variable(computed_value=centerbody_tip_chord_computed, desired_value=centerbody_tip_chord)
    #     parameterization_solver.add_state(centerbody_tip_chord_stretch)
    # if 'centerbody_chords' in variables.keys():
    #     centerbody_chords = variables['centerbody_chords']
    #     geometric_variables.add_variable(computed_value=centerbody_chords_computed, desired_value=centerbody_chords)
    #     parameterization_solver.add_state(centerbody_chord_stretch_dofs)
    
    # if not enforce_thickness_to_chord_ratio:
    #     # if 'wing_root_thickness_to_chord_ratio' in variables.keys():
    #     #     wing_root_thickness_to_chord_ratio = variables['wing_root_thickness_to_chord_ratio']
    #     #     geometric_variables.add_variable(computed_value=wing_root_thickness_to_chord_ratio_computed, desired_value=wing_root_thickness_to_chord_ratio)
    #     #     parameterization_solver.add_state(wing_root_thickness_stretch)
    #     # if 'wing_tip_thickness_to_chord_ratio' in variables.keys():
    #     #     wing_tip_thickness_to_chord_ratio = variables['wing_tip_thickness_to_chord_ratio']
    #     #     geometric_variables.add_variable(computed_value=wing_tip_thickness_to_chord_ratio_computed, desired_value=wing_tip_thickness_to_chord_ratio)
    #     #     parameterization_solver.add_state(wing_tip_thickness_stretch)
    #     if 'wing_thickness_to_chord_ratios' in variables.keys():
    #         wing_thickness_to_chord_ratios = variables['wing_thickness_to_chord_ratios']
    #         geometric_variables.add_variable(computed_value=wing_thickness_to_chord_ratios_computed, desired_value=wing_thickness_to_chord_ratios)
    #         parameterization_solver.add_state(wing_thickness_stretch_dofs)        
    #     if 'centerbody_thickness_to_chord_ratios' in variables.keys():
    #         centerbody_thickness_to_chord_ratios = variables['centerbody_thickness_to_chord_ratios']
    #         geometric_variables.add_variable(computed_value=centerbody_thickness_to_chord_ratios_computed, desired_value=centerbody_thickness_to_chord_ratios)
    #         parameterization_solver.add_state(centerbody_thickness_stretch_dofs)
    
    if len(geometric_variables.desired_value) > 0 or len(parameterization_solver.optimization.equality_constraints) > 0:
        # parameterization_solver.evaluate(geometric_variables)
        for i in range(len(geometric_variables.computed_value)):
            computed_value = geometric_variables.computed_value[i]
            desired_value = geometric_variables.desired_value[i]
            penalty_value = geometric_variables.penalty_value[i]
            parameterization_solver.add_variable(computed_value, desired_value, penalty_value)

        parameterization_solver.setup()
        # parameterization_solver.optimization.setup()

        num_dvs = 0
        for dv in parameterization_solver.states:
            num_dvs += dv.state.shape[0]
        num_constraints = 0
        for constraint in parameterization_solver.optimization.equality_constraints:
            num_constraints += constraint.constraint.shape[0]

        kkt_matrix = np.zeros(shape=(num_dvs + num_constraints, num_dvs + num_constraints))


        # Compute d^f/dx^2 term of KKT matrix (assume second derivative of constraints is 0)
        counter = 0
        for design_variable in parameterization_solver.states:
            design_variable_size = design_variable.state.shape[0]
            design_variable_cost = design_variable.cost
            kkt_matrix[counter:counter+design_variable_size, counter:counter+design_variable_size] = 2 * np.eye(design_variable_size) * design_variable_cost
            counter += design_variable_size

        constraint_counter = 0
        t1 = time.time()
        for constraint in parameterization_solver.optimization.equality_constraints:
            constraint_size = constraint.constraint.shape[0]
            design_variable_counter = 0
            for design_variable in parameterization_solver.states:
                design_variable_size = design_variable.state.shape[0]
                
                constraint_jacobian = csdl.derivative(constraint.constraint, design_variable.state).value
                kkt_matrix[design_variable_counter:design_variable_counter+design_variable_size, num_dvs+constraint_counter:num_dvs+constraint_counter+constraint_size] = constraint_jacobian.T
                kkt_matrix[num_dvs+constraint_counter:num_dvs+constraint_counter+constraint_size, design_variable_counter:design_variable_counter+design_variable_size] = constraint_jacobian
                design_variable_counter += design_variable_size
            constraint_counter += constraint_size

        t2 = time.time()
        print("Time to compute constraint jacobians: ", t2-t1)

        lagrangian_gradient = csdl.Variable(shape=(num_dvs + num_constraints,), value=0.)

        design_variable_counter = 0
        for design_variable in parameterization_solver.states:
            design_variable_size = design_variable.state.shape[0]
            # We only need objective gradient since this converges in one iteration anyways and lagrange multiplier initial guesses are 0 (0 * dc/dx = 0)
            objective_gradient = parameterization_solver.optimization.compute_objective_gradient(design_variables=design_variable.state).flatten()
            lagrangian_gradient = lagrangian_gradient.set(csdl.slice[design_variable_counter:design_variable_counter+design_variable_size], objective_gradient)
            design_variable_counter += design_variable_size
        constraint_counter = design_variable_counter
        for constraint in parameterization_solver.optimization.equality_constraints:
            constraint_size = constraint.constraint.shape[0]
            lagrangian_gradient = lagrangian_gradient.set(csdl.slice[constraint_counter:constraint_counter+constraint_size], constraint.constraint)
            constraint_counter += constraint_size

        t3 = time.time()
        print("Time to compute RHS: ", t3-t2)

        inverted_kkt_matrix = np.linalg.inv(kkt_matrix)
        t4 = time.time()
        print("Time to invert KKT matrix: ", t4-t3)
        change_in_design_variables = csdl.matvec(inverted_kkt_matrix, -lagrangian_gradient)
        t5 = time.time()
        print("Time to compute change in design variables: ", t5-t4)

        design_variable_counter = 0
        # new_design_variable_values = []
        for design_variable in parameterization_solver.states:
            design_variable_size = design_variable.state.shape[0]
            change_in_design_variable = change_in_design_variables[design_variable_counter:design_variable_counter+design_variable_size]
            # new_design_variable_values.append(design_variable + change_in_design_variable)
            states[design_variable.state.name] = design_variable.state + change_in_design_variable
            design_variable_counter += design_variable_size

    if 'wing_twist_coefficients' in variables.keys() or \
        'centerbody_twist_coefficients' in variables.keys() or \
            'airfoil_thickness_coefficients' in variables.keys() or \
            'airfoil_camber_coefficients' in variables.keys() or \
            'elevator_deflection' in variables.keys():
        post_solve_deformation_in_problem = True
    else:
        post_solve_deformation_in_problem = False
    
    if len(geometric_variables.desired_value) > 0 or len(parameterization_solver.optimization.equality_constraints) > 0 \
        or post_solve_deformation_in_problem:
        # Run the forward model again with the updated design variables to get the final geometry
        geometry = parameterization_forward_model(initial_refit_geometry, variables, states,
                                                  plot=plot, apply_post_solve_deformations=True)

    

    if plot:
        openvsp_geometry_plot = initial_refit_geometry.plot(opacity=0.7, show=False)
        geometry.plot(show=True, color='red', opacity=0.7, additional_plotting_elements=[openvsp_geometry_plot])

    return geometry

