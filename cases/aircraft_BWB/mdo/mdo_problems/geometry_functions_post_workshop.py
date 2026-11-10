import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import numpy as np
import scipy.sparse.linalg as spsl
import time
from typing import Optional

enforce_no_centerbody_trailing_edge_sweep = True
enforce_no_wing_leading_edge_sweep = True
enforce_thickness_to_chord_ratio = True
use_baseline_twist = True

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


