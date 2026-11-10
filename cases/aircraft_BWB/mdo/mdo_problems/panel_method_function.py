from VortexAD import PanelMethod
import csdl_alpha as csdl
from additional_solvers import compute_static_margin, estimate_CDw, estimate_Cf, estimate_fuel_volume
import numpy as np

def compute_panel_method(panel_method_inputs):
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------

    pm_solver_inputs = panel_method_inputs['pm_solver_inputs']

    mesh_data = panel_method_inputs['mesh_data']
    panel_mesh = mesh_data['panel_mesh']
    cell_adjacency_data = mesh_data['cell_adjacency_data']
    TE_properties = mesh_data['TE_properties']

    flight_conditions = panel_method_inputs['flight_conditions']
    rho_array = flight_conditions['rho']
    mu_array = flight_conditions['mu']
    V_inf_array = flight_conditions['V_inf']
    mach_array = flight_conditions['mach']

    reference_geometry = panel_method_inputs['reference_geometry']
    planform_area = reference_geometry['planform_area']
    MAC = reference_geometry['MAC']
    wing_sweep = reference_geometry['wing_sweep']
    LE_points_CB_TR = reference_geometry['LE_points_CB_TR']

    strip_data = panel_method_inputs['strip_data']
    sec_chord_st = strip_data['sectional_avg_chord']
    sec_span_st = strip_data['sectional_span']
    num_centerbody_stations = strip_data['num_centerbody_stations']
    num_center_wing_spanwise_design_dofs = strip_data['num_centerbody_stations']
    num_wing_spanwise_design_dofs = strip_data['num_wing_spanwise_design_dofs']

    mission_data = panel_method_inputs['mission_data']
    num_cruise = mission_data['num_cruise']
    num_sizing = mission_data['num_sizing']

    # These can be inferred instead of passed through the dictionary.
    num_nodes = V_inf_array.shape[0]
    num_sections = sec_chord_st.shape[0]

    # The transition strip follows the last pure centerbody strip.
    transition_strip_index = num_centerbody_stations - 1

    panel_method = PanelMethod(
        solver_input_dict=pm_solver_inputs,
        skip_geometry=True # not running geometry
    )
    # inserting grid data from above
    panel_method.insert_grid_data(
        mesh=panel_mesh[0,:],
        cell_adjacency_data=cell_adjacency_data,
        TE_properties=TE_properties
    )

    panel_method.declare_outputs([
        'Cp',
        'L',
        'Di',
        'M',
        'panel_forces',
        'Di_Trefftz',
    ])
    outputs = panel_method.evaluate()


    Cp = outputs['Cp']
    Cp.add_name('Cp')
    # Cp.save()
    L = outputs['L']
    L.add_name('L')
    # L.save()
    # Di = outputs['Di']
    # Di.add_name('Di')
    Di = outputs['Di_Trefftz']
    Di.add_name('Di')
    # Di.save()
    M = outputs['M'][:,1]
    M.add_name('M_y')
    # M.save()
    panel_forces = outputs['panel_forces'][num_cruise:num_cruise+num_sizing,:]
    # panel_normal = outputs['panel_normal'][0,:] # (num_panels, 3)

    # computing coefficients
    CL = L/(0.5*rho_array*V_inf_array**2*planform_area)
    CL.add_name('CL')
    # CL.save()
    CDi = Di/(0.5*rho_array*V_inf_array**2*planform_area)
    CDi.add_name('CDi')
    # CDi.save()
    CM = M/(0.5*rho_array*V_inf_array**2*planform_area*MAC)
    CM.add_name('CMy')
    # CM.save()

    # skin friction drag coefficient + strip theory
    sec_plan_area_st = sec_span_st*sec_chord_st
    exp_shape = (num_cruise, sec_chord_st.shape[0])
    exp_shape = (V_inf_array.shape[0], sec_chord_st.shape[0])
    sec_chord_st_nn = csdl.expand(sec_chord_st, exp_shape, 'i->ai')
    sec_plan_area_st_nn = csdl.expand(sec_plan_area_st, exp_shape, 'i->ai')

    V_inf_array_exp = csdl.expand(V_inf_array, (num_nodes, sec_chord_st.shape[0]), 'i->ia')
    rho_array_exp = csdl.expand(rho_array, (num_nodes, sec_chord_st.shape[0]), 'i->ia')
    mu_array_exp = csdl.expand(mu_array, (num_nodes, sec_chord_st.shape[0]), 'i->ia')

    Re=rho_array_exp*V_inf_array_exp*sec_chord_st_nn/mu_array_exp
    Cf = estimate_Cf(Re,M=0)
    Cf.add_name('Cf')
    # Cf.save()
    plan_2_wetted_area_conv = 2.1
    wetted_area = sec_plan_area_st_nn*plan_2_wetted_area_conv # use 2x the sectional planform area here since we say flat plate
    FF = 1.5 # add a form factor equation here
    # Df_strip = rho_array_exp*V_inf_array_exp**2*wetted_area*Cf*FF

    x_c_max = csdl.Variable(
        value=np.array(
            [0.3]*(num_center_wing_spanwise_design_dofs-1) + \
            [(0.3+0.35)/2] + \
            [0.35]*(num_wing_spanwise_design_dofs-1)
        )
    )
    x_c_max_exp = x_c_max.expand(exp_shape, 'i->ai')

    # getting points on LE at nose, centerbody-transition point, and transition-wing point
    LE_nose = LE_points_CB_TR[0]
    LE_CB_edge = LE_points_CB_TR[2]
    LE_TR_edge = LE_points_CB_TR[4]

    # deltas between points (to compute arctan)
    LE_CB_line = LE_CB_edge-LE_nose
    LE_TR_line = LE_TR_edge-LE_CB_edge

    CB_LE_sweep = csdl.arctan(LE_CB_line[0]/LE_CB_line[1])*180/np.pi
    CB_LE_sweep.add_name('centerbody_LE_sweep')
    TR_LE_sweep = csdl.arctan(LE_TR_line[0]/LE_TR_line[1])*180/np.pi
    TR_LE_sweep.add_name('transition_LE_sweep')

    Lambda_m = csdl.Variable(value=np.zeros((sec_chord_st.shape[0])))
    Lambda_m = Lambda_m.set(csdl.slice[:num_center_wing_spanwise_design_dofs-1], CB_LE_sweep) # center body sections # 63 to start
    Lambda_m = Lambda_m.set(csdl.slice[num_center_wing_spanwise_design_dofs-1], TR_LE_sweep) # transition section # 58 to start
    Lambda_m = Lambda_m.set(csdl.slice[num_center_wing_spanwise_design_dofs:], wing_sweep) # wing sections
    Lambda_m_exp = Lambda_m.expand(exp_shape, 'i->ai')

    t_c = csdl.Variable(value=np.zeros((sec_chord_st.shape[0])))
    t_c = t_c.set(csdl.slice[:num_center_wing_spanwise_design_dofs-1], 0.15) # center body sections
    t_c = t_c.set(csdl.slice[num_center_wing_spanwise_design_dofs-1], (0.15+.11336)/2) # transition section
    t_c = t_c.set(csdl.slice[num_center_wing_spanwise_design_dofs:], 0.11336) # wing sections
    t_c_exp = t_c.expand(exp_shape, 'i->ai')

    mach_array_exp = csdl.expand(mach_array, exp_shape, 'i->ia')

    FF = (1+0.6/x_c_max_exp*t_c_exp+100*t_c_exp**4) * \
        (1.34*mach_array_exp**0.18*(csdl.cos(Lambda_m_exp*np.pi/180.))**0.28)
    FF.add_name('form_factor')

    CD0 = Cf*wetted_area/sec_plan_area_st_nn*FF
    CD0.add_name('CD0')
    Df_strip = 0.5*rho_array_exp*V_inf_array_exp**2*sec_plan_area_st_nn*CD0

    Df = csdl.sum(Df_strip, axes=(1,)) * 2 # multiply by two to account for other side
    Df.add_name('Df')
    # Df.save()

    # wave drag
    t_c = (.15+.11336)/2 # TODO: UPDATE USING STRIP THEORY RATHER THAN AVERAGE t/c
    t_c = .15 # TODO: UPDATE USING STRIP THEORY RATHER THAN AVERAGE t/c
    # CDw = estimate_CDw(t_c, CL, cruise_mach, wing_sweep)
    CDw = estimate_CDw(t_c, CL, mach_array, wing_sweep)
    # maybe use the normal component of cruise mach to wing via the sweep angle
    CDw.add_name('CDw')
    # CDw.save()
    Dw = 0.5*rho_array*V_inf_array**2*planform_area*CDw
    Dw.add_name('Dw')
    # Dw.save()

    D_total = Di + Df + Dw
    D_total.add_name('D_total')
    # D_total.save()

    return {
        # Primary outputs
        'panel_forces': panel_forces,
        # 'panel_forces_all': panel_forces_all,
        'D_total': D_total,
        'L': L,
        'M': M,

        # Panel-method quantities
        'Cp': Cp,
        'Di': Di,
        # 'Di_near_field': Di_near_field,

        # Coefficients
        'CL': CL,
        'CDi': CDi,
        'CMy': CM,
        'CD0': CD0,
        'CDw': CDw,

        # Viscous/wave-drag details
        'Cf': Cf,
        # 'form_factor': form_factor,
        'Df_strip': Df_strip,
        'Df': Df,
        'Dw': Dw,
    }