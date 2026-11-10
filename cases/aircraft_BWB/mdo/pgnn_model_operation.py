# ============================== GNN MODEL ================================
from build_gnn import *
import pickle

# Panel preparation:
from build_panel import build_panel_functions, load_panel_topology
panel_mesh_info = load_panel_topology()
panel_functions = build_panel_functions(
    panel_mesh_info, {"U0": v_ref, "p0": p_ref, "T0": T_ref},
    device=constants['panel_device'], flow_condition=flow_condition,
)
panel_gnn_dict = get_lowfi_panel_mesh(str(MODEL_DIR / "CONNECTIVITY_NEW.pkl"), topology, {"verbose": True, "run_checks": True, "visualize": False},)
panel_projections_path = os.path.join(dafoam_directory, "panel_projections_NEWHASH.pickle")
with open(panel_projections_path, "rb") as f:
    panel_projections = pickle.load(f)

# NEW TCP1B PGRANS
load_model_path = 'PG_RANS_GNN_trained_paper.pickle'

ml_func, d_ml_func, ml_func_full = build_ml_nn_eval_func(
    topology, constants, flow_condition, load_model_path,
    lowfi=panel_gnn_dict,
)

# Create custom operation
import csdl_alpha as csdl
class PGRANSGNN(csdl.CustomExplicitOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # To save derivative time
        self._reverse_cache = {'surf_x': None, 'pitch': None, 'global_x_vol': None,
                               'panel_mesh': None, 'lowfi_data': None}

    def evaluate(self, surf_x, pitch, panel_mesh):
        # inputs are the 25-rank-projected surface points
        self.declare_input('surf_x', surf_x)    # shape=(num_points_25r, 3))
        self.declare_input('pitch', pitch)      # shape=(1,))
        self.declare_input('panel_mesh', panel_mesh) # body-fixed points, shape=(num_panel_points, 3)

        # Outputs include nodal forces, scalars, 
        forces = self.create_output('surface_forces', (num_wall_faces, 3))
        points = self.create_output('surface_points', (num_wall_faces, 3))
        L, D, M = self.create_output('L', (1,)), self.create_output('D', (1,)), self.create_output('M', (1,))
        
        # dictionary format output
        outputs = {
            'surface_forces': forces, # outputs in GLOBAL frame
            'surface_points': points, # outputs in GLOBAL frame
            'L': L, 'D': D, 'M': M    # outputs in GLOBAL frame
        }
        return outputs

    def compute(self, inputs, outputs):
        # preprocessing
        jax.config.update("jax_enable_x64", False)
        
        surf_x = inputs['surf_x']
        pitch = inputs['pitch'] # degrees
        flow_condition_vol = {'pitch': pitch}
        panel_mesh = inputs['panel_mesh']

        # perform panel method
        result = panel_functions.forward(panel_mesh, pitch)
        lowfi_data = {key: result[key] for key in ('pressure', 'velocity')}

        # perform volume-warping
        new_surf_pts = surf_x[surf_25r_to_1r_map]
        new_vol_pts = compute_volume_pts(new_surf_pts, flow_condition_vol)

        # compute ML-NN
        # Panel and volume builders have already applied pitch.
        L, D, M, surf_aux = ml_func(new_vol_pts, pitch, lowfi_data=lowfi_data) 

        # finalize and set outputs
        jax.config.update("jax_enable_x64", True)
        outputs['surface_forces'] = np.asarray(surf_aux['surface_forces'], dtype=np.float64)
        outputs['surface_points'] = np.asarray(surf_aux['surface_points'], dtype=np.float64)
        outputs['L'] = np.asarray(L, dtype=np.float64)
        outputs['D'] = np.asarray(D, dtype=np.float64)
        outputs['M'] = np.asarray(M, dtype=np.float64)

        # cache:
        self._reverse_cache['surf_x'] = np.array(surf_x, copy=True)
        self._reverse_cache['pitch'] = np.array(pitch, copy=True)
        self._reverse_cache['global_x_vol'] = new_vol_pts
        self._reverse_cache['panel_mesh'] = np.array(panel_mesh, copy=True)
        self._reverse_cache['lowfi_data'] = lowfi_data

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        if mode != 'rev':
            raise NotImplementedError('PGRANSGNN supports reverse-mode derivatives only')
        jax.config.update("jax_enable_x64", False)
        

        # Reuse the forward volume and panel fields only at the same inputs.
        # cache check
        surf_x = inputs['surf_x']
        pitch = inputs['pitch']
        panel_mesh = inputs['panel_mesh']
        cache_hit = (
            self._reverse_cache['global_x_vol'] is not None # make sure the cache exists
            and np.array_equal(surf_x, self._reverse_cache['surf_x']) # ... and first input matches 
            and np.array_equal(pitch, self._reverse_cache['pitch']) # ... second input matches
            and np.array_equal(panel_mesh, self._reverse_cache['panel_mesh'])
            and self._reverse_cache['lowfi_data'] is not None
        )
        if cache_hit:
            global_x_vol = self._reverse_cache['global_x_vol']
            lowfi_data = self._reverse_cache['lowfi_data']
        else:
            flow_condition_vol = {'pitch': pitch}
            new_surf_pts = surf_x[surf_25r_to_1r_map]
            global_x_vol = compute_volume_pts(new_surf_pts, flow_condition_vol)
            result = panel_functions.forward(panel_mesh, pitch)
            lowfi_data = {key: result[key] for key in ('pressure', 'velocity')}

            # update the cache with the newly evaluated point
            self._reverse_cache['surf_x'] = np.array(surf_x, copy=True)
            self._reverse_cache['pitch'] = np.array(pitch, copy=True)
            self._reverse_cache['global_x_vol'] = global_x_vol
            self._reverse_cache['panel_mesh'] = np.array(panel_mesh, copy=True)
            self._reverse_cache['lowfi_data'] = lowfi_data

        # reverse through the GNN first
        d_new_vol_pts, d_pitch, d_lowfi_data = d_ml_func(
            global_x_vol, pitch,
            d_outputs['L'].squeeze(), d_outputs['D'].squeeze(), d_outputs['M'].squeeze(), 
            {
                'surface_forces': d_outputs['surface_forces'],
                'surface_points': d_outputs['surface_points'],
            },
            lowfi_data=lowfi_data,
        )

        # Reverse pressure/velocity through the panel solve, including its pitch rotation.
        d_panel_mesh, d_panel_pitch = panel_functions.vjp(panel_mesh, pitch, d_lowfi_data)
        
        # then reverse through the volume warper
        surf_x = inputs['surf_x']
        new_surf_pts = surf_x[surf_25r_to_1r_map]
        flow_condition_vol = {'pitch': pitch}
        d_new_surf_pts, d_flow = d_compute_volume_pts(new_surf_pts, flow_condition_vol, d_new_vol_pts)

        # global_x_vol
        np.add.at(d_inputs['surf_x'], surf_25r_to_1r_map, np.asarray(d_new_surf_pts))
        d_inputs['surf_x'] = np.asarray(d_inputs['surf_x'], dtype=np.float64)
        d_inputs['panel_mesh'] += np.asarray(d_panel_mesh, dtype=np.float64)
        d_inputs['pitch'] += (np.asarray(d_flow['pitch'], dtype=np.float64)
                              + np.asarray(d_pitch, dtype=np.float64)
                              + np.asarray(d_panel_pitch, dtype=np.float64))
        jax.config.update("jax_enable_x64", True)

    def visualize_mesh(self):
        pass

if __name__ == '__main__':
    run_geometry = True
    # run_geometry = False
    sweep_vals = {'L':[], 'D':[], 'M':[]}
    # pitches = np.linspace(-30, 30, 40)
    # # pitches = np.array([0.0])
    # geometry_vals = {var_name: {'value': var.value} for var_name, var in geometry_config_map.items()}
    
    recorder = csdl.Recorder(inline=False, debug=True)
    recorder.start()

    scale_var = csdl.Variable(name='surf_x_scale', value = 1.0)

    # test:
    if run_geometry:
        recorder.inline = True
        from geometry import build_surface_csdl
        csdl_surf_points, geometry_config_map, bwb_geometry = build_surface_csdl(
            recorder,
            geometry_pickle_file_path,
            stp_file_path,
            surface_mesh_projection_file_path,
            dafoam_directory,
            x_surf_hash,
            index_unique = False,
            return_geometry = True
        )
        csdl_surf_points_val_full = csdl_surf_points
        panel_mesh_val = bwb_geometry.evaluate(panel_projections, plot=False)
        recorder.inline = False
    else:
        csdl_surf_points_val_full = np.load(f"{dafoam_directory}/surf_dafoam_initial_47466_25.npy")
        panel_mesh_val = panel_functions.topology.reference_points

    surface_var = csdl_surf_points_val_full*scale_var
    panel_surface_var = panel_mesh_val*scale_var
    pitch_dv_var = csdl.Variable(name='pitch_dv', value = np.array([1.0, -1.0]))
    pitch_var = pitch_dv_var[0]
    pitch_var1 = pitch_dv_var[1]

    # .............................................................................
    # ==================================== GNN 1 ====================================
    pgransgnn = PGRANSGNN()
    gnn_outputs = pgransgnn.evaluate(surface_var, pitch_var, panel_surface_var)
    
    # process scalar forces
    L_half, D_half, My_half = gnn_outputs['L'], gnn_outputs['D'], gnn_outputs['M']
    L, D, My = 2*L_half, 2*D_half, 2*My_half
    
    body_fixed_rotation_angle = -pitch_var
    # rotate nodal pts back to local frame (origin 0,0,0) and mirror for full body
    half_surface_points = rotate_pitch(gnn_outputs['surface_points'].flatten(), body_fixed_rotation_angle, use_csdl=True).reshape(-1,3)
    other_half_surface_points = half_surface_points.set(csdl.slice[:,1], value = -half_surface_points[:,1])
    surface_points = csdl.concatenate((half_surface_points, other_half_surface_points))

    # rotate forces pts back to local frame and mirror for full body (pretty sure I can just reuse the pitch func)
    half_forces = rotate_pitch(gnn_outputs['surface_forces'].flatten(), body_fixed_rotation_angle, use_csdl=True).reshape(-1,3)
    other_half_forces = half_forces.set(csdl.slice[:,1], value = -half_forces[:,1])
    surface_forces = csdl.concatenate((half_forces, other_half_forces))

    # final usuable variables:
    L.add_name('L')
    D.add_name('D')
    My.add_name('My')
    surface_points.add_name('surface_points')
    surface_forces.add_name('surface_forces')
    # ==================================== GNN 1 ====================================
    # .............................................................................

    # .............................................................................
    # ==================================== GNN 2 ====================================
    pgransgnn = PGRANSGNN()
    gnn_outputs = pgransgnn.evaluate(surface_var, pitch_var1, panel_surface_var)
    
    # process scalar forces
    L_half, D_half, My_half = gnn_outputs['L'], gnn_outputs['D'], gnn_outputs['M']
    L2, D2, My2 = 2*L_half, 2*D_half, 2*My_half
    
    body_fixed_rotation_angle = -pitch_var1
    # rotate nodal pts back to local frame (origin 0,0,0) and mirror for full body
    half_surface_points = rotate_pitch(gnn_outputs['surface_points'].flatten(), body_fixed_rotation_angle, use_csdl=True).reshape(-1,3)
    other_half_surface_points = half_surface_points.set(csdl.slice[:,1], value = -half_surface_points[:,1])
    surface_points2 = csdl.concatenate((half_surface_points, other_half_surface_points))

    # rotate forces pts back to local frame and mirror for full body (pretty sure I can just reuse the pitch func)
    half_forces = rotate_pitch(gnn_outputs['surface_forces'].flatten(), body_fixed_rotation_angle, use_csdl=True).reshape(-1,3)
    other_half_forces = half_forces.set(csdl.slice[:,1], value = -half_forces[:,1])
    surface_forces2 = csdl.concatenate((half_forces, other_half_forces))

    # final usuable variables:
    L2.add_name('L')
    D2.add_name('D')
    My2.add_name('My')
    surface_points2.add_name('surface_points')
    surface_forces2.add_name('surface_forces')
    # ==================================== GNN 1 ====================================
    # ...............................................................................

    L_tot = L + L2
    D_tot = D + D2
    My_tot = My + My2

    recorder.stop()

    js = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs = (scale_var, pitch_dv_var),
        additional_outputs = [L, D, My, L2, D2, My2, L_tot, D_tot, My_tot],
        # additional_outputs = [L_tot, D_tot, My_tot],
        gpu = False,
    )
    # js = csdl.experimental.PySimulator(
        # recorder=recorder,
    # )

    # js.compute_totals()
    # exit()
    # js.check_totals(step_size = 1e-1)
    # js.check_totals(step_size = 1e-2)
    # js.check_totals(step_size = 1e-3)
    # exit()

    pitches = np.linspace(-6, 6, 13)
    scales = np.ones((13,))

    # scales = np.linspace(0.8, 1.2, 5)

    # pitches = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    # scales = [1.0, 1.0]
    for scale, pitch in zip(scales, pitches):
        # for scale in scales:
        # js[pitch_dv_var] = pitch
        js[pitch_dv_var] = np.array([pitch, pitch])
        js[scale_var] = scale
        js.run()

        print(
            f"scale={scale}, deform_full_volume={deform_full_volume}, "
            f"L={js[L]}, D={js[D]}, My={js[My]}, "
            f"L_tot={js[L_tot]}, D_tot={js[D_tot]}, My_tot={js[My_tot]}"
        )

        # test nondeterminism
        # for i in range(10):
        #     js.run()
        #     L_val = js[L]
        #     print(f"  iteration {i}: L={L_val}")


        # test nondeterminism
        # for i in range(20):
        #     totals = js.compute_totals()
        #     dL = totals[L, scale_var]
        #     dD = totals[D, scale_var]
        #     dMy = totals[My, scale_var]
        # print(f"  iteration {i}: dL={dL}, dD={dD}, dMy={dMy}")
        # exit()

        sweep_vals['L'].append(js[L])
        sweep_vals['D'].append(js[D])
        sweep_vals['M'].append(js[My])

    print((sweep_vals['L']))
    print((sweep_vals['D']))
    print((sweep_vals['M']))
    for name in ("L", "D", "M"):
        values = np.asarray(jax.device_get(sweep_vals[name]), dtype=float).flatten()
        values_str = np.array2string(values, separator=", ", precision=8, threshold=np.inf, max_line_width=np.inf)
        print(f"{name} = np.array({values_str})")