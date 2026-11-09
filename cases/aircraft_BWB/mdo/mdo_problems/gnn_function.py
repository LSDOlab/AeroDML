import sys
from pathlib import Path
import pickle
import numpy as np
import csdl_alpha as csdl

# Reuse the repository backend and its selected trained weights.
rgnn_directory = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(rgnn_directory))

def evaluate_cfd_surface(geometry):
    from build_gnn import surface_mesh_projection_file_path
    # Evaluate mesh:
    with open(surface_mesh_projection_file_path, 'rb') as handle:
        projected_surf_mesh_dafoam = pickle.load(handle)
    x_surf_dafoam_full_geo = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)
    surface_var = x_surf_dafoam_full_geo
    return surface_var

def compute_rans_gnn(surface_var, pitch_var, panel_surface_var = None):
    # .............................................................................
    from build_gnn import rotate_pitch
    # ==================================== GNN ====================================
    if panel_surface_var is None:
        from gnn_model_operation import RANSGNN
        ransgnn = RANSGNN()
        gnn_outputs = ransgnn.evaluate(surface_var, pitch_var)
    else:
        from pgnn_model_operation import PGRANSGNN
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
    L.add_name('L_gnn')
    D.add_name('D_gnn')
    My.add_name('M_gnn')
    surface_points.add_name('surface_points_gnn')
    surface_forces.add_name('surface_forces_gnn')
    # ==================================== GNN ====================================
    # .............................................................................

    return {
        'L':L,
        'D':D,
        'M':My,
        'surface_points':surface_points,
        'surface_forces':surface_forces,
    }


_panel_functions = None


def compute_panel(pitch_var, panel_surface_var):
    """Full-aircraft panel loads at fixed flow conditions; pitch is in degrees.

    D is induced drag only. Mesh points must use the panel topology ordering.
    """
    from panel_model_operation import PANEL, build_default_panel_functions

    global _panel_functions
    if _panel_functions is None:
        # The builder uses its own recorder and restores the active recorder.
        _panel_functions = build_default_panel_functions()

    panel_outputs = PANEL(_panel_functions).evaluate(panel_surface_var, pitch_var)
    for name in ('L', 'D', 'M', 'surface_forces'):
        panel_outputs[name].add_name(f'{name}_panel')
    return panel_outputs
