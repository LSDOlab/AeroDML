"""Geometry-only graph deformation viewer. Run in the aerodml environment.

Default: interactive, linked cameras. Press S to save camera, window size,
and debug PNG. Use Left/Right to step the deformation. Set EXPORT_FRAMES=True after choosing the camera to
write PNG frames and a GIF. No model parameters, inference, or CFD are loaded.
"""
from pathlib import Path
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MDO_DIR = HERE.parent
CASE_DIR = MDO_DIR.parent
BWB_DIR = CASE_DIR.parent / 'datasets' / 'BWB'
sys.path.insert(0, str(MDO_DIR))
sys.path.insert(0, str(CASE_DIR.parents[1] / 'src'))

# ---------------------------- Settings ----------------------------------
EXPORT_FRAMES = True
NUM_FRAMES = 41
FRAME_DURATION_MS = 70
WINDOW_SIZE = (1800, 750)
OUTPUT_DIR = HERE / 'output'
CAMERA_FILE = OUTPUT_DIR / 'camera.json'
# Chord and wing sweep move together; all other geometry inputs stay fixed.
CHORD_LOWER = np.array([0.9, 0.9, 0.8, 0.8])
CHORD_UPPER = np.array([1.10, 0.97, 0.97, 0.94])
# Degrees, matching visualize_sweeps. Set equal to hold wing sweep fixed.
WING_SWEEP_LOWER = 15.0
WING_SWEEP_UPPER = 42.0
DEBUG_FRACTION = 0.5
# Keep the near-aircraft positive-y half-volume readable. None shows all edges.
# Membership is selected once on the reference mesh, so edges never flicker.
GRAPH_BOUNDS = (-7., 42., 0., 33., -7., 10.)  # xmin,xmax,ymin,ymax,zmin,zmax
EDGE_COLOR = '#354d65'
EDGE_OPACITY = 0.35
LINE_WIDTH = 1.0
SURFACE_COLOR = 'lightgray'
# ------------------------------------------------------------------------


def build_deformation():
    import csdl_alpha as csdl
    import jax
    from geometry import build_surface_csdl
    from mdo_utils import read_mesh
    from idwarp_jax.vol_warper import build_volume_pts_func
    from aerodml.load_mesh import build_geometry_case

    mesh_dir = CASE_DIR / 'model' / 'mesh_639k'
    geometry_dir = BWB_DIR / 'bwb_geometry'
    topology = read_mesh(str(mesh_dir), {
        'verbose': False, 'run_checks': True, 'visualize': False,
    })['geometry_topology']
    # Geometry imports cache STEP data relative to cwd, as in visualize_sweeps.
    old_cwd = Path.cwd()
    os.chdir(HERE)
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        surface, inputs = build_surface_csdl(
            recorder, geometry_dir / 'bwb_new_stored_refit.pickle',
            geometry_dir / 'bwbv2_no_wingtip_coarse_refined_flat.stp',
            mesh_dir / 'projected_surface_mesh_NEWHASH.pickle',
            str(mesh_dir), 'NEWHASH', index_unique=False,
        )
    finally:
        recorder.stop()
        os.chdir(old_cwd)
    simulator = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=list(inputs.values()),
        additional_outputs=[surface], gpu=False,
    )
    gids = np.load(mesh_dir / 'surf_gid_full_a63d3d5d35e5809c.npy')
    _, surface_indices = np.unique(gids, return_index=True)
    # Use CPU by default, with the same volume warper and cached routes as MDO.
    with jax.enable_x64(False):
        warp, _ = build_volume_pts_func(
            topology['foam_mesh'], np.array(topology['foam_mesh']['points']),
            jax.devices('gpu')[0],
            cached_routes=str(mesh_dir / 'cached_routes_NEWHASH'),
            print_timings=True, progress=True,
        )

    def evaluate(fraction):
        ratios = (1. - fraction) * CHORD_LOWER + fraction * CHORD_UPPER
        variable = inputs['centerbody_chord_stretch_ratio']
        simulator[variable] = ratios.reshape(variable.shape)
        sweep = (1. - fraction) * WING_SWEEP_LOWER + fraction * WING_SWEEP_UPPER
        sweep_variable = inputs['wing_sweep']
        simulator[sweep_variable] = np.full(sweep_variable.shape, sweep)
        simulator.run()
        with jax.enable_x64(False):
            points = warp(np.asarray(simulator[surface])[surface_indices],
                          {'pitch': np.array([0.], dtype=np.float32)})
            geometry = build_geometry_case(topology, points, False)
            result = {'points': np.asarray(points)}
            for key in topology['hierarchy']:
                result[key] = np.asarray(geometry[key]['dual_cells'])
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError('Nonfinite deformed coordinates')
        return result, ratios

    return topology, evaluate


def graph_lines(graph):
    """Pack actual graph connectivity as VTK lines, without tubes or glyphs."""
    points = np.asarray(graph['dual_cells'])
    edges = np.asarray(graph['dual_edges'], dtype=np.int64)
    keep = points[:, 1] >= -1.e-6
    if GRAPH_BOUNDS is not None:
        bounds = np.asarray(GRAPH_BOUNDS).reshape(3, 2)
        keep &= ((points >= bounds[:, 0] - 1.e-6) &
                 (points <= bounds[:, 1] + 1.e-6)).all(axis=1)
    edges = edges[keep[edges].all(axis=1)]
    if not len(edges):
        raise ValueError('GRAPH_BOUNDS excludes every edge; enlarge it.')
    ids, inverse = np.unique(edges, return_inverse=True)
    lines = np.column_stack((np.full(len(edges), 2), inverse.reshape(-1, 2)))
    return ids, lines.ravel()


def main():
    import pyvista as pv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    topology, evaluate = build_deformation()
    data, ratios = evaluate(DEBUG_FRACTION)
    foam = topology['foam_mesh']
    wall_faces = [np.asarray(face, dtype=np.int64)
                  for face, kind in zip(foam['faces']['points'], foam['faces']['type'])
                  if kind == 1]
    wall_ids, inverse = np.unique(np.concatenate(wall_faces), return_inverse=True)
    packed_faces = []
    offset = 0
    for face in wall_faces:
        packed_faces.extend([len(face), *inverse[offset:offset + len(face)]])
        offset += len(face)

    camera = json.loads(CAMERA_FILE.read_text()) if CAMERA_FILE.exists() else None
    window_size = WINDOW_SIZE
    if camera is not None:
        if 'window_size' in camera:
            window_size = tuple(camera['window_size'])
        elif (OUTPUT_DIR / 'debug_frame.png').exists():
            # Older saves omitted window dimensions; recover from their screenshot.
            from PIL import Image
            with Image.open(OUTPUT_DIR / 'debug_frame.png') as saved_frame:
                window_size = saved_frame.size
    plotter = pv.Plotter(shape=(1, 3), window_size=window_size,
                         off_screen=EXPORT_FRAMES, border=False)
    actors = []
    state = {'fraction': DEBUG_FRACTION, 'ratios': ratios}
    for index, key in enumerate(topology['hierarchy']):
        plotter.subplot(0, index)
        plotter.set_background('white')
        half = pv.PolyData(data['points'][wall_ids], packed_faces)
        mirror = half.copy()
        mirror.points[:, 1] *= -1
        for mesh in (half, mirror):
            plotter.add_mesh(mesh, color=SURFACE_COLOR, smooth_shading=True)
        ids, lines = graph_lines(topology[key])
        graph = pv.PolyData(data[key][ids], lines=lines)
        plotter.add_mesh(graph, color=EDGE_COLOR, opacity=EDGE_OPACITY,
                         line_width=LINE_WIDTH, lighting=False)
        actors.append((half, mirror, graph, ids, key))
    plotter.link_views()
    plotter.subplot(0, 0)
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    if camera is not None:
        plotter.camera_position = camera['position']
        plotter.camera.parallel_projection = camera['parallel_projection']
        plotter.camera.parallel_scale = camera['parallel_scale']
        plotter.camera.view_angle = camera['view_angle']

    def update(fraction):
        fraction = float(fraction)
        if fraction != state['fraction']:
            frame, ratios = evaluate(fraction)
            for half, mirror, graph, ids, key in actors:
                half.points = frame['points'][wall_ids]
                mirrored = half.points.copy()
                mirrored[:, 1] *= -1
                mirror.points = mirrored
                graph.points = frame[key][ids]
            state.update(fraction=fraction, ratios=ratios)
        plotter.subplot(0, 0)
        for renderer in plotter.renderers:
            renderer.reset_camera_clipping_range()
        plotter.render()

    def save():
        plotter.subplot(0, 0)
        camera = {
            'position': [list(v) for v in plotter.camera_position],
            'parallel_projection': bool(plotter.camera.parallel_projection),
            'parallel_scale': float(plotter.camera.parallel_scale),
            'view_angle': float(plotter.camera.view_angle),
            'window_size': [int(v) for v in plotter.window_size],
        }
        CAMERA_FILE.write_text(json.dumps(camera, indent=2) + '\n')
        plotter.render()
        plotter.screenshot(str(OUTPUT_DIR / 'debug_frame.png'))
        print(f'Saved camera and debug_frame.png in {OUTPUT_DIR}', flush=True)

    update(DEBUG_FRACTION)
    if EXPORT_FRAMES:
        if NUM_FRAMES < 2:
            raise ValueError('NUM_FRAMES must be at least 2')
        from PIL import Image
        frames = []
        # Ease to zero speed at both ends, then play back without duplicate ends.
        fractions = 0.5 * (1. - np.cos(np.linspace(0., np.pi, NUM_FRAMES)))
        for index, fraction in enumerate(fractions):
            print(f'Frame {index + 1}/{NUM_FRAMES}', flush=True)
            update(fraction)
            path = OUTPUT_DIR / f'frame_{index:04d}.png'
            pixels = plotter.screenshot(str(path), return_img=True)
            frames.append(Image.fromarray(pixels))
            path.with_suffix('.json').write_text(json.dumps({
                'fraction': float(fraction),
                'centerbody_chord_stretch_ratio': state['ratios'].tolist(),
                'wing_sweep_degrees': float(
                    (1. - fraction) * WING_SWEEP_LOWER + fraction * WING_SWEEP_UPPER),
            }, indent=2) + '\n')
        frames[0].save(OUTPUT_DIR / 'graph_deformation.gif', save_all=True,
                       append_images=frames[1:] + frames[-2:0:-1],
                       duration=FRAME_DURATION_MS, loop=0)
        plotter.close()
    else:
        plotter.add_key_event('s', save)
        plotter.add_key_event('Left', lambda: update(max(0., state['fraction'] - 0.05)))
        plotter.add_key_event('Right', lambda: update(min(1., state['fraction'] + 0.05)))
        print('Drag to choose camera; Left/Right to deform. Press S to save camera, size + PNG.',
              flush=True)
        plotter.show()


if __name__ == '__main__':
    main()
