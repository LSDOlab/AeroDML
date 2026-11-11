"""Sweep BWB geometry and render pressure using the same plot_field as run.py.

Run this file from any directory in the aerodml environment. Edit settings below;
there are no command-line arguments. Model paths, devices and freestream settings
come from the parent mdo modules. No gradients are evaluated.
"""
from pathlib import Path
import importlib
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

# ------------------------------ Settings ---------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODEL = "pgnn"  # One of: panel, gnn, pgnn. run_sweeps.sh runs all three.
NUM_FRAMES = 41
FRAME_DURATION_MS = 150
all_at_once = False
all_at_once = True

# Enable for two frames of the selected model. It sweeps only TEST_PARAMETER
# and holds every other parameter at ref_value.
TEST_RUN = True
TEST_RUN = False
TEST_PARAMETER = "wing_sweep"

# Copied from datasets/BWB/data/sample_dataset/build_configs.py.
# Remove BOTH lower and upper to hold a parameter fixed at ref_value.
# With all_at_once=False, inactive parameters also use ref_value.
# Arrays interpolate elementwise as one parameter (not one sweep per element).
# Descending bounds, such as wing_position, intentionally run backwards.
PARAMETERS = {
    # Pitch is in degrees. Uncomment the bounds to sweep it like geometry.
    "pitch": {
        "ref_value": 0.0,
        "lower": -3.0, "upper": 3.0,
    },
    "centerbody_chord_stretch_ratio": {
        "ref_value": [1.0, 0.9265333333333333, 0.8718160886458484, 0.8330375933644204],
        # "lower": [0.9, 0.9, 0.8, 0.8],
        # "upper": [1.10, 0.97, 0.97, 0.94],
    },
    "wing_position": {
        "ref_value": 20.186592500296157,
        # "lower": 22.186592500296157, "upper": 13.186592500296157,
    },
    "wing_sweep": {
        "ref_value": 35.067400250175616,
        "lower": 15.0, "upper": 42.0
    },
    "wing_half_span": {
        "ref_value": 15.961024804157153,
        # "lower": 8.0, "upper": 20.0
    },
    "centerbody_half_span": {
        "ref_value": 5.000000000000001,
        "lower": 3.0, "upper": 7.0
    },
    "transition_half_span": {
        "ref_value": 4.890738003669036,
        "lower": 3.5, "upper": 7.0
    },
    "wing_chord_stretch_coefficients": {
        "ref_value": [0.0, 0.0, 0.0, 0.0],
        "lower": [-3.0000005, -2.49952453, -1.99904855, -1.49857258],
        "upper": [3.0000005, 2.49952453, 1.99904855, 1.49857258],
    },
    "wing_twist_coefficients": {
        "ref_value": [0.0] * 4,
        # "lower": [-5.0] * 4, "upper": [5.0] * 4,
    },
    "airfoil_thickness_coefficients": {
        "ref_value": np.zeros((8, 8)),
        # "lower": np.full((8, 8), -0.05), "upper": np.full((8, 8), 0.05),
    },
    "airfoil_camber_coefficients": {
        "ref_value": np.zeros((6, 8)),
        # "lower": np.full((6, 8), -0.03), "upper": np.full((6, 8), 0.03),
    },
    "elevator_deflection": {
        "ref_value": 0.0,
        # "lower": 0.0, "upper": 7.0
    },
}

# Same section and view as model/run.py. Any XCWindow plane can be used:
# ('y', (0, 5, 0)), ('x', (20, 0, 0)), or ((nx, ny, nz), (x, y, z)).
SECTIONS = [{"plane": ("y", (0, 16, 0)), "center": "auto"}]
CAMERA = "pos_3d_paper"
PRESSURE_LIMITS = (10000.0, 40000.0)  # Pa; identical across models and frames
WINDOW_SIZE = (1200, 900)
GIF_WIDTH = 900
# -------------------------------------------------------------------------

MDO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MDO_DIR))


def make_sweeps(parameters, num_frames, together):
    """Return (name, active parameters, frames) for each requested sweep."""
    if num_frames < 2:
        raise ValueError("Use at least two frames to include both endpoints.")
    references = {}
    varying = []
    for name, settings in parameters.items():
        reference = np.asarray(settings["ref_value"], dtype=float)
        if name == "pitch" and reference.size != 1:
            raise ValueError("pitch must be a single angle in degrees")
        if not np.isfinite(reference).all():
            raise ValueError(f"{name}: ref_value must be finite")
        references[name] = reference
        if ("lower" in settings) != ("upper" in settings):
            raise ValueError(f"{name}: provide both lower and upper, or neither")
        if "lower" in settings:
            for bound in ("lower", "upper"):
                value = np.asarray(settings[bound], dtype=float)
                if value.shape != reference.shape or not np.isfinite(value).all():
                    raise ValueError(f"{name}: {bound} must match ref_value's shape and be finite")
            varying.append(name)
    groups = [varying] if together else [[name] for name in varying]
    if not varying:
        groups = [[]]  # A fixed-geometry animation is allowed.
    sweeps = []
    for active in groups:
        frames = []
        for fraction in np.linspace(0.0, 1.0, num_frames):
            values = {name: value.copy() for name, value in references.items()}
            for name in active:
                lower = np.asarray(parameters[name]["lower"], dtype=float)
                upper = np.asarray(parameters[name]["upper"], dtype=float)
                values[name] = (1.0 - fraction) * lower + fraction * upper
            frames.append(values)
        name = "all_at_once" if together and active else (active[0] if active else "fixed")
        sweeps.append((name, active, frames))
    return sweeps


def build_geometry(model, panel_projections):
    """Build a geometry-only CPU simulator, reused for every frame."""
    import csdl_alpha as csdl
    from geometry import build_surface_csdl

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    surface, inputs, geometry = build_surface_csdl(
        recorder, model.geometry_pickle_file_path, model.stp_file_path,
        model.surface_mesh_projection_file_path, model.dafoam_directory,
        model.x_surf_hash, index_unique=False, return_geometry=True,
    )
    panel_points = None
    outputs = [surface]
    if panel_projections is not None:
        panel_points = geometry.evaluate(panel_projections, plot=False)
        outputs.append(panel_points)
    recorder.stop()
    simulator = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=list(inputs.values()),
        additional_outputs=outputs, gpu=False,
    )
    return simulator, inputs, surface, panel_points


def evaluate_nn(model, volume_points, pitch, lowfi_data=None):
    """Run full-field inference on the parent's GNN device; return host arrays."""
    import jax

    def to_device(value):
        return jax.device_put(np.asarray(value, dtype=np.float32), model.constants["gnn_device"])

    with jax.enable_x64(False):
        result = model.ml_func_full(
            to_device(volume_points), to_device(pitch),
            lowfi_data=jax.tree.map(to_device, lowfi_data),
        )
        return jax.device_get(result[-1])


def panel_plot_data(panel_functions, points, result, pitch):
    """Adapt surface panels to plot_field without inventing a volume field.

    plot_field mirrors a half aircraft. Keep the positive-y panels and use
    polygon cells for plot_xc: the section then contains only airfoil lines.
    Panel connectivity and pressure share VortexAD's cell-dictionary ordering.
    """
    import pyvista as pv
    from bwb_helper_functions import rotate_pitch

    points = rotate_pitch(points.flatten(), float(pitch[0]), use_csdl=False).reshape(-1, 3)
    cells = panel_functions.topology.cell_adjacency_data[1]
    faces = [np.asarray(face, dtype=int) for block in cells.values() for face in block]
    keep = np.array([points[face, 1].mean() >= 0.0 for face in faces])
    faces = [face for face, selected in zip(faces, keep) if selected]
    packed = np.concatenate([np.r_[len(face), face] for face in faces])
    celltypes = np.full(len(faces), pv.CellType.POLYGON, dtype=np.uint8)
    topology = {"foam_mesh": {"faces": {
        "points": faces, "type": np.ones(len(faces), dtype=int),
        "surface_neighbors": np.arange(len(faces)),
    }}}
    return topology, {"foam_mesh": {"points": points}}, np.asarray(result["pressure"])[keep], (packed, celltypes)


def render_frame(model_name, base, topology, geometry, pressure, mesh, frame_id, folder, caption):
    """Call run.py's renderer, then stack the surface and section PNGs."""
    from aerodml.postprocessing_viz import (
        plot_field, XCWindow, _cfd_wall_surface, _xc_plane_geometry,
    )

    # plot_field expects normalized model outputs. Supply the already computed
    # pressure through its NN interface so plotting never reruns the model.
    normalized_pressure = (pressure - base.constants["p"]["mean"]) / base.constants["p"]["std"]

    def cached_prediction(params, flow, geo):
        return None, None, normalized_pressure, None

    config = {
        "case_id": f"{frame_id:04d}", "topology": topology,
        "warped_geo": geometry, "flow_condition": base.flow_condition,
        # Used only to construct the wall surface; these are NOT CFD results.
        "data": {"p": pressure},
    }
    # A span sweep may move the wing completely inside a fixed section plane.
    # Keep a blank, labeled section in that frame rather than aborting the GIF.
    windows = [XCWindow(**section) for section in SECTIONS]
    surface = _cfd_wall_surface(config)
    intersects = []
    for window in windows:
        normal, origin = _xc_plane_geometry(window.plane)
        intersects.append(surface.slice(normal=normal, origin=origin).n_points > 0)
    paths = plot_field(
        cached_prediction, None, config, base.constants,
        screenshot=True, model_name=model_name, model_type="NN",
        plots_dir=str(folder / "plots"), state="p", camera=CAMERA,
        cross_section_stations=[window for window, hit in zip(windows, intersects) if hit],
        pyvista_mesh_topology=mesh, window_size=WINDOW_SIZE,
        smooth_scalars=True, cmap="jet", clim=PRESSURE_LIMITS,
        parallel_projection=False, lighting=True, contour_levels=None,
    )
    selected = [path for path in paths if Path(path).suffix == ".png" and "_raw_" not in Path(path).name]
    rendered = []
    for path in selected:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = round(image.height * GIF_WIDTH / image.width)
            rendered.append(image.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS))
    images = [rendered[0]]  # Surface view with section windows.
    source_paths = [selected[0]]
    section_paths = iter(selected[1:])
    section_images = iter(rendered[1:])
    for window, hit in zip(windows, intersects):
        if hit:
            images.append(next(section_images))
            source_paths.append(next(section_paths))
        else:
            blank = Image.new("RGB", (GIF_WIDTH, round(GIF_WIDTH / window.aspect_ratio)), "white")
            ImageDraw.Draw(blank).text(
                (20, 30), f"Section {window.plane}: outside the aircraft", fill="black",
            )
            images.append(blank)
            source_paths.append(None)
    header_height = 28 + 18 * len(caption)
    frame = Image.new("RGB", (GIF_WIDTH, header_height + sum(image.height for image in images)), "white")
    draw = ImageDraw.Draw(frame)
    draw.text((12, 8), model_name.upper() + (" (surface pressure only)" if model_name == "panel" else ""), fill="black")
    for index, line in enumerate(caption):
        draw.text((12, 28 + 18 * index), line, fill="black")
    y = header_height
    for image in images:
        frame.paste(image, (0, y))
        y += image.height
    # Record source plots instead of saving duplicate composite PNGs.
    metadata_path = folder / f"{model_name}_{frame_id:04d}.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["caption"] = caption
    metadata["plot_images"] = [
        {"path": str(Path(path).relative_to(folder)) if path is not None else None,
         "size": list(image.size)}
        for path, image in zip(source_paths, images)
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return frame


def main():
    # Validate the inexpensive settings before loading geometry or models.
    parameters = {name: dict(settings) for name, settings in PARAMETERS.items()}
    # The bash launcher selects one model per process; direct runs use MODEL.
    model = os.environ.get("BWB_SWEEP_MODEL", MODEL)
    models = [model]
    num_frames = NUM_FRAMES
    if TEST_RUN:
        num_frames = 2
        for name, settings in parameters.items():
            if name != TEST_PARAMETER:
                settings.pop("lower", None)
                settings.pop("upper", None)
        if TEST_PARAMETER not in parameters or "lower" not in parameters[TEST_PARAMETER]:
            raise ValueError("TEST_PARAMETER must name a parameter with bounds")
    if model not in {"panel", "gnn", "pgnn"}:
        raise ValueError("MODEL must be panel, gnn, or pgnn")
    sweeps = make_sweeps(parameters, num_frames, all_at_once)
    root = OUTPUT_DIR / ("test" if TEST_RUN else "sweeps")

    run_sweeps(models, parameters, sweeps, root, num_frames)


def run_sweeps(models, parameters, sweeps, root, num_frames):
    """Evaluate one model's sweeps in its own process, then release its GPU."""
    import jax
    import pyvista as pv
    import build_gnn as base
    from aerodml.visualize_mesh import build_pyvista_mesh_topology

    pv.OFF_SCREEN = True
    neural_models = {name: importlib.import_module(f"{name}_model_operation")
                     for name in models if name != "panel"}
    panel_functions = None
    panel_projections = None
    if "pgnn" in neural_models:
        panel_functions = neural_models["pgnn"].panel_functions
        panel_projections = neural_models["pgnn"].panel_projections
    elif "panel" in models:
        from panel_model_operation import build_default_panel_functions
        panel_functions = build_default_panel_functions(device=base.constants["panel_device"])
        with open(Path(base.dafoam_directory) / f"panel_projections_{base.x_surf_hash}.pickle", "rb") as handle:
            panel_projections = pickle.load(handle)

    simulator, inputs, surface, panel_surface = build_geometry(base, panel_projections)
    unknown = set(parameters) - set(inputs) - {"pitch"}
    if unknown:
        raise ValueError(f"Unknown geometry parameters: {sorted(unknown)}")
    volume_mesh = build_pyvista_mesh_topology(base.topology["foam_mesh"]) if neural_models else None
    # If pitch is omitted from PARAMETERS, keep the parent's incidence.
    default_pitch = np.asarray(base.flow_condition["alpha"], dtype=float).reshape(1)

    for sweep_name, active, frames in sweeps:
        folder = root / sweep_name
        folder.mkdir(parents=True, exist_ok=True)
        animations = {name: [] for name in models}
        pitch_history = []
        for frame_id, values in enumerate(frames):
            print(f"\n{sweep_name}: frame {frame_id + 1}/{num_frames}", flush=True)
            pitch = np.asarray(values.get("pitch", default_pitch), dtype=float).reshape(1)
            pitch_history.append(float(pitch[0]))
            for name, value in values.items():
                if name != "pitch":
                    simulator[inputs[name]] = value.reshape(inputs[name].shape)
            simulator.run()
            # Save every actual geometry input, including fixed/default values
            # and full arrays, rather than the rounded min/max caption text.
            # Write per frame so completed work remains usable after a failure.
            geometry_values = {
                name: np.asarray(simulator[variable]).tolist()
                for name, variable in inputs.items()
            }
            for name in models:
                frame_values = {
                    "model": name,
                    "sweep": sweep_name,
                    "frame_index": frame_id,
                    "frame_duration_ms": FRAME_DURATION_MS,
                    "swept_parameters": active,
                    "geometry": geometry_values,
                    "pitch_degrees": float(pitch[0]),
                }
                (folder / f"{name}_{frame_id:04d}.json").write_text(
                    json.dumps(frame_values, indent=2) + "\n"
                )
            panel_points = None if panel_surface is None else simulator[panel_surface]
            panel_result = None
            if panel_functions is not None:
                panel_result = panel_functions.forward(panel_points, pitch)
            volume_points = None
            if neural_models:
                with jax.enable_x64(False):
                    surface_points = simulator[surface][base.surf_25r_to_1r_map]
                    volume_points = base.compute_volume_pts(surface_points, {"pitch": pitch})
            caption = [f"{sweep_name}: frame {frame_id + 1}/{num_frames}; pitch = {pitch[0]:g} deg"]
            for name in active:
                value = values[name]
                summary = f"{value.item():.5g}" if value.size == 1 else f"min {value.min():.5g}, max {value.max():.5g}"
                caption.append(f"{name}: {summary}")
            for name in models:
                print(f"  Evaluating/rendering {name}", flush=True)
                if name == "panel":
                    topology, geometry, pressure, mesh = panel_plot_data(panel_functions, panel_points, panel_result, pitch)
                else:
                    lowfi = None if name == "gnn" else {key: panel_result[key] for key in ("pressure", "velocity")}
                    result = evaluate_nn(neural_models[name], volume_points, pitch, lowfi)
                    topology, geometry = base.topology, result["warped_geo"]
                    pressure, mesh = result["p"], volume_mesh
                if not np.isfinite(pressure).all():
                    raise ValueError(f"{name}: nonfinite pressure in frame {frame_id}")
                frame = render_frame(name, base, topology, geometry, pressure, mesh, frame_id, folder, caption)
                animations[name].append(frame)
        manifest = {
            "frames": [{name: value.tolist() for name, value in values.items()} for values in frames],
            "pitch": pitch_history, "sections": SECTIONS,
            "pressure_limits": PRESSURE_LIMITS, "frame_duration_ms": FRAME_DURATION_MS,
        }
        for name, images in animations.items():
            path = folder / f"{name}.gif"
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=FRAME_DURATION_MS, loop=0, disposal=2)
            path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
