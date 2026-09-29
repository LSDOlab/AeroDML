import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from aerodeltanet.utils import print_content
# ============================================================
# Edit these settings only
# ============================================================

DATASET_DIR = "sample_case"
SPLIT_DIR = "split_0"
CFD_MESH = {"rans": "mesh_639k", "euler": "mesh_112k"}

# Any subset/order of: "rans", "euler", "panel"
SOLVERS = ("rans",)
# SOLVERS = ("rans", "euler")
# SOLVERS = ("rans", "euler", 'panel')

# CFD: "p", "pressure", "T", "nuTilda", or "U_mag".
# Panel: "p", "pressure", "U_mag", or "velocity_magnitude".
# STATE = "p"
STATE = "T"

# None = all cases. Otherwise use 0-based case ids, e.g. (0, 3, 7).
CASES = None

MODE = "interactive"       # "interactive" or "video"
MODE = "video"       # "interactive" or "video"
FPS = 1
USE_XVFB = False            # Set True only if your HPC needs Xvfb and has it installed.

# None = auto scale. Tuple = fixed scale, e.g. (1.0e4, 4.0e4).
CLIM = None
SHARED_CLIM = True          # When CLIM is None, share one auto scale across plotted solvers per case.
AUTO_PERCENTILES = (1, 99)  # Use None for true min/max.

# Either a key from available_cameras or a camera dict with position/view_angle/clipping_range.
# CAMERA = "pos_2"
CAMERA = "TE"
# CAMERA = "TE_underside"

MIRROR_CFD = True           # Mirror RANS/Euler across y=0. Panel is left unmirrored.
CMAP = "viridis"
SHOW_SCALAR_BAR = True
SMOOTH_SCALARS = False
WINDOW_WIDTH = 1200         # For mp4, keep width and row height divisible by 16.
ROW_HEIGHT = 512

VIDEO_PATH = f"plots/CFD_surface_{CAMERA}_{STATE}_{DATASET_DIR}_{SPLIT_DIR}.mp4"
# ============================================================

if MODE == "video":
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv

from run import DATA_DIR, get_graph, load_cfd_data
from aerodeltanet.load_panel_data import get_panel_mesh, load_panel_data
from aerodeltanet.visualize_mesh import available_cameras, visualize_surface


LOAD_SETTINGS = {"verbose": False, "run_checks": True, "visualize": False}
PANEL_CONNECTIVITY = "CONNECTIVITY.pkl"


def camera_name(camera):
    if isinstance(camera, str):
        if camera not in available_cameras:
            raise ValueError(f"Unknown CAMERA={camera!r}. Available: {tuple(available_cameras)}")
        return camera

    required = {"position", "view_angle", "clipping_range"}
    missing = required.difference(camera)
    if missing:
        raise ValueError(f"Custom CAMERA is missing {missing}")

    available_cameras["input_camera"] = camera
    return "input_camera"


def state_name():
    return {
        "pressure": "p",
        "velocity_magnitude": "U_mag",
        "|U|": "U_mag",
    }.get(STATE, STATE)


def load_data():
    dataset_path = f"{DATA_DIR}/{DATASET_DIR}"
    split_path = f"{dataset_path}/{SPLIT_DIR}"
    loaded = {}

    for solver in SOLVERS:
        if solver in CFD_MESH:
            topo = get_graph(
                CFD_MESH[solver],
                mesh_load_settings=LOAD_SETTINGS,
                dual_load_settings=LOAD_SETTINGS,
                coarse_load_settings=LOAD_SETTINGS,
            )
            loaded[solver] = {
                "topology": topo,
                "data": load_cfd_data(dir=f"{dataset_path}/{solver}", split=split_path, num_cases=NUM_CASES),
            }
            # print_content(topo)
            # print_content(loaded[solver]['data'][0])
            # exit()

            continue

        if solver == "panel":
            # get_panel_mesh needs the RANS topology, matching the original plotting script.
            rans_topo = loaded.get("rans", {}).get("topology") or get_graph(
                CFD_MESH["rans"],
                mesh_load_settings=LOAD_SETTINGS,
                dual_load_settings=LOAD_SETTINGS,
                coarse_load_settings=LOAD_SETTINGS,
            )
            loaded["panel"] = {
                "topology": get_panel_mesh(PANEL_CONNECTIVITY, rans_topo, LOAD_SETTINGS),
                "data": load_panel_data(dir=f"{dataset_path}/panel", split_dir=split_path),
            }
            continue

        raise ValueError(f"Unknown solver {solver!r}. Use only 'rans', 'euler', and/or 'panel'.")

    sizes = {solver: len(pack["data"]) for solver, pack in loaded.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Selected solvers have different case counts: {sizes}")
    return loaded


def cfd_scalars(case):
    state = state_name()
    if state == "U_mag":
        return np.linalg.norm(np.asarray(case["U"]), axis=1)

    values = np.asarray(case[state])
    if values.ndim != 1:
        raise ValueError(f"STATE={STATE!r} is not scalar. Use STATE='U_mag' for velocity magnitude.")
    return values


def panel_scalars(case):
    state = state_name()
    if state == "p":
        return np.asarray(case["pressure"])
    if state == "U_mag":
        return np.linalg.norm(np.asarray(case["velocity"]), axis=1)
    if state == 'T':
        return np.ones_like(np.asarray(case["pressure"]))*228
    raise ValueError("Panel supports only STATE='p', 'pressure', 'T', 'U_mag', or 'velocity_magnitude'.")

def surface_scalars(topology, scalars):
    wall = np.asarray(topology["foam_mesh"]["faces"]["type"]) == 1
    if scalars.shape[0] == int(wall.sum()):
        return scalars
    return scalars[np.asarray(topology["foam_mesh"]["faces"]["surface_neighbors"])]


def case_clim(loaded, case_id):
    if CLIM is not None:
        return tuple(CLIM)
    if not SHARED_CLIM:
        return None

    values = []
    for solver, pack in loaded.items():
        case = pack["data"][case_id]
        if solver == "panel":
            values.append(panel_scalars(case).ravel())
        else:
            values.append(surface_scalars(pack["topology"], cfd_scalars(case)).ravel())

    values = np.concatenate(values)
    if AUTO_PERCENTILES is None:
        return float(np.nanmin(values)), float(np.nanmax(values))
    return tuple(float(x) for x in np.nanpercentile(values, AUTO_PERCENTILES))


def add_cfd_surface(plotter, solver, pack, case_id, clim, camera):
    case = pack["data"][case_id]
    points = np.asarray(case["vertex_coordinates"])
    title = f"{solver.upper()} — {case.get('case_id', f'case_{case_id}')}"

    visualize_surface(
        pack["topology"],
        {"foam_mesh": {"points": points}},
        cfd_scalars(case),
        title=title,
        cmap=CMAP,
        clim=clim,
        camera=camera,
        p=plotter,
        enable_picking=(MODE == "interactive"),
        show_scalar_bar=SHOW_SCALAR_BAR,
        smooth_scalars=SMOOTH_SCALARS,
    )

    if not MIRROR_CFD:
        return

    mirrored = points.copy()
    mirrored[:, 1] *= -1.0
    visualize_surface(
        pack["topology"],
        {"foam_mesh": {"points": mirrored}},
        cfd_scalars(case),
        title=None,
        cmap=CMAP,
        clim=clim,
        camera=camera,
        p=plotter,
        enable_picking=False,
        show_scalar_bar=False,
        smooth_scalars=SMOOTH_SCALARS,
        add_axes=False,
    )


def add_panel_surface(plotter, pack, case_id, clim, camera):
    case = pack["data"][case_id]
    tris = np.asarray(pack["topology"]["triangles"], dtype=np.int64)
    faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int64), tris]).ravel()

    surf = pv.PolyData(np.asarray(case["panel_mesh"]), faces)
    surf.cell_data["field"] = panel_scalars(case)

    plotter.add_mesh(surf, scalars="field", cmap=CMAP, clim=clim, show_scalar_bar=SHOW_SCALAR_BAR)
    plotter.add_title(f"PANEL — {case.get('case_id', f'case_{case_id}')}")
    plotter.add_axes()

    cam = available_cameras[camera]
    plotter.camera_position = cam["position"]
    plotter.camera.view_angle = cam["view_angle"]
    plotter.camera.clipping_range = cam["clipping_range"]


def make_plotter(loaded, case_id, camera, off_screen=False):
    plotter = pv.Plotter(
        shape=(len(loaded), 1),
        window_size=(WINDOW_WIDTH, ROW_HEIGHT * len(loaded)),
        border=False,
        off_screen=off_screen,
    )
    clim = case_clim(loaded, case_id)

    for row, (solver, pack) in enumerate(loaded.items()):
        plotter.subplot(row, 0)
        if solver == "panel":
            add_panel_surface(plotter, pack, case_id, clim, camera)
        else:
            add_cfd_surface(plotter, solver, pack, case_id, clim, camera)

    if len(loaded) > 1:
        plotter.link_views()
    return plotter


def write_video(loaded, case_ids, camera):
    import imageio.v2 as imageio

    pv.OFF_SCREEN = True
    if USE_XVFB:
        pv.start_xvfb(wait=0.5)

    with imageio.get_writer(VIDEO_PATH, fps=FPS, macro_block_size=16) as writer:
        for i, case_id in enumerate(case_ids, start=1):
            print(f"Rendering frame {i}/{len(case_ids)}: case {case_id}")
            plotter = make_plotter(loaded, case_id, camera, off_screen=True)
            frame = plotter.screenshot(return_img=True)
            plotter.close()

            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            writer.append_data(frame)

    print(f"Wrote {VIDEO_PATH}")


def main():
    if MODE not in ("interactive", "video"):
        raise ValueError("MODE must be 'interactive' or 'video'.")

    loaded = load_data()
    camera = camera_name(CAMERA)
    n_cases = len(next(iter(loaded.values()))["data"])
    case_ids = list(range(n_cases)) if CASES is None else list(CASES)

    if MODE == "video":
        write_video(loaded, case_ids, camera)
        return

    for case_id in case_ids:
        plotter = make_plotter(loaded, case_id, camera, off_screen=False)
        plotter.show()
        plotter.close()


if __name__ == "__main__":
    main()