"""Run one RANS case using geometry coefficients stored in an HDF5 iteration.

The HDF5 geometry format follows make_gnn_animations_new.py:
each iteration contains a ``geometry_coefficients`` variable and, optionally,
a ``pitch`` variable.  CFD is deliberately opt-in through ``RUN_CFD`` in the
settings block; when false, this script only reconstructs the surface points.
"""

import sys
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from mpi4py import MPI

from bwb_cfd import build_cfd_func
from bwb_geometry import compute_geometry_csdl
from bwb_helper_functions import read_simple_pickle


U0 = 227.3805
P0 = 30089.6
T0 = 228.714
NU_TILDA0 = 4.5e-5

# ============================== USER SETTINGS ==============================
# Set these values before launching with mpirun from cases/datasets/BWB,
# using the same rank count as the decomposed mesh (as in batch_rans.sh).
HDF5_PATH = ""  # Required: set the input HDF5 file path (relative or absolute).
OUTPUT_DIR = Path("data/hdf5_rans_sep10")
CASE_NAME = "hdf5_geometry"
MESH = "mesh_639k"
X_SURF_HASH_STR = "NEWHASH"  # Match generate_rans_single.py.
ITERATION = 138

# Keep False while checking paths and geometry. Set True only to run DAFoam.
# RUN_CFD = False
RUN_CFD = True
# Set False to skip dataset output files; OpenFOAM may still write working files in MESH.
SAVE_OUTPUTS = True
# SAVE_OUTPUTS = False
# ============================================================================


def build_da_options():
    wall_list = [
        "wall_body_lower", "wall_body_upper",
        "wall_wing_lower", "wall_transition_lower",
        "wall_wing_upper", "wall_transition_upper", "wall_wing_cap",
    ]
    farfield_list = ["inout"]
    return {
        "designSurfaces": wall_list,
        "solverName": "DARhoSimpleCFoam",
        "primalMinResTol": 5e-9,
        "primalMinResTolDiff": 1e2,
        "primalBC": {
            "U0": {"variable": "U", "patches": farfield_list, "value": [U0, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": farfield_list, "value": [P0]},
            "T0": {"variable": "T", "patches": farfield_list, "value": [T0]},
            "nuTilda0": {"variable": "nuTilda", "patches": farfield_list, "value": [NU_TILDA0]},
            "useWallFunction": True,
        },
        "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
        "function": {
            "drag": {
                "type": "force", "source": "patchToFace", "patches": wall_list,
                "directionMode": "parallelToFlow", "patchVelocityInputName": "patch_velocity",
                "scale": 1.0,
            },
            "lift": {
                "type": "force", "source": "patchToFace", "patches": wall_list,
                "directionMode": "normalToFlow", "patchVelocityInputName": "patch_velocity",
                "scale": 1.0,
            },
        },
        "adjStateOrdering": "cell",
        "adjEqnOption": {"gmresRelTol": 1e-4, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
        "transonicPCOption": 2,
        "adjPCLag": 5,
        "normalizeStates": {"U": U0, "p": P0, "T": T0, "nuTilda": NU_TILDA0 * 10.0, "phi": 1.0},
        "inputInfo": {
            "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
            "patch_velocity": {
                "type": "patchVelocity", "patches": farfield_list, "flowAxis": "x",
                "normalAxis": "z", "components": ["solver", "function"],
            },
            "pressure": {
                "type": "patchVar", "varName": "p", "varType": "scalar",
                "patches": farfield_list, "components": ["solver", "function"],
            },
            "temperature": {
                "type": "patchVar", "varName": "T", "varType": "scalar",
                "patches": farfield_list, "components": ["solver", "function"],
            },
        },
        "checkMeshThreshold": {
            "maxAspectRatio": 10000.0, "maxNonOrth": 80.0,
            "maxSkewness": 10.0, "maxIncorrectlyOrientedFaces": 0,
        },
    }


def load_hdf5_surface(hdf5_path, iteration, mesh_dir):
    """Reconstruct DAFoam-ordered surface points from one HDF5 iteration."""
    root = Path(__file__).resolve().parents[3]
    geometry_dir = root / "cases" / "datasets" / "BWB" / "bwb_geometry"
    projection_path = Path(mesh_dir).resolve() / f"projected_surface_mesh_{X_SURF_HASH_STR}.pickle"
    if not projection_path.is_file():
        raise FileNotFoundError(f"Surface mesh projection does not exist: {projection_path}")

    recorder = csdl.Recorder(inline=True, debug=False)
    recorder.start()
    geometry, _, _, _ = compute_geometry_csdl(
        None,
        geometry_dir / "bwb_new_stored_refit.pickle",
        geometry_dir / "bwbv2_no_wingtip_coarse_refined_flat.stp",
        projection_path,
        None,
        None,
        recorder,
        timing_enabled=False,
        return_baseline_projections=True,
    )
    recorder.inline = True

    variables = csdl.inline_import(str(hdf5_path), f"iteration_{iteration}")
    if "geometry_coefficients" not in variables:
        raise KeyError(f"iteration_{iteration} has no geometry_coefficients variable")

    # The HDF5 coefficients are the deformed geometry used by the animation script.
    geometry_coefficients = variables["geometry_coefficients"].value
    geometry.unstack_coefficients(geometry_coefficients)
    projected_surface = read_simple_pickle(projection_path)
    surface_points = geometry.evaluate(projected_surface, plot=False).value
    recorder.stop()

    pitch = float(np.asarray(variables["pitch"].value).reshape(-1)[0]) if "pitch" in variables else 0.0
    reference = {}
    for label, aliases in {
        "L": ("L", "lift"),
        "D": ("D_total", "D", "drag"),
        "L/D": ("L_D", "L_over_D"),
    }.items():
        for key in aliases:
            if key in variables:
                values = np.asarray(variables[key].value).reshape(-1)
                if values.size:
                    reference[label] = (float(values[0]), f"{key}[0]")
                    break
    if "L/D" not in reference and "L" in reference and "D" in reference:
        if reference["D"][0] != 0:
            reference["L/D"] = (reference["L"][0] / reference["D"][0], "L[0]/D[0]")
    return np.asarray(surface_points), pitch, reference


def print_force_comparison(forces, reference):
    """Compare the first HDF5 condition, matching the pitch selected above."""
    # OpenFOAM solves half the aircraft; HDF5 forces are for the full aircraft.
    lift = 2.0 * float(np.asarray(forces["lift"]).reshape(-1)[0])
    drag = 2.0 * float(np.asarray(forces["drag"]).reshape(-1)[0])
    computed = {"L": lift, "D": drag, "L/D": lift / drag if drag != 0 else np.nan}
    print(f"\nRANS vs HDF5: {HDF5_PATH}, iteration_{ITERATION}, condition index 0")
    print("RANS lift and drag multiplied by 2 for the full aircraft (half-model CFD).")
    print("Forces in N; delta = RANS - HDF5; delta % = 100 * delta / abs(HDF5).")
    print(f"{'Quantity':<10} {'RANS':>15} {'HDF5':>15} {'Delta':>15} {'Delta %':>12}  HDF5 source")
    for label, value in computed.items():
        if label not in reference:
            print(f"{label:<10} {value:>15.7g} {'unavailable':>15}")
            continue
        expected, source = reference[label]
        delta = value - expected
        percent = f"{100 * delta / abs(expected):.3f}" if expected != 0 else "n/a"
        print(f"{label:<10} {value:>15.7g} {expected:>15.7g} {delta:>15.7g} {percent:>12}  {source}")
    print(flush=True)


def main():
    if not HDF5_PATH:
        raise ValueError("Set HDF5_PATH in run_rans_from_hdf5.py before running.")
    comm = MPI.COMM_WORLD

    # Geometry reconstruction uses CSDL/HDF5 and only needs to happen once.
    # Broadcast both the result and failures so non-root ranks cannot hang.
    surface_points = None
    pitch = None
    load_error = None
    if comm.rank == 0:
        try:
            if RUN_CFD:
                warm_start_dir = Path(MESH) / "processor0" / "9999"
                if not warm_start_dir.is_dir():
                    raise FileNotFoundError(f"Warm-start directory does not exist: {warm_start_dir}")
            surface_points, pitch, reference = load_hdf5_surface(HDF5_PATH, ITERATION, MESH)
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"

    load_error = comm.bcast(load_error, root=0)
    if load_error is not None:
        raise RuntimeError(f"Rank 0 could not reconstruct HDF5 geometry: {load_error}")

    surface_points = comm.bcast(surface_points, root=0)
    pitch = comm.bcast(pitch, root=0)
    if comm.rank == 0:
        print(f"Loaded iteration {ITERATION}: surface shape={surface_points.shape}, pitch={pitch}", flush=True)

    if not RUN_CFD:
        print("Dry run only. Set RUN_CFD = True to invoke DAFoam.", flush=True)
        return

    operating_reference = {
        "U0": U0, "p0": P0, "T0": T0, "nuTilda0": NU_TILDA0,
        "CL_target": 0.5, "aoa0": 0, "pitch0": 0, "A0": 518,
        "rho0": P0 / T0 / 287,
    }
    flow_condition = {
        "v_inf": {"value": np.array([U0, 0.0, 0.0])},
        "p_inf": {"value": np.array([P0])},
        "T_inf": {"value": np.array([T0])},
        "pitch": {"value": np.array([pitch])},
    }
    cfd_runner = build_cfd_func(
        MESH, build_da_options(), operating_reference,
        save_primal_time=None, x_surf_hash_str=X_SURF_HASH_STR,
    )
    forces = cfd_runner["model"](
        name=f"{CASE_NAME}_iteration_{ITERATION}",
        configuration={},
        flow_condition=flow_condition,
        dir=str(OUTPUT_DIR),
        surface_pts=surface_points,
        write=SAVE_OUTPUTS,
    )
    if comm.rank == 0:
        print_force_comparison(forces, reference)

if __name__ == "__main__":
    sys.exit(main())
