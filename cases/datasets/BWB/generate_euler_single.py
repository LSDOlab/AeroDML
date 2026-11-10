from bwb_cfd import build_cfd_func
from data_config_builder import get_dataset_configurations
import numpy as np

if __name__ == '__main__':
    # region Dafoam options
    wall_list = [
        'wall',
    ]
    farfield_list = [
        'farfield',
    ]
    x_surf_hash_str = 'NEWHASH_EULER' # read and choose hash from mesh_dir
    # x_surf_hash_str = None # read and choose hash from mesh_dir

    # Initial/reference values for DAFoam (best to use base conditions)
    # These correspond to M=0.75 @ 30k feet
    U0        = 227.3805         # used for normalizing CD and CL
    # U0        = 197.04         # M = 0.65
    p0        = 30089.6
    T0        = 228.714
    nuTilda0  = 4.5e-5
    CL_target = 0.5
    aoa0      = 0
    pitch0    = 0
    A0        = 518           # Projected area of entire BWB. Used for normalizing CD and CL
    rho0      = p0 / T0 / 287 # used for normalizing CD and CL

    operating_reference = {
        'U0': U0,
        'p0': p0,
        'T0': T0,
        'nuTilda0': nuTilda0,
        'CL_target': CL_target,
        'aoa0': aoa0,
        'pitch0': pitch0,
        'A0': A0,
        'rho0': rho0,
    }

    da_options = {
        "designSurfaces": wall_list,
        "solverName": "DARhoSimpleCFoam",
        "primalMinResTol": 1.0e-10,
        # "primalMinResTol": 1e-7,
        "primalBC": {
            "U0": {"variable": "U", "patches": farfield_list, "value": [U0, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": farfield_list, "value": [p0]},
            "T0": {"variable": "T", "patches": farfield_list, "value": [T0]},
            # "nuTilda0": {"variable": "nuTilda", "patches": farfield_list, "value": [nuTilda0]},
            "useWallFunction": False,
        },
        "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
        "function": {
            "drag": {
                "type": "force",
                "source": "patchToFace",
                "patches": wall_list,
                "directionMode": "parallelToFlow",
                "patchVelocityInputName": "patch_velocity",
                "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
            },
            "lift": {
                "type": "force",
                "source": "patchToFace",
                "patches": wall_list,
                "directionMode": "normalToFlow",
                "patchVelocityInputName": "patch_velocity",
                "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
            },
        },
        "adjStateOrdering": "cell",
        "adjEqnOption": {"gmresRelTol": 1.0e-4, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
        # transonic preconditioner to speed up the ff convergence
        "transonicPCOption": 2,
        "adjPCLag": 5,
        # "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
        # # transonic preconditioner to speed up the adjoint convergence
        # "transonicPCOption": 1,
        "normalizeStates": {
            "U": U0,
            "p": p0,
            "T": T0,
            # "nuTilda": nuTilda0 * 10.0,
            "phi": 1.0,
        },
        "inputInfo": {
            "aero_vol_coords": {
                "type": "volCoord", 
                "components": ["solver", "function"],
            },
            "patch_velocity": {
                "type": "patchVelocity",
                "patches": farfield_list,
                "flowAxis": "x",
                "normalAxis": "z",
                "components": ["solver", "function"],
            },
            "pressure": {
                "type": "patchVar",
                "varName": "p",
                "varType": "scalar",
                "patches": farfield_list,
                "components": ["solver", "function"],
            },
            "temperature": {
                "type": "patchVar",
                "varName": "T",
                "varType": "scalar",
                "patches": farfield_list,
                "components": ["solver", "function"],
            },
        },
        "checkMeshThreshold": { 
                "maxAspectRatio": 10000.0,
                "maxNonOrth": 80.0,
                "maxSkewness": 10.0,
                "maxIncorrectlyOrientedFaces": 0,
        }
    }

    # gather geometry dataset
    # gather one geometry dataset case
    import sys
    from pathlib import Path

    data_dir = sys.argv[1]          # e.g. data/sampling_200
    case_id  = sys.argv[2]          # exact case key (including numeric keys)


    dataset_configurations = get_dataset_configurations(data_dir)

    matches = [d for d in dataset_configurations if str(d['key']) == case_id]
    if len(matches) != 1:
        raise ValueError(f"Could not find exactly one case with key={case_id!r}")
    dataset_input = matches[0]

    mesh = sys.argv[3] if len(sys.argv) > 3 else 'mesh_112k'
    cfd_runner_dict = build_cfd_func(
        mesh,
        da_options,
        operating_reference,
        save_primal_time = 9999 if case_id == "0" else None,
        x_surf_hash_str = x_surf_hash_str,
    )

    surf_pts = None
    surf_pts_path = Path(data_dir) / "surf_pts_euler" / f"{dataset_input['key']}.npy"
    if surf_pts_path.is_file():
        print(f"Loading surface points from: {surf_pts_path}", flush=True)
        surf_pts = np.load(surf_pts_path)
    else:
        print(f"No surface points found at: {surf_pts_path}", flush=True)

    cfd_writer = cfd_runner_dict["model"]

    ##### Desktop on docker #####
    # HOST_MOUNT_ROOT = Path("/home/dafoamuser/mount")
    # output_dir = HOST_MOUNT_ROOT / "_cfd_outputs" / data_dir / "euler"

    ##### HPC native #####
    output_dir = f"{data_dir}/euler"

    print(f"Running single EULER case: data_dir={data_dir}, key={dataset_input['key']}", flush=True)

    # run
    cfd_writer(
        name = dataset_input['key'],
        configuration = dataset_input['geometry_configuration'],
        flow_condition = dataset_input['flow_conditions'],
        dir = output_dir,
        surface_pts = surf_pts,
    )

    # sim = cfd_runner_dict["sim"]
    # sim.run()