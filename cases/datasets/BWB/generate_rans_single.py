# import os
# os.environ.setdefault("JAX_PLATFORMS", "cpu")
from bwb_cfd import build_cfd_func
from data_config_builder import get_dataset_configurations
import pickle
from pathlib import Path
import numpy as np
import time

from mpi4py import MPI
comm           = MPI.COMM_WORLD

if __name__ == '__main__':
    # ================= 639k mesh: region Dafoam options =================
    wall_list = [
        'wall_body_lower', 'wall_body_upper', 
        'wall_wing_lower', 'wall_transition_lower', 
        'wall_wing_upper', 'wall_transition_upper','wall_wing_cap',
    ]
    farfield_list = ['inout',]
    # x_surf_hash_str = 'a63d3d5d35e5809c' # read and choose hash from mesh_dir
    # x_surf_hash_str = None
    x_surf_hash_str = 'NEWHASH' # read and choose hash from mesh_dir
    # x_surf_hash_str = None # read and choose hash from mesh_dir

    # ================= 3M mesh: region Dafoam options =================
    # wall_list = ['wall',]
    # farfield_list = ['farfield',]

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

        # Stopping criteria
        # "primalMinResTol": 1.0e-8,
        "primalMinResTol": 5e-9,
        "primalMinResTolDiff": 1e2,
        # "primalFuncStdTol": {
        #     "tol": 1e-4,
        #     "funcName": "drag",
        #     "nSteps": 100,
        # },

        # BCs etc
        "primalBC": {
            "U0": {"variable": "U", "patches": farfield_list, "value": [U0, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": farfield_list, "value": [p0]},
            "T0": {"variable": "T", "patches": farfield_list, "value": [T0]},
            "nuTilda0": {"variable": "nuTilda", "patches": farfield_list, "value": [nuTilda0]},
            "useWallFunction": True,
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
            "nuTilda": nuTilda0 * 10.0,
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

    # gather one geometry dataset case
    import sys
    from pathlib import Path

    data_dir = sys.argv[1]          # e.g. data/sampling_200
    case_id  = sys.argv[2]          # exact case key (including numeric keys)
    mesh = sys.argv[3]  

    cfd_runner_dict = build_cfd_func(
        mesh,
        da_options,
        operating_reference,
        save_primal_time = 9999 if str(case_id) == "0" else None,
        x_surf_hash_str = x_surf_hash_str,
    )

    dataset_configurations = get_dataset_configurations(data_dir)
    
    # Run the CFD
    matches = [d for d in dataset_configurations if str(d['key']) == case_id]
    if len(matches) != 1:
        raise ValueError(f"Could not find exactly one case with key={case_id!r}")
    dataset_input = matches[0]

    surf_pts = None
    # look for the resolved case key in data_dir/surf_pts
    # if it exists, load it as surf_pts
    surf_pts_path = Path(data_dir) / "surf_pts" / f"{dataset_input['key']}.npy"

    if surf_pts_path.is_file():
        print(f"Loading surface points from: {surf_pts_path}", flush=True)
        surf_pts = np.load(surf_pts_path)
    else:
        print(f"No surface points found at: {surf_pts_path}", flush=True)

    cfd_writer = cfd_runner_dict["model"]

    ##### Desktop on docker #####
    # HOST_MOUNT_ROOT = Path("/home/dafoamuser/mount")
    # output_dir = HOST_MOUNT_ROOT / "_cfd_outputs" / data_dir / "rans"

    ##### HPC native #####
    output_dir = f"{data_dir}/rans"

    print(f"Running single RANS case: data_dir={data_dir}, key={dataset_input['key']}", flush=True)

    cfd_writer(
        name = dataset_input['key'],
        configuration = dataset_input['geometry_configuration'],
        flow_condition = dataset_input['flow_conditions'],
        dir = output_dir,
        surface_pts = surf_pts,
    )

    # sim = cfd_runner_dict["sim"]
    # sim.run()