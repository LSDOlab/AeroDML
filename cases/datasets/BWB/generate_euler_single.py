from bwb_cfd import build_cfd_func
from data_config_builder import get_dataset_configurations

if __name__ == '__main__':
    # region Dafoam options
    wall_list = [
        'wall',
    ]
    farfield_list = [
        'farfield',
    ]

    # Initial/reference values for DAFoam (best to use base conditions)
    # These correspond to M=0.75 @ 30k feet
    U0        = 227.3805         # used for normalizing CD and CL
    # U0        = 197.04         # M = 0.65
    p0        = 30089.6
    T0        = 228.714
    nuTilda0  = 4.5e-5
    CL_target = 0.5
    aoa0      = 0
    A0        = 518           # Projected area of entire BWB. Used for normalizing CD and CL
    rho0      = p0 / T0 / 287 # used for normalizing CD and CL

    operating_reference = {
        'U0': U0,
        'p0': p0,
        'T0': T0,
        'nuTilda0': nuTilda0,
        'CL_target': CL_target,
        'aoa0': aoa0,
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

    # build model
    cfd_runner_dict = build_cfd_func(
        'mesh_112k',
        da_options,
        operating_reference,
    )
    # sim = cfd_runner_dict["sim"]
    # sim.run()
    # exit()

    # gather geometry dataset
    # gather one geometry dataset case
    import sys
    from pathlib import Path

    data_dir = sys.argv[1]          # e.g. data/sampling_200
    case_id  = sys.argv[2]          # either an integer index or an exact key


    dataset_configurations = get_dataset_configurations(data_dir)

    if case_id.isdigit():
        dataset_input = dataset_configurations[int(case_id)]
    else:
        matches = [d for d in dataset_configurations if str(d['key']) == case_id]
        if len(matches) != 1:
            raise ValueError(f"Could not find exactly one case with key={case_id!r}")
        dataset_input = matches[0]

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
        dir = output_dir
    )

    # sim = cfd_runner_dict["sim"]
    # sim.run()