from bwb_panel import build_panel_func
from data_config_builder import get_dataset_configurations
if __name__ == '__main__':

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

    cfd_runner_dict = build_panel_func(
        operating_reference,
    )
    cfd_writer = cfd_runner_dict["model"]

    # sim = cfd_runner_dict["sim"]
    # sim.run()

    # gather one geometry dataset case
    import sys
    from pathlib import Path

    data_dir = sys.argv[1]          # e.g. data/sampling_200

    dataset_configurations = get_dataset_configurations(data_dir)

    # output_dir = "data" / data_dir / "panel"
    output_dir = Path(f"{data_dir}") / "panel"
    print(f"Writing panel datasets to {output_dir}")
    
    print(type(dataset_configurations))
    for dataset_input in dataset_configurations:
        cfd_writer(
            name = dataset_input['key'],
            configuration = dataset_input['geometry_configuration'],
            flow_condition = dataset_input['flow_conditions'],
            sample_points = None,
            dir = output_dir
        )