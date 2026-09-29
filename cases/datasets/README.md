# Data Generation
This directory contains the scripts used to generate CFD training data via CSDL and DAFoam. The workflow creates geometry configurations, runs a CFD solution for each generated configuration, reads solver convergence information, and finally produces training and test splits.

## Directory Structure
Directory structure for BWB data: 
```text
BWB/
├── batch_euler.sh                         — Runs all configured Euler cases with MPI [optional]
├── batch_panel.sh                         — Runs all configured panel-method cases [optional]
├── batch_rans.sh                          — Runs all configured RANS cases with MPI
├── bwb_cfd.py                             — Defines the CSDL/DAFoam CFD model [called from generate_rans_single]
├── bwb_geometry.py                        — Loads and parameterizes the BWB geometry
├── bwb_helper_functions.py                — Contains geometry and MPI utility functions
├── bwb_panel.py                           — Defines the BWB panel-method model.
├── data_config_builder.py                 — Creates and reads case configurations.
├── generate_euler_single.py               — Generates one Euler CFD case [optional]
├── generate_panel.py                      — Generates the panel-method dataset [optional]
├── generate_rans_single.py                — Generates one RANS CFD case [called from batch_rans]
└── data/
    ├── foam_output_parser.py              — Parses CFD logs and convergence histories.
    └── sample_case/
        ├── build_configs.py               — Creates input samples for CFD generation via LHS.
        ├── build_dataset_split.py         — Creates training, testing splits after CFD data is generated.
        ├── cases.json                     — Stores generated case configurations. [generated from build_configs]
        ├── split_0.json                   — Stores split [can have multiple splits] [generated from build_dataset_split]
        └── rans/
            ├── 0.npz                      — Stores the output for RANS case 0. [generated from bwb_cfd]
            ├── 1.npz                      — Stores the output for RANS case 1. [...]
            ├── 2.npz                      — Stores the output for RANS case 2. [...]
            ├── ...                        — (Additional generated cases)
            ├── parser.py                  — Parses RANS convergence and force histories.
            └── parser_plots/              — Stores RANS convergence plots [generated]
```
For different configurations, the directory structure and code can be copied over exactly except for bwb-specific files: bwb_cfd (just the CSDL model needs to be changed), bwb_geometry, and bwb_helper_functions.

### Tested package dependencies
- DAFoam: v4 + dependencies
- CSDL: branch dev_andrew
- lsdo_function_spaces: https://github.com/ed-low/lsdo_function_spaces.git@vectorize_direction_projection

# Training data preparation (using BWB example):
#### 1. Save OpenFOAM RANS case directory with mesh in BWB directory
 - Ideally ControlDict is defined such that it writes force outputs during solve
#### 2. Edit CSDL model in `bwb_cfd.py` and geometry setup/parameterization in `bwb_geometry.py`
 - `geometry_config_map` output in `bwb_geometry.py` variable names will be used for latin hypercube sampling
#### 3. Edit `generate_rans_single.py` to match OpenFOAM case:
- wall_list and farfield_list much match OpenFOAM boundaries/patches
- update freestream reference values if needed
- update `primalMinResTol` and `primalMinResTolDiff` stopping criterion if needed

# Training data generation (using BWB example):
Different directories in `data` points to a different dataset for the BWB geometry for training.

For a new dataset (assume we will call it `sample_case`):
#### 1. Create new empty directory `sample_case` containing `build_configs.py` and `build_dataset_split.py` in `data`
 - Specify data generation variable ranges and # of samples in `build_configs.py` to perform latin hypercube sampling over
#### 2. Run `build_configs.py` while in `sample_case` to generate LHS samples `cases.json`
#### 3. Run `batch_rans.sh` (or create sbatch script if on HPC) to load and run CFD cases defined in `cases.json`
 - CFD solutions will be saved under directory `rans`
 - Arguments: `bash batch_rans.sh <# of MPI ranks> data/<sample directory> <mesh>`
   - For this case:`bash batch_rans.sh 15 data/sample_case mesh_639k`
 - If warm-start solution exists for time 9999, edit bash sript to start after case 0
#### 4. Run `parser.py` (copy from existing) in `sample_case/rans` to read CFD convergence history
 - Convergence history plots will appear under `sample_case/rans/parser_plots`
#### 5. Run `build_dataset_split.py` to determine usable and unusable data and training/test splits
 - Recommended to manually check each solution and add invalid cases if necessary
    - Can be done via `visualize_data.py`
 - Multiple test/train splits for this dataset can be created by rerunning with different # of cases, residual tol cutoffs, random seed etc.

