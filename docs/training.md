# Data generation and training

## Software

[AeroDML installation](package.md#installation) is required for training. Generating new
CFD data also requires a working **OpenFOAM and DAFoam** environment, including
MPI, `mpi4py`, PETSc, `petsc4py`, and IDWarp. The geometry scripts use
`csdl_alpha`, `lsdo_geo`, and `lsdo_function_spaces`; the panel scripts also use
`VortexAD`.
These tools are separate from the core package installation. More information on these packages are noted in the [MDO](mdo.md#tested-environment-snapshot-for-geometry-sweep-visualization) section.

Training uses JAX and Optax. For GPU training, a CUDA-enabled JAX
build compatible with your machine is required.

## Dataset generation

The BWB data and data generation scripts are in `cases/datasets/BWB/`.

### 1. Input sampling

From `cases/datasets/BWB/`, copy the helper scripts into a new dataset directory
(replace `YOUR_DATASET` with your dataset name):

```bash
mkdir -p data/YOUR_DATASET
cp data/sample_dataset/build_configs.py data/sample_dataset/build_dataset_split.py data/YOUR_DATASET/
cd data/YOUR_DATASET
```

Edit `build_configs.py` to set the geometry ranges, flow conditions, and sample
count. Keep `data_dir=""` to write into the current directory, then run:

```bash
python build_configs.py
```

This creates `cases.json` using Latin hypercube sampling which records the generated sample inputs.

### 2. Geometry and mesh preparation

Use a common OpenFOAM reference mesh and deform it for each geometry so every
case keeps the same connectivity. Review the geometry mapping, boundary patches,
and solver settings. The RANS batch script expects a warm-start solution at
`<MESH_DIR>/processor0/9999`.

### 3. CFD dataset generation

From `cases/datasets/BWB/`, generate RANS data for the configured cases:

```bash
bash batch_rans.sh <MPI_RANKS> data/YOUR_DATASET <MESH_DIR>
```

For panel-guided or Euler-guided training, generate the matching low-fidelity
cases with `generate_panel.py` or `generate_euler_single.py` and their batch
scripts.

### 4. Dataset splitting

From `cases/datasets/BWB/data/YOUR_DATASET/`, summarize the RANS solver logs:

```bash
cd rans
python ../../foam_output_parser.py
cd ..
```

For an Euler dataset, repeat those commands with `cd euler` instead of `cd rans`.
Inspect the summaries and saved fields, then edit `build_dataset_split.py` to
set the residual threshold and any manually excluded cases. The default
`test_ratio=0.20` holds out 20% of valid cases and uses the rest for training.
From the dataset directory, run:

```bash
python build_dataset_split.py
```

This writes `split_cfd.json` with `train` and `test` labels; the training driver
uses the held-out cases for validation. Use `--split split_cfd` when training
with this file.

## Training and analysis

The shared driver, `cases/aircraft_BWB/model/run.py`, loads the data, prepares
the graph hierarchy, and handles training and analysis. From that directory:

```bash
python train_rans_gnn.py --dataset YOUR_DATASET --split YOUR_SPLIT
```

Use `train_pg_rans_gnn.py` or `train_eg_rans_gnn.py` for panel-guided or
Euler-guided training. Review the model settings, checkpoint paths, and output
names in the scripts before a full run. Weights and training histories are
saved under `saved_parameters/` and `training_stats/`.

Training fits normalized RANS fields, with extra weight on pressure near the
aircraft surface. After training, use the analysis functions in `run.py` to
compare flow fields and reconstructed lift and drag. The dashboard scripts
plot training histories.

The released three-case sample is for evaluation only. Training needs a
separate dataset with training and validation cases.
