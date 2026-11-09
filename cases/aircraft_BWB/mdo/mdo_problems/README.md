# BWB optimization problems

- `bwb_mdo.py`: coupled aircraft/beam/mission optimization.
- `bwb_camber_opt.py`: cruise drag minimization using wing twist and camber.
- `make_visualizations_post_workshop.py`: history plots and geometry/flow videos.

The scripts, supporting helpers, geometry, and starting caches are included here.
**Keep this folder inside this repository.** It reuses the aerodynamic backend in
`../`, mesh/connectivity data and trained weights in `../../model/`, and BWB
utilities/data in `../../../datasets/BWB/`.
GNN weights remain those selected by `../gnn_model_operation.py` and
`../pgnn_model_operation.py`.

## Dependencies

Use your own Python environment and install this repository with `pip install -e .`
from the repository root. Its base dependencies alone are not sufficient.
The versions below were recorded from the development environment on 2026-10-30
(Python **3.11.15**); they are a reference, not a freshly validated installation recipe.

| Package | Version |
| --- | --- |
| jax, jaxlib | 0.5.0 |
| jax-cuda12-plugin, jax-cuda12-pjrt (GPU) | 0.5.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| meshio | 5.3.5 |
| h5py | 3.16.0 |
| matplotlib | 3.10.8 |
| pyvista | 0.47.2 |
| imageio | 2.37.3 |
| imageio-ffmpeg | 0.6.0 |
| pyslsqp | 0.1.4 |

GPU runs require a compatible NVIDIA driver and CUDA-enabled JAX installation.
History plots use h5py, NumPy, and Matplotlib; geometry/video rendering also uses
PyVista, imageio, and its FFmpeg backend.

Install these source packages and their dependencies as well:

| Package | Version | Source commit (HEAD) |
| --- | --- | --- |
| csdl_alpha | 0.0.0a2 | `06eb5164da35b491e2933b2dbda41b3d7dafc089` |
| lsdo_function_spaces | 0.1.4 | `c6cf4febafba1d9dff696969507bbd116e334485` |
| lsdo_geo | 0.1.0 | `eb6e1495b2b9809208198f77b29380f1f654381b` |
| modopt | 0.1.0 | `093a1540dfa15d69cdf46ef7f93e6ce9dbd38037` |
| VortexAD | 0.0.0 | `6ba17018990cb588d09938bc87b6b5c67fae7bb1` |
| aframe | 0.1.0 | `001dd77291ec06e6b3a129fec1629257145dfb08` |
| aeroelastic_coupling_utils | 0.0.1 | `e02a8f826ccf2cfe10a45f8636e95525c3eea74c` |
| idwarp_jax (mesh-warping package) | 0.1.0 | `99c52600da979983ff2da7a8aa6a67b7dcdf6bca` |

**Reproducibility note:** the recorded VortexAD checkout includes uncommitted
solver and packaging changes. modopt and aeroelastic_coupling_utils include
packaging changes; lsdo_function_spaces contains three untracked source files.
The commit IDs alone therefore do not fully capture the development environment.
The scripts require `lsdo_function_spaces.import_file_patched`, modopt's
`InteriorPoint` and `PySLSQP` interfaces, and `aeroelastic_coupling_utils.NodalMap`.
Here `idwarp_jax` refers to the required Python package, not an environment name.

## Run

Activate your Python environment, then run from the repository root inside an
appropriate GPU allocation:

```bash
cd cases/aircraft_BWB/mdo/mdo_problems
python bwb_mdo.py Panel
python bwb_camber_opt.py PG-RANS-GNN
```

Run one optimization at a time. Both scripts require a model argument:
`Panel`, `RANS-GNN`, or `PG-RANS-GNN`. Both retain the original InteriorPoint
settings and optimization formulation. The batch files are TSCC templates:
replace `YOUR_ACCOUNT` and `YOUR_ENV_NAME`, and adapt the partition, QoS, node
exclusions, and module commands for your cluster. Conda must be on `PATH`.
Submit from this folder:

```bash
sbatch batch-opt.sb RANS-GNN
sbatch batch-opt-camber.sb PG-RANS-GNN
```

Each batch file defaults to `Panel` and requests one GPU, 8 CPUs, 64 GB, and one
hour. A full optimization may need longer; arrange an appropriate time limit
before submitting. Logs are `slurm-<jobid>.out`.

The Python entry scripts switch their working directory to this folder, even
when launched elsewhere. HDF5 histories, optimizer output folders, and geometry
caches stay here. Existing optimization results are preserved by incrementing
run names. Historical HDF5 files and generated plots are not included.

## Visualize

Pass the history file produced by your run (replace `RUN.hdf5`):

```bash
python make_visualizations_post_workshop.py RUN.hdf5 --histories-only
python make_visualizations_post_workshop.py RUN.hdf5 --geometry-only
python make_visualizations_post_workshop.py RUN.hdf5 --model PG-RANS-GNN
```

History plots need only CPU plotting packages. Full flow rendering reruns the
selected GNN and needs its GPU environment; match `--model` to the optimization.
Panel results support history and geometry views, not full flow rendering in
this tool. Relative input/output paths are resolved from your launch directory;
plots default to `plots_<history-name>/` beside the history. Use `--help` for options.
