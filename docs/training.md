# Training

The training scripts are in `cases/aircraft_BWB/model/`. They train the RANS,
panel-guided, or Euler-guided GNN on a dataset with a common mesh topology.
The released three-case sample is evaluation-only; create a separate dataset
with training and validation cases.

## Packages

From the repository root, `python -m pip install -e .` installs the packages
declared in `pyproject.toml`: JAX, NumPy, SciPy, h5py, Optax, Matplotlib,
PyVista, Ofpp, and idwarp-jax. GPU training also needs a CUDA-enabled JAX
installation appropriate for the machine; the tested CUDA 12 setup used:

```bash
python -m pip install 'jax[cuda12]==0.10.2'
```

Creating new CFD data also needs its separate OpenFOAM, DAFoam, geometry, and
cluster software. Those tools are not installed by AeroDML's `pyproject.toml`.

## Run

From `cases/aircraft_BWB/model/`, choose a dataset directory under
`cases/datasets/BWB/data/` and a split JSON name:

```bash
python train_rans_gnn.py --dataset YOUR_DATASET --split YOUR_SPLIT
```

Use `train_pg_rans_gnn.py` for panel-guided training or
`train_eg_rans_gnn.py` for Euler-guided training. Review the scripts' model,
checkpoint, and output settings before starting a full run. Training can
require substantial GPU memory.
