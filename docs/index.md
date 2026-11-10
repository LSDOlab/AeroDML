# AeroDML software

AeroDML predicts three-dimensional aerodynamic flow fields with graph neural
networks. The BWB case includes three model choices: a RANS GNN, a panel-guided
RANS GNN, and an Euler-guided RANS GNN.

## Install

Use Python 3.11 and run from the repository root:

```bash
python -m pip install -e .
```

This installs the core software packages declared in `pyproject.toml`. Running
the supplied models also requires the matching release assets and a CUDA-enabled
JAX environment. Download `aero-dml-assets-mesh.zip`,
`aero-dml-assets-data.zip`, and `SHA256SUMS` from the same
[GitHub release](https://github.com/LSDOlab/AeroDML/releases), place them in
the repository root, and run:

```bash
sha256sum -c SHA256SUMS
unzip aero-dml-assets-mesh.zip
unzip aero-dml-assets-data.zip
python package_assets.py
```

## Evaluate the supplied models

From `cases/aircraft_BWB/model/`, evaluate the three-case sample with:

```bash
python evaluate_sample.py --model rans
```

Choose `panel` or `euler` instead of `rans` for the other models. The sample is
for evaluation only; it cannot be used to train a model. See [Training](training.md)
for training on a new dataset and [MDO](mdo.md) for optimization examples.

## CPU tests

From the repository root:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```
