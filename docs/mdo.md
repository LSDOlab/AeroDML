# Multidisciplinary design optimization (MDO)

The BWB optimization examples are in `cases/aircraft_BWB/mdo/mdo_problems/`.
`bwb_mdo.py` couples aircraft, beam, and mission calculations;
`bwb_camber_opt.py` optimizes wing twist and camber for cruise drag.

## Packages

Install AeroDML from the repository root with `python -m pip install -e .`.
The MDO scripts additionally import these packages, which are **not** included
in AeroDML's `pyproject.toml`:

- `csdl_alpha`, `lsdo_function_spaces`, and `lsdo_geo` for geometry and models
- `modopt` and `pyslsqp` for optimization
- `VortexAD`, `aframe`, and `aeroelastic_coupling_utils` for panel and structural models
- `meshio` for mesh handling

GPU runs also need a compatible CUDA-enabled JAX installation. The MDO source
packages were recorded from a development environment, but that environment
included local source changes; its recorded commit IDs are not a verified
installation recipe.

## Run

Use a suitable GPU allocation and run from `cases/aircraft_BWB/mdo/mdo_problems/`:

```bash
python bwb_mdo.py Panel
python bwb_camber_opt.py PG-RANS-GNN
```

The accepted model arguments are `Panel`, `RANS-GNN`, and `PG-RANS-GNN`.
The optimization scripts use the matching model weights and geometry assets
from the repository and release ZIPs. Run one optimization at a time.
