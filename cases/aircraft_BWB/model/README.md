## Directory Structure

```text
aircraft_BWB/
├── mdo/
│   └── mdo_opt.py                      — Placeholder.
│
└── model/
    ├── dashboard.py                    — Displays training plots [can be ran during training]
    ├── dashboard_overlay.py            — Compares multiple training histories [can be ran during training]
    ├── run.py                          — Builds, trains and analyzes the model
    ├── train_rans_gnn.py               — Launches RANS-GNN training
    ├── train_eg_rans_gnn.py            — Launches Euler-guided RANS-GNN training [optional] 
    ├── train_pg_rans_gnn.py            — Launches panel-guided RANS-GNN training [optional]
    ├── visualize_data.py               — Visualizes generated data for a specific dataset and split
    │
    └── saved_parameters/               — Stores generated model checkpoints.
```

## Training:
Make sure OpenFOAM case to build graph is located in this directory.

Run `run.py` or `train_rans_gnn.py` while specifying:
- dataset directory (which directory in cases/datasets/BWB/data)
- split (which split in dataset directory)
- name (training stats and model parameters will be saved using this name)
- whether to load from existing parameters (for warm start)