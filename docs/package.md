# Package and model

The reusable Python package lives in `src/aerodml/`. It handles mesh loading,
graph construction, the GNN, training, and aerodynamic postprocessing.
The BWB application driver, `cases/aircraft_BWB/model/run.py`, combines these
tools with the aircraft's meshes, datasets, and model settings.

## From mesh to graph

Graph construction is defined in `src/aerodml/load_mesh.py`.

The model works on a graph built from an unstructured CFD volume mesh. Each
cell center becomes a node, and cells that share a face are connected by an
edge. Extra boundary nodes sit at wall, farfield, and symmetry face centers
and connect to their neighboring volume cells.

Two coarser graphs are built by clustering nearby connected volume nodes while
preserving boundary nodes. The mapping between fine and coarse nodes is saved
for pooling and unpooling. This gives the network three levels for exchanging
information over both short and long distances. Connectivity is reused across
cases; node positions and geometric features change with the deformed mesh.

Node features describe the input flow state, node type, approximate wall
distance, and direction to the nearest wall. Wall nodes also carry face area
and surface normal. Edge features describe direction, length, and boundary type.

## The GNN

The architecture is defined in `src/aerodml/model.py`, with neural-network
layers and parameter handling in `src/aerodml/networks.py`.

The network follows a U-shaped path through the three graph levels. Separate
encoders turn wall and non-wall features into learned node representations.
Message passing exchanges information between neighbors, then pooling moves
that information to the coarser graphs. Learned updates bring it back to the
fine graph, using saved fine-level information along the way.

Each message uses the two connected node representations and their edge
features. Messages travel in both directions and are averaged at each node.
The node update also uses a global average of the volume-node representations,
giving it information about the wider flow. Pooling averages nodes within each
cluster; unpooling combines the coarse representation with the saved fine-level
representation through a learned update.

A decoder predicts pressure, three velocity components, and temperature at
volume nodes. Aerodynamic postprocessing in `src/aerodml/postprocessing.py`
computes lift and drag from the predicted fields.

The BWB example has three variants:

- **RANS-GNN:** starts from freestream inputs and predicts the RANS field.
- **PG-RANS-GNN:** uses a panel-method solution mapped onto the graph and
  learns a correction toward RANS.
- **EG-RANS-GNN:** uses an interpolated Euler solution as its starting point
  and learns the RANS correction.

## Model API and hyperparameters

`get_model()` in `src/aerodml/model.py` returns a prediction function and a
parameter manager. This example assumes the normalization constants, flow
conditions, graph coordinates, and topology have already been prepared, as in
`cases/aircraft_BWB/model/run.py`:

```python
from aerodml.model import get_model

predict, parameter_info = get_model(
    constants,
    multifidelity_model=False,
    add_cell_sdf=True,
    root="cases/aircraft_BWB/model/saved_parameters",
    model_config={"cell_latent_dim": 45, "mp_steps": (1, 3, 5, 3, 1)},
)
params = parameter_info.load("checkpoint", custom_path="RANS_GNN_trained_paper.pickle")
velocity, temperature, pressure, *_ = predict(
    params, flow_condition, graph_points, geometry
)
```

`root` is the weights directory, relative to your working directory;
`custom_path` selects a checkpoint file within it.
`geometry` holds the fixed graph topology and cluster mappings; `graph_points`
holds the coordinates for the current case. For panel- or Euler-guided
prediction, set `multifidelity_model=True` and supply the mapped solver fields
in `graph_points["low_fidelity_inputs"]`. Outputs are normalized; convert them
back to physical units using the checkpoint's normalization constants before
analysis. Model settings and normalization must match the loaded checkpoint.

The main `model_config` options and their defaults are:

| Option | Default | Controls |
| --- | --- | --- |
| `cell_latent_dim` | `45` | Size of each learned node representation |
| `encoder_width`, `broadcast_width`, `decoder_width` | `48` each | Hidden-layer widths for encoding, unpooling, and decoding |
| `processor_width` | `85` | Hidden-layer width for messages and node updates |
| `mlp_hidden_layers` | `2` | Hidden layers in each network |
| `mp_steps` | `(1, 3, 5, 3, 1)` | Message-passing steps at fine, middle, coarse, middle, and fine levels |
| `share_mp_weights` | `False` | Reuse message and update weights across stages |
| `aggregate_method` | `"grouped"` | Aggregation implementation: `grouped`, `sequential`, or `scatter` |

`add_cell_sdf=True` includes approximate wall distance and direction features.
To train a different architecture, pass the chosen settings to `get_model()`
in the BWB driver and use the same settings when evaluating its checkpoints.

## Installation

Use Python 3.11 and run from the repository root:

```bash
python -m pip install -e .
```

This installs the core dependencies listed in `pyproject.toml`. GPU runs also
need a compatible CUDA-enabled JAX installation.

Download `aero-dml-assets-mesh.zip`, `aero-dml-assets-data.zip`, and `SHA256SUMS`
from the matching [GitHub release](https://github.com/LSDOlab/AeroDML/releases).
Place them in a clean checkout at the repository root, then run:

```bash
sha256sum -c SHA256SUMS
unzip aero-dml-assets-mesh.zip
unzip aero-dml-assets-data.zip
python package_assets.py
```

To evaluate the supplied sample, run from `cases/aircraft_BWB/model/`:

```bash
python evaluate_sample.py
```

This evaluates the RANS, panel-guided, and Euler-guided models in sequence.
