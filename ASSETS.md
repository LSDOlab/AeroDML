# Release assets

Assets are retained as ordinary files for local testing and packaged into two ZIPs
under `release_assets/`. Both archives are required.
`asset_paths.txt` is the exact list of files to package separately from source code;
`.gitignore` excludes those files from Git. `asset_manifest.json` records sizes and
SHA-256 checksums. The JSON configurations, this documentation, manifests, and
packaging/evaluation scripts belong in the code repository.

## Included files

- `cases/datasets/BWB/data/sample_dataset/`: cases **38, 132, 155**, selected from
  the valid test cases in `sample_tcp1_b/split_2`. Original case IDs are preserved.
  Each has RANS, Euler, panel, graph vertices, RANS surface points, and Euler
  surface points. `cases.json` and `sample_split.json` contain only those three cases;
  all three are test cases and none are training cases.
- `cases/aircraft_BWB/model/saved_parameters/`: exactly three checkpoints:
  `RANS_GNN_trained_paper.pickle`, `PG_RANS_GNN_trained_paper.pickle`, and
  `EG_RANS_GNN_trained_paper.pickle`.
- Model `mesh_639k/` and `mesh_112k/`, including existing graph/deformation caches,
  plus `CONNECTIVITY_NEW.pkl`.
- CFD `mesh_639k/` and `mesh_112k/`: initial conditions, constants/mesh, solver
  settings, and existing surface projection/index arrays. Solver result times,
  `processor*` directories, and postprocessing histories are excluded.
- BWB CAD/refit files, panel geometry, and existing `storage/` and `stored_files/`
  geometry caches, following the previous release's asset categories.

Total asset size: **2,631,647,503 bytes (2.632 GB / 2.451 GiB)**.
The sample arrays account for 345,436,326 bytes. All source symlinks were
resolved into ordinary destination files. Other production datasets, the old
sample test ZIP, plots/videos, and unrelated checkpoints were not copied.

## Evaluate the sample

Use the project's GPU environment and an authorized GPU allocation. From the
repository root:

```bash
python cases/aircraft_BWB/model/evaluate_sample.py --model rans
python cases/aircraft_BWB/model/evaluate_sample.py --model panel
python cases/aircraft_BWB/model/evaluate_sample.py --model euler
```

Each command loads one checkpoint and evaluates all three test cases without
training or saving parameters. It uses the existing force-analysis/plotting
pipeline. The dedicated entry point invokes `run.main` and disables validation-case
exclusion. `run.py` has no direct-execution block. The user reports running the
GPU tests; their results have not been independently reviewed during packaging.

## Verify, package, and extract

```bash
python package_assets.py
# Creates both ZIPs, checks sizes, and tests clean extraction with unzip:
python package_assets.py --output-dir release_assets
```

Packaging refuses to overwrite existing archives. `aero-dml-assets-mesh.zip`
contains the model's `mesh_639k/` directory. `aero-dml-assets-data.zip` contains
all other listed assets: sample arrays, checkpoints, histories, Euler mesh,
CFD meshes, and geometry. Each archive must be below **2,000,000,000 bytes**;
the packaging script enforces this and emits `SHA256SUMS` after verification.

For installation, download both archives and `SHA256SUMS` into a clean matching
source checkout, then run from its root:

```bash
sha256sum -c SHA256SUMS
unzip aero-dml-assets-mesh.zip
unzip aero-dml-assets-data.zip
python package_assets.py
```

The two archives have disjoint file lists and can be extracted in either order.
Source files, including sample configuration/split JSON and the manifests, are
provided by the code release. The asset ZIPs exclude Git metadata and generated
plots. Packaging tests extraction and checksums; it does not rerun GPU inference.

This inventory is not completion of the final release cleanup or validation
of CFD/MDO workflows; existing code and serialized caches still need the final
release review described in the release instructions.

## Paper training histories

`model/training_stats/training_aux_info_<MODEL_NAME>_trained_paper.jsonl`
contains the full training histories for the three released checkpoints, selected
from the final three-model comparison in the original dashboard overlay. These
histories describe the original training/validation data, not the three-case
release sample. The files are included in the asset manifest and checksums.

Run from the repository root:

```bash
python cases/aircraft_BWB/model/dashboard.py --model rans
python cases/aircraft_BWB/model/dashboard.py --model panel
python cases/aircraft_BWB/model/dashboard.py --model euler
python cases/aircraft_BWB/model/dashboard_overlay.py
```

Plots are written under `cases/aircraft_BWB/model/plots/`. Set `MPLBACKEND=Agg`
for headless plotting. The single-model dashboard keeps independent runs separate;
the overlay compares all three. No training data or GPU is required for dashboards.
