# BWB dataset generation

The BWB example under `BWB/` generates geometry configurations, RANS solutions,
optional Euler and panel solutions, convergence summaries, and dataset splits.
The released `BWB/data/sample_dataset/` contains three **evaluation-only**
cases. Its arrays are distributed in the release asset ZIPs.

## Layout and entry points

| Path | Purpose |
| --- | --- |
| `BWB/data/sample_dataset/build_configs.py` | Example geometry and flow ranges; exits before writing so it cannot overwrite the release sample. |
| `BWB/data_config_builder.py` | Creates and reads `cases.json`. |
| `BWB/batch_rans.sh`, `BWB/generate_rans_single.py` | Runs configured RANS cases. |
| `BWB/batch_euler.sh`, `BWB/generate_euler_single.py` | Runs optional Euler cases. |
| `BWB/batch_panel.sh`, `BWB/generate_panel.py` | Runs optional panel cases. |
| `BWB/data/foam_output_parser.py` | Parses solver logs into `parsed_case_summary.json`. |
| `BWB/data/sample_dataset/build_dataset_split.py` | Example split builder; write to a new dataset directory. |

## Generate a new dataset

1. Copy the two helper scripts from `sample_dataset/` into a **new** directory
   below `BWB/data/`. Set ranges, output path, and sample count in
   `build_configs.py`, then remove its protective `exit()`. Run it from the new
   directory to create `cases.json`. Case 0 is the reference configuration.
2. Prepare a common OpenFOAM reference mesh and a warm-start time at
   `<mesh>/processor0/9999`. Update BWB geometry, CFD settings, patch names,
   solver tolerances, and cluster environment for the new geometry.
3. From `cases/datasets/BWB/`, run
   `bash batch_rans.sh <MPI_RANKS> data/<NEW_DATASET> <MESH_DIR>`.
   The script accepts optional start and end case indices after the mesh path.
   Generate matching Euler or panel cases if the chosen model uses them.
4. From each populated `rans/` or `euler/` directory, run
   `python ../../foam_output_parser.py` to create
   `parsed_case_summary.json`. Inspect convergence and fields; the parser's
   NaN detection uses log text and does not inspect NPZ arrays.
5. Adapt and run `build_dataset_split.py` for the new directory. Review
   invalid cases and ensure the resulting split has training and validation
   cases before training.

All cases used by one model must share mesh connectivity. Create geometry
variants by deforming a common reference mesh. DAFoam, CSDL, OpenFOAM, and
geometry package dependencies are separate from the core `aerodml` install.
See `BWB/readme.md` for the original environment notes and
`../aircraft_BWB/model/README.md` for model entry points.
