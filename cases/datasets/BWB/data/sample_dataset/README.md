# Release sample and data-generation helpers

The bundled dataset contains evaluation cases 38, 132, and 155. `cases.json`
and `sample_split.json` describe those cases; the arrays arrive in the release
asset ZIPs. There are no training cases.

The adjacent helper scripts are templates for generating a new dataset, not
commands for recreating or overwriting this release sample:

- `build_configs.py` contains geometry/flow ranges for 350 new configurations.
  It currently exits before writing. Adapt its settings and output directory,
  then remove that explicit `exit()` when ready to generate a separate dataset.
- `build_dataset_split.py` reads `rans/parsed_case_summary.json` and optionally
  `euler/parsed_case_summary.json` from its output directory. Those summaries are
  not bundled here. Run it in a new generated dataset directory, or call
  `create_dataset_split(output_dir=...)`. Its default output is `split_cfd.json`.

Neither script is needed to evaluate the shipped sample. Preserve the supplied
`cases.json` and `sample_split.json` when using the released checkpoints.
