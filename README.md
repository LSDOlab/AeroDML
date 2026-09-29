The development of this software was supported (, in part,) by the Air Force Research Laboratory through the Collaborative Center for the Design and Research Of InterDisciplinary Systems (CC DROIDS). Distribution Statement A: Approved for public release; distribution is unlimited. PA# AFRL-2026-3662. This authorization applies to Git commit XXXXX.

For data generation guidelines, follow README located in /cases/datasets
For ML training/analysis guidelines, follow README located in /cases/aircraft_BWB/model

## Download release assets

Large data and model assets are distributed separately from the Git repository.

1. Clone or check out the desired repository release.
2. Download the matching `aero-delta-net-assets-vX.Y.Z.zip` file from that GitHub release. Do not download GitHub's automatically generated "Source code" ZIP for the assets.
3. Extract the asset archive from the repository root:

```bash
cd aero-delta-net
unzip aero-delta-net-assets-vX.Y.Z.zip
```

The archive preserves repository-relative paths, so the files will be placed in the required directories automatically. The asset version should match the repository release or tag.
