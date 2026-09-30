The development of this software was supported (, in part,) by the Air Force Research Laboratory through the Collaborative Center for the Design and Research Of InterDisciplinary Systems (CC DROIDS). Distribution Statement A: Approved for public release; distribution is unlimited. PA# AFRL-2026-3662. The authorized Git commit is identified in the corresponding GitHub release.

<!-- ## Reference -->
## **AeroDML** (Aerodynamic Deep Multifidelity Learning)

This repository contains the RANS-GNN implementation associated with [Multifidelity Surrogate Modeling for 3D Aerodynamic Flow Field Prediction Using Graph Neural Networks](https://www.researchgate.net/publication/406026389_Multifidelity_Surrogate_Modeling_for_3D_Aerodynamic_Flow_Field_Prediction_Using_Graph_Neural_Networks).

For data generation guidelines, follow README located in /cases/datasets
For ML training/analysis guidelines, follow README located in /cases/aircraft_BWB/model

## Download release assets

Large data and model assets are distributed separately from the Git repository.

1. Clone or check out the desired repository release.
2. Download the matching `aero-dml-assets-vX.Y.Z.zip` file from that GitHub release. Do not download GitHub's automatically generated "Source code" ZIP for the assets.
3. Extract the asset archive from the repository root:

```bash
cd AeroDML
unzip aero-dml-assets-vX.Y.Z.zip
```

The archive preserves repository-relative paths, so the files will be placed in the required directories automatically. The asset version should match the repository release or tag.
