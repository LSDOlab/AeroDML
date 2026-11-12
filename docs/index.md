# AeroDML

AeroDML (Aerodynamic Deep Multifidelity Learning) uses graph neural networks
to predict three-dimensional aerodynamic flow fields for aircraft design.
High-fidelity CFD can be a major cost in multidisciplinary design optimization,
where many aerodynamic evaluations are needed. Predicting the full flow field
offers a faster way to estimate forces while retaining the spatial information
needed for tasks such as transferring aerodynamic loads to a structural model.

The multifidelity approach combines lower-cost aerodynamic solutions with
high-fidelity RANS training data. A panel-method or Euler solution provides an
approximate flow field, and the GNN learns a correction toward RANS. In the
paper's BWB study, these inputs improved prediction accuracy and reduced the
amount of high-fidelity training data needed. A purely data-driven GNN is also
included for comparison.

![Case 38 pressure comparison: RANS reference, GNN predictions, and prediction errors](assets/pressure_comparison_case_38.png)

*Surface and cross-section pressure predictions and errors for the single-fidelity, panel-guided, and Euler-guided GNN models compared with RANS for a sample test case.*

![BWB geometry sweep and aerodynamic load comparison across Panel Method, RANS-GNN, and PG-RANS GNN](assets/bwb_sweep_comparison.gif)

*Geometry sweeps showing predicted surface pressure fields, cross sections, and aerodynamic coefficients across Panel Method (VortexAD), RANS-GNN, and PG-RANS GNN models.*

This application was developed for high-performance computing (HPC) use on a
blended-wing-body (BWB) configuration. It includes parallel CFD data generation,
GPU training and evaluation, and gradient-based optimization examples. It can
be modified for other configurations by adapting the geometry parameterization,
reference meshes, data-generation setup, and training data.

The repository includes:

- **[Package and model](package.md):** the `aerodml` Python package for building
  mesh-based graphs, defining the GNN, and evaluating flow predictions.
- **[Data generation and training](training.md):** scripts for generating RANS
  CFD data, adding optional panel or Euler solutions, and training the models.
- **[Optimization](mdo.md):** aerodynamic and multidisciplinary design
  optimization examples using the trained models.

## Main workflow

1. **Generate data.** Sample geometry and flow conditions, deform a reference
   mesh, and run CFD. Add low-fidelity solutions for multifidelity training.
2. **Train.** Build the graph and train a GNN against the RANS flow fields.
3. **Analyze.** Compare predicted fields, lift, and drag with reference results.
4. **Optimize.** Use the trained model in a differentiable design workflow.
