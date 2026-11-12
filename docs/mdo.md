# Optimization

The BWB examples use aerodynamic predictions in gradient-based design
optimization. `bwb_camber_opt.py` adjusts wing twist and camber to reduce cruise
drag. `bwb_mdo.py` couples aerodynamics, beam structures, and mission
calculations in a multidisciplinary design optimization (MDO) problem.

![BWB geometry sweep and aerodynamic load comparison across Panel Method, RANS-GNN, and PG-RANS GNN](assets/bwb_sweep_comparison.gif)

*Model comparison during geometry variations.*

## Software

Install [AeroDML and the matching release assets](package.md#installation),
then set up the additional packages used by the optimization examples:

- `csdl_alpha`, `lsdo_geo`, and `lsdo_function_spaces` for differentiable models
  and geometry.
- `modopt` and `pyslsqp` for optimization.
- `VortexAD`, `aframe`, and `aeroelastic_coupling_utils` for panel aerodynamics,
  structures, and load transfer.
- `meshio` for mesh handling, plus CUDA-enabled JAX for GPU runs.

These dependencies are separate from the core package. The README in
`cases/aircraft_BWB/mdo/mdo_problems/` records the development environment and
required source interfaces; some packages used local changes, so the recorded
versions alone do not reproduce that environment.

## Workflow

1. **Choose the problem and aerodynamic model.** Both examples accept `Panel`,
   `RANS-GNN`, or `PG-RANS-GNN`. The GNN options load their matching trained weights.
2. **Review the design setup.** Set design variables, bounds, objectives, and
   constraints in the chosen script. Keep the examples in the repository so
   their relative geometry, mesh, and model paths resolve correctly.
3. **Run the optimization.** Each iteration updates the geometry, evaluates
   aerodynamics and any coupled models, and computes derivatives for the
   optimizer. The GNN predicts flow fields from which forces are reconstructed.
4. **Inspect the result.** Review the optimization history, geometry, and flow
   predictions. Re-evaluate the final design with RANS CFD to assess accuracy.

From `cases/aircraft_BWB/mdo/mdo_problems/`, in a suitable GPU allocation:

```bash
python bwb_mdo.py Panel
python bwb_camber_opt.py PG-RANS-GNN
```

Run one optimization at a time. To plot a completed run's history:

```bash
python make_visualizations_post_workshop.py RUN.hdf5 --histories-only
```

Euler-guided prediction is available for analysis, but is not included in these
optimization examples because the current workflow does not provide derivatives
through the Euler solver.
