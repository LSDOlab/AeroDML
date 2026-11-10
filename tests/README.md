# Validation

Use Python 3.10 or newer with the project dependencies installed. From the
repository root, run the CPU tests with:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

The CPU suite tests `run.main()` control flow for RANS, panel, and Euler models,
the sample evaluator's asset selection, and logging indices for small training
and validation splits. Expensive model and mesh operations are mocked. The
checkpoint test checks the release manifest and, when present, the data ZIP; it
does not require extracting the assets.

GPU smoke tests require a CUDA-enabled JAX environment, an authorized GPU
allocation, and **both** release asset ZIPs extracted into the repository. Run
one model at a time from the `tests/` directory:

```bash
cd tests
AERODML_GPU_TESTS=1 AERODML_TEST_MODEL=rans python -m unittest -v test_run.GPUSmokeTests
```

Set `AERODML_TEST_MODEL` to `panel` or `euler` for the other variants.
To run only evaluation or training, append `.test_real_evaluation` or
`.test_two_training_iterations` to `test_run.GPUSmokeTests`.

Evaluation checks finite predictions for all three sample cases. Training
performs two optimizer iterations using two cases for training and one for
validation, checks that parameters changed, and checks that a checkpoint was
saved. Compilation may take minutes and training can exceed available GPU
memory. Tests write temporary checkpoints and logs outside the repository.
They do not establish predictive accuracy, convergence, or CFD/MDO correctness.
