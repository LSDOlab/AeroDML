# Small run.py test suite

These tests have been written but **not run**. Use the project's ML environment
with the release dependencies installed. No pytest dependency is needed.
Commands below run from this `tests/` directory.

## CPU unit tests

```bash
JAX_PLATFORMS=cpu python -m unittest -v test_run.DriverTests
```

Tests actual `run.main()` control flow for all three model variants: checkpoint
loading, initialization, training dispatch, saving updated parameters, low-fidelity
loader selection, and keeping all sample cases. Heavy computations are mocked;
these tests do not establish numerical correctness. The sample evaluator's model
selection and asset paths are checked too. No training outputs are written.

## GPU smoke tests (explicit opt-in)

Run only inside an authorized GPU allocation, with GPU JAX and `optax` installed.
Do not set `JAX_PLATFORMS=cpu` for these commands. Start with RANS:

```bash
AERODML_GPU_TESTS=1 AERODML_TEST_MODEL=rans python -m unittest -v test_run.GPUSmokeTests
```

Repeat with `AERODML_TEST_MODEL=panel` and `AERODML_TEST_MODEL=euler`, one process
at a time. To run only training or evaluation, append `.test_two_training_iterations`
or `.test_real_evaluation` to `test_run.GPUSmokeTests` in the command.

These load the real meshes/data/checkpoint and call `run.main()`. Evaluation
checks finite predictions for every sample case. Training performs two actual
optimizer iterations, checks finite and changed parameters, and verifies that a
checkpoint was saved. Compilation can take minutes despite only two iterations.
These are smoke tests, not accuracy, convergence, plotting, or CFD/MDO tests.

The release sample remains evaluation-only on disk. For the training smoke test,
two cases are temporarily used for training and one is held out. The two training
cases are repeated in memory to accommodate the existing training logger's fixed
indices 2 and 5. This does not establish support for arbitrary small training sets.
Checkpoints and logs go into an automatically deleted temporary directory; shipped
checkpoints and JSON files are unchanged. Set `TMPDIR` to scratch if desired.
Tests constrain package imports to this release's source directory, preventing
fallback to the private repository's editable package.



```
# Initialize conda and any GPU modules required by your system first.
conda activate YOUR_ENV_NAME
# From the repository root:
cd tests

# CPU unit tests—all three passed
JAX_PLATFORMS=cpu python -m unittest -v test_run.DriverTests

# GPU evaluation only—RANS passed
AERODML_GPU_TESTS=1 AERODML_TEST_MODEL=rans \
python -m unittest -v test_run.GPUSmokeTests.test_real_evaluation

# Two training iterations—RANS hit GPU out-of-memory
AERODML_GPU_TESTS=1 AERODML_TEST_MODEL=rans \
python -m unittest -v test_run.GPUSmokeTests.test_two_training_iterations
```