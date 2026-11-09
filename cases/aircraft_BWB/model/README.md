# BWB model examples

`run.py` provides the shared model setup, training, and analysis functions.
Use `evaluate_sample.py --model rans` (or `panel`, `euler`) to evaluate the bundled
three-case `sample_dataset/sample_split` with the matching paper checkpoint.
GPU evaluation requires the project dependencies and a GPU environment.

Use `dashboard.py --model rans` (or `panel`, `euler`) for one paper training history,
or `dashboard_overlay.py` to compare all three. These CPU plotting scripts read
bundled histories independently of the evaluation sample. See the root ASSETS.md.

Training requires a separate dataset containing training cases. For example:

```bash
python train_rans_gnn.py --dataset YOUR_DATASET --split YOUR_SPLIT
```

The panel and Euler training scripts accept the same arguments. Edit the explicit
flags, checkpoint, and `save_training_path` settings in each script as needed. Install `optax`
for training. The scripts reject evaluation-only splits before loading the model.
Training outputs are saved under `saved_parameters/` and `training_stats/`.

## Batch submission

Initialize conda and cluster modules before submission, and replace
`conda activate YOUR_ENV_NAME` in the chosen batch script. Submit from this
model directory, supplying your cluster's scheduler settings and dataset:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION train-slurm-data.sb --dataset YOUR_DATASET --split YOUR_SPLIT
```

Add `--qos=YOUR_QOS` if required. Resource requests and wall times are examples
to adapt to your allocation. CFD environments also require their OpenFOAM/DAFoam
shell setup; activate or source that setup for your installation before submission.
