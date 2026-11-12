#!/usr/bin/env bash
# Run each complete sweep in a fresh process so GPU memory is released between
# models. Geometry, plotting and test settings live in generate_sweep_gifs.py.
set -euo pipefail

# Replace with your own conda environment
# CONDA_ENV="aerodml" 
CONDA_ENV="official_aerodml"

MODELS=(pgnn panel gnn)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for model in "${MODELS[@]}"; do
    echo "Starting ${model} sweep"
    BWB_SWEEP_MODEL="$model" conda run --no-capture-output -n "$CONDA_ENV" \
        python -u "$SCRIPT_DIR/generate_sweep_gifs.py"
    echo "Finished ${model} sweep"
done

echo "All model sweeps finished. Run combine_gifs.py to combine the GIFs."
