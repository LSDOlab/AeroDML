#!/usr/bin/env bash
set -e

RANKS=${1:-15}
DATA_DIR=${2:-data/sampling_200}

N=$(python -c "from data_config_builder import get_dataset_configurations as g; print(len(g('$DATA_DIR')))")

for i in $(seq 0 $((N-1))); do
    echo "Running case $i / $((N-1))"
    mpirun -n "$RANKS" --output-filename "$DATA_DIR/euler/logs/output_logs_$i" python generate_euler_single.py "$DATA_DIR" "$i"
done

# mpirun -n 25 python generate_euler_single.py data/sample_b 1