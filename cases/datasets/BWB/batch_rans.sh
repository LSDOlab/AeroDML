#!/usr/bin/env bash
set -e

RANKS=${1:-15}
DATA_DIR=${2:-data/sampling_200}
MESH_DIR=${3:-"None"}
N=$(python -c "from data_config_builder import get_dataset_configurations as g; print(len(g('$DATA_DIR')))")

# # Check if the warm-start directory exists
# if [[ ! -d "$MESH_DIR/processor0/9999" ]]; then
#     echo "ERROR: Warm-start directory $MESH_DIR/processor0/9999 does not exist."
#     exit 1
# fi

# for ((i=1; i<N; i++)); do
for i in $(seq 0 $((N-1))); do
    echo "Running case $i / $((N-1))"
    mpirun -n "$RANKS" --output-filename "$DATA_DIR/rans/logs/output_logs_$i" python generate_rans_single.py "$DATA_DIR" "$i" "$MESH_DIR"
    foamListTimes -case "$MESH_DIR" -processor -time ':9998' -rm
done