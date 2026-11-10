#!/usr/bin/env bash
set -e

RANKS=${1:-15}
DATA_DIR=${2:-data/sampling_200}
MESH_DIR=${3:-mesh_112k}
START=${4:-0}

N=$(python -c "from data_config_builder import get_dataset_configurations as g; print(len(g('$DATA_DIR')))")

END=${5:-$((N-1))}  # inclusive list index

(( START >= 0 && START <= END && END < N )) || {
    echo "ERROR: Invalid range $START-$END; valid indices are 0-$((N-1))"
    exit 1
}

for ((i=START; i<=END; i++)); do
    key=$(python -c "import sys; from data_config_builder import get_dataset_configurations as g; print(g(sys.argv[1])[int(sys.argv[2])]['key'])" "$DATA_DIR" "$i")
    # Key 0 can initialize the warm-start fields on a fresh Euler mesh.
    if [[ "$key" != "0" && ! -d "$MESH_DIR/processor0/9999" ]]; then
        echo "ERROR: Warm-start directory $MESH_DIR/processor0/9999 does not exist; run key 0 first."
        exit 1
    fi
    echo "Running case $i / $((N-1))"
    mpirun -n "$RANKS" --output-filename "$DATA_DIR/euler/logs/output_logs_$key" python generate_euler_single.py "$DATA_DIR" "$key" "$MESH_DIR"
    foamListTimes -case "$MESH_DIR" -processor -time ':9998' -rm
done

# mpirun -n 25 python generate_euler_single.py data/sample_b 1