#!/usr/bin/env bash
set -e

RANKS=${1:-15}
DATA_DIR=${2:-data/sampling_200}
MESH_DIR=${3:-"None"}
START=${4:-0}
N=$(python -c "from data_config_builder import get_dataset_configurations as g; print(len(g('$DATA_DIR')))")
END=${5:-$((N-1))}  # inclusive

# make sure the start and end are valid
(( START >= 0 && START <= END && END < N )) || {
    echo "ERROR: Invalid range $START-$END; valid indices are 0-$((N-1))"
    exit 1
}

# Check if the warm-start directory exists
if [[ ! -d "$MESH_DIR/processor0/9999" ]]; then
    echo "ERROR: Warm-start directory $MESH_DIR/processor0/9999 does not exist."
    exit 1
fi

# for i in $(seq 0 $((N-1))); do
# for ((i=1; i<N; i++)); do
for ((i=START; i<=END; i++)); do
    key=$(python -c "import sys; from data_config_builder import get_dataset_configurations as g; print(g(sys.argv[1])[int(sys.argv[2])]['key'])" "$DATA_DIR" "$i")
    echo "Running case $i / $((N-1))"
    mpirun -n "$RANKS" --output-filename "$DATA_DIR/rans/logs/output_logs_$key" python generate_rans_single.py "$DATA_DIR" "$key" "$MESH_DIR"
    foamListTimes -case "$MESH_DIR" -processor -time ':9998' -rm
done

# mpirun -n 1 --output-filename data/sample_b/rans/logs/output_logs_1 python generate_rans_single.py data/sample_b 1
# mpirun -n 25 --output-filename data/sample_c/rans/logs/output_logs_136 python generate_rans_single.py data/sample_c 136 mesh_639k
# ==== SINGLE EXAMPLES ====
# mpirun -n 25 --output-filename "data/sample_tcp1_a/rans/logs/output_logs_0" python generate_rans_single.py data/sample_tcp1_a 0 mesh_639k
# mpirun -n 25 python -u generate_rans_single.py data/sample_tcp1_b 1 mesh_639k
# ==== BASH EXAMPLES ====
# bash batch_rans.sh 25 data/sample_c mesh_3M