#!/usr/bin/env bash
set -e

DATA_DIR=${1:-data/sampling_200}

mpirun -n 1 python generate_panel.py "$DATA_DIR"