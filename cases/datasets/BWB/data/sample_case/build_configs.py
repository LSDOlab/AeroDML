# project/ directory
import numpy as np
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# allow imports from project/A
sys.path.insert(0, str(ROOT))
from data_config_builder import write_dataset_config

if __name__ == "__main__":

    geometry_ranges = {
        "span": {
            "ref_value": 25.852 - 9.891,
            "delta": 7.0,
        },
        "sweep": {
            "ref_value": 0.0,
            "delta": 6.0,
        },
        "wing_chord": {
            "ref_value": [0.0, 0.0],
            "delta": 1.5,
        },
        "root_twist": {
            "ref_value": [0.0],
            "delta": 7.0 * np.pi / 180.0,
        },
        "mid_twist": {
            "ref_value": [0.0, 0.0],
            "delta": 7.0 * np.pi / 180.0,
        },
        "tip_twist": {
            "ref_value": [0.0],
            "delta": 7.0 * np.pi / 180.0,
        },
        "%_thickness_change_wing": {
            "ref_value": np.zeros((8, 4)).tolist(),
            "delta": 10.0,
        },
        "%_camber_change_wing": {
            "ref_value": np.zeros((6, 4)).tolist(),
            "delta": 10.0,
        },
        "center_chord": {
            "ref_value": [0.0, 0.0, 0.0],
            "delta": 8.0,
        },
        "center_span": {
            "ref_value": 10.0,
            "delta": 2.0,
        },
        "center_twist": {
            "ref_value": [0.0, 0.0],
            "delta": 5.0 * np.pi / 180.0,
        },
        "center_twist_root": {
            "ref_value": [0.0],
            "delta": 5.0 * np.pi / 180.0,
        },
        "%_thickness_change_body": {
            "ref_value": np.zeros((8, 4)).tolist(),
            "delta": 10.0,
        },
        "%_camber_change_body": {
            "ref_value": np.zeros((6, 4)).tolist(),
            "delta": 10.0,
        },
        "transition_span": {
            "ref_value": 4.891,
            "delta": 2.0,
        },
    }

    flow_conditions = {
        "v_inf": {
            "value": np.array([227.3805, 0.0, 0.0]),
        },
        "p_inf": {
            "value": 30089.6,
        },
        "T_inf": {
            "value": 228.714,
        },
    }

    write_dataset_config(
        data_dir="",
        n_samples=3,
        geometry_ranges=geometry_ranges,
        flow_ranges=flow_conditions,
        seed=9999,
    )