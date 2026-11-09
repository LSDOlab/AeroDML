# project/ directory
import numpy as np
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# allow imports from project/A
sys.path.insert(0, str(ROOT))
from data_config_builder import write_dataset_config, atmos_model

if __name__ == "__main__":
    # Sample geometry ranges for the dataset
    geometry_ranges = {
        "centerbody_chord_stretch_ratio": {
            "ref_value": [1.0, 0.9265333333333333, 0.8718160886458484, 0.8330375933644204],
            "lower": [0.9, 0.9, 0.8, 0.8],
            "upper": [1.10, 0.97, 0.97, 0.94],
        },
        "wing_position": {
            "ref_value": 20.186592500296157,
            "lower": 22.186592500296157,
            "upper": 13.186592500296157,
        },
        "wing_sweep": {
            "ref_value": 35.067400250175616,
            "lower": 15.0,
            "upper": 42.0,
        },
        "wing_half_span": {
            "ref_value": 15.961024804157153,
            "lower": 8.0,
            "upper": 20.0,
        },
        "centerbody_half_span": {
            "ref_value": 5.000000000000001,
            "lower": 3.0,
            "upper": 7.0,
        },
        "transition_half_span": {
            "ref_value": 4.890738003669036,
            "lower": 3.5,
            "upper": 7.0,
        },
        "wing_chord_stretch_coefficients": {
            "ref_value": [0.0, 0.0, 0.0, 0.0],
            "lower": [-3.0000005, -2.49952453, -1.99904855, -1.49857258],
            "upper": [3.0000005, 2.49952453, 1.99904855, 1.49857258],
        },
        "wing_twist_coefficients": {
            "ref_value": [0.0, 0.0, 0.0, 0.0],
            "lower": [-5.0, -5.0, -5.0, -5.0],
            "upper": [5.0, 5.0, 5.0, 5.0],
        },
        "airfoil_thickness_coefficients": {
            "ref_value": np.zeros((8, 8)).tolist(),
            "lower": (np.full((8, 8), -0.05)).tolist(),
            "upper": (np.full((8, 8), 0.05)).tolist(),
        },
        "airfoil_camber_coefficients": {
            "ref_value": np.zeros((6, 8)).tolist(),
            "lower": (np.full((6, 8), -0.03)).tolist(),
            "upper": (np.full((6, 8), 0.03)).tolist(),
        },
        "elevator_deflection": {
            "ref_value": 0.0,
            "lower": 0.0,
            "upper": 7.0,
        },
    }
    

    rho_inf, sos_inf, mu_inf, T_inf, p_inf = atmos_model(9.144)
    Ma = 0.75
    v_inf = Ma*sos_inf
    p_inf = p_inf
    T_inf = T_inf

    flow_conditions = {
        "v_inf": {
            "value": np.array([v_inf, 0.0, 0.0]),
        },
        "p_inf": {
            "value": p_inf,
        },
        "T_inf": {
            "value": T_inf,
        },
        "pitch": {
            "ref_value": 0.0,
            "lower": -5,
            "upper": 5,
        }
    }

    exit()
    write_dataset_config(
        data_dir="",
        n_samples=350,
        geometry_ranges=geometry_ranges,
        flow_ranges=flow_conditions,
        seed=71,
    )