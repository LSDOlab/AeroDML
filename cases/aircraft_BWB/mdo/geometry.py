# paths
from pathlib import Path
HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parents[0]          # aircraft_BWB
CASES_DIR = HERE.parents[1]         # cases

BWB_DIR = CASES_DIR / "datasets" / "BWB"
import sys
sys.path.insert(0, str(BWB_DIR))
from bwb_geometry import compute_geometry_csdl

import numpy as np

def build_surface_csdl(
        recorder,
        geometry_pickle_file_path,
        stp_file_path,
        surface_mesh_projection_file_path,
        dafoam_directory,
        x_surf_hash,
        index_unique = True,
        return_geometry = False,
    ):
    bwb_geometry, geometry_config_map, x_surf_dafoam_full_geo, x_surf_dafoam_full_geo_baseline  = compute_geometry_csdl(
        None,
        geometry_pickle_file_path,
        stp_file_path,
        surface_mesh_projection_file_path,
        None,
        None,
        recorder,
        timing_enabled=True,
        return_baseline_projections = True,
    )
    recorder.inline = False
    recorder.inline = True

    if index_unique:
        surf_indices = np.load(f"{dafoam_directory}/surf_gid_full_{x_surf_hash}.npy")
        arr = np.asarray(surf_indices)
        unique_count = len(np.unique(arr))
        print("Unique elements:", unique_count)
        print("Total elements:", len(arr))
        unique_gids, surf_indices_list = np.unique(
            surf_indices,
            return_index=True,
        )
        csdl_surf_points = x_surf_dafoam_full_geo[surf_indices_list.tolist()]
        if return_geometry:
            return csdl_surf_points, geometry_config_map, bwb_geometry
        else:
            return csdl_surf_points, geometry_config_map
    else:
        if return_geometry:
            return x_surf_dafoam_full_geo, geometry_config_map, bwb_geometry
        else:
            return x_surf_dafoam_full_geo, geometry_config_map