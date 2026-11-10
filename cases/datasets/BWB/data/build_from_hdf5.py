"""Build cases.json and scratch-backed output directories from optimization history.

Run this script from any working directory. Edit CASE_NAME to select an existing
sibling directory containing hdf5_specs.py. Specs define HDF5_ITERS,
GEOMETRY_VARIABLES_MAP, FLOW_CONDITION_VARIABLES_MAP, and flow_conditions.
Map values of None use the configuration variable's name in the HDF5 file.
Optional FLOW_CONDITION_INDEX (default 0) selects the recorded flow condition.
Case keys are str(file_index * 100000 + iteration), with zero-based file_index
in HDF5_ITERS insertion order (e.g. first file iteration 10: "10"; second file
iteration 5: "100005"). At most nine files and iterations 0 through 9999 are allowed.
Existing output directories and their contents are preserved.
"""

import json
import os
from pathlib import Path
import runpy
import tempfile

import h5py
import numpy as np


CASE_NAME = 'SEP10_OPT'
CASE_NAME = 'SEP14_OPTS'
CASE_NAME = 'SEP17_OPTS'
# CASE_NAME = 'sample_tcp1_c'
CASE_NAME = 'SEP20_2031_PGNN'
CASE_NAME = 'SEP20_2031_GNN'
CASE_NAME = 'SEP20_2031_GNN'

# Cameber opt
CASE_NAME = 'FINAL_CAMBER_OPT_panel_sept27'
CASE_NAME = 'FINAL_CAMBER_OPT_pgrgnn_sept27'
CASE_NAME = 'FINAL_CAMBER_OPT_rgnn_sept27'

# full MDO opt
CASE_NAME = 'FINAL_MDO_panel_sept27'
CASE_NAME = 'FINAL_MDO_pgrgnn_sept27'
CASE_NAME = 'FINAL_MDO_rgnn_sept27'

HOME_ROOT = ""  # Required: home root used to compute the existing relative layout.
SCRATCH_ROOT = ""  # Required: scratch root for the mirrored output directories.
OUTPUT_DIRS = ('rans', 'panel', 'euler', 'surf_pts', 'vertices', 'surf_pts_euler')


def json_value(value):
    """Preserve arrays, converting singleton geometry/flow values to scalars."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError('Expected finite numeric values')
    return array.item() if array.size == 1 else array.tolist()


def read_variable(group, config_name, source_name):
    source_name = source_name or config_name
    if source_name not in group or not isinstance(group[source_name], h5py.Dataset):
        raise KeyError(f'{group.file.filename}:{group.name}: missing dataset {source_name!r}')
    return group[source_name][()]


def build_config(case_dir):
    """Read and validate all selected iterations without writing output."""
    specs_path = case_dir / 'hdf5_specs.py'
    if not specs_path.is_file():
        raise FileNotFoundError(f'Missing specs: {specs_path}')
    specs = runpy.run_path(str(specs_path))
    condition_index = specs.get('FLOW_CONDITION_INDEX', 0)
    if not isinstance(condition_index, int) or condition_index < 0:
        raise ValueError('FLOW_CONDITION_INDEX must be a nonnegative integer')
    cases = []
    sources = []
    hdf5_iters = specs['HDF5_ITERS']
    if len(hdf5_iters) > 9:
        raise ValueError('HDF5_ITERS must contain at most nine files')
    for file_index, (filename, iterations) in enumerate(hdf5_iters.items()):
        filename = Path(filename).expanduser()
        if not filename.is_absolute():
            filename = case_dir / filename
        filename = filename.resolve(strict=True)
        with h5py.File(filename, 'r') as history:
            for iteration in iterations:
                if not isinstance(iteration, int) or not 0 <= iteration <= 9999:
                    raise ValueError(f'Invalid iteration: {iteration!r}; expected an integer from 0 to 9999')
                group_name = f'iteration_{iteration}'
                if group_name not in history:
                    raise KeyError(f'{filename}: missing {group_name}')
                group = history[group_name]
                geometry = {
                    name: {'value': json_value(read_variable(group, name, source))}
                    for name, source in specs['GEOMETRY_VARIABLES_MAP'].items()
                }
                flow = {
                    name: {'value': json_value(item['value'])}
                    for name, item in specs['flow_conditions'].items()
                }
                for name, source in specs['FLOW_CONDITION_VARIABLES_MAP'].items():
                    value = np.asarray(read_variable(group, name, source))
                    # Recorded flow variables use their first axis for conditions.
                    if value.ndim:
                        if condition_index >= value.shape[0]:
                            raise IndexError(f'{filename}:{group_name}/{name}: '
                                             f'no condition {condition_index}, shape={value.shape}')
                        value = value[condition_index]
                    flow[name] = {'value': json_value(value)}
                key = str(file_index * 100000 + iteration)
                cases.append({'key': key, 'geometry_configuration': geometry,
                              'flow_conditions': flow})
                sources.append({'key': key, 'hdf5_file': str(filename),
                                'iteration': iteration, 'flow_condition_index': condition_index})
    if not cases:
        raise ValueError('HDF5_ITERS must select at least one iteration')
    # The existing loader reads only "cases"; provenance stays in this same file.
    return {'cases': cases, 'sources': sources}


def prepare_output_dirs(case_dir):
    """Mirror the home path on scratch, checking conflicts before creating links."""
    if not HOME_ROOT or not SCRATCH_ROOT:
        raise ValueError("Set HOME_ROOT and SCRATCH_ROOT in build_from_hdf5.py before running.")
    scratch_case = Path(SCRATCH_ROOT).expanduser().resolve() / case_dir.relative_to(Path(HOME_ROOT).expanduser().resolve())
    pending = []
    for name in OUTPUT_DIRS:
        link = case_dir / name
        target = scratch_case / name
        if link.is_dir():
            continue
        if link.is_symlink():
            # Path.exists() alone would mistake a dangling link for an absent entry.
            if link.resolve() != target.resolve():
                raise FileExistsError(f'Unexpected broken or non-directory symlink: {link}')
        elif link.exists():
            raise FileExistsError(f'Output path is not a directory: {link}')
        if target.is_symlink() and not target.is_dir():
            raise FileExistsError(f'Invalid scratch target symlink: {target}')
        if target.exists() and not target.is_dir():
            raise FileExistsError(f'Scratch target is not a directory: {target}')
        pending.append((link, target))
    for link, target in pending:
        target.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        print(f'{link} -> {target}')


def main():
    if not HOME_ROOT or not SCRATCH_ROOT:
        raise ValueError("Set HOME_ROOT and SCRATCH_ROOT in build_from_hdf5.py before running.")
    if not CASE_NAME or Path(CASE_NAME).name != CASE_NAME or CASE_NAME in ('.', '..'):
        raise ValueError('CASE_NAME must be a single directory name')
    case_dir = Path(__file__).resolve().parent / CASE_NAME
    if not case_dir.is_dir():
        raise FileNotFoundError(f'Case directory must already exist: {case_dir}')
    config = build_config(case_dir)
    contents = json.dumps(config, indent=2, allow_nan=False) + '\n'
    prepare_output_dirs(case_dir)
    # Replace the config only after extraction and directory setup succeed.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=case_dir, suffix='.tmp',
                                         prefix='cases.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        os.replace(temporary, case_dir / 'cases.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Wrote {len(config["cases"])} cases to {case_dir / "cases.json"}')


if __name__ == '__main__':
    main()
