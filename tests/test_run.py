"""Driver unit tests and explicitly enabled GPU smoke tests. See README.md."""
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'cases/aircraft_BWB/model'
# Prevent the editable private repository from filling in missing release modules.
package = types.ModuleType('aerodml')
package.__path__ = [str(ROOT / 'src/aerodml')]
sys.modules['aerodml'] = package
sys.path.insert(0, str(MODEL))
import run
import evaluate_sample
import numpy as np
import jax
from aerodml import postprocessing_viz as viz


class DriverTests(unittest.TestCase):
    """Exercise actual run.main control flow with expensive operations mocked."""

    def exercise(self, fidelity, train=False, load=True, save=False):
        info = Mock()
        info.load.return_value = {'weight': 1}
        info.initialize.return_value = {'weight': 0}
        cases = [{'case_id': str(i), 'topology': {}} for i in (38, 132, 155)]
        replacements = {
            'get_graph': Mock(return_value={}),
            'get_lowfi_panel_mesh': Mock(return_value={}),
            'get_lowfi_CFD_mesh': Mock(return_value={}),
            'get_model': Mock(return_value=(Mock(), info)),
            'load_cfd_data': Mock(return_value=[]),
            'load_lowfi_panel_data': Mock(),
            'load_lowfi_cfd_data': Mock(),
            'build_geometry_data': Mock(return_value=(cases[:1] if train else [], cases)),
            'build_residual_function': Mock(return_value=Mock()),
            'train_model': Mock(return_value={'weight': 2}),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(run.jax, 'devices', return_value=[object()]))
            stack.enter_context(patch.object(run.jax, 'jit', side_effect=lambda f, **kw: f))
            stack.enter_context(patch.dict(sys.modules, {'optax': types.SimpleNamespace(piecewise_constant_schedule=Mock())}))
            for name, value in replacements.items():
                stack.enter_context(patch.object(run, name, value))
            analyze = stack.enter_context(patch.object(viz, 'analyze_aero_forces'))
            run.main(load=load, train=train, save=save,
                     load_model_path='test.pickle' if load else None,
                     as_multifidelity_model=fidelity, exclude_validation_cases=False)
        self.assertEqual(info.load.call_count, int(load))
        self.assertEqual(info.initialize.call_count, int(not load))
        self.assertEqual(replacements['train_model'].call_count, int(train))
        self.assertEqual(info.save.call_count, int(save))
        if save:
            self.assertEqual(info.save.call_args.args[0], {'weight': 2})
        analyzed = analyze.call_args.args[2]
        self.assertEqual({c['case_id'] for c in analyzed}, {'38', '132', '155'})
        self.assertEqual(replacements['load_lowfi_panel_data'].call_count, int(fidelity == 'panel'))
        self.assertEqual(replacements['load_lowfi_cfd_data'].call_count, int(fidelity == 'euler'))
        return replacements

    def test_evaluation_all_models(self):
        for fidelity in (False, 'panel', 'euler'):
            with self.subTest(fidelity=fidelity):
                self.exercise(fidelity)

    def test_training_from_scratch_and_checkpoint(self):
        for fidelity in (False, 'panel', 'euler'):
            for load in (False, True):
                with self.subTest(fidelity=fidelity, load=load):
                    self.exercise(fidelity, train=True, load=load, save=True)

    def test_evaluator_selects_shipped_assets(self):
        for name, (checkpoint, fidelity) in evaluate_sample.MODELS.items():
            with self.subTest(model=name), patch.object(sys, 'argv', ['evaluate_sample.py', '--model', name]), patch.object(run, 'main') as main:
                evaluate_sample.main()
                options = main.call_args.kwargs
                self.assertFalse(options['train'])
                self.assertFalse(options['save'])
                self.assertFalse(options['exclude_validation_cases'])
                self.assertEqual(options['as_multifidelity_model'], fidelity)
                self.assertTrue((MODEL / 'saved_parameters' / checkpoint).is_file())
                self.assertTrue((run.DATA_DIR / options['dataset_dir'] / (options['split'] + '.json')).is_file())


@unittest.skipUnless(os.environ.get('AERODML_GPU_TESTS') == '1', 'GPU smoke tests require explicit opt-in')
class GPUSmokeTests(unittest.TestCase):
    """Real mesh loading, prediction, gradients, updates and checkpoint saving."""

    def smoke(self, training):
        name = os.environ.get('AERODML_TEST_MODEL', 'rans')
        checkpoint, fidelity = evaluate_sample.MODELS[name]
        self.assertTrue(jax.devices('gpu'), 'A GPU allocation is required')
        real_build = run.build_geometry_data
        real_train = run.train_model
        predictions = []

        def build(*args, **kwargs):
            _, cases = real_build(*args, **kwargs)
            self.assertEqual(len(cases), 3)
            if not training:
                return [], cases
            # Existing training logger indexes training cases 2 and 5.
            # Repeat two cases in memory only; the third remains held out.
            return cases[:2] * 3, cases[2:]

        def train(*args):
            args = list(args)
            before = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(args[2])]
            args[6] = dict(args[6], max_iter=2, print_stride=1, save='smoke')
            args[8] = [0]
            result = real_train(*args)
            after = [np.asarray(x) for x in jax.tree_util.tree_leaves(result)]
            self.assertTrue(all(np.isfinite(x).all() for x in after))
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, after)), 'Training did not change parameters')
            return result

        def analyze(model, params, cases, constants, **kwargs):
            # Test numerical predictions without producing publication plots.
            for case in cases:
                values = model(params, case['flow_condition'], case['warped_geo'])
                arrays = [np.asarray(v) for v in jax.tree_util.tree_leaves(values)]
                self.assertTrue(arrays and all(a.size and np.isfinite(a).all() for a in arrays))
                predictions.append(case['case_id'])

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='aerodml-smoke-') as directory:
            work = Path(directory)
            params = work / 'saved_parameters'
            params.mkdir()
            (work / 'training_stats').mkdir()
            shutil.copy2(MODEL / 'saved_parameters' / checkpoint, params / checkpoint)
            try:
                os.chdir(work)
                with patch.object(run, 'PARAM_DIR', params), patch.object(run, 'build_geometry_data', side_effect=build), patch.object(run, 'train_model', side_effect=train), patch.object(viz, 'analyze_aero_forces', side_effect=analyze):
                    run.main(load=True, train=training, save=training,
                             dataset_dir='sample_dataset', split='sample_split',
                             load_model_path=checkpoint, as_multifidelity_model=fidelity,
                             save_training_path='smoke', exclude_validation_cases=False)
                self.assertEqual(set(predictions), {'38', '132', '155'})
                if training:
                    self.assertTrue(list(params.glob('smoke_gnn_*.pickle')))
            finally:
                os.chdir(cwd)

    def test_real_evaluation(self):
        self.smoke(False)

    def test_two_training_iterations(self):
        self.smoke(True)


if __name__ == '__main__':
    unittest.main()
