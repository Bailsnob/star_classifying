"""Behavioral checks for clustering selection and label bookkeeping."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".testing-deps").exists():
    sys.path.insert(0, str(ROOT / ".testing-deps"))
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import unsupervised as u


class BookkeepingTests(unittest.TestCase):
    def test_prepare_drops_undefined_logs_and_preserves_alignment(self):
        df = pd.DataFrame({"teff_gspphot": [5000., 0., np.inf, 6000., 4000.],
                           "lum_flame": [1., 1., 1., -1., .1],
                           u.TARGET: ["G", "A", "B", "F", "K"],
                           "source_id": [10, 11, 12, 13, 14]})
        prepared = u.prepare(df)
        self.assertEqual(prepared.source_id.tolist(), [10, 14])
        self.assertEqual(prepared[u.TARGET].tolist(), ["G", "K"])
        np.testing.assert_allclose(prepared.log_lum, [0., -1.])

    def test_spectral_labels_do_not_change_features(self):
        df = pd.DataFrame({"teff_gspphot": [4000., 5000., 6000.],
                           "lum_flame": [.1, 1., 10.], u.TARGET: ["K", "G", "F"]})
        original = u.scale_hr(u.prepare(df))
        df[u.TARGET] = ["O", "M", "CSTAR"]
        np.testing.assert_array_equal(original, u.scale_hr(u.prepare(df)))

    def test_relabel_preserves_partition_and_orders_temperature(self):
        old = np.array([8, 2, 8, 5, 2, 5])
        new = u.relabel_hot_to_cool(old, np.array([4., 3., 4.1, 3.5, 3.1, 3.6]), 3, "test")
        np.testing.assert_array_equal(new, [0, 2, 0, 1, 2, 1])
        self.assertEqual(adjusted_rand_score(old, new), 1.)

    def test_contingency_is_normalized_by_type(self):
        frac = u.row_fractions(np.array(["G", "G", "K", "K"]),
                               np.array([0, 1, 1, 1]), ["G", "K"])
        np.testing.assert_allclose(frac.to_numpy(), [[.5, .5], [0., 1.]])
        np.testing.assert_allclose(frac.sum(axis=1), 1.)

    def test_degenerate_silhouettes_return_nan(self):
        X = np.array([[0.], [1.], [2.]])
        self.assertTrue(np.isnan(u.safe_silhouette(X, np.array([0, 0, 0]))))
        self.assertTrue(np.isnan(u.safe_silhouette(X, np.array([0, 1, 2]))))

    def test_silhouette_tolerance_and_all_invalid(self):
        rows = [{"k": 2, "silhouette": .495}, {"k": 3, "silhouette": .5},
                {"k": 4, "silhouette": np.nan}]
        self.assertEqual(u.choose_k_silhouette(rows), 2)
        with self.assertRaises(RuntimeError):
            u.choose_k_silhouette([rows[-1]])

    def test_bic_ignores_failed_and_nonfinite_candidates(self):
        rows = [{"k": 2, "bic": np.nan, "converged": True},
                {"k": 3, "bic": -100., "converged": False},
                {"k": 4, "bic": -50., "converged": True},
                {"k": 5, "bic": -50., "converged": True}]
        self.assertEqual(u.choose_k_bic(rows), 4)
        with self.assertRaises(RuntimeError):
            u.choose_k_bic(rows[:2])

    def test_gmm_baseline_rejects_nonconvergence(self):
        with patch.object(u, "GaussianMixture") as factory:
            factory.return_value.fit_predict.return_value = np.array([0, 1, 1])
            factory.return_value.converged_ = False
            with self.assertRaises(RuntimeError):
                u.fit_temperature_only(np.array([[0.], [1.], [2.]]), 2, "gmm")

    def test_search_does_not_select_unconverged_low_bic(self):
        with patch.object(u, "K_RANGE", range(2, 4)), patch.object(u, "GaussianMixture") as factory:
            failed = MagicMock(converged_=False, n_iter_=3)
            valid = MagicMock(converged_=True, n_iter_=3)
            for model, bic in [(failed, -100.), (valid, -50.)]:
                model.fit_predict.return_value = np.array([0, 0, 1, 1])
                model.bic.return_value = bic
            factory.side_effect = [failed, valid]
            with warnings.catch_warnings(record=True):
                table, k, _ = u.search_gmm(np.array([[0.], [.1], [4.], [4.1]]))
            self.assertEqual(k, 3)
            self.assertEqual(table.converged.tolist(), [False, True])


class SyntheticRecoveryTests(unittest.TestCase):
    def test_both_searches_recover_three_well_separated_groups(self):
        X, truth = make_blobs(n_samples=600, centers=[[-6., -6.], [0., 6.], [6., -6.]],
                              cluster_std=.25, random_state=42)
        with threadpool_limits(limits=2), patch.object(u, "K_RANGE", range(2, 5)):
            _, km_k, km_labels = u.search_kmeans(X, np.arange(len(X)))
            _, gm_k, gm_labels = u.search_gmm(X)
        self.assertEqual(km_k, 3)
        self.assertEqual(gm_k, 3)
        self.assertGreater(adjusted_rand_score(truth, km_labels), .99)
        self.assertGreater(adjusted_rand_score(truth, gm_labels), .99)


if __name__ == "__main__":
    unittest.main()
