"""Unit tests for src.scorers module."""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scorers import robust_z, memo_scores, unsupervised_scores, two_tailed_precision


class TestRobustZ:
    def test_basic(self):
        """Test robust_z returns correct shape and reasonable values."""
        X = np.random.randn(100, 5)
        Z = robust_z(X)
        assert Z.shape == X.shape
        # Robust z-scores should have median ≈ 0 and MAD-based scale
        assert abs(np.median(Z)) < 0.5

    def test_constant_feature(self):
        """Test robust_z handles constant features (MAD=0)."""
        X = np.ones((100, 3))
        X[:, 1] = np.random.randn(100)  # Only middle column varies
        Z = robust_z(X)
        assert Z.shape == X.shape
        # Constant columns should have z=0
        assert np.allclose(Z[:, 0], 0)
        assert np.allclose(Z[:, 2], 0)

    def test_outliers(self):
        """Test robust_z is less affected by outliers than standard z."""
        X = np.random.randn(100, 1)
        X[0] = 100  # Extreme outlier
        Z = robust_z(X)
        # The outlier's z-score should be large but finite
        assert Z[0] > 10
        assert np.isfinite(Z[0])


class TestMemoScores:
    def test_basic(self):
        """Test memo_scores with simple DataFrame."""
        df = pd.DataFrame({
            'loss_mean': np.random.randn(100),
            'entropy': np.random.randn(100),
            'other': np.random.randn(100),
        })
        feats_sign = {'loss_mean': -1, 'entropy': -1}  # Low = memorized
        sub, scores, cols = memo_scores(df, feats_sign)

        assert len(sub) == 100
        assert len(scores) == 100
        assert cols == ['loss_mean', 'entropy']

    def test_missing_features(self):
        """Test memo_scores gracefully handles missing features."""
        df = pd.DataFrame({'loss_mean': np.random.randn(50)})
        feats_sign = {'loss_mean': -1, 'nonexistent': -1}
        sub, scores, cols = memo_scores(df, feats_sign)

        assert cols == ['loss_mean']
        assert len(scores) == 50

    def test_nan_handling(self):
        """Test memo_scores drops rows with NaN."""
        df = pd.DataFrame({
            'loss_mean': [1.0, np.nan, 3.0],
            'entropy': [0.5, 0.6, 0.7],
        })
        feats_sign = {'loss_mean': -1, 'entropy': -1}
        sub, scores, cols = memo_scores(df, feats_sign)

        assert len(sub) == 2  # Middle row dropped


class TestUnsupervisedScores:
    def test_all_methods_run(self):
        """Test all unsupervised methods return scores."""
        X = np.random.randn(100, 5)
        results = unsupervised_scores(X, seed=42)

        expected_keys = {'iforest', 'iforest_score', 'mahalanobis',
                        'zscore_max', 'zscore_mean'}
        assert set(results.keys()) == expected_keys

        for key, scores in results.items():
            assert len(scores) == 100
            if key != 'iforest':  # iforest returns binary labels
                assert np.isfinite(scores).any()


class TestTwoTailedPrecision:
    def test_basic(self):
        """Test two-tailed precision with known distribution."""
        # Noise at both tails
        y = np.array([1]*10 + [0]*80 + [1]*10)
        scores = np.concatenate([np.arange(10), np.arange(80)+20, np.arange(10)+100])

        p = two_tailed_precision(y, scores, budget=0.20)
        assert p == 1.0  # All 20 dropped samples are noise

    def test_single_tail(self):
        """Test two-tailed fails when noise is only at one tail."""
        y = np.array([1]*10 + [0]*90)
        scores = np.arange(100)  # Noise at low end only

        p = two_tailed_precision(y, scores, budget=0.20)
        assert p == 0.5  # Half the budget wasted on clean tail


if __name__ == "__main__":
    # Run all test classes
    test_classes = [TestRobustZ, TestMemoScores, TestUnsupervisedScores, TestTwoTailedPrecision]

    total_tests = 0
    passed_tests = 0

    for test_cls in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_cls.__name__}")
        print('='*60)

        test_obj = test_cls()
        test_methods = [m for m in dir(test_obj) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_obj, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print('='*60)

    if passed_tests == total_tests:
        print("✓✓✓ All tests passed ✓✓✓")
    else:
        sys.exit(1)
