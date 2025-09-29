import numpy as np

from hawkes.diagnostics import ks_test, qq_points


def test_ks_test_handles_empty():
    result = ks_test(np.array([]))
    assert not result["pass"]


def test_qq_points_shapes():
    residuals = np.linspace(0.1, 5.0, 10)
    empirical, theoretical = qq_points(residuals, n_points=5)
    assert empirical.shape == theoretical.shape
