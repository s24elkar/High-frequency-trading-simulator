import numpy as np
import pytest

from hawkes.exp_bivar import omega_matrix, pack_params, unpack_params


def test_pack_unpack_roundtrip():
    params = {
        "mu_b": 0.3,
        "mu_s": 0.2,
        "alpha_bb": 0.4,
        "beta_bb": 2.0,
        "alpha_bs": 0.1,
        "beta_bs": 3.0,
        "alpha_sb": 0.05,
        "beta_sb": 4.0,
        "alpha_ss": 0.6,
        "beta_ss": 5.0,
    }
    vec = pack_params(params)
    recovered = unpack_params(vec)
    assert pytest.approx(params["alpha_bb"]) == recovered.alpha_bb
    omega = omega_matrix(recovered)
    assert omega.shape == (2, 2)
    assert np.isfinite(omega).all()
