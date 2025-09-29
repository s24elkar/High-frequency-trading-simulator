import numpy as np
import pytest

from hawkes.pow_bivar import PowerLawParams, omega_matrix, pack_params, unpack_params


def test_pack_unpack_roundtrip():
    params = {
        "mu_b": 0.25,
        "mu_s": 0.3,
        "eta_bb": 0.4,
        "c_bb": 0.1,
        "gamma_bb": 0.7,
        "eta_bs": 0.2,
        "c_bs": 0.1,
        "gamma_bs": 0.8,
        "eta_sb": 0.2,
        "c_sb": 0.1,
        "gamma_sb": 0.9,
        "eta_ss": 0.3,
        "c_ss": 0.2,
        "gamma_ss": 0.6,
    }
    vec = pack_params(params)
    recovered = unpack_params(vec)
    assert pytest.approx(params["c_ss"]) == recovered.c_ss
    omega = omega_matrix(recovered)
    assert omega.shape == (2, 2)
    assert np.isfinite(omega).all()
