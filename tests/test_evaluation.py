"""Loss functions and Mincer-Zarnowitz regression."""

import numpy as np
import pytest

from volforecast.evaluation import metrics, mincer_zarnowitz, qlike


def test_qlike_zero_iff_perfect():
    r = np.array([0.04, 0.09])
    assert np.allclose(qlike(r, r), 0)
    assert (qlike(r, r * 1.3) > 0).all() and (qlike(r, r * 0.7) > 0).all()


def test_qlike_hand_computed_and_asymmetric():
    assert qlike(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(2 - np.log(2) - 1)
    # Under-predicting by half costs more than over-predicting by double.
    under = qlike(np.array([1.0]), np.array([0.5]))[0]
    over = qlike(np.array([1.0]), np.array([2.0]))[0]
    assert under > over


def test_mincer_zarnowitz_perfect_forecast():
    f = np.linspace(0.01, 0.2, 50)
    mz = mincer_zarnowitz(f, f)
    assert mz["mz_alpha"] == pytest.approx(0, abs=1e-12)
    assert mz["mz_beta"] == pytest.approx(1)
    assert mz["mz_r2"] == pytest.approx(1)


def test_metrics_values():
    m = metrics([0.04, 0.09], [0.04, 0.04])
    assert m["mse_var"] == pytest.approx(0.05**2 / 2)
    assert m["mae_vol"] == pytest.approx(0.1 / 2)
    assert m["n_obs"] == 2
