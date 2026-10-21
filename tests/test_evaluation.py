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


# --- Diebold-Mariano --------------------------------------------------------------

from statsmodels.stats.sandwich_covariance import cov_hac  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from volforecast.evaluation import diebold_mariano, long_run_variance  # noqa: E402


def ma4_noise(n, rng):
    """MA(4) noise, like the loss differential of overlapping 5-day targets."""
    e = rng.normal(size=n + 4)
    return sum(e[k:k + n] for k in range(5)) / np.sqrt(5)


def test_long_run_variance_matches_statsmodels():
    d = ma4_noise(2000, np.random.default_rng(0))
    res = sm.OLS(d, np.ones(len(d))).fit()
    sm_var = cov_hac(res, nlags=8, use_correction=False)[0, 0] * len(d)
    assert long_run_variance(d, 8) == pytest.approx(sm_var, rel=1e-10)


def test_hln_correction_hand_computed():
    rng = np.random.default_rng(1)
    d = ma4_noise(300, rng) + 0.1
    r = diebold_mariano(d, np.zeros_like(d), horizon=5)
    raw = d.mean() / np.sqrt(long_run_variance(d, r["lags"]) / len(d))
    n, h = len(d), 5
    assert r["dm_stat"] == pytest.approx(raw * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))


def test_dm_size_and_power():
    """Equal accuracy: rejects at roughly the nominal rate. Real difference: rejects."""
    rng = np.random.default_rng(2)
    rejections = sum(diebold_mariano(ma4_noise(1000, rng), np.zeros(1000), 5)["p_value"] < 0.05
                     for _ in range(400))
    assert rejections / 400 < 0.10
    r = diebold_mariano(ma4_noise(1000, rng) - 0.3, np.zeros(1000), 5)
    assert r["p_value"] < 0.01 and r["dm_stat"] < 0   # negative = model better


def test_dm_size_with_volatility_persistence():
    """Overlapping sums of a persistent process: the sqrt(n) bandwidth keeps the
    rejection rate of a true null close to the nominal 5%."""
    rng = np.random.default_rng(4)

    def persistent(n, phi=0.9):
        u = np.zeros(n + 204)
        e = rng.normal(size=len(u))
        for t in range(1, len(u)):
            u[t] = phi * u[t - 1] + e[t] * np.sqrt(1 - phi**2)
        x = u[200:] + rng.normal(size=n + 4)
        return sum(x[k:k + n] for k in range(5))

    rej = np.mean([diebold_mariano(persistent(1500), np.zeros(1500), 5)["p_value"] < 0.05
                   for _ in range(300)])
    assert rej < 0.10
