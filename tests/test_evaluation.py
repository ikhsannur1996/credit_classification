import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    binomial_backtest,
    bootstrap_ci,
    calibration_slope_intercept,
    corrected_mean_ci,
    corrected_resampled_ttest,
    delong_auc_ci,
    delong_test,
    holm_correction,
    hosmer_lemeshow,
    ks_statistic,
    mcnemar_test,
    psi,
    woe_iv,
)

rng = np.random.default_rng(0)
N = 2000
Y = rng.integers(0, 2, N)
P_GOOD = np.clip(0.5 * Y + rng.normal(0.25, 0.2, N), 0.001, 0.999)
P_BAD = rng.uniform(0, 1, N)


def test_delong_auc_matches_sklearn():
    auc, lo, hi = delong_auc_ci(Y, P_GOOD)
    assert auc == pytest.approx(roc_auc_score(Y, P_GOOD))
    assert lo < auc < hi


def test_delong_detects_better_model():
    res = delong_test(Y, P_GOOD, P_BAD)
    assert res["diff"] > 0 and res["p_value"] < 1e-6


def test_delong_same_model_not_significant():
    assert delong_test(Y, P_GOOD, P_GOOD + 1e-12)["p_value"] > 0.5


def test_ks_matches_scipy():
    expected = stats.ks_2samp(P_GOOD[Y == 1], P_GOOD[Y == 0]).statistic
    assert ks_statistic(Y, P_GOOD) == pytest.approx(expected)


def test_psi_identical_is_zero_and_shift_is_large():
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-9)
    assert psi(x, x + 1) > 0.25


def test_bootstrap_ci_contains_estimate():
    ci = bootstrap_ci(Y, P_GOOD, 0.5, n_boot=100)
    assert ((ci["ci_lower"] <= ci["estimate"]) & (ci["estimate"] <= ci["ci_upper"])).all()


def test_calibrated_model_passes_hosmer_lemeshow():
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert hosmer_lemeshow(y, p)["p_value"] > 0.01
    cal = calibration_slope_intercept(y, p)
    assert cal["slope"] == pytest.approx(1, abs=0.1)


def test_binomial_backtest_traffic_light():
    assert binomial_backtest(1000, 100, 0.10)["traffic_light"] == "green"
    assert binomial_backtest(1000, 200, 0.10)["traffic_light"] == "red"


def test_mcnemar_identical_predictions():
    pred = (P_GOOD > 0.5).astype(int)
    assert mcnemar_test(Y, pred, pred)["p_value"] == 1.0


def test_corrected_ttest_is_more_conservative():
    a = 0.80 + rng.normal(0, 0.01, 25)
    b = a - 0.004 + rng.normal(0, 0.005, 25)
    corrected = corrected_resampled_ttest(a, b, n_train=6400, n_test=1600)["p_value"]
    naive = stats.ttest_rel(a, b).pvalue
    assert corrected > naive


def test_woe_iv_strong_vs_noise():
    x_strong = pd.Series(P_GOOD)
    x_noise = pd.Series(P_BAD)
    _, iv_strong = woe_iv(x_strong, pd.Series(Y))
    _, iv_noise = woe_iv(x_noise, pd.Series(Y))
    assert iv_strong > 0.3 > 0.02 > iv_noise


def test_holm_is_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd"))
    adj = holm_correction(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert adj["a"] == pytest.approx(0.04)


def test_corrected_ci_wider_than_naive():
    from src.evaluation import mean_ci
    v = 0.85 + rng.normal(0, 0.01, 25)
    _, lo_c, hi_c = corrected_mean_ci(v, 6400, 1600)
    _, lo_n, hi_n = mean_ci(v)
    assert (hi_c - lo_c) > (hi_n - lo_n)


def test_reference_coding_and_perfect_collinearity():
    from src.evaluation import gvif, reference_coding, resolve_perfect_collinearity
    n = 600
    a = rng.choice(["x", "y", "z"], n, p=[0.5, 0.3, 0.2])
    b = np.where(a == "x", "p", "q")                     # b sepenuhnya ditentukan oleh a
    num = rng.normal(size=n)
    Z = pd.get_dummies(pd.DataFrame({"a": a, "b": b}), dtype=float).assign(num=num)
    D, groups, refs = reference_coding(Z, {"a": "a_", "b": "b_", "num": "num"})
    expected_b = "b_p" if (b == "p").sum() >= (b == "q").sum() else "b_q"   # acuan = kategori terbanyak
    assert refs == {"a": "a_x", "b": expected_b} and "a_x" not in D
    D2, groups2, dropped = resolve_perfect_collinearity(D, groups)
    assert dropped == ["b"] and set(groups2) == {"a", "num"}
    g = gvif(D2, groups2)
    assert (g["GVIF_adj"] < 1.2).all()


def test_gvif_matches_eigenvalue_formula():
    from src.evaluation import gvif
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    X["c"] = X["a"] * 0.8 + rng.normal(scale=0.3, size=500)
    groups = {"g1": ["a", "b"], "g2": ["c"], "g3": ["d"]}
    R = np.corrcoef(X.to_numpy(), rowvar=False)
    ld = lambda M: np.sum(np.log(np.linalg.eigvalsh(M)))  # noqa: E731
    expected = np.exp(ld(R[np.ix_([0, 1], [0, 1])]) + ld(R[np.ix_([2, 3], [2, 3])]) - ld(R))
    assert gvif(X, groups).loc["g1", "GVIF"] == pytest.approx(expected, rel=1e-9)
