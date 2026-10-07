"""Fungsi evaluasi statistik untuk model risiko kredit.

Dipakai di notebook (Bagian uji statistik fitur, cross-validation lanjutan,
evaluasi statistik test set, dan backtesting). Hanya bergantung pada numpy,
pandas, scipy, dan scikit-learn.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================================ metrik dasar
def ks_statistic(y_true, proba) -> float:
    """Kolmogorov-Smirnov: jarak maksimum antara CDF skor kelas default vs non-default."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    return float(stats.ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic)


def gini(y_true, proba) -> float:
    """Gini (Accuracy Ratio) = 2 * AUC - 1, metrik standar credit scoring."""
    return 2 * roc_auc_score(y_true, proba) - 1


def classification_metrics(y_true, proba, threshold: float) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "gini": gini(y_true, proba),
        "ks": ks_statistic(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
    }


# ============================================================================ bootstrap
def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Confidence interval metrik dengan bootstrap percentile (resample baris dengan pengembalian).

    Resample dibuat stratified per kelas agar setiap sampel bootstrap punya kedua kelas
    dengan proporsi yang sama seperti data asli.
    """
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    point = classification_metrics(y_true, proba, threshold)
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        samples.append(classification_metrics(y_true[idx], proba[idx], threshold))
    boot = pd.DataFrame(samples)
    return pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_lower": boot.quantile(alpha / 2),
        "ci_upper": boot.quantile(1 - alpha / 2),
        "std_error": boot.std(),
    })


# ============================================================================ DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y_true, scores_list):
    y_true = np.asarray(y_true)
    pos, neg = scores_list[:, y_true == 1], scores_list[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores_list.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y_true, proba_a, proba_b) -> dict:
    """Uji DeLong (1988): apakah AUC dua model pada data yang SAMA berbeda signifikan.

    Memperhitungkan korelasi antar prediksi kedua model (data berpasangan).
    """
    aucs, cov = _delong_components(y_true, np.vstack([proba_a, proba_b]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-15:  # prediksi identik -> tidak ada perbedaan
        return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "z": 0.0, "p_value": 1.0,
                "ci_lower": diff, "ci_upper": diff}
    se = np.sqrt(var)
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "z": z, "p_value": p,
            "ci_lower": diff - 1.96 * se, "ci_upper": diff + 1.96 * se}


def delong_auc_ci(y_true, proba, alpha: float = 0.05) -> tuple[float, float, float]:
    aucs, cov = _delong_components(y_true, np.atleast_2d(proba))
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], aucs[0] - z * se, aucs[0] + z * se


# ============================================================================ McNemar
def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Uji McNemar (dengan koreksi kontinuitas): apakah tingkat kesalahan dua model berbeda.

    b = A benar & B salah, c = A salah & B benar. Hanya pasangan yang berbeda yang informatif.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": chi2, "p_value": stats.chi2.sf(chi2, 1)}


# ============================================================================ CV comparison
def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio, 2003) untuk skor repeated K-fold berpasangan.

    t-test biasa terlalu optimis karena fold CV saling tumpang tindih (data train dipakai ulang).
    Koreksi: variance dikalikan (1/k + n_test/n_train).
    """
    d = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": d.mean(), "t": t, "df": k - 1, "p_value": p}


def mean_ci(values, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean dan confidence interval berbasis distribusi t."""
    v = np.asarray(values)
    half = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def corrected_mean_ci(values, n_train: int, n_test: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """CI untuk rata-rata skor repeated K-fold dengan koreksi Nadeau-Bengio.

    CI t-biasa terlalu sempit karena fold saling tumpang tindih; standard error dikoreksi
    dengan faktor sqrt(1/k + n_test/n_train).
    """
    v = np.asarray(values)
    k = len(v)
    se = np.sqrt((1 / k + n_test / n_train) * v.var(ddof=1))
    half = stats.t.ppf(1 - alpha / 2, k - 1) * se
    return v.mean(), v.mean() - half, v.mean() + half


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    ci = stats.binomtest(int(successes), int(n)).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return ci.low, ci.high


# ============================================================================ calibration
def hosmer_lemeshow(y_true, proba, n_groups: int = 10) -> dict:
    """Uji Hosmer-Lemeshow: H0 = model terkalibrasi (PD prediksi = default rate aktual per grup)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["group"] = pd.qcut(df["p"].rank(method="first"), n_groups, labels=False)
    g = df.groupby("group").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"), mean_pd=("p", "mean"))
    g["observed_rate"] = g["observed"] / g["n"]
    num = (g["observed"] - g["expected"]) ** 2
    den = g["expected"] * (1 - g["expected"] / g["n"])
    hl = float((num / den).sum())
    dof = n_groups - 2
    return {"statistic": hl, "df": dof, "p_value": stats.chi2.sf(hl, dof), "table": g}


def calibration_slope_intercept(y_true, proba) -> dict:
    """Regresi logistik y ~ logit(p). Ideal: slope = 1 dan intercept = 0.

    slope < 1: prediksi terlalu ekstrem (overfit); slope > 1: prediksi terlalu 'malu-malu'.
    intercept > 0: model under-predict risiko; intercept < 0: over-predict.
    """
    p = np.clip(np.asarray(proba), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y_true)
    # intercept "calibration-in-the-large": slope dikunci 1 (offset)
    intercept_large = np.log(np.mean(y_true) / (1 - np.mean(y_true))) - np.log(np.mean(p) / (1 - np.mean(p)))
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
            "calibration_in_the_large": float(intercept_large)}


def binomial_backtest(n: int, defaults: int, mean_pd: float) -> dict:
    """Uji binomial (Basel): apakah jumlah default aktual konsisten dengan PD rata-rata.

    Traffic light: hijau p >= 0.05, kuning 0.01 <= p < 0.05, merah p < 0.01.
    """
    p_two = stats.binomtest(defaults, n, mean_pd, alternative="two-sided").pvalue
    p_under = stats.binomtest(defaults, n, mean_pd, alternative="greater").pvalue  # PD terlalu rendah?
    light = "green" if p_two >= 0.05 else ("yellow" if p_two >= 0.01 else "red")
    return {"n": n, "defaults": defaults, "observed_dr": defaults / n, "mean_pd": mean_pd,
            "p_two_sided": p_two, "p_underestimate": p_under, "traffic_light": light}


# ============================================================================ stability
def psi(expected, actual, bins: int = 10, edges=None) -> float:
    """Population Stability Index. <0.1 stabil, 0.1-0.25 perlu perhatian, >0.25 bergeser signifikan."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).fillna("missing").value_counts(normalize=True)
    a = pd.Series(actual).fillna("missing").value_counts(normalize=True)
    cats = e.index.union(a.index)
    e, a = e.reindex(cats, fill_value=1e-6).clip(lower=1e-6), a.reindex(cats, fill_value=1e-6).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def stability_label(value: float) -> str:
    return "stabil" if value < 0.1 else ("perlu perhatian" if value < 0.25 else "bergeser signifikan")


# ============================================================================ feature statistics
def woe_iv(feature: pd.Series, target: pd.Series, bins: int = 5) -> tuple[pd.DataFrame, float]:
    """Weight of Evidence & Information Value.

    Numerik -> dibagi kuantil; kategorikal -> per kategori; NaN -> grup 'missing'.
    WoE = ln(%non-default / %default). IV = sum((%non-default - %default) * WoE).
    """
    if pd.api.types.is_numeric_dtype(feature) and feature.nunique() > bins:
        grp = pd.qcut(feature, bins, duplicates="drop")  # tetap kategorikal agar urutan bin terjaga
        grp = grp.cat.rename_categories([str(c) for c in grp.cat.categories]).cat.add_categories("missing").fillna("missing")
    else:
        grp = feature.astype(object).where(feature.notna(), "missing")
    t = pd.crosstab(grp, target)
    t.columns = ["good", "bad"]
    dist_good = (t["good"] / t["good"].sum()).clip(lower=1e-6)
    dist_bad = (t["bad"] / t["bad"].sum()).clip(lower=1e-6)
    t["bad_rate"] = t["bad"] / (t["good"] + t["bad"])
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t, float(t["iv"].sum())


def iv_label(iv: float) -> str:
    if iv < 0.02:
        return "tidak prediktif"
    if iv < 0.1:
        return "lemah"
    if iv < 0.3:
        return "sedang"
    if iv < 0.5:
        return "kuat"
    return "sangat kuat (cek leakage)"


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Chi-square test of independence + Cramér's V (besar efek 0-1)."""
    table = pd.crosstab(x.fillna("missing"), y)
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, p, v


def mann_whitney(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Mann-Whitney U (non-parametrik) + rank-biserial correlation sebagai besar efek (-1..1).

    Positif = nilai fitur cenderung LEBIH TINGGI pada nasabah default.
    """
    a, b = x[y == 1].dropna(), x[y == 0].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    return u, p, r


def vif(df_numeric: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor. >5 multikolinearitas sedang, >10 tinggi."""
    X = df_numeric.fillna(df_numeric.median())
    out = {}
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out, name="VIF").sort_values(ascending=False)


def holm_correction(p_values: pd.Series) -> pd.Series:
    """Koreksi Holm-Bonferroni untuk banyak uji sekaligus (mengontrol family-wise error)."""
    p = p_values.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(p.values)])
    return pd.Series(adj, index=p.index).reindex(p_values.index)
