"""Feature engineering yang ikut tersimpan di dalam pipeline model.

Modul ini wajib bisa di-import saat model di-load (notebook, API, dan Docker),
karena `joblib` menyimpan referensi ke class di sini.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "years_employed", "debt_to_income", "num_credit_lines", "num_late_payments_12m",
    "loan_amount", "loan_term_months", "interest_rate",
]
CATEGORICAL_COLS = [
    "occupation", "employment_type", "employee_status",
    "marital_status", "education", "home_ownership", "loan_purpose",
]

# Binning berbasis domain: batas ditentukan aturan bisnis, bukan dari data
DOMAIN_BINS = {
    "age": {
        "edges": [-np.inf, 25, 35, 45, 55, np.inf],
        "labels": ["18-25", "26-35", "36-45", "46-55", "56+"],
    },
    "credit_score": {  # rentang skor ala FICO
        "edges": [-np.inf, 579, 669, 739, 799, np.inf],
        "labels": ["poor", "fair", "good", "very_good", "excellent"],
    },
    "monthly_salary": {  # juta Rp/bulan; 3,5 jt ~ kisaran UMR, 0 = tidak berpenghasilan
        "edges": [-np.inf, 0, 3.5, 5, 10, 20, np.inf],
        "labels": ["no_salary", "below_3.5m", "3.5m-5m", "5m-10m", "10m-20m", "above_20m"],
    },
}
# Binning berbasis data: kuantil (distribusi pendapatan sangat skewed)
QUANTILE_BIN_COLS = ["annual_income"]
QUANTILE_N_BINS = 5

FEATURES = list(DOMAIN_BINS) + QUANTILE_BIN_COLS + NUMERIC_COLS + CATEGORICAL_COLS


class DomainBinner(BaseEstimator, TransformerMixin):
    """Ubah kolom numerik jadi kategori berdasarkan batas tetap (pd.cut).

    Nilai kosong diberi label "missing" agar tetap menjadi informasi bagi model.
    """

    def __init__(self, bins: dict):
        self.bins = bins

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(self.bins))
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = pd.DataFrame(index=X.index)
        for col, spec in self.bins.items():
            binned = pd.cut(X[col].astype(float), bins=spec["edges"], labels=spec["labels"])
            out[f"{col}_bin"] = binned.astype(object).where(binned.notna(), "missing")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_bin" for c in self.bins])


def build_preprocessor() -> ColumnTransformer:
    domain_bin = Pipeline([
        ("bin", DomainBinner(DOMAIN_BINS)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    quantile_bin = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("bin", KBinsDiscretizer(n_bins=QUANTILE_N_BINS, encode="onehot-dense", strategy="quantile")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("domain_bin", domain_bin, list(DOMAIN_BINS)),
            ("quantile_bin", quantile_bin, QUANTILE_BIN_COLS),
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
