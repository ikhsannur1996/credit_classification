"""Serialisasi pipeline model ke JSON + inferensi tanpa scikit-learn.

`export_pipeline(pipeline, metadata)` mengubah pipeline hasil training
(ColumnTransformer dari `src/features.py` + LogisticRegression / RandomForestClassifier)
menjadi dict yang bisa disimpan sebagai JSON.

`JsonModel` membaca file JSON tersebut dan menghitung `predict_proba` hanya dengan
numpy & pandas, sehingga API tidak bergantung pada pickle/joblib maupun versi scikit-learn.

Format JSON:
{
  "format": "credit-default-json-model", "format_version": 1,
  "metadata": {...},
  "preprocessing": [ {"name": ..., "type": ..., "columns": [...], ...}, ... ],  # urutan = urutan kolom output
  "feature_names_out": [...],
  "model": {"type": "logistic_regression", "coef": [...], "intercept": ...}
         | {"type": "random_forest", "trees": [{"children_left": [...], ...}, ...]}
}
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_NAME = "credit-default-json-model"
FORMAT_VERSION = 1


# ----------------------------------------------------------------------------- helpers
def _edges_to_json(edges) -> list:
    """JSON tidak mendukung inf, jadi -inf/inf disimpan sebagai null."""
    return [None if np.isinf(e) else float(e) for e in edges]


def _edges_from_json(edges) -> np.ndarray:
    out = np.array([np.nan if e is None else e for e in edges], dtype=float)
    out[0] = -np.inf if np.isnan(out[0]) else out[0]
    out[-1] = np.inf if np.isnan(out[-1]) else out[-1]
    return out


def _to_list(arr) -> list:
    return np.asarray(arr).tolist()


# ----------------------------------------------------------------------------- export
def _export_preprocessor(ct) -> list[dict]:
    steps = []
    for name, transformer, columns in ct.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        columns = list(columns)
        if name == "domain_bin":
            binner, onehot = transformer.named_steps["bin"], transformer.named_steps["onehot"]
            steps.append({
                "name": name,
                "type": "domain_bin_onehot",
                "columns": columns,
                "bins": {c: {"edges": _edges_to_json(binner.bins[c]["edges"]), "labels": list(binner.bins[c]["labels"])}
                         for c in columns},
                "categories": [_to_list(c) for c in onehot.categories_],
            })
        elif name == "quantile_bin":
            imputer, kbins = transformer.named_steps["imputer"], transformer.named_steps["bin"]
            steps.append({
                "name": name,
                "type": "quantile_bin_onehot",
                "columns": columns,
                "impute_values": _to_list(imputer.statistics_),
                "bin_edges": [_to_list(e) for e in kbins.bin_edges_],
            })
        elif name == "num":
            imputer, scaler = transformer.named_steps["imputer"], transformer.named_steps["scaler"]
            steps.append({
                "name": name,
                "type": "impute_standard_scale",
                "columns": columns,
                "impute_values": _to_list(imputer.statistics_),
                "mean": _to_list(scaler.mean_),
                "scale": _to_list(scaler.scale_),
            })
        elif name == "cat":
            imputer, onehot = transformer.named_steps["imputer"], transformer.named_steps["onehot"]
            steps.append({
                "name": name,
                "type": "impute_onehot",
                "columns": columns,
                "fill_value": imputer.fill_value,
                "categories": [_to_list(c) for c in onehot.categories_],
            })
        else:
            raise ValueError(f"Transformer '{name}' belum didukung untuk ekspor JSON")
    return steps


def _export_tree(tree) -> dict:
    t = tree.tree_
    value = t.value[:, 0, :]
    proba = value / value.sum(axis=1, keepdims=True)
    return {
        "children_left": _to_list(t.children_left),
        "children_right": _to_list(t.children_right),
        "feature": _to_list(t.feature),
        "threshold": _to_list(t.threshold),
        "proba": _to_list(proba[:, 1]),  # P(kelas 1) di setiap node
    }


def _export_estimator(est) -> dict:
    kind = type(est).__name__
    if kind == "LogisticRegression":
        return {"type": "logistic_regression", "coef": _to_list(est.coef_[0]), "intercept": float(est.intercept_[0])}
    if kind == "RandomForestClassifier":
        return {"type": "random_forest", "trees": [_export_tree(t) for t in est.estimators_]}
    raise ValueError(f"Model '{kind}' belum didukung untuk ekspor JSON")


def export_pipeline(pipeline, metadata: dict) -> dict:
    ct = pipeline.named_steps["preprocess"]
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "preprocessing": _export_preprocessor(ct),
        "feature_names_out": _to_list(ct.get_feature_names_out()),
        "model": _export_estimator(pipeline.named_steps["model"]),
    }


def save_json(model_dict: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(model_dict, indent=1, allow_nan=False))


# ----------------------------------------------------------------------------- inference
def _onehot(values: pd.Series, categories: list) -> np.ndarray:
    """One-hot dengan handle_unknown='ignore': kategori tak dikenal -> semua 0."""
    cats = np.asarray(categories, dtype=object)
    return (values.to_numpy(dtype=object)[:, None] == cats[None, :]).astype(float)


class JsonModel:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT_NAME:
            raise ValueError("Bukan file model JSON yang valid")
        if spec.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"format_version {spec.get('format_version')} tidak didukung")
        self.spec = spec
        self.metadata = spec["metadata"]
        self.features = self.metadata["features"]
        self.feature_names_out = spec["feature_names_out"]
        self._model = spec["model"]
        if self._model["type"] == "random_forest":
            self._trees = [{k: np.asarray(v) for k, v in t.items()} for t in self._model["trees"]]

    @classmethod
    def load(cls, path) -> "JsonModel":
        return cls(json.loads(Path(path).read_text()))

    # --- preprocessing ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for step in self.spec["preprocessing"]:
            kind, cols = step["type"], step["columns"]
            if kind == "domain_bin_onehot":
                for col, cats in zip(cols, step["categories"]):
                    spec = step["bins"][col]
                    binned = pd.cut(X[col].astype(float), bins=_edges_from_json(spec["edges"]), labels=spec["labels"])
                    binned = binned.astype(object).where(binned.notna(), "missing")
                    blocks.append(_onehot(binned, cats))
            elif kind == "quantile_bin_onehot":
                for col, fill, edges in zip(cols, step["impute_values"], step["bin_edges"]):
                    x = X[col].astype(float).fillna(fill).to_numpy()
                    edges = np.asarray(edges)
                    # sama seperti KBinsDiscretizer: toleransi numerik lalu searchsorted
                    eps = 1e-8 + 1e-5 * np.abs(x)
                    idx = np.clip(np.searchsorted(edges[1:-1], x + eps, side="right"), 0, len(edges) - 2)
                    blocks.append(np.eye(len(edges) - 1)[idx])
            elif kind == "impute_standard_scale":
                x = X[cols].astype(float).to_numpy()
                fill = np.asarray(step["impute_values"])
                x = np.where(np.isnan(x), fill, x)
                blocks.append((x - np.asarray(step["mean"])) / np.asarray(step["scale"]))
            elif kind == "impute_onehot":
                for col, cats in zip(cols, step["categories"]):
                    values = X[col].astype(object).where(X[col].notna(), step["fill_value"])
                    blocks.append(_onehot(values, cats))
            else:
                raise ValueError(f"Tipe preprocessing '{kind}' tidak dikenal")
        out = np.hstack(blocks)
        assert out.shape[1] == len(self.feature_names_out), "Jumlah kolom hasil transformasi tidak cocok"
        return out

    # --- model -----------------------------------------------------------------
    def _tree_proba(self, tree: dict, Xt: np.ndarray) -> np.ndarray:
        node = np.zeros(len(Xt), dtype=np.int64)
        rows = np.arange(len(Xt))
        while True:
            left = tree["children_left"][node]
            active = left != -1
            if not active.any():
                return tree["proba"][node]
            r, n = rows[active], node[active]
            go_left = Xt[r, tree["feature"][n]] <= tree["threshold"][n]
            node[r] = np.where(go_left, tree["children_left"][n], tree["children_right"][n])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return array shape (n, 2) seperti scikit-learn: [P(0), P(1)]."""
        Xt = self.transform(X[self.features])
        if self._model["type"] == "logistic_regression":
            z = Xt @ np.asarray(self._model["coef"]) + self._model["intercept"]
            p1 = 1.0 / (1.0 + np.exp(-z))
        else:
            Xt32 = Xt.astype(np.float32).astype(np.float64)  # tree sklearn membandingkan dalam float32
            p1 = np.mean([self._tree_proba(t, Xt32) for t in self._trees], axis=0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.metadata["threshold"]).astype(int)
