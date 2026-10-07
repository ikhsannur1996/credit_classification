"""FastAPI service untuk model prediksi gagal bayar kredit (credit default)."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LoanApplication,
    Prediction,
)
from src.json_model import JsonModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.json"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("credit-default-api")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = JsonModel.load(MODEL_PATH)
    state["model"] = model
    state["metadata"] = model.metadata

    if set(LoanApplication.model_fields) != set(state["metadata"]["features"]):
        raise RuntimeError("Field di schemas.py tidak sama dengan fitur di models/model.json")

    logger.info(
        "Model JSON '%s' versi %s dimuat dari %s (threshold=%s)",
        state["metadata"]["model_name"], state["metadata"]["model_version"], MODEL_PATH, state["metadata"]["threshold"],
    )
    yield
    state.clear()


app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Prediksi probabilitas gagal bayar dari 18 fitur aplikasi kredit (termasuk pekerjaan, gaji, status karyawan). "
        "Binning (umur, skor kredit, gaji, pendapatan) dan one-hot encoding dilakukan di dalam pipeline model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_band(p: float, threshold: float) -> str:
    if p >= threshold:
        return "high"
    return "medium" if p >= threshold / 2 else "low"


def _predict(rows: list[LoanApplication]) -> list[Prediction]:
    meta = state["metadata"]
    X = pd.DataFrame([r.model_dump() for r in rows])[meta["features"]]
    X = X.fillna(value=np.nan)  # None -> NaN supaya ditangani imputer
    for col in meta["numeric_features"]:
        X[col] = X[col].astype(float)

    try:
        proba = state["model"].predict_proba(X)[:, 1]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inferensi gagal")
        raise HTTPException(status_code=500, detail="Inferensi model gagal") from exc

    threshold = meta["threshold"]
    return [
        Prediction(
            prediction=int(p >= threshold),
            label=meta["labels"][str(int(p >= threshold))],
            probability_default=round(float(p), 6),
            risk_band=_risk_band(float(p), threshold),
            threshold=threshold,
        )
        for p in proba
    ]


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Credit Default Prediction API. Buka /docs untuk dokumentasi."}


@app.get("/health", response_model=HealthResponse)
def health():
    meta = state.get("metadata")
    return HealthResponse(
        status="ok",
        model_loaded="model" in state,
        model_version=meta["model_version"] if meta else None,
    )


@app.get("/model-info")
def model_info():
    meta = state["metadata"]
    info = {k: meta[k] for k in (
        "model_name", "model_version", "trained_at", "threshold", "labels", "selection_metric",
        "compared_models", "cv_metrics", "test_metrics", "features", "categorical_features",
        "n_features_after_preprocessing",
    )}
    preprocessing = {step["name"]: step for step in state["model"].spec["preprocessing"]}
    info["binning"] = {
        "domain": preprocessing["domain_bin"]["bins"],
        "quantile": dict(zip(preprocessing["quantile_bin"]["columns"], preprocessing["quantile_bin"]["bin_edges"])),
    }
    return info


@app.post("/predict", response_model=Prediction)
def predict(payload: LoanApplication):
    return _predict([payload])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest):
    return BatchResponse(
        model_version=state["metadata"]["model_version"],
        predictions=_predict(payload.instances),
    )
