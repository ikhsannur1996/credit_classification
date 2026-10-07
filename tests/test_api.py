import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())  # sampel risiko tertinggi di test set
BATCH = json.loads((ROOT / "sample_batch_request.json").read_text())  # [risiko tinggi, risiko rendah]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] in body["compared_models"]
    assert len(body["features"]) == 18
    assert set(body["binning"]["domain"]) == {"age", "credit_score", "monthly_salary"}


def test_predict_high_risk(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "default"
    assert body["risk_band"] == "high"
    assert 0.0 <= body["probability_default"] <= 1.0


def test_predict_batch(client):
    r = client.post("/predict/batch", json=BATCH)
    assert r.status_code == 200
    labels = [p["label"] for p in r.json()["predictions"]]
    assert labels == ["default", "no_default"]


def test_nullable_fields_are_imputed(client):
    payload = {**SAMPLE, "credit_score": None, "years_employed": None, "education": None, "monthly_salary": None}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200


def test_missing_required_feature_rejected(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "annual_income"}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_category_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "occupation": "astronaut"}).status_code == 422


def test_out_of_range_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "credit_score": 900}).status_code == 422


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "customer_id": "C001"}).status_code == 422


def test_inconsistent_employment_rejected(client):
    unemployed_but_permanent = {
        **SAMPLE, "occupation": "unemployed", "employment_type": "unemployed", "employee_status": "permanent",
    }
    assert client.post("/predict", json=unemployed_but_permanent).status_code == 422

    salaried_without_status = {
        **SAMPLE, "occupation": "private_employee", "employment_type": "salaried", "employee_status": "not_applicable",
    }
    assert client.post("/predict", json=salaried_without_status).status_code == 422

    job_type_mismatch = {**SAMPLE, "occupation": "unemployed", "employment_type": "salaried"}
    assert client.post("/predict", json=job_type_mismatch).status_code == 422
