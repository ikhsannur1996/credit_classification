"""Skema request/response API. Nama field harus sama dengan `metadata.features` di models/model.json."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXAMPLE = {
    "age": 29,
    "annual_income": 96.5,
    "monthly_salary": 6.8,
    "years_employed": 3.0,
    "credit_score": 610,
    "debt_to_income": 0.42,
    "num_credit_lines": 6,
    "num_late_payments_12m": 2,
    "loan_amount": 85.0,
    "loan_term_months": 36,
    "interest_rate": 16.5,
    "occupation": "retail_service",
    "employment_type": "salaried",
    "employee_status": "contract",
    "marital_status": "single",
    "education": "diploma",
    "home_ownership": "rent",
    "loan_purpose": "debt_consolidation",
}


class LoanApplication(BaseModel):
    """18 fitur aplikasi kredit. Binning & one-hot encoding dilakukan otomatis di dalam model."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    age: int = Field(..., ge=18, le=100, description="Umur (tahun), di-binning ke kelompok umur")
    annual_income: float = Field(..., gt=0, description="Pendapatan tahunan (juta Rp), di-binning per kuantil")
    monthly_salary: float | None = Field(
        None, ge=0, le=10_000,
        description="Gaji/penghasilan bulanan (juta Rp), 0 = tidak berpenghasilan; boleh null. Di-binning per rentang UMR",
    )
    years_employed: float | None = Field(None, ge=0, le=60, description="Lama bekerja (tahun); boleh null")
    credit_score: float | None = Field(None, ge=300, le=850, description="Skor kredit; null = tidak punya riwayat")
    debt_to_income: float = Field(..., ge=0, le=1, description="Rasio cicilan / pendapatan")
    num_credit_lines: int = Field(..., ge=0, le=50)
    num_late_payments_12m: int = Field(..., ge=0, le=100)
    loan_amount: float = Field(..., gt=0, description="Plafon pinjaman (juta Rp)")
    loan_term_months: int = Field(..., ge=1, le=360)
    interest_rate: float = Field(..., ge=0, le=100, description="Suku bunga (% p.a.)")
    occupation: Literal[
        "civil_servant", "soe_employee", "private_employee", "professional", "factory_worker",
        "retail_service", "driver_ojol", "entrepreneur", "unemployed",
    ] = Field(..., description="Pekerjaan / sektor")
    employment_type: Literal["salaried", "self_employed", "unemployed"]
    employee_status: Literal["permanent", "contract", "probation", "outsourcing", "not_applicable"] = Field(
        ..., description="Status karyawan; 'not_applicable' untuk wiraswasta / tidak bekerja"
    )
    marital_status: Literal["single", "married", "divorced"]
    education: Literal["high_school", "diploma", "bachelor", "master"] | None = None
    home_ownership: Literal["rent", "mortgage", "own"]
    loan_purpose: Literal["debt_consolidation", "home_improvement", "business", "education", "car", "personal"]


    @model_validator(mode="after")
    def check_employment_consistency(self):
        if (self.occupation == "unemployed") != (self.employment_type == "unemployed"):
            raise ValueError("occupation 'unemployed' harus berpasangan dengan employment_type 'unemployed'")
        if (self.employment_type == "salaried") == (self.employee_status == "not_applicable"):
            raise ValueError(
                "employee_status wajib diisi (permanent/contract/probation/outsourcing) untuk karyawan 'salaried', "
                "dan harus 'not_applicable' untuk self_employed / unemployed"
            )
        return self


class BatchRequest(BaseModel):
    instances: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = default (gagal bayar), 0 = no_default")
    label: str
    probability_default: float
    risk_band: Literal["low", "medium", "high"]
    threshold: float


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
