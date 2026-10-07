"""Generate dataset sintetis credit default (18 fitur + target) -> data/credit_default.csv.

Pola risiko dibuat mengikuti logika perbankan: skor kredit rendah, DTI tinggi,
riwayat telat bayar, rasio pinjaman/penghasilan besar, pekerjaan & status kerja
yang tidak stabil menaikkan peluang gagal bayar.

Hubungan antar kolom pekerjaan dijaga konsisten:
    occupation      -> employment_type (salaried / self_employed / unemployed)
    employment_type -> employee_status (hanya karyawan yang punya status tetap/kontrak/dst.)
    occupation + education + pengalaman -> monthly_salary -> annual_income

    python scripts/generate_data.py --n 10000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "credit_default.csv"

# pekerjaan: (proporsi, gaji dasar juta Rp/bulan, efek risiko)
OCCUPATIONS = {
    "civil_servant":    (0.12, 6.5, -0.50),  # PNS / ASN / TNI / Polri
    "soe_employee":     (0.07, 9.0, -0.35),  # karyawan BUMN
    "private_employee": (0.25, 8.0, 0.00),   # karyawan swasta (kantor)
    "professional":     (0.08, 16.0, -0.30), # dokter, pengacara, akuntan, engineer
    "factory_worker":   (0.12, 4.8, 0.15),   # buruh pabrik / manufaktur
    "retail_service":   (0.10, 4.2, 0.25),   # retail, F&B, sales, jasa
    "driver_ojol":      (0.08, 4.5, 0.35),   # pengemudi / ojek online (mitra)
    "entrepreneur":     (0.12, 11.0, 0.30),  # wiraswasta / UMKM
    "unemployed":       (0.06, 0.0, 1.00),   # tidak bekerja
}

# status karyawan per pekerjaan (hanya untuk employment_type = salaried)
EMPLOYEE_STATUS_P = {
    "civil_servant":    {"permanent": 1.00},
    "soe_employee":     {"permanent": 0.85, "contract": 0.15},
    "private_employee": {"permanent": 0.55, "contract": 0.30, "probation": 0.08, "outsourcing": 0.07},
    "professional":     {"permanent": 0.75, "contract": 0.25},
    "factory_worker":   {"permanent": 0.35, "contract": 0.40, "outsourcing": 0.25},
    "retail_service":   {"permanent": 0.30, "contract": 0.45, "probation": 0.10, "outsourcing": 0.15},
}
STATUS_RISK = {"permanent": -0.20, "contract": 0.25, "probation": 0.35, "outsourcing": 0.40, "not_applicable": 0.0}


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda options, probs: rng.choice(options, n, p=probs)  # noqa: E731

    age = rng.integers(21, 66, n)
    education = pick(["high_school", "diploma", "bachelor", "master"], [0.30, 0.20, 0.38, 0.12])
    marital_status = pick(["single", "married", "divorced"], [0.38, 0.52, 0.10])
    home_ownership = pick(["rent", "mortgage", "own"], [0.42, 0.38, 0.20])
    loan_purpose = pick(["debt_consolidation", "home_improvement", "business", "education", "car", "personal"],
                        [0.28, 0.14, 0.12, 0.10, 0.18, 0.18])

    # --- pekerjaan, jenis & status kerja ----------------------------------------------
    occ_names = list(OCCUPATIONS)
    occupation = pick(occ_names, [OCCUPATIONS[o][0] for o in occ_names])
    # profesional bisa karyawan atau praktik mandiri
    prof_self = (occupation == "professional") & (rng.uniform(0, 1, n) < 0.4)
    employment_type = np.select(
        [occupation == "unemployed", np.isin(occupation, ["driver_ojol", "entrepreneur"]) | prof_self],
        ["unemployed", "self_employed"],
        default="salaried",
    )
    employee_status = np.full(n, "not_applicable", dtype=object)
    for occ, dist in EMPLOYEE_STATUS_P.items():
        mask = (occupation == occ) & (employment_type == "salaried")
        employee_status[mask] = rng.choice(list(dist), mask.sum(), p=list(dist.values()))

    # --- pengalaman kerja ------------------------------------------------------------
    years_employed = np.clip(np.round(rng.uniform(0, 1, n) * (age - 20) * 0.7, 1), 0, None)
    years_employed[employee_status == "probation"] = np.round(rng.uniform(0, 1, (employee_status == "probation").sum()), 1)
    years_employed[employment_type == "unemployed"] = 0

    # --- gaji bulanan & pendapatan tahunan -----------------------------------------
    edu_mult = pd.Series(education).map({"high_school": 0.8, "diploma": 0.95, "bachelor": 1.15, "master": 1.45}).to_numpy()
    base = pd.Series(occupation).map({k: v[1] for k, v in OCCUPATIONS.items()}).to_numpy()
    spread = np.where(employment_type == "self_employed", 0.6, 0.3)  # penghasilan wiraswasta lebih bervariasi
    monthly_salary = np.round(base * edu_mult * (1 + years_employed / 40) * rng.lognormal(0, spread, n), 2)

    bonus_months = pd.Series(employee_status).map(
        {"permanent": 2.0, "contract": 1.0, "probation": 0.5, "outsourcing": 1.0, "not_applicable": 0.0}
    ).to_numpy()
    other_income = rng.lognormal(np.log(8), 0.8, n)  # sewa, usaha sampingan, dll. (juta Rp/tahun)
    annual_income = np.round(monthly_salary * (12 + bonus_months) + other_income, 1)

    # --- profil kredit ---------------------------------------------------------------
    credit_score = np.clip(np.round(rng.normal(680, 70, n) + years_employed * 1.5), 300, 850).astype(int)
    num_late_payments_12m = rng.poisson(np.clip((720 - credit_score) / 60, 0.05, None))
    num_credit_lines = rng.poisson(4, n) + 1
    debt_to_income = np.round(np.clip(rng.beta(2, 5, n) * 0.9 + num_credit_lines * 0.01, 0.01, 0.95), 3)

    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.20, 0.35, 0.15, 0.20])
    loan_amount = np.round(np.clip(annual_income * rng.uniform(0.1, 1.2, n), 5, None), 1)
    interest_rate = np.round(np.clip(22 - (credit_score - 300) / 40 + rng.normal(0, 1.5, n) + loan_term_months / 60, 6, 28), 2)

    # --- target ------------------------------------------------------------------------
    loan_to_income = loan_amount / annual_income
    logit = (
        -2.3
        - (credit_score - 680) / 45
        + 3.0 * (debt_to_income - 0.3)
        + 0.45 * num_late_payments_12m
        + 1.1 * (loan_to_income - 0.6)
        + 0.06 * (interest_rate - 12)
        - 0.04 * years_employed
        - 0.25 * np.log1p(monthly_salary / 5)
        + 0.35 * (age < 26)
        + pd.Series(occupation).map({k: v[2] for k, v in OCCUPATIONS.items()}).to_numpy()
        + pd.Series(employee_status).map(STATUS_RISK).to_numpy()
        + pd.Series(employment_type).map({"salaried": 0.0, "self_employed": 0.2, "unemployed": 0.4}).to_numpy()
        + pd.Series(home_ownership).map({"rent": 0.3, "mortgage": 0.0, "own": -0.3}).to_numpy()
        + pd.Series(loan_purpose).map({"debt_consolidation": 0.3, "home_improvement": -0.1, "business": 0.45,
                                       "education": 0.1, "car": -0.2, "personal": 0.15}).to_numpy()
        + pd.Series(marital_status).map({"single": 0.1, "married": -0.1, "divorced": 0.2}).to_numpy()
        + rng.normal(0, 0.6, n)
    )
    default = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "monthly_salary": monthly_salary,
        "years_employed": years_employed,
        "credit_score": credit_score.astype(float),
        "debt_to_income": debt_to_income,
        "num_credit_lines": num_credit_lines,
        "num_late_payments_12m": num_late_payments_12m,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "occupation": occupation,
        "employment_type": employment_type,
        "employee_status": employee_status,
        "marital_status": marital_status,
        "education": education,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    # missing value realistis (data tidak lengkap / tidak dilaporkan)
    for col, rate in [("years_employed", 0.04), ("credit_score", 0.03), ("education", 0.02), ("monthly_salary", 0.03)]:
        df.loc[rng.uniform(0, 1, n) < rate, col] = np.nan

    df["application_date"] = assign_application_dates(credit_score, occupation, seed)
    return df


def assign_application_dates(credit_score, occupation, seed: int, start="2023-01-01", n_months=36) -> pd.Series:
    """Tanggal pengajuan (BUKAN fitur model, hanya untuk backtesting / out-of-time).

    Memakai RNG terpisah agar nilai kolom lain tidak berubah. Untuk mensimulasikan
    *population drift*, pemohon di periode akhir sedikit lebih berisiko (skor kredit
    lebih rendah, lebih banyak pekerja gig/informal). Hubungan fitur -> default tidak
    berubah, jadi yang bergeser adalah populasinya (covariate shift), bukan konsepnya.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(credit_score)
    gig = np.isin(occupation, ["driver_ojol", "retail_service", "unemployed"]).astype(float)
    drift_score = -(credit_score - credit_score.mean()) / credit_score.std() * 0.12 + gig * 0.15 + rng.normal(0, 1, n)

    month_idx = np.empty(n, dtype=int)
    month_idx[np.argsort(drift_score, kind="stable")] = np.arange(n) * n_months // n  # volume per bulan sama
    month_start = pd.date_range(start, periods=n_months, freq="MS")
    days = rng.integers(0, 28, n)
    dates = month_start[month_idx] + pd.to_timedelta(days, unit="D")
    return pd.Series(dates.strftime("%Y-%m-%d"), name="application_date")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate(args.n, args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"{data.shape} -> {OUT_PATH} | default rate {data['default'].mean():.2%}")
