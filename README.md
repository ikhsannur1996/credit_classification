# Prediksi Gagal Bayar Kredit (Credit Default): End-to-End

Proyek klasifikasi perbankan dari notebook sampai API: memprediksi apakah pemohon pinjaman akan **gagal bayar (default)**. Dua model dibandingkan, model terbaik disajikan lewat **FastAPI**, lalu dijalankan di **Docker lokal**.

**Highlight:**
- Dataset **18 kolom fitur** (11 numerik + 7 kategorikal) + target `default`, 10.000 baris, default rate ~23%
- Termasuk data **pekerjaan** (`occupation`), **gaji bulanan** (`monthly_salary`), dan **status karyawan** (`employee_status`), dengan kombinasi yang divalidasi konsisten di API
- **Binning** dan **one-hot encoding** ada di dalam `sklearn.Pipeline`, jadi API cukup menerima data mentah
- Missing value ditangani di pipeline (API menerima `null` untuk `credit_score`, `monthly_salary`, `years_employed`, `education`)
- Threshold keputusan dituning (bukan default 0.5)
- **Evaluasi statistik lengkap**: uji fitur (Mann-Whitney, χ², IV/WoE, VIF), cross-validation lanjutan (repeated 5×5, nested, uji signifikansi), bootstrap CI, uji DeLong & McNemar, kalibrasi (Hosmer-Lemeshow, Brier), KS/Gini
- **Backtesting**: out-of-time, walk-forward 8 kuartal, uji binomial per grade (traffic light Basel), PSI/CSI
- **Model disimpan dalam format JSON** (`models/model.json`, ~11 KB, termasuk ringkasan evaluasi): bukan pickle/joblib, dan API tidak butuh scikit-learn

## Hasil

| Model | CV ROC-AUC (5-fold) |
|---|---|
| **Logistic Regression** (C=10) ✅ terpilih | **0.8819** |
| Random Forest | 0.8762 |

Model terpilih di **test set** (20%, threshold 0.30): ROC-AUC 0.884 · PR-AUC 0.722 · Recall 0.714 · Precision 0.606 · F1 0.655 · Accuracy 0.827.

> Model dipilih dari skor **cross-validation**, threshold dipilih dari prediksi **out-of-fold** data train. Test set hanya dipakai sekali di akhir.

### Ringkasan Evaluasi Statistik (22 ✅ · 2 ⚠️ · 0 ❌)

| Aspek | Hasil | 95% CI / p-value | Status |
|---|---|---|---|
| Repeated CV 5×5 ROC-AUC | 0.8812 ± 0.0060 | 0.875–0.888 (terkoreksi) | ✅ stabil |
| Nested CV ROC-AUC | 0.8808 (optimisme +0.001) | 0.864–0.898 | ✅ tidak optimis |
| LR vs RF (repeated CV, corrected t-test) | Δ AUC +0.0048, LR menang 24/25 fold | p = 0.016 | ✅ signifikan |
| LR vs RF (test set, DeLong / McNemar) | Δ AUC +0.0058 | p = 0.062 / 0.79 | ⚠️ tidak signifikan (selisih kecil) |
| Gini / KS (test) | 0.768 / 0.610 | 0.736–0.801 / 0.581–0.651 | ✅ sangat baik |
| Kalibrasi (test) | PD 22.6% vs DR 23.1%, slope 0.97, BSS 0.38 | Hosmer-Lemeshow p = 0.16 | ✅ terkalibrasi |
| Out-of-time 2025Q3–Q4 | AUC 0.871 (in-time 0.884) | penurunan p = 0.23 | ✅ tidak signifikan |
| Kalibrasi OOT (uji binomial) | 457 default aktual vs 449 diharapkan | p = 0.66 | ✅ hijau |
| Walk-forward 8 kuartal | AUC 0.868–0.889, tanpa tren | slope p = 0.94 | ✅ 8/8 hijau |
| Stabilitas (PSI skor / CSI maks) | 0.034 / 0.040 | < 0.1 | ✅ stabil |
| Multikolinearitas gaji–pendapatan | VIF 25 | > 10 | ⚠️ hanya memengaruhi interpretasi |

Rincian setiap uji, cara membacanya, dan rekomendasi monitoring ada di notebook Bagian 4, 10, 14, 16, dan 17.

## Fitur & Feature Engineering

| # | Kolom | Tipe | Perlakuan di pipeline |
|---|---|---|---|
| 1 | `age` | numerik | **Binning domain** → `18-25 / 26-35 / 36-45 / 46-55 / 56+` → one-hot |
| 2 | `credit_score` (300–850) | numerik | **Binning domain** → `poor / fair / good / very_good / excellent / missing` → one-hot |
| 3 | `monthly_salary` 🆕 (gaji, juta Rp/bulan) | numerik | **Binning domain** (rentang UMR) → `no_salary / below_3.5m / 3.5m-5m / 5m-10m / 10m-20m / above_20m / missing` → one-hot |
| 4 | `annual_income` (juta Rp/tahun) | numerik | Imputasi → **binning kuantil** (5 bin, `KBinsDiscretizer`) → one-hot |
| 5 | `years_employed` | numerik | Imputasi median → scaling |
| 6 | `debt_to_income` (0–1) | numerik | Scaling |
| 7 | `num_credit_lines` | numerik | Scaling |
| 8 | `num_late_payments_12m` | numerik | Scaling |
| 9 | `loan_amount` (juta Rp) | numerik | Scaling |
| 10 | `loan_term_months` | numerik | Scaling |
| 11 | `interest_rate` (% p.a.) | numerik | Scaling |
| 12 | `occupation` 🆕 (pekerjaan) | kategorikal | **One-hot**: civil_servant (PNS/TNI/Polri), soe_employee (BUMN), private_employee, professional, factory_worker, retail_service, driver_ojol, entrepreneur, unemployed |
| 13 | `employment_type` | kategorikal | **One-hot**: salaried, self_employed, unemployed |
| 14 | `employee_status` 🆕 (status karyawan) | kategorikal | **One-hot**: permanent (tetap/PKWTT), contract (PKWT), probation, outsourcing, not_applicable |
| 15 | `marital_status` | kategorikal | **One-hot**: single, married, divorced |
| 16 | `education` | kategorikal | Imputasi `missing` → **one-hot**: high_school, diploma, bachelor, master |
| 17 | `home_ownership` | kategorikal | **One-hot**: rent, mortgage, own |
| 18 | `loan_purpose` | kategorikal | **One-hot**: debt_consolidation, home_improvement, business, education, car, personal |

**Aturan konsistensi pekerjaan** (divalidasi API, kalau dilanggar → HTTP 422):
- `occupation = unemployed` ⇔ `employment_type = unemployed`
- `employee_status` wajib diisi (permanent/contract/probation/outsourcing) untuk `salaried`, dan harus `not_applicable` untuk `self_employed` / `unemployed`
- `annual_income` = gaji × 12 + bonus/THR + penghasilan lain, jadi tetap > 0 walaupun gaji 0

18 kolom mentah menjadi **64 kolom** setelah transformasi. Pipeline training ada di [`src/features.py`](src/features.py), lalu semua parameternya diekspor ke JSON.

```
ColumnTransformer
├── domain_bin   [age, credit_score, monthly_salary] DomainBinner (pd.cut) → OneHotEncoder
├── quantile_bin [annual_income]       SimpleImputer → KBinsDiscretizer(quantile, onehot)
├── num          [7 kolom numerik]     SimpleImputer(median) → StandardScaler
└── cat          [7 kolom kategorikal] SimpleImputer('missing') → OneHotEncoder(handle_unknown='ignore')
```

## Struktur Proyek

```
credit_classification/
├── data/credit_default.csv               # dataset 10.000 baris: 18 fitur + target + application_date (untuk backtest)
├── scripts/generate_data.py              # generator dataset sintetis perbankan
├── notebooks/credit_default_end_to_end.ipynb
├── src/
│   ├── features.py                       # pipeline training: binning + one-hot + imputasi + scaling
│   ├── evaluation.py                     # fungsi uji statistik: bootstrap, DeLong, McNemar, HL, PSI, IV/WoE, VIF, binomial
│   └── json_model.py                     # ekspor pipeline → JSON + inferensi JSON (numpy/pandas)
├── models/
│   └── model.json                        # model lengkap: metadata + preprocessing + parameter model
├── app/
│   ├── main.py                           # FastAPI app
│   └── schemas.py                        # validasi input (Pydantic)
├── tests/
│   ├── test_api.py                       # test endpoint & validasi input
│   └── test_evaluation.py                # test fungsi statistik (DeLong vs sklearn, PSI, HL, dst.)
├── sample_request.json / sample_batch_request.json
├── requirements.txt                      # dependensi API (tanpa scikit-learn)
├── requirements-dev.txt                  # + scikit-learn, notebook & testing
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## 1. Data & Training (Notebook)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# (opsional) buat ulang dataset
python scripts/generate_data.py --n 10000 --seed 42

jupyter lab notebooks/credit_default_end_to_end.ipynb
```

Jalankan semua cell (*Run All*, ±3–4 menit karena nested CV & bootstrap). Hasilnya akan menimpa `models/model.json` dan `sample_*.json`.

Isi notebook (21 bagian; setiap cell kode didahului kotak **Alur data: Input → Proses → Output → Berikutnya**):

| Bagian | Isi |
|---|---|
| 0–3 | Peta alur data, import, load & kamus data, EDA (termasuk profil pekerjaan/gaji/status karyawan) |
| **4** | **Uji statistik fitur**: Mann-Whitney U + rank-biserial, χ² + Cramér's V (koreksi Holm), Information Value & WoE, VIF |
| 5–9 | Split stratified, pipeline binning + one-hot, 2 model, GridSearchCV, perbandingan CV |
| **10** | **Cross-validation lanjutan**: repeated stratified K-fold 5×5 (CI terkoreksi Nadeau-Bengio), corrected resampled t-test + Wilcoxon, nested CV, learning curve |
| 11–13 | Pilih model, tuning threshold (out-of-fold), evaluasi test (confusion matrix, ROC, PR, lift) |
| **14** | **Evaluasi statistik test**: bootstrap CI 1.000×, CI DeLong, uji DeLong & McNemar, KS, kalibrasi (reliability diagram, Brier skill score, Hosmer-Lemeshow, slope/intercept), trade-off approval vs bad rate & threshold berbasis biaya |
| 15 | Interpretasi: koefisien, catatan multikolinearitas, **permutation importance** |
| **16** | **Backtesting**: timeline portofolio, out-of-time validation, walk-forward per kuartal, uji binomial per grade (traffic light), PSI skor & CSI fitur |
| **17** | **Ringkasan evaluasi statistik** (24 pengujian, status ✅/⚠️/❌) + kesimpulan & rekomendasi monitoring |
| 18–20 | Ekspor model JSON (beserta ringkasan evaluasi), validasi identik, alur data di API |

> Untuk memakai **data asli bank**, ganti `data/credit_default.csv` dengan file yang punya nama dan tipe kolom yang sama.

## 2. Menjalankan API Lokal (tanpa Docker)

Jalankan dari root proyek (agar modul `src` bisa di-import):

```bash
uvicorn app.main:app --reload --port 8000
```

Buka **http://localhost:8000/docs**. Untuk test:

```bash
pytest -q
```

## 3. Deploy ke Docker Lokal

### Prasyarat
- Docker Desktop terpasang dan **berjalan** (cek: `docker info`)
- `models/model.json` sudah ada (hasil notebook)

### Opsi A: Docker CLI

```bash
# 1. Build image
docker build -t credit-default-api:latest .

# 2. Jalankan container
docker run -d --name credit-default-api -p 8000:8000 credit-default-api:latest

# 3. Cek status (tunggu STATUS = healthy) & log
docker ps
docker logs -f credit-default-api
```

### Opsi B: Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

### Uji API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info

# prediksi 1 aplikasi
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# prediksi batch (maks. 1000)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @sample_batch_request.json
```

Contoh request:

```json
{
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
  "loan_purpose": "debt_consolidation"
}
```

Response:

```json
{"prediction": 1, "label": "default", "probability_default": 0.839344, "risk_band": "high", "threshold": 0.302}
```

`risk_band`: `high` jika probabilitas ≥ threshold, `medium` jika ≥ threshold/2, selain itu `low`.

Dari Python:

```python
import json, requests

payload = json.load(open("sample_request.json"))
print(requests.post("http://localhost:8000/predict", json=payload).json())
```

### Stop & Bersihkan

```bash
docker stop credit-default-api && docker rm credit-default-api   # Opsi A
docker compose down                                               # Opsi B
docker rmi credit-default-api:latest                              # hapus image (opsional)
```

### Update Model

Setelah notebook dijalankan ulang:

```bash
docker compose up -d --build
curl http://localhost:8000/health   # model_version berubah
```

Atau tanpa rebuild image, mount folder model sebagai volume:

```bash
docker run -d --name credit-default-api -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" credit-default-api:latest
docker restart credit-default-api   # setelah model diganti
```

> Jika format atau langkah preprocessing berubah (mis. ada transformer baru di `src/features.py`), `src/json_model.py` juga harus diperbarui, lalu image di-rebuild.

## Format Model JSON

`models/model.json` berisi seluruh pipeline dalam satu file:

```json
{
  "format": "credit-default-json-model",
  "format_version": 1,
  "metadata": { "model_name": "logistic_regression", "threshold": 0.302, "features": [...], "test_metrics": {...} },
  "preprocessing": [
    { "name": "domain_bin",   "type": "domain_bin_onehot",     "columns": ["age", "credit_score", "monthly_salary"],
      "bins": { "credit_score": { "edges": [null, 579, 669, 739, 799, null], "labels": ["poor", "fair", "good", "very_good", "excellent"] } },
      "categories": [...] },
    { "name": "quantile_bin", "type": "quantile_bin_onehot",   "columns": ["annual_income"],
      "impute_values": [118.6], "bin_edges": [[1.1, 72.25, 102.9, 136.6, 192.6, 1771.9]] },
    { "name": "num",          "type": "impute_standard_scale", "columns": [...], "impute_values": [...], "mean": [...], "scale": [...] },
    { "name": "cat",          "type": "impute_onehot",         "columns": [...], "fill_value": "missing", "categories": [...] }
  ],
  "feature_names_out": ["domain_bin__age_bin_18-25", "..."],
  "model": { "type": "logistic_regression", "coef": [...], "intercept": -0.54 }
}
```

`null` pada `edges` berarti -∞/+∞. Jika Random Forest yang terpilih, `model` berisi struktur setiap pohon (`children_left`, `children_right`, `feature`, `threshold`, `proba`).

Memakai model JSON di luar API:

```python
import pandas as pd
from src.json_model import JsonModel

model = JsonModel.load("models/model.json")
df = pd.read_json("sample_batch_request.json")["instances"].apply(pd.Series)
print(model.predict_proba(df)[:, 1])   # probabilitas default
print(model.predict(df))               # 0/1 sesuai threshold di metadata
```

**Keunggulan dibanding pickle/joblib:** aman di-load (tidak mengeksekusi kode), tidak terikat versi scikit-learn/Python, bisa dibaca & di-diff untuk audit model kredit, dan image Docker lebih ringan karena tanpa scikit-learn. Notebook memvalidasi bahwa prediksi model JSON **identik** dengan pipeline scikit-learn (selisih 0) pada seluruh test set, termasuk kasus nilai kosong dan kategori yang tidak dikenal.

---

## Referensi API

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | Status service & versi model |
| GET | `/model-info` | Model, metrik, threshold, kategori valid, batas bin |
| POST | `/predict` | Prediksi 1 aplikasi kredit |
| POST | `/predict/batch` | `{"instances": [ {...}, ... ]}` |
| GET | `/docs` | Swagger UI |

**Validasi input** (response 422 jika gagal): semua fitur wajib ada, kecuali `credit_score`, `monthly_salary`, `years_employed`, dan `education` yang boleh `null`. Nilai kategori harus termasuk daftar yang valid, kombinasi pekerjaan harus konsisten, angka harus dalam rentang wajar (mis. `credit_score` 300–850, `debt_to_income` 0–1), dan field yang tidak dikenal ditolak.

| Env var | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `models/model.json` | Lokasi file model JSON |
| `LOG_LEVEL` | `INFO` | Level logging |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Cannot connect to the Docker daemon` | Buka Docker Desktop dan tunggu sampai *running* |
| `port is already allocated` | Ganti port: `-p 8080:8000`, lalu akses `localhost:8080` |
| `ModuleNotFoundError: No module named 'src'` | Jalankan uvicorn dari root proyek; di Docker pastikan `src/json_model.py` ikut di-COPY |
| `Transformer '...' belum didukung untuk ekspor JSON` | Ada langkah baru di pipeline; tambahkan ekspor & inferensinya di `src/json_model.py` |
| Response 422 | Cek nama field, nilai kategori, dan rentang angka di `/docs` atau `/model-info` |
| Build gagal di Apple Silicon | Tambahkan `--platform linux/arm64` (atau `linux/amd64`) |

> ⚠️ Dataset bersifat sintetis untuk pembelajaran. Model kredit di produksi perlu validasi tambahan (fairness, stabilitas/PSI, monitoring drift, dan persetujuan risk management).
