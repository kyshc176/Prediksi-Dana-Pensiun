# 💰 Prediksi Dana Pensiun — Streamlit App

Aplikasi web interaktif untuk mengestimasi dana pensiun seseorang berdasarkan profil keuangan saat ini, menggunakan model Machine Learning (XGBoost). Dibangun sebagai Capstone Project.

🔗 **Live demo:** [_(Live Demo)_](https://prediksi-dana-pensiun.streamlit.app)
📓 **Notebook training:** [`notebooks/Model.ipynb`](notebooks/Model.ipynb)

---

## 📌 Ringkasan Proyek

| | |
|---|---|
| **Masalah** | Memprediksi estimasi dana pensiun individu berdasarkan data keuangan pribadi |
| **Tipe masalah** | Regresi |
| **Model** | XGBoost Regressor (dipilih dari 4 model yang dibandingkan) |
| **Performa (test set)** | R² = 0.997, MAPE = 2.5%, dievaluasi di skala Rupiah asli |
| **Fitur input** | Usia (→ `YearsToRetirement`), pendapatan bulanan, cicilan/utang bulanan, saldo tabungan, net worth, skor kredit, lama bekerja |
| **Output** | Estimasi nominal dana pensiun (Rupiah) + grafik proyeksi pertumbuhan dana per tahun |
| **Deployment** | Streamlit Cloud |

---

## 🧠 Penjelasan & Validasi Model

### Kenapa R² bisa setinggi 0.997?

Ini bukan hasil yang "terlalu bagus untuk dipercaya" karena kebocoran data — target (`TargetRetirementFund`) memang **dihitung dari rumus keuangan deterministik** (*future value*: lump sum + anuitas majemuk dari tabungan & kontribusi bulanan), bukan angka pensiun riil yang penuh noise dunia nyata. Noise acak ±8% (distribusi normal) ditambahkan ke target supaya tidak sepenuhnya deterministik, tapi pola utamanya tetap sangat bisa dipelajari oleh model nonlinear seperti XGBoost. Dengan kata lain: **R² tinggi di sini mencerminkan seberapa baik model menangkap rumus finansial di balik data simulasi — bukan bukti bahwa model ini bisa memprediksi masa depan finansial seseorang di dunia nyata.**

### Metodologi validasi

- **Train/test split 80/20** (`random_state` tetap → reproducible), `StandardScaler` di-*fit* hanya pada data train untuk menghindari data leakage.
- **5-fold cross-validation** dipakai saat membandingkan model, bukan cuma satu kali split.
- Target ditransformasi `log1p` (skewness turun signifikan) sebelum training, lalu hasil prediksi dikonversi balik (`expm1`) untuk evaluasi metrik di skala Rupiah asli.
- Fitur redundan dibuang sebelum training: `Age` diganti `YearsToRetirement`, `TotalAssets`/`TotalLiabilities` dibuang karena sudah terwakili oleh `NetWorth` (korelasi absolut maksimum antar fitur final < 0.7 → aman dari multikolinearitas).
- Dicek dengan **residual plot** (actual vs predicted) dan **feature importance** — bukan cuma dilihat dari skor R²/MAPE saja.

### Perbandingan 4 model (5-fold CV + test set)

| Model | Catatan |
|---|---|
| **XGBoost** ✅ | Performa terbaik (R² & MAPE terendah) — dipakai di app ini |
| Random Forest | Kompetitif, sedikit di bawah XGBoost |
| Neural Network | Di bawah kedua model tree-based, meski masih jauh lebih baik dari Linear Regression |
| Linear Regression | **Tidak layak dipakai** — R² di skala log terlihat oke (~0.83), tapi jeblok begitu dibalik ke skala asli, karena target bersifat nonlinear/eksponensial akibat efek *compounding*. Ini jadi bukti empiris kenapa model linear tidak cukup untuk kasus ini. |

Insight ini (bukan cuma "pakai XGBoost karena paling akurat") adalah bagian penting dari notebook — proses eliminasi model justru menunjukkan pemahaman tentang *kenapa* pendekatan nonlinear dibutuhkan, bukan sekadar ikut tren.

> ⚠️ **Batasan:** Model dilatih dari data simulasi, bukan data pensiun riil — cocok sebagai *tooling edukasi/simulasi* perencanaan keuangan, **bukan** dasar keputusan finansial tanpa validasi dari data riil/ahli keuangan.

---

## 🖥️ Fitur Aplikasi

- Input profil keuangan (usia, pendapatan, utang, tabungan, net worth, skor kredit, lama bekerja) dalam Rupiah
- Prediksi estimasi dana pensiun secara instan
- Visualisasi interaktif proyeksi pertumbuhan dana dari usia sekarang sampai usia pensiun (Plotly)
- Penjelasan singkat tentang cara kerja & batasan model di dalam app

---

## 🛠️ Tech Stack

- **Model**: XGBoost (Scikit-learn API), StandardScaler
- **App**: Streamlit, Plotly
- **Bahasa**: Python

---

## 🚀 Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Cara Deploy ke Streamlit Cloud

1. Push repo ini (minimal `app.py`, `requirements.txt`, `models/scaler.pkl`, `models/xgboost_model.pkl`) ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo → pilih `app.py` sebagai main file.
3. Deploy.

---

## 📁 Struktur Repo

```
retirement-fund-prediction/
├── app.py                              # Aplikasi Streamlit utama (input & output dalam Rupiah)
├── requirements.txt                    # Daftar dependency
├── README.md
├── models/
│   ├── scaler.pkl                      # StandardScaler hasil training
│   └── xgboost_model.pkl               # Model terbaik dari 4 model yang dibandingkan (R²=0.997, MAPE=2.5%)
└── notebooks/
    └── capstone_project_advanced.ipynb # EDA, feature engineering, training & evaluasi 4 model
```

> 📝 Rencana rapikan: pindahkan `scaler.pkl` & `xgboost_model.pkl` ke folder `models/`, rename notebook (hilangkan spasi & `(2)`) ke `notebooks/capstone_project_advanced.ipynb`, lalu sesuaikan path load di `app.py` (`models/scaler.pkl`, `models/xgboost_model.pkl`).

---

## 📝 Catatan Teknis

Model dilatih pada skala data asli dataset (bukan langsung Rupiah). Input Rupiah dari user dikonversi dulu ke skala internal (dibagi kurs Rp16.000) sebelum masuk model, lalu hasil prediksi dikonversi balik ke Rupiah. Ini murni penyesuaian skala tampilan — tidak mengubah cara kerja model sama sekali.

Model Neural Network (`retirement_nn_model.keras` / `.tflite`) dilatih & divalidasi di notebook (termasuk sanity check kesesuaian output Keras vs TFLite untuk keperluan deployment mobile), tapi **tidak dipakai** di app Streamlit ini — XGBoost dipilih karena performanya lebih baik dan dependency-nya jauh lebih ringan (tanpa TensorFlow), cocok untuk deployment gratis di Streamlit Cloud.

---

## 👤 Author

**Tika Putri Marsanti**
[LinkedIn](https://linkedin.com/in/tika-putri-marsanti) · [GitHub](https://github.com/kyshc176)
