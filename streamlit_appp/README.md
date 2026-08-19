# Prediksi Dana Pensiun — Streamlit App (Rupiah)

## Cara jalanin lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara deploy ke Streamlit Cloud
1. Push folder ini (app.py, requirements.txt, scaler.pkl, xgboost_model.pkl) ke repo GitHub.
2. Buka https://share.streamlit.io -> New app -> pilih repo -> pilih `app.py` sebagai main file.
3. Deploy.

## Isi folder
- `app.py` — aplikasi utama, semua input & output dalam Rupiah
- `scaler.pkl` — StandardScaler hasil training (wajib, dipakai untuk transform input)
- `xgboost_model.pkl` — model XGBoost terbaik (R²≈0.997, MAPE≈2.5%) dari notebook
- `requirements.txt` — daftar dependency

## Catatan teknis
Model dilatih pada skala data asli dataset (bukan langsung Rupiah). Input Rupiah dari
user dikonversi dulu ke skala internal (dibagi kurs Rp16.000) sebelum masuk model, lalu
hasil prediksi dikonversi balik ke Rupiah. Ini murni penyesuaian skala tampilan — tidak
mengubah cara kerja model sama sekali.

Model Neural Network (`retirement_nn_model.keras`) tidak dipakai di app ini karena XGBoost
performanya lebih baik dan dependency-nya jauh lebih ringan (tanpa TensorFlow) — cocok
untuk deployment gratis di Streamlit Cloud.
