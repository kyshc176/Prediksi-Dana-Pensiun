import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi Dana Pensiun", page_icon="💰", layout="centered")

# ---------------------------------------------------------------------------
# Load model & scaler (di-cache biar cuma di-load sekali per sesi)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    model = joblib.load(BASE_DIR / "xgb_model.pkl")
    return scaler, model

scaler, model = load_artifacts()

FINAL_FEATURES = [
    "YearsToRetirement", "MonthlyIncome", "MonthlyDebtPayments",
    "SavingsAccountBalance", "NetWorth", "CreditScore", "JobTenure"
]
RETIREMENT_AGE = 60

# Model dilatih pada skala internal dataset asli (bukan Rupiah langsung), jadi input
# Rupiah dari user dikonversi dulu ke skala internal sebelum masuk model, lalu hasil
# prediksinya dikonversi balik ke Rupiah. Ini murni skala pembanding, tidak mengubah
# logika model sama sekali.
KURS = 16000

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💰 Prediksi Dana Pensiun")
st.caption(
    "Estimasi dana pensiun berbasis Machine Learning (XGBoost) — dilatih pada data simulasi "
    "keuangan dengan pendekatan *future value* (compound interest)."
)

with st.expander("ℹ️ Tentang model & batasan ini", expanded=False):
    st.markdown(
        """
        - Model ini dilatih dari data simulasi, bukan data pensiun riil. Target dihitung memakai
          rumus *future value* (nilai tabungan saat ini + kontribusi bulanan yang dikompaun
          dengan asumsi return investasi tahunan 5%).
        - **R² ≈ 0.997, MAPE ≈ 2.5%** pada data uji.
        - Cocok dipakai sebagai *tooling edukasi/simulasi* perencanaan keuangan, **bukan**
          sebagai dasar keputusan finansial tanpa validasi dari data riil/ahli keuangan.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Form input (semua dalam Rupiah)
# ---------------------------------------------------------------------------
st.subheader("Masukkan Profil Keuangan")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Usia saat ini", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input(
        "Pendapatan Bulanan (Rp)", min_value=0, value=80_000_000, step=1_000_000, format="%d"
    )
    monthly_debt = st.number_input(
        "Cicilan/Utang Bulanan (Rp)", min_value=0, value=6_000_000, step=500_000, format="%d"
    )
    savings_balance = st.number_input(
        "Saldo Tabungan Saat Ini (Rp)", min_value=0, value=128_000_000, step=1_000_000, format="%d"
    )

with col2:
    net_worth = st.number_input(
        "Kekayaan Bersih / Net Worth (Rp)", min_value=0, value=960_000_000, step=10_000_000, format="%d"
    )
    credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, value=650)
    job_tenure = st.slider("Lama Bekerja di Pekerjaan Saat Ini (tahun)", min_value=0, max_value=40, value=5)

years_to_retirement = max(RETIREMENT_AGE - age, 1)
st.caption(f"➡️ Sisa waktu menuju usia pensiun ({RETIREMENT_AGE} tahun): **{years_to_retirement} tahun**")

st.divider()

# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
if st.button("🔮 Prediksi Dana Pensiun", type="primary", use_container_width=True):
    # Konversi input Rupiah -> skala internal model (dibagi kurs)
    input_df = pd.DataFrame([{
        "YearsToRetirement": years_to_retirement,
        "MonthlyIncome": monthly_income / KURS,
        "MonthlyDebtPayments": monthly_debt / KURS,
        "SavingsAccountBalance": savings_balance / KURS,
        "NetWorth": net_worth / KURS,
        "CreditScore": credit_score,
        "JobTenure": job_tenure,
    }])[FINAL_FEATURES]

    input_scaled = scaler.transform(input_df)
    pred_log = model.predict(input_scaled)[0]
    pred_idr = float(np.expm1(pred_log)) * KURS

    st.success("Prediksi berhasil dihitung!")
    st.metric("Estimasi Dana Pensiun", f"Rp {pred_idr:,.0f}")

    # ---------------------------------------------------------------------
    # Grafik proyeksi pertumbuhan dana per tahun (ilustratif, memakai rumus
    # future value yang sama seperti pembuatan target — dihitung langsung
    # dalam Rupiah, tidak perlu konversi)
    # ---------------------------------------------------------------------
    st.subheader("📈 Proyeksi Pertumbuhan Dana per Tahun")

    annual_return = 0.05
    savings_rate = 0.20
    disposable = max(monthly_income - monthly_debt, 0)
    monthly_contribution = disposable * savings_rate
    monthly_rate = annual_return / 12

    years_range = np.arange(0, years_to_retirement + 1)
    months_range = years_range * 12
    fv_lump = savings_balance * (1 + monthly_rate) ** months_range
    fv_annuity = np.where(
        months_range == 0,
        0,
        monthly_contribution * (((1 + monthly_rate) ** months_range - 1) / monthly_rate),
    )
    projection = fv_lump + fv_annuity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age + years_range, y=projection, mode="lines+markers",
        line=dict(color="#2E7D32", width=3), name="Proyeksi Dana"
    ))
    fig.update_layout(
        xaxis_title="Usia", yaxis_title="Dana Terkumpul (Rp)",
        yaxis_tickformat=",.0f",
        template="plotly_white", height=400, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Grafik ini ilustratif — memakai asumsi return investasi 5%/tahun dan tingkat "
        "tabungan 20% dari disposable income, konsisten dengan formula yang dipakai saat "
        "membangun target model."
    )

st.divider()
st.caption("Dibangun dengan XGBoost (R²≈0.997) · Capstone Project — Prediksi Dana Pensiun")
