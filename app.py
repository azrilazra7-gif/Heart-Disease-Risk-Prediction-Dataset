import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PENGATURAN LOKASI FILE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"

# ==========================================
# 2. KONFIGURASI HALAMAN & ULTIMATE CSS
# ==========================================

st.set_page_config(
    page_title="Cardio-Score AI | Medical Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
    /* Global Theme Styling */
    .stApp {
        background: radial-gradient(circle at top right, #111827, #030712);
        color: #f3f4f6;
    }
    
    /* Header Banner Utama */
    .hero-box {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        margin-bottom: 30px;
    }
    .hero-box h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    .hero-box p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1f2937;
    }

    /* Tombol Utama */
    .stButton > button {
        background: linear-gradient(135deg, #f43f5e, #e11d48);
        color: white;
        font-weight: 700;
        border-radius: 14px;
        padding: 16px 24px;
        border: none;
        box-shadow: 0 8px 25px rgba(244, 63, 94, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(244, 63, 94, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MEMUAT MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("⚠️ Model belum ditemukan. Jalankan file main.py terlebih dahulu.")
    st.stop()

model, feature_columns = load_model()

# ==========================================
# 4. SIDEBAR DASHBOARD
# ==========================================

with st.sidebar:
    st.markdown("## 🫀 Cardio-Score AI")
    st.markdown("---")
    st.markdown("### 📊 Status Sistem")
    st.success("🟢 Model AI: Terhubung")
    st.info("Algoritma: Logistic Regression & Random Forest Ensemble.")
    st.markdown("---")
    st.caption("Developed for Machine Learning Project & Healthcare Analysis.")

# ==========================================
# 5. KONTEN UTAMA
# ==========================================

st.markdown("""
<div class="hero-box">
    <h1>❤️ Cardio-Score: AI Heart Risk Predictor</h1>
    <p>Platform analisis cerdas untuk mendeteksi tingkat risiko kesehatan jantung secara cepat, akurat, dan transparan.</p>
</div>
""", unsafe_allow_html=True)

binary_features = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations", 
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea", 
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking", 
    "Obesity", "Sedentary_Lifestyle", "Family_History", "Chronic_Stress"
]

labels = {
    "Chest_Pain": "Nyeri dada",
    "Shortness_of_Breath": "Sesak napas",
    "Fatigue": "Kelelahan",
    "Palpitations": "Jantung berdebar",
    "Dizziness": "Pusing",
    "Swelling": "Pembengkakan",
    "Pain_Arms_Jaw_Back": "Nyeri lengan, rahang, atau punggung",
    "Cold_Sweats_Nausea": "Keringat dingin atau mual",
    "High_BP": "Tekanan darah tinggi",
    "High_Cholesterol": "Kolesterol tinggi",
    "Diabetes": "Diabetes",
    "Smoking": "Merokok",
    "Obesity": "Obesitas",
    "Sedentary_Lifestyle": "Kurang aktivitas fisik",
    "Family_History": "Riwayat keluarga",
    "Chronic_Stress": "Stres kronis"
}

with st.form("prediction_form"):
    st.markdown("### 👤 Informasi Dasar Pasien")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (Tahun)", min_value=20, max_value=84, value=30, step=1)
    with col2:
        gender = st.selectbox(
            "Jenis Kelamin / Gender",
            options=[0, 1],
            format_func=lambda x: "Laki-laki (Kode 1)" if x == 1 else "Perempuan (Kode 0)"
        )

    st.markdown("---")
    st.markdown("### 🩺 Gejala & Indikator Klinis")

    input_data = {}
    c1, c2 = st.columns(2)

    for i, feature in enumerate(binary_features):
        with c1 if i % 2 == 0 else c2:
            input_data[feature] = int(st.checkbox(labels[feature], key=feature))

    st.write("")
    submit = st.form_submit_button("🚀 ANALISIS RISIKO SEKARANG", use_container_width=True)

# ==========================================
# 6. HASIL PREDIKSI
# ==========================================

if submit:
    input_data["Gender"] = gender
    input_data["Age"] = age

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.markdown("### 📈 Hasil Evaluasi Model")

    if prediction == 0:
        st.success("🟢 **Risiko Rendah (Low Risk):** Berdasarkan parameter yang dimasukkan, kondisi terpantau aman dan stabil.")
    else:
        st.error("🔴 **Risiko Tinggi (High Risk):** Terdeteksi indikasi risiko yang memerlukan perhatian serta konsultasi medis.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📊 Skor Keyakinan Model", value=f"{probability[int(prediction)] * 100:.2f}%")
    with col2:
        st.metric(label="🩺 Status Klasifikasi", value="Risiko Rendah" if prediction == 0 else "Risiko Tinggi")

    st.caption("💡 Catatan: Skor ini merupakan hasil keluaran analitik dari model Machine Learning, bukan diagnosis medis resmi.")

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()
st.markdown("<p style='text-align: center; color: #4b5563; font-size: 0.85rem;'>Cardio-Score AI Dashboard &copy; 2026 | Powered by Python & Streamlit</p>", unsafe_allow_html=True)