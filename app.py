import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_FILE = "Data_Indeks_Ketahanan_Pangan_Provinsi.csv" 
MODEL_FILE = "random_forest_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

def load_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f172a, #020617);
    }

    .main-title {
        background: linear-gradient(90deg, #b4d0f4, #eaf1ff);
        padding: 32px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 40px;
    }

    .main-title h1 {
        color: #1f3b73;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .main-title p {
        color: #375a9e;
        font-size: 16px;
        margin: 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #9ec5fe 0%,
            #b4d0f4 45%,
            #eaf1ff 100%
        );
        border-right: 2px solid #c7dcf8;
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f3b73 !important;
        font-weight: 800;
    }

    div[role="radiogroup"] label {
        padding: 8px 6px;
        border-radius: 10px;
    }

    div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.6);
    }

    div[role="radiogroup"] input:checked + div {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
    }

    .card {
        background-color: #f8fbff;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.25);
        margin-bottom: 20px;
        color: #0f172a;
    }

    .metric-box {
        background-color: #eaf1ff;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }

    .metric-box h3 {
        color: #1f3b73;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .metric-box p {
        color: #334155;
        font-size: 15px;
        margin: 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4a6fa5, #1f3b73);
        color: white;
        font-weight: bold;
        border-radius: 16px;
        height: 55px;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1f3b73, #020617);
    }

    h2, h3 {
        color: #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

# DATA CLEANING
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    # Imputasi modus untuk Komposit_Status yang hilang
    df['Indeks_Komposit'].fillna(df['Indeks_Komposit'].mode()[0], inplace=True)

    month_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    df['Bulan_Angka'] = df['Bulan Rilis'].map(month_map)
    df['Tanggal'] = pd.to_datetime(
        df['Tahun'].astype(str) + '-' + df['Bulan_Angka'].astype(str) + '-01'
    )

    df = df.rename(columns={
        'Indeks_Ketersediaan': 'Ketersediaan',
        'Indeks_Keterjangkauan': 'Keterjangkauan',
        'Indeks_Pemanfaatan': 'Pemanfaatan',
        'Indeks_Komposit': 'Komposit_Status'
    })

    df['Komposit_Status'] = df['Komposit_Status'].astype(int)
    status_map = {1: 'Rendah / Kritis', 2: 'Sedang', 3: 'Tinggi / Baik'}
    df['Status_Label'] = df['Komposit_Status'].map(status_map)
    return df

# EDA (DENGAN GRAFIK)
def plot_eda(df):
    st.markdown("## 📊 Gambaran Umum Ketahanan Pangan")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>{df['Provinsi'].nunique()}</h3><p>Provinsi Terdata</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>{df['Tanggal'].dt.year.nunique()}</h3><p>Tahun Pengamatan (2022-2025)</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>{df['Status_Label'].mode()[0]}</h3><p>Status Dominan</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # GRAFIK 1: TREN RATA-RATA IKP NASIONAL
    st.markdown("### 📈 Tren Rata-rata Indeks Komposit Nasional")
    df_trend = df.groupby('Tanggal')['Komposit_Status'].mean().reset_index()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(
        data=df_trend,
        x='Tanggal',
        y='Komposit_Status',
        marker='o',
        linewidth=2,
        color='#9ec5fe',
        ax=ax
    )

    ax.set_title('Rata-rata Status IKP dari Waktu ke Waktu (2022-2025)', fontsize=16, color='#e5e7eb')
    ax.set_xlabel('Waktu', fontsize=12, color='#e5e7eb')
    ax.set_ylabel('Rata-rata Status IKP', fontsize=12, color='#e5e7eb')
    
    ax.set_ylim(1.0, 3.0) 
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1 (Rendah)', '2 (Sedang)', '3 (Tinggi)'])
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    st.pyplot(fig)


    # GRAFIK 2: DISTRIBUSI STATUS IKP (Bar Plot)
    st.markdown("### 📊 Distribusi Status Ketahanan Pangan")

    prov_filter = st.selectbox(
        "Pilih Provinsi untuk melihat distribusi:",
        ['NASIONAL'] + sorted(df['Provinsi'].unique())
    )

    if prov_filter == 'NASIONAL':
        df_plot = df
        title = 'Distribusi Status IKP (NASIONAL)'
    else:
        df_plot = df[df['Provinsi'] == prov_filter]
        title = f'Distribusi Status IKP ({prov_filter})'

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    status_order = ['Rendah / Kritis', 'Sedang', 'Tinggi / Baik']
    palette_map = {'Rendah / Kritis': '#ef4444', 'Sedang': '#f97316', 'Tinggi / Baik': '#10b981'}

    sns.countplot(
        data=df_plot,
        x='Status_Label',
        order=status_order,
        palette=palette_map,
        ax=ax2
    )

    ax2.set_title(title, fontsize=16, color='#e5e7eb')
    ax2.set_xlabel('Status Ketahanan Pangan', fontsize=12, color='#e5e7eb')
    ax2.set_ylabel('Jumlah Data', fontsize=12, color='#e5e7eb')

    ax2.set_facecolor('#0f172a')
    fig2.patch.set_facecolor('#0f172a')
    
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='#e5e7eb')
    
    st.pyplot(fig2)

    st.markdown("### Contoh Data Mentah")
    st.dataframe(df.head())

# PREDIKSI
def predict_status(model, feature_names, input_data):
    # Buat DataFrame kosong dengan semua nama fitur
    df_pred = pd.DataFrame(columns=feature_names, data=[np.zeros(len(feature_names))])
    
    # Isi nilai sub-indeks (numerik)
    df_pred['Ketersediaan'] = input_data['Ketersediaan']
    df_pred['Keterjangkauan'] = input_data['Keterjangkauan']
    df_pred['Pemanfaatan'] = input_data['Pemanfaatan']

    # One-Hot Encoding untuk Provinsi dan Bulan
    provinsi_col = f"Provinsi_{input_data['Provinsi']}"
    bulan_col = f"Bulan_Nomor_{input_data['Bulan']}"
    
    if provinsi_col in df_pred.columns:
        df_pred[provinsi_col] = 1
    if bulan_col in df_pred.columns:
        df_pred[bulan_col] = 1

    # Prediksi
    return model.predict(df_pred)[0]

# LOAD MODEL & FEATURE NAMES
try:
    rf_model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_NAMES_FILE)
except FileNotFoundError:
    st.error("Error: File model (random_forest_model.pkl) atau feature names (feature_names.pkl) tidak ditemukan. Pastikan sudah melatih model dan menyimpannya.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# MAIN APP
def main():
    st.set_page_config(page_title="Ketahanan Pangan Indonesia", layout="wide")
    load_css()

    st.markdown("""
    <div class="main-title">
        <h1>🌾 Dashboard Cerdas: Prediksi Status Ketahanan Pangan Indonesia</h1>
        <p>Berbasis Real-Time Menggunakan Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## 📌 Menu Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman",
        ["Dashboard Informasi", "Prediksi Ketahanan Pangan"]
    )

    if page == "Dashboard Informasi":
        df = load_and_clean_data(RAW_DATA_FILE)
        plot_eda(df)
        
    else:
        st.markdown("## 🔮 Prediksi Status Ketahanan Pangan")
        st.info("Masukkan kondisi wilayah (Indeks Ketersediaan, Keterjangkauan, dan Pemanfaatan: 1=Rendah, 2=Sedang, 3=Tinggi) untuk memprediksi status komposit.")

        # Ambil daftar provinsi dari data mentah
        df_raw = pd.read_csv(RAW_DATA_FILE, delimiter=';')
        provinsi = sorted(df_raw['Provinsi'].unique())

        # Kolom Input
        col1, col2 = st.columns(2)
        with col1:
            prov = st.selectbox("Provinsi", provinsi)
            bulan = st.selectbox("Bulan (1–12)", list(range(1, 13)))
        with col2:
            k1 = st.slider("Indeks Ketersediaan Pangan", 1, 3, 2)
            k2 = st.slider("Indeks Keterjangkauan Pangan", 1, 3, 2)
            k3 = st.slider("Indeks Pemanfaatan Pangan", 1, 3, 2)

        if st.button("🔍 Prediksi Sekarang", use_container_width=True):
            hasil = predict_status(
                rf_model,
                feature_cols,
                {
                    'Provinsi': prov,
                    'Bulan': bulan,
                    'Ketersediaan': k1,
                    'Keterjangkauan': k2,
                    'Pemanfaatan': k3
                }
            )

            label = {1: "RENDAH / KRITIS", 2: "SEDANG", 3: "TINGGI / BAIK"}
            color_hex = {1: "#ef4444", 2: "#f97316", 3: "#10b981"}

            st.markdown(f"""
            <div class="card">
                <h2 style="text-align:center; color:#0f172a;">Hasil Prediksi Status IKP</h2>
                <h1 style="text-align:center; color:{color_hex[hasil]};">{label[hasil]}</h1>
                <p style="text-align:center; color:#334155;">
                    Diprediksi untuk Provinsi <b>{prov}</b> pada Bulan ke-<b>{bulan}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Model Klasifikasi Random Forest • Akurasi Pengujian $\pm 98.25\%$")

if __name__ == "__main__":
    main()