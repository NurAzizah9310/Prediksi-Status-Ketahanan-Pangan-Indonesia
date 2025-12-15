import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Library yang umum digunakan untuk menyimpan model (lebih baik dari pickle)
import os

# --- 1. SETTING FILE & PARAMETER ---
CLEAN_DATA_FILE = "Data_Ketahanan_Pangan_Clean.csv"
MODEL_OUTPUT_FILE = "random_forest_model.pkl"
RANDOM_STATE = 42

if not os.path.exists(CLEAN_DATA_FILE):
    print(f"ERROR: File {CLEAN_DATA_FILE} tidak ditemukan. Mohon jalankan langkah pembersihan data terlebih dahulu.")
else:
    # --- 2. MUAT DATA BERSIH ---
    df_model = pd.read_csv(CLEAN_DATA_FILE)
    print(f"Data berhasil dimuat. Jumlah baris: {len(df_model)}")

    # --- 3. FEATURE ENGINEERING & ENCODING ---
    
    # Ekstrak nomor bulan dari kolom 'Tanggal' untuk menangkap efek musiman/bulan (seasonality)
    df_model['Tanggal'] = pd.to_datetime(df_model['Tanggal'])
    df_model['Bulan_Nomor'] = df_model['Tanggal'].dt.month

    # Tentukan Variabel Target (y) dan Fitur (X)
    y = df_model['Komposit_Status']
    X_features = ['Ketersediaan', 'Keterjangkauan', 'Pemanfaatan', 'Provinsi', 'Bulan_Nomor']
    X = df_model[X_features]

    # One-Hot Encoding untuk 'Provinsi' dan 'Bulan_Nomor'
    # 'drop_first=True' untuk menghindari multicollinearity
    X_encoded = pd.get_dummies(X, columns=['Provinsi', 'Bulan_Nomor'], drop_first=True)
    
    # Simpan nama kolom fitur yang sudah di-encoded. INI PENTING untuk Streamlit Deployment!
    feature_names = X_encoded.columns.tolist()
    
    print(f"\nJumlah Fitur (kolom) setelah Encoding: {len(feature_names)}")

    # --- 4. PEMBAGIAN DATA ---
    # Stratify=y memastikan proporsi kelas (1, 2, 3) pada training dan testing set seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Data Training: {len(X_train)} baris, Data Testing: {len(X_test)} baris.")

    # --- 5. PELATIHAN MODEL (Random Forest Classifier) ---
    print("\nMemulai pelatihan model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")

    # --- 6. EVALUASI MODEL ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=============================================")
    print(f"AKURASI MODEL PREDIKSI: {accuracy:.4f}")
    print("=============================================")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['1 (Rendah)', '2 (Sedang)', '3 (Tinggi)']))

    # --- 7. PENYIMPANAN MODEL & FEATURE NAMES ---
    # Simpan model Random Forest ke file .pkl menggunakan joblib
    joblib.dump(model, MODEL_OUTPUT_FILE)

    # Simpan juga daftar nama fitur (feature_names) yang digunakan
    # Ini krusial agar input Streamlit memiliki format kolom yang sama dengan saat training!
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print(f"\nModel telah disimpan sebagai '{MODEL_OUTPUT_FILE}'")
    print(f"Daftar nama fitur telah disimpan sebagai 'feature_names.pkl'")