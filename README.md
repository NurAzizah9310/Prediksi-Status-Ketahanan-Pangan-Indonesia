# Prediksi Status Ketahanan Pangan Indonesia

Proyek ini merupakan *Tugas Ujian Akhir Semester (UAS) Mata Kuliah Analisis Data* dengan tema *Pangan*.

## Anggota Kelompok

* **Nur Azizah Santoso** (2400655)
* **Rianti Agustini** (2400146)

---

## Deskripsi Proyek

Proyek ini bertujuan untuk mengatasi tantangan evaluasi Status Indeks Komposit Ketahanan Pangan (IKP) Provinsi di Indonesia yang cenderung bersifat **reaktif**.

Kami menganalisis data IKP historis periode 2022–2025 yang mencakup tiga sub-indeks utama (*Ketersediaan, Keterjangkauan, Pemanfaatan*). Solusi yang dikembangkan adalah **sistem prediksi proaktif** melalui **Model Klasifikasi *Random Forest*** yang diintegrasikan ke dalam **Dashboard Streamlit**.

Hasilnya adalah *Dashboard Cerdas* yang dapat memprediksi status risiko pangan (*Rendah, Sedang, atau Tinggi*) secara instan (*real-time*), berfungsi sebagai alat *early warning system*.

## Dataset

Dataset yang digunakan adalah data IKP Provinsi Indonesia yang telah melalui proses *data cleaning* dan *Feature Engineering* (termasuk One-Hot Encoding).

**Variabel Kunci:**
* Indeks Ketersediaan
* Indeks Keterjangkauan
* Indeks Pemanfaatan
* Provinsi dan Bulan (sebagai faktor musiman)
* **Target:** Status Komposit IKP (1, 2, atau 3)

## Tools & Teknologi

| Kategori | Tool/Library | Fungsi Utama |
| :--- | :--- | :--- |
| **Bahasa Pemrograman** | Python | Bahasa utama pengembangan. |
| **Data Processing** | Pandas, Numpy | Pembersihan, Pra-pemrosesan, dan *Feature Engineering*. |
| **Pemodelan ML** | Scikit-learn (RandomForestClassifier) | Melatih Model Klasifikasi dengan Akurasi Pengujian **$\approx 98.25\%$**. |
| **Deployment** | Streamlit, Joblib | Kerangka kerja untuk membangun Dashboard Interaktif dan menyimpan/memuat model. |

## Model Machine Learning

| Aspek | Detail |
| :--- | :--- |
| **Jenis Model** | **Klasifikasi (Random Forest Classifier)** |
| **Target Prediksi** | **Status Komposit IKP** (Rendah/1, Sedang/2, Tinggi/3) |
| **Metode Deployment** | Model disimpan dalam format **`.pkl`** dan diintegrasikan ke Streamlit untuk fungsionalitas prediksi *real-time*. |

## Fitur Dashboard

* **Visualisasi EDA:** Menampilkan tren historis IKP dan distribusi status per provinsi.
* **Modul Prediksi Real-Time:** Memungkinkan pengguna memasukkan nilai sub-indeks, provinsi, dan bulan untuk mendapatkan prediksi status IKP instan.
* **Tampilan Interaktif:** Hasil prediksi disajikan dengan label status dan kode warna yang intuitif.

## Cara Menjalankan Aplikasi

1.  **Instalasi Dependensi:**
    Pastikan Anda telah menginstal semua *library* yang dibutuhkan:

    ```bash
    pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
    ```

2.  **Jalankan Aplikasi Streamlit:**
    Arahkan ke direktori proyek Anda di terminal, lalu jalankan:

    ```bash
    streamlit run app.py
    ```

3.  **Akses Dashboard:**
    Aplikasi akan terbuka secara otomatis di *web browser* Anda (biasanya di `http://localhost:8501`).
