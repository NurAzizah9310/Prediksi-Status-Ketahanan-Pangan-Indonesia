import pandas as pd

# 1. Memuat Data Mentah
# Pastikan menggunakan delimiter=';' karena data asli menggunakan tanda titik koma.
file_name = "Data_Indeks_Ketahanan_Pangan_Provinsi.csv"
df = pd.read_csv(file_name, delimiter=';')

print("Informasi Data Sebelum Cleaning (df.info()):")
df.info()

# 2. Penanganan Nilai Hilang (Indeks_Komposit)
# Terdapat 1 nilai NaN pada Indeks_Komposit. Kita imputasi dengan mode (nilai paling sering muncul), yaitu 3.0.
mode_composite = df['Indeks_Komposit'].mode()[0]
df['Indeks_Komposit'].fillna(mode_composite, inplace=True)

# 3. Rekayasa Fitur Waktu (Feature Engineering)
# Menggabungkan kolom Tahun dan Bulan Rilis menjadi satu kolom tanggal yang dapat dibaca.
month_map = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

# Buat kolom angka bulan
df['Bulan_Angka'] = df['Bulan Rilis'].map(month_map)

# Gabungkan Tahun dan Bulan Angka menjadi objek datetime 'Tanggal'
df['Tanggal'] = pd.to_datetime(
    df['Tahun'].astype(str) + '-' + df['Bulan_Angka'].astype(str) + '-01'
)

# 4. Standardisasi dan Pemilihan Kolom Akhir
# Ganti nama kolom agar lebih ringkas dan sesuai untuk model
df_clean = df.rename(columns={
    'Indeks_Ketersediaan': 'Ketersediaan',
    'Indeks_Keterjangkauan': 'Keterjangkauan',
    'Indeks_Pemanfaatan': 'Pemanfaatan',
    'Indeks_Komposit': 'Komposit_Status' # Ini adalah Variabel Target (y)
}).drop(columns=['Tahun', 'Bulan Rilis', 'Bulan_Angka', 'Kode_Provinsi']) # Drop kolom yang tidak diperlukan

# Pastikan variabel target bertipe integer
df_clean['Komposit_Status'] = df_clean['Komposit_Status'].astype(int)

# 5. Simpan Data Bersih
output_file_name = "Data_Ketahanan_Pangan_Clean.csv"
df_clean.to_csv(output_file_name, index=False)

print("\nData Bersih (df_clean) Sudah Siap:")
print(df_clean.head())
print("\nData Bersih sudah disimpan sebagai:", output_file_name)