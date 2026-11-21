
import pandas as pd

# Membaca dataset CSV
df = pd.read_csv("Mall_Customers.csv")

print("===== INFORMASI DATASET =====")

# Menampilkan 5 data pertama
print("\n5 Data Pertama:")
print(df.head())

# Menampilkan jumlah baris dan kolom
print("\nJumlah Baris dan Kolom:")
print(df.shape)

# Menampilkan nama kolom
print("\nNama Kolom:")
print(df.columns.tolist())

# Memeriksa nilai kosong
print("\nJumlah Data Kosong:")
print(df.isnull().sum())

# Memeriksa data duplikat
print("\nJumlah Data Duplikat:")
print(df.duplicated().sum())

# Memeriksa duplikasi ID pelanggan
print("\nID Pelanggan Duplikat:")
print(df["CustomerID"].duplicated().sum())

# Informasi tipe data
print("\nInformasi Tipe Data:")
print(df.info())

print("\n===== PEMERIKSAAN SELESAI =====")
