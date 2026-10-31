"""
Data pipeline untuk dataset automobileEDA.

Merangkum proses pada data-pipeline.ipynb ke dalam beberapa function:
1. load_data       : membaca raw dataset
2. inspect_data    : memeriksa kondisi awal dataset
3. clean_data      : data cleaning
   - drop duplicate records
   - drop baris dengan missing values
   - menghapus spasi berlebih & menyeragamkan huruf pada kolom kategorikal
   - menyeragamkan label kategori yang bermakna sama (`awd` -> `4wd`)
   - memperbaiki tipe data (`transaction_date`, `num-of-doors`, `num-of-cylinders`)
   - memeriksa nilai tidak wajar pada kolom numerik
4. transform_data  : data transformation
   - min-max scaling pada `curb-weight`, `peak-rpm`, `price` -> kolom baru `<kolom>_scaled`
   - one-hot encoding kolom `horsepower-binned`
5. save_data       : simpan hasil data pipeline ke file CSV baru

Cara menjalankan (dari root project):
    python src/pipeline.py
    python src/pipeline.py --input data/raw/file.csv --output data/processed/file.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / 'data' / 'raw' / 'automobileEDA_dirty_training.csv'
DEFAULT_OUTPUT = BASE_DIR / 'data' / 'processed' / 'automobileEDA_processed.csv'

DATE_COLUMN = 'transaction_date'
DATE_FORMATS = ['%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y', '%d-%b-%Y']
OUTPUT_DATE_FORMAT = '%d-%m-%Y'
CATEGORY_ALIASES = {
    'drive-wheels': {'awd': '4wd'},
}
WORD_TO_NUMBER = {
    'num-of-doors': {'two': 2, 'four': 4},
    'num-of-cylinders': {'two': 2, 'three': 3, 'four': 4, 'five': 5,
                         'six': 6, 'eight': 8, 'twelve': 12},
}
# Kolom ukuran fisik / performa yang tidak mungkin bernilai <= 0
POSITIVE_COLUMNS = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',
                    'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
                    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price',
                    'city-L/100km']
SCALED_COLUMNS = ['curb-weight', 'peak-rpm', 'price']
ENCODED_COLUMN = 'horsepower-binned'


def section(title):
    print(f'\n{"=" * 70}\n{title}\n{"=" * 70}')


def categorical_columns(df):
    return [col for col in df.select_dtypes(exclude=['number', 'datetime']).columns
            if col != DATE_COLUMN]


# 1. Loading raw dataset
def load_data(path):
    df = pd.read_csv(path)
    print(f'[load] {path} -> {df.shape[0]} baris, {df.shape[1]} kolom')
    return df


# 2. Pemeriksaan kondisi awal dataset
def inspect_data(df):
    section('PEMERIKSAAN AWAL DATASET')

    print('\n[inspect] Lima baris pertama (ditampilkan transpose):')
    print(df.head().T.to_string())

    print(f'\n[inspect] Jumlah baris: {df.shape[0]}, jumlah kolom: {df.shape[1]}')

    print('\n[inspect] Nama kolom:')
    print(', '.join(df.columns))

    print('\n[inspect] Tipe data setiap kolom:')
    print(df.dtypes.to_string())

    print('\n[inspect] Jumlah missing values setiap kolom:')
    print(df.isna().sum().to_string())
    print(f'Total missing values: {df.isna().sum().sum()}')

    print(f'\n[inspect] Jumlah duplicate records: {df.duplicated().sum()}')

    print('\n[inspect] Nilai unik kolom kategorikal:')
    for col in categorical_columns(df):
        print(f'- {col} ({df[col].nunique()}): {df[col].dropna().unique().tolist()}')
    print(f'- {DATE_COLUMN}: contoh format -> {df[DATE_COLUMN].dropna().head(4).tolist()}')

    # Sebaran data numerikal untuk melihat nilai tidak wajar
    print('\n[inspect] Statistik kolom numerik:')
    print(df.describe().T[['min', 'mean', 'max']].to_string())


# 3. Data cleaning
def drop_duplicates(df):
    # Drop duplikat, simpan 1 baris original
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates(keep='first')
    print(f'[clean] drop {n_dup} baris duplikat')
    return df, n_dup


def drop_missing_values(df):
    # Jumlah baris dengan missing value sedikit (< 10% data) sehingga dilakukan drop rows
    n_before = len(df)
    df = df.dropna()
    print(f'[clean] drop {n_before - len(df)} baris yang memiliki missing value')
    return df.copy()


def standardize_categories(df):
    # Hapus spasi di awal dan akhir string, ubah semua huruf menjadi huruf kecil,
    # lalu seragamkan label yang bermakna sama
    columns = categorical_columns(df)
    for col in columns:
        df[col] = df[col].str.strip().str.lower()
    for col, aliases in CATEGORY_ALIASES.items():
        df[col] = df[col].replace(aliases)
    print(f'[clean] standarisasi teks (strip + lowercase) pada kolom: {", ".join(columns)}')
    return df


def parse_transaction_date(df):
    # Parsing `transaction_date` dengan beberapa tipe format tanggal yang berbeda
    date_strip = df[DATE_COLUMN].str.strip()
    parsed = pd.Series(pd.NaT, index=date_strip.index, dtype='datetime64[ns]')

    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(date_strip, format=fmt, errors='coerce'))

    n_failed = parsed.isna().sum()
    if n_failed:
        print(f'[clean] peringatan: {n_failed} tanggal gagal di-parse')

    df[DATE_COLUMN] = parsed
    print(f'[clean] konversi `{DATE_COLUMN}` ke tipe datetime')
    return df


def convert_words_to_numbers(df):
    # Angka dalam bentuk kata (two, four, ...) diubah menjadi integer
    for col, mapping in WORD_TO_NUMBER.items():
        unknown = set(df[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f'Nilai `{col}` tidak dikenali: {sorted(unknown)}')
        df[col] = df[col].map(mapping).astype(int)
    print(f'[clean] konversi angka string ke integer: {", ".join(WORD_TO_NUMBER)}')
    return df


def check_invalid_values(df):
    # Ukuran fisik / harga tidak mungkin bernilai nol atau negatif
    invalid = (df[POSITIVE_COLUMNS] <= 0).sum()
    invalid = invalid[invalid > 0]
    if invalid.empty:
        print('[clean] tidak ditemukan nilai tidak wajar (<= 0) pada kolom numerik')
    else:
        print(f'[clean] peringatan: nilai <= 0 ditemukan -> {invalid.to_dict()}')
    return df


def report_cleaning(raw, clean, n_dup):
    section('RINGKASAN CLEANING')
    print(f'Jumlah data      : {len(raw)} -> {len(clean)} baris '
          f'({len(raw) - len(clean)} baris dihapus)')
    print(f'Missing values   : {raw.isna().sum().sum()} -> {clean.isna().sum().sum()}')
    print(f'Duplikat dihapus : {n_dup} baris')

    print('Kolom yang berubah:')
    before = raw.loc[clean.index]
    for col in clean.columns:
        n_changed = (before[col].astype(str) != clean[col].astype(str)).sum()
        dtype_changed = before[col].dtype != clean[col].dtype
        if n_changed or dtype_changed:
            print(f'- {col}: {n_changed} nilai berubah, tipe {before[col].dtype} -> {clean[col].dtype}')


def clean_data(df):
    section('DATA CLEANING')
    raw = df

    df, n_dup = drop_duplicates(df)
    df = drop_missing_values(df)
    df = standardize_categories(df)
    df = parse_transaction_date(df)
    df = convert_words_to_numbers(df)
    df = check_invalid_values(df)

    report_cleaning(raw, df, n_dup)
    return df


# 4. Data transformation
def min_max_scaling(df):
    # Hasil scaling disimpan di kolom baru agar nilai asli tetap bisa dibandingkan
    scaler = MinMaxScaler()
    for col in SCALED_COLUMNS:
        df[f'{col}_scaled'] = scaler.fit_transform(df[[col]]).ravel()
    print(f'[transform] min-max scaling pada kolom: {", ".join(SCALED_COLUMNS)}')
    return df


def encode_horsepower_binned(df):
    dummies = pd.get_dummies(df[ENCODED_COLUMN], prefix='horsepower', dtype=int)
    print(f'[transform] one-hot encoding `{ENCODED_COLUMN}` -> {", ".join(dummies.columns)}')
    return pd.concat([df, dummies], axis=1)


def report_transformation(df):
    section('PERBANDINGAN SEBELUM & SESUDAH TRANSFORMASI (5 baris pertama)')
    scaled = [c for col in SCALED_COLUMNS for c in (col, f'{col}_scaled')]
    print(df[scaled].head().to_string())
    print()
    encoded = [ENCODED_COLUMN] + [c for c in df.columns if c.startswith('horsepower_')]
    print(df[encoded].head().to_string())


def transform_data(df):
    section('DATA TRANSFORMATION')
    df = min_max_scaling(df)
    df = encode_horsepower_binned(df)
    report_transformation(df)
    return df


# 5. Simpan hasil data pipeline ke file CSV baru
def save_data(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format=OUTPUT_DATE_FORMAT)
    section('SIMPAN HASIL')
    print(f'[save] {path} -> {df.shape[0]} baris, {df.shape[1]} kolom')


def run_pipeline(input_path, output_path):
    df = load_data(input_path)
    inspect_data(df)
    df = clean_data(df)
    df = transform_data(df)
    save_data(df, output_path)
    return df


def main():
    parser = argparse.ArgumentParser(description='Data pipeline automobileEDA')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Path raw dataset (CSV)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Path processed dataset (CSV)')
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == '__main__':
    main()
