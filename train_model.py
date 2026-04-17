"""
train_model.py
--------------
Script untuk:
1. Membersihkan dataset employee_data.csv
2. Melatih model Random Forest untuk prediksi Attrition
3. Menyimpan model pipeline (model.pkl) dan data bersih (employee_data_cleaned.csv)
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
print("=" * 60)
print("FASE 1: Memuat Dataset")
print("=" * 60)

df = pd.read_csv('employee_data.csv')
print(f"Total baris data   : {len(df)}")
print(f"Total kolom        : {len(df.columns)}")
print(f"Missing Attrition  : {df['Attrition'].isna().sum()}")
print()


# 2. PREPROCESSING

print("=" * 60)
print("FASE 2: Preprocessing Data")
print("=" * 60)

drop_cols = ['EmployeeId', 'EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=drop_cols, errors='ignore')
print(f"Kolom dihapus (tidak berguna): {drop_cols}")

df_labeled = df[df['Attrition'].notna()].copy()
df_unlabeled = df[df['Attrition'].isna()].copy()

print(f"Data berlabel      : {len(df_labeled)}")
print(f"Data tanpa label   : {len(df_unlabeled)}")

df_labeled['Attrition'] = df_labeled['Attrition'].astype(float).astype(int)
print(f"\nDistribusi Attrition:")
print(f"  Bertahan (0): {(df_labeled['Attrition'] == 0).sum()}")
print(f"  Resign   (1): {(df_labeled['Attrition'] == 1).sum()}")
print()

# 3. ENCODE CATEGORICAL FEATURES
print("=" * 60)
print("FASE 3: Encoding Fitur Kategorikal")
print("=" * 60)

# Identifikasi kolom kategorikal
cat_cols = df_labeled.select_dtypes(include=['object']).columns.tolist()
print(f"Kolom kategorikal: {cat_cols}")

# Buat LabelEncoders untuk setiap kolom kategorikal
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit pada gabungan data labeled + unlabeled agar semua kategori tercakup
    all_values = pd.concat([df_labeled[col], df_unlabeled[col]]).astype(str)
    le.fit(all_values)
    df_labeled[col] = le.transform(df_labeled[col].astype(str))
    df_unlabeled[col] = le.transform(df_unlabeled[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

print()

# 4. FEATURE SELECTION & MODEL TRAINING
print("=" * 60)
print("FASE 4: Training Model")
print("=" * 60)

feature_cols = [c for c in df_labeled.columns if c != 'Attrition']
X = df_labeled[feature_cols]
y = df_labeled['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set  : {len(X_train)} baris")
print(f"Testing set   : {len(X_test)} baris")

# Train model menggunakan Gradient Boosting (lebih akurat)
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bertahan', 'Resign']))

# ==========================================
# 5. FEATURE IMPORTANCE - PILIH TOP FEATURES
# ==========================================
print("=" * 60)
print("FASE 5: Top Feature Importance")
print("=" * 60)

importances = model.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Fitur Terpenting:")
for i, row in feat_imp.head(15).iterrows():
    bar = "#" * int(row['importance'] * 100)
    print(f"  {row['feature']:<28} {row['importance']:.4f} {bar}")

# Pilih top features untuk web form prediksi
# Kita pilih fitur yang paling penting DAN mudah dipahami user
TOP_N = 8
top_features = feat_imp.head(TOP_N)['feature'].tolist()
print(f"\nTop {TOP_N} fitur untuk form prediksi: {top_features}")

# ==========================================
# 6. TRAIN FINAL MODEL DENGAN TOP FEATURES
# ==========================================
print("\n" + "=" * 60)
print("FASE 6: Training Model Final (Top Features)")
print("=" * 60)

X_top_train = X_train[top_features]
X_top_test = X_test[top_features]

final_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

final_model.fit(X_top_train, y_train)
y_pred_final = final_model.predict(X_top_test)
accuracy_final = accuracy_score(y_test, y_pred_final)
print(f"Akurasi Model Final (Top {TOP_N}): {accuracy_final:.4f} ({accuracy_final*100:.1f}%)")

# ==========================================
# 7. PREDIKSI DATA UNLABELED
# ==========================================
print("\n" + "=" * 60)
print("FASE 7: Prediksi Data Tanpa Label")
print("=" * 60)

X_unlabeled = df_unlabeled[feature_cols]
predictions = model.predict(X_unlabeled)
df_unlabeled['Attrition'] = predictions

print(f"Data tanpa label berhasil diprediksi: {len(df_unlabeled)} baris")
print(f"  Diprediksi Bertahan: {(predictions == 0).sum()}")
print(f"  Diprediksi Resign  : {(predictions == 1).sum()}")

# ==========================================
# 8. BUAT CLEANED CSV (DECODE KEMBALI)
# ==========================================
print("\n" + "=" * 60)
print("FASE 8: Menyimpan Data Bersih")
print("=" * 60)

# Gabungkan kembali
df_full = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

# Decode kembali kolom kategorikal ke bentuk aslinya untuk CSV bersih
df_clean = df_full.copy()
for col in cat_cols:
    le = label_encoders[col]
    df_clean[col] = le.inverse_transform(df_clean[col].astype(int))

# Ganti 0/1 ke label yang jelas
df_clean['Attrition'] = df_clean['Attrition'].map({0: 'No', 1: 'Yes'})

df_clean.to_csv('employee_data_cleaned.csv', index=False)
print(f"Disimpan: employee_data_cleaned.csv ({len(df_clean)} baris)")

# ==========================================
# 9. SIMPAN MODEL & METADATA
# ==========================================
print("\n" + "=" * 60)
print("FASE 9: Menyimpan Model & Metadata")
print("=" * 60)

# Simpan model
with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Disimpan: model.pkl")

# Simpan metadata (fitur, encoders, dll)
# Kita perlu info ini di backend untuk decode input user
metadata = {
    'top_features': top_features,
    'feature_info': {},
    'accuracy': float(accuracy_final),
    'label_encoders': {}
}

# Simpan info setiap fitur untuk backend form
for feat in top_features:
    if feat in cat_cols:
        le = label_encoders[feat]
        metadata['feature_info'][feat] = {
            'type': 'categorical',
            'options': list(le.classes_),
            'mapping': {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        }
    else:
        metadata['feature_info'][feat] = {
            'type': 'numerical',
            'min': float(df_full[feat].min()),
            'max': float(df_full[feat].max()),
            'mean': float(df_full[feat].mean())
        }

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Disimpan: model_metadata.json")

print("\n" + "=" * 60)
print("SELESAI! Semua file berhasil dibuat.")
print("=" * 60)
print(f"\nFile yang dihasilkan:")
print(f"  1. employee_data_cleaned.csv  - Data bersih untuk Looker Studio")
print(f"  2. model.pkl                  - Model prediksi untuk backend")
print(f"  3. model_metadata.json        - Metadata fitur untuk form web")
