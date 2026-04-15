"""
modeling.py - Training Model Attrition dengan MLflow Tracking
Script ini melatih model Gradient Boosting untuk prediksi attrition karyawan
dan mencatat semua parameter, metrik, serta artefak ke MLflow + DagsHub.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

# ============================================================
# KONFIGURASI DAGSHUB
# ============================================================
DAGSHUB_USERNAME = "anndaanhr"
DAGSHUB_REPO_NAME = "utsds"

# Set MLflow tracking ke DagsHub
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI

# Token DagsHub (ambil dari: DagsHub > Profile > Settings > Tokens > Generate)
# Jika belum di-set, script akan meminta input
DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN', '')
if not DAGSHUB_TOKEN:
    DAGSHUB_TOKEN = input("Masukkan DagsHub Token Anda: ").strip()

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# ============================================================
# LOAD & PREPROCESSING DATA
# ============================================================
print("=" * 50)
print("LOADING DATA...")
print("=" * 50)

df = pd.read_csv("employee_data_cleaned.csv")
print(f"Total data: {len(df)} baris, {len(df.columns)} kolom")

# Hapus kolom yang tidak relevan untuk modeling
drop_cols = ['EmployeeId', 'EmployeeNumber']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Encode semua kolom kategorikal
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Pisahkan fitur dan target
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

# ============================================================
# TRAINING & LOGGING KE MLFLOW
# ============================================================
print("\n" + "=" * 50)
print("TRAINING MODEL + LOGGING KE MLFLOW & DAGSHUB...")
print("=" * 50)

# Parameter model
params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "random_state": 42,
    "test_size": 0.2
}

# Set nama experiment
mlflow.set_experiment("Attrition_Prediction")

with mlflow.start_run(run_name="GradientBoosting_v1"):

    # 1) Log parameter
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])
    mlflow.log_param("learning_rate", params["learning_rate"])
    mlflow.log_param("min_samples_split", params["min_samples_split"])
    mlflow.log_param("min_samples_leaf", params["min_samples_leaf"])
    mlflow.log_param("random_state", params["random_state"])
    mlflow.log_param("test_size", params["test_size"])
    mlflow.log_param("total_features", X.shape[1])
    mlflow.log_param("total_samples", len(df))

    # 2) Training model
    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)

    # 3) Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 4) Log metrik
    mlflow.log_metric("accuracy", round(accuracy, 4))
    mlflow.log_metric("precision", round(precision, 4))
    mlflow.log_metric("recall", round(recall, 4))
    mlflow.log_metric("f1_score", round(f1, 4))

    # 5) Log model ke MLflow
    mlflow.sklearn.log_model(model, "attrition_model")

    # 6) Log dataset sebagai artefak
    mlflow.log_artifact("employee_data_cleaned.csv")

    # 7) Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fi_path = "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    # Print hasil
    print(f"\nHasil Training:")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nTop 5 Fitur Terpenting:")
    print(feature_importance.head().to_string(index=False))

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    print(f"\nLihat hasil di DagsHub:")
    print(f"  https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}/experiments")

print("\n" + "=" * 50)
print("SELESAI! Cek DagsHub untuk melihat hasilnya.")
print("=" * 50)
