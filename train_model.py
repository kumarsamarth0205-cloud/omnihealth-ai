"""
train_model.py
--------------
Trains a Random Forest Classifier on the generated patient dataset.
Saves the trained model and column metadata so the Streamlit app can use it.

Run this AFTER generating data:
    python train_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "patients.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")
SCALER_PATH= os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH  = os.path.join(MODEL_DIR, "metadata.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Features used for training ─────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"


def load_data():
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found. Run  python data/generate_data.py  first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return X, y, df


def train_disease_model(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


def train_anomaly_model(X):
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.08,   # expect ~8% anomalies
        random_state=42
    )
    iso.fit(X)
    return iso


def main():
    print("\n[OmniHealth] Model Training Pipeline")
    print("=" * 45)

    # 1. Load
    df = load_data()
    X, y, df_clean = preprocess(df)

    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Train disease classifier
    print("\n[AI] Training Random Forest Classifier...")
    clf = train_disease_model(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # 5. Train anomaly detector
    print("[AI] Training Isolation Forest (Anomaly Detector)...")
    iso = train_anomaly_model(X_scaled)
    anomaly_labels = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_count = (anomaly_labels == -1).sum()
    print(f"   Detected {anomaly_count} anomalous patients in training data")

    # 6. Feature importance
    feature_importance = dict(zip(FEATURE_COLS, clf.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: -x[1])
    print("\n[INFO] Top 5 Most Important Features:")
    for feat, imp in top_features[:5]:
        print(f"   {feat:<12} -> {imp*100:.1f}%")

    # 7. Save everything
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({
        "feature_cols":       FEATURE_COLS,
        "accuracy":           round(acc * 100, 1),
        "feature_importance": feature_importance,
        "total_patients":     len(df_clean),
        "disease_count":      int(y.sum()),
        "healthy_count":      int((y == 0).sum()),
    }, META_PATH)

    print(f"\n[OK] Model saved   -> {MODEL_PATH}")
    print(f"[OK] Scaler saved  -> {SCALER_PATH}")
    print(f"[OK] Metadata saved-> {META_PATH}")
    print("\n[DONE] Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
