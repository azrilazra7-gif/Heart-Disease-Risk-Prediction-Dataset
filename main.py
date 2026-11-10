# ============================================================
# ANALISIS RISIKO PENYAKIT JANTUNG
# Tahapan: Preprocessing Data, Modeling, Evaluasi Model, Results
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# Folder tempat file main.py berada
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "heart_disease_risk_dataset_earlymed.csv"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 65)
    print("ANALISIS RISIKO PENYAKIT JANTUNG")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. PREPROCESSING DATA
    # --------------------------------------------------------
    print("\n1. PREPROCESSING DATA")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset tidak ditemukan. Pastikan file CSV berada satu folder dengan main.py."
        )

    df = pd.read_csv(DATA_FILE)
    print("\n5 data pertama:")
    print(df.head())
    print("\nUkuran data awal:", df.shape)
    print("\nJumlah nilai kosong per kolom:")
    print(df.isnull().sum())
    print("\nJumlah baris duplikat:", df.duplicated().sum())

    # Hapus baris duplikat agar data yang sama tidak berulang.
    df = df.drop_duplicates().copy()
    print("Ukuran data setelah duplikat dihapus:", df.shape)

    # Pastikan kolom target tersedia.
    target = "Heart_Risk"
    if target not in df.columns:
        raise ValueError("Kolom target 'Heart_Risk' tidak ditemukan dalam dataset.")

    # Pisahkan fitur (X) dan target (y).
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    print("\nDistribusi target (0 = risiko rendah, 1 = risiko tinggi):")
    print(y.value_counts().sort_index())

    # Simpan grafik distribusi target.
    ax = y.value_counts().sort_index().plot(kind="bar", rot=0)
    ax.set_title("Distribusi Target Heart_Risk")
    ax.set_xlabel("Kategori Heart_Risk")
    ax.set_ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "distribusi_target.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 2. PEMBAGIAN DATA DAN MODELING
    # --------------------------------------------------------
    print("\n2. MODELING")
    # Baris dengan fitur yang sama dikelompokkan agar tidak tersebar
    # sekaligus ke data latih dan data uji (mengurangi data leakage).
    groups = X.astype(str).agg("|".join, axis=1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Jumlah data latih: {len(X_train)}")
    print(f"Jumlah data uji  : {len(X_test)}")

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=42,
                n_jobs=-1, class_weight="balanced"
            ))
        ])
    }

    scores = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"\nMelatih model: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        predictions[name] = pred
        probabilities[name] = prob
        scores[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }

    # --------------------------------------------------------
    # 3. EVALUASI MODEL
    # --------------------------------------------------------
    print("\n3. EVALUASI MODEL")
    results_df = pd.DataFrame(scores).T
    print("\nTabel hasil evaluasi:")
    print(results_df.round(4))
    results_df.to_csv(RESULT_DIR / "hasil_evaluasi.csv", index=True)

    for name in models:
        print(f"\nLaporan klasifikasi: {name}")
        print(classification_report(
            y_test, predictions[name],
            target_names=["Risiko rendah (0)", "Risiko tinggi (1)"],
            zero_division=0
        ))

    # Grafik perbandingan metrik.
    results_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Perbandingan Kinerja Model")
    plt.xlabel("Model")
    plt.ylabel("Nilai")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "perbandingan_model.png", dpi=300)
    plt.close()

        # Confusion matrix untuk model dengan F1-score tertinggi.
    best_name = results_df["F1-Score"].idxmax()
    best_model = models[best_name]

    print(f"\nModel terbaik berdasarkan F1-Score: {best_name}")

    # Menyimpan model terbaik
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    # Menyimpan nama fitur
    feature_path = MODEL_DIR / "feature_columns.pkl"
    joblib.dump(list(X.columns), feature_path)

    print("\nModel berhasil disimpan!")
    print("Lokasi model:", model_path)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions[best_name],
        display_labels=["Risiko rendah", "Risiko tinggi"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Kurva ROC kedua model.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in models:
        RocCurveDisplay.from_predictions(
            y_test, probabilities[name], name=name, ax=ax
        )
    ax.set_title("ROC Curve")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # Feature importance jika model terbaik adalah Random Forest.
    if best_name == "Random Forest":
        rf = best_model.named_steps["model"]
        importance = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(10)
        importance.to_csv(RESULT_DIR / "feature_importance.csv", header=["Importance"])
        importance.sort_values().plot(kind="barh", figsize=(8, 6))
        plt.title("10 Fitur Paling Berpengaruh - Random Forest")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_importance.png", dpi=300)
        plt.close()

    # --------------------------------------------------------
    # 4. RESULTS / HASIL AKHIR
    # --------------------------------------------------------
    print("\n4. RESULTS / HASIL AKHIR")
    print("Model terbaik berdasarkan F1-Score:", best_name)
    print("Nilai evaluasi model terbaik:")
    print(results_df.loc[best_name].round(4))
    print("\nSemua hasil dan grafik tersimpan di folder:", RESULT_DIR)
    print("Selesai. Periksa folder results untuk melihat file hasil.")


if __name__ == "__main__":
    main()
