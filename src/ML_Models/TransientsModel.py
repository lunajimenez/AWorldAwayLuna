#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransientsModel.py — NASA Exoplanet Harmonized ML Pipeline
-------------------------------------------------------------
Entrenamiento y evaluación de un modelo ML sobre el dataset armonizado 
de exoplanetas provenientes de misiones KEPLER, K2 y TESS.

Objetivo:
    - Predecir la clase final del tránsito (`final_disposition`):
        * CONFIRMED
        * CANDIDATE
        * FALSE_POSITIVE
    - Usar solo las variables armonizadas físicas observables.
    - Generar un modelo explicativo y reproducible para análisis posterior.

Dataset esperado:
    Located at: ../FilteredData/FinalDataHarmonization.csv
    Columns:
        orbital_period_days, transit_duration_hours, transit_depth_ppm, 
        planet_radius_earth, equilibrium_temperature_K, insolation_flux_Earth, 
        stellar_radius_solar, stellar_temperature_K, final_disposition, source_mission
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("[WARN] LightGBM no disponible; se usará RandomForest.", file=sys.stderr)

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    print("[WARN] imbalanced-learn no disponible; SMOTE desactivado.", file=sys.stderr)

# === CONFIGURACIÓN DE VARIABLES GLOBALES === #

FEATURES_GLOBAL = [
    "orbital_period_days",
    "transit_duration_hours",
    "transit_depth_ppm",
    "planet_radius_earth",
    "equilibrium_temperature_K",
    "insolation_flux_Earth",
    "stellar_radius_solar",
    "stellar_temperature_K",
]

TARGET = "final_disposition"
DEFAULT_CSV = "../../FilteredData/FinalDataHarmonization.csv"


# === FUNCIONES DE PREPARACIÓN === #

def load_and_prepare(csv_path: str):
    """Carga y prepara el dataset armonizado, imputando valores faltantes."""
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset leído: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Asegurar que las columnas esenciales estén
    missing = [c for c in FEATURES_GLOBAL + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas esenciales: {missing}")

    X = df[FEATURES_GLOBAL]
    y = df[TARGET].astype(str)

    # Imputación robusta
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Codificar etiquetas
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"🧠 Clases detectadas: {class_names}")

    return X_imp, y_enc, class_names, imputer, X.columns.tolist()


def build_model(random_state: int = 42):
    """Crea el modelo base (LightGBM o RandomForest)."""
    if _HAS_LGBM:
        print("[INFO] Usando LightGBM")
        model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    else:
        print("[INFO] Usando RandomForest")
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    return model


def cross_validate_model(model, X, y, folds=5):
    """Realiza validación cruzada con F1-macro."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"[CV] F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def train_and_evaluate(model, X, y, class_names, use_smote=False, outdir="outputs"):
    """Entrena, evalúa y genera reportes visuales."""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/figs", exist_ok=True)
    os.makedirs(f"{outdir}/reports", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if use_smote and _HAS_SMOTE:
        print("[INFO] Aplicando SMOTE para balancear clases.")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    print("\n📊 === Evaluación del modelo ===")
    print(classification_report(y_test, y_pred, target_names=class_names))
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(f"{outdir}/figs/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[INFO] Matriz de confusión guardada en: {outdir}/figs/confusion_matrix.png")

    # Importancia de características
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(indices)), importances[indices])
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(np.array(X.columns)[indices], rotation=90)
        ax.set_title("Feature Importances")
        plt.tight_layout()
        plt.savefig(f"{outdir}/figs/feature_importances.png", dpi=150)
        plt.close()
        print(f"[INFO] Importancias de características guardadas en: {outdir}/figs/feature_importances.png")
    except Exception as e:
        print(f"[WARN] No se pudieron graficar importancias: {e}", file=sys.stderr)

    # Guardar modelo
    joblib.dump(model, f"{outdir}/transients_model.joblib")
    print(f"✅ Modelo guardado en: {outdir}/transients_model.joblib")
    return model


# === MAIN PIPELINE === #

def main():
    parser = argparse.ArgumentParser(description="NASA Transient Exoplanets ML Pipeline 🌍")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Ruta al CSV armonizado (por defecto: FinalDataHarmonization.csv)")
    parser.add_argument("--smote", action="store_true", help="Aplicar SMOTE para balancear clases")
    parser.add_argument("--outdir", default="outputs", help="Carpeta de salida (default: outputs)")
    args = parser.parse_args()

    X, y, class_names, imputer, feature_names = load_and_prepare(args.csv)
    model = build_model()

    cross_validate_model(model, X, y)
    train_and_evaluate(model, X, y, class_names, use_smote=args.smote, outdir=args.outdir)


if __name__ == "__main__":
    main()