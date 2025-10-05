#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransientsModel_Stacking.py
---------------------------
Stacking ensemble (LightGBM + RandomForest) para clasificar `final_disposition`
a partir del dataset armonizado de exoplanetas.

Salida:
  - outputs/stacking_model.joblib       -> modelo entrenado (stack)
  - outputs/reports/metrics.txt         -> reporte texto con métricas
  - outputs/figs/confusion_matrix.png   -> matriz de confusión
  - outputs/figs/perm_importances.png   -> importancias por permutación
  - outputs/ (otros archivos de diagnóstico)

Usage:
  python TransientsModel_Stacking.py [--csv PATH] [--smote] [--outdir outputs]
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

# --- FEATURES & DEFAULTS (ajusta si cambias columnas) --- #
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

# default CSV paths to try (script lives in src/ML_Models)
CSV_TRIES = [
    Path("../../FilteredData/FinalDataHarmonization.csv"),  # likely when running from src/ML_Models
    Path("../FilteredData/FinalDataHarmonization.csv"),
    Path("./FilteredData/FinalDataHarmonization.csv"),
    Path("../../FilteredData/Harmonized_Intersection.csv"),
    Path("../FilteredData/Harmonized_Intersection.csv"),
]

# -------------------------
# Utility & I/O
# -------------------------
def resolve_csv_path(user_path: str = None) -> Path:
    """Return a Path to the CSV: prefer user_path, else try canned locations."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        else:
            raise FileNotFoundError(f"CSV especificado no existe: {user_path}")

    for p in CSV_TRIES:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró FinalDataHarmonization.csv en las rutas esperadas. "
                            f"Buscadas: {CSV_TRIES}. Usa --csv para dar la ruta exacta.")

def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data loading & prep
# -------------------------
def load_and_prepare(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], SimpleImputer, List[str]]:
    """Carga CSV, selecciona features, imputa y codifica etiquetas.
    Devuelve: X_imp (DataFrame), y_enc (ndarray), class_names, imputer, feature_names
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset cargado: {csv_path} -> {df.shape[0]} filas × {df.shape[1]} columnas")

    # verificar columnas
    missing = [c for c in FEATURES_GLOBAL + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas esenciales en el CSV: {missing}")

    X = df[FEATURES_GLOBAL].copy()
    y = df[TARGET].astype(str).copy()

    # Imputación simple (mediana) para reproducibilidad rápida
    imputer = SimpleImputer(strategy="median")
    X_imp_array = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp_array, columns=X.columns)

    # Label encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"[INFO] Clases encontradas: {class_names}")

    return X_imp, y_enc, class_names, imputer, X_imp.columns.tolist()

# -------------------------
# Models: base learners & stacking
# -------------------------
def build_base_learners(random_state: int = 42):
    """Crea base learners: LGBM (si disponible) y RandomForest."""
    learners = []
    # LightGBM
    if _HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            class_weight='balanced',
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1
        )
        learners.append(('lgbm', lgbm))
    else:
        print("[WARN] LightGBM no disponible; solo RF será base learner.", file=sys.stderr)

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )
    learners.append(('rf', rf))

    return learners

def build_stacking_estimator(random_state: int = 42):
    """Construye un StackingClassifier con meta-estimador logístico."""
    estimators = build_base_learners(random_state=random_state)
    # Meta-estimador: logística con regularización C=1.0
    final_estimator = LogisticRegression(max_iter=500, class_weight='balanced', random_state=random_state)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,               # cross-validation folds internally for stacking
        n_jobs=-1,
        passthrough=False,  # no pasar features originales al meta-estimator; cambia si quieres
    )
    return stack

# -------------------------
# Eval & Train
# -------------------------
def cross_validate_model(model, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"[CV] F1-macro: mean={scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

def train_and_evaluate(model, X, y, class_names, use_smote=False, outdir: Path = Path("outputs")):
    """Entrena el stacking estimator, evalúa y guarda outputs."""
    ensure_dirs([outdir, outdir / "figs", outdir / "reports"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # SMOTE opcional
    if use_smote:
        if not _HAS_SMOTE:
            print("[WARN] SMOTE solicitado pero imblearn no instalado; saltando SMOTE.", file=sys.stderr)
        else:
            print("[INFO] Aplicando SMOTE al conjunto de entrenamiento...")
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

    print("[INFO] Entrenando el estimador de stacking...")
    model.fit(X_train, y_train)

    # predicción y métricas
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print("\n[TEST] Classification report:\n", report)
    print(f"[TEST] Balanced accuracy: {bal_acc:.4f}")

    # ROC-AUC si hay probabilidades
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba is not None:
                if y_proba.shape[1] == len(class_names) and len(class_names) > 2:
                    roc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                else:
                    roc = roc_auc_score(y_test, y_proba[:, 1])
                print(f"[TEST] ROC-AUC: {roc:.4f}")
    except Exception as e:
        print(f"[WARN] No se pudo calcular ROC-AUC: {e}", file=sys.stderr)

    # Guardar report
    rpt_path = outdir / "reports" / "metrics.txt"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("Classification report\n")
        f.write(report + "\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n")
    print(f"[INFO] Guardado reporte en: {rpt_path}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
    fig.tight_layout()
    cm_path = outdir / "figs" / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Matriz de confusión guardada en: {cm_path}")

    # Permutation importance (global, sobre test set)
    try:
        print("[INFO] Calculando importancias por permutación (puede tardar)...")
        result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.bar(range(len(importances)), importances[indices])
        ax2.set_xticks(range(len(importances)))
        ax2.set_xticklabels(np.array(X.columns)[indices], rotation=90)
        ax2.set_title("Permutation Importances (test set)")
        fig2.tight_layout()
        imp_path = outdir / "figs" / "perm_importances.png"
        fig2.savefig(imp_path, dpi=150)
        plt.close(fig2)
        print(f"[INFO] Importancias por permutación guardadas en: {imp_path}")

        # Print top 10
        print("[INFO] Top features (permutation importance):")
        for rank in range(min(10, len(indices))):
            idx = indices[rank]
            print(f"  {rank+1:2d}. {X.columns[idx]}  ({importances[idx]:.6f})")
    except Exception as e:
        print(f"[WARN] No se pudieron calcular importancias por permutación: {e}", file=sys.stderr)

    # Guardar el stack completo
    model_bundle = {
        "model": model,
        "features": list(X.columns),
        "class_names": class_names,
        "target": TARGET
    }
    model_path = outdir / "stacking_model.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"[INFO] Bundle guardado en: {model_path}")

    return model

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stacking ensemble: LightGBM + RandomForest for transients")
    p.add_argument("--csv", default=None, help="Ruta al CSV armonizado (si omites, se intenta resolver automáticamente)")
    p.add_argument("--smote", action="store_true", help="Aplicar SMOTE al entrenamiento (si está disponible)")
    p.add_argument("--outdir", default="outputs", help="Directorio de salida")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        csv_path = resolve_csv_path(args.csv)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    X, y, class_names, imputer, feature_names = load_and_prepare(csv_path)
    stack = build_stacking_estimator(random_state=42)

    print("[INFO] Cross-validando el stacking (F1-macro, 5-fold)...")
    cross_validate_model(stack, X, y, folds=5)

    stack_fit = train_and_evaluate(stack, X, y, class_names, use_smote=args.smote, outdir=Path(args.outdir))

if __name__ == "__main__":
    main()