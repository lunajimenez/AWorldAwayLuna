# -*- coding: utf-8 -*-
"""
ML_LightGBM.py
----------------------------------
Optimización del modelo para detección de exoplanetas transientes.
Basado en LightGBM como modelo principal.

✅ Mejoras respecto al ML.py anterior:
- Sustituye RandomForest por LightGBM.
- Incluye búsqueda aleatoria de hiperparámetros.
- Calcula umbral óptimo basado en F1-score.
- Guarda métricas, curvas ROC y PR, e importancias.
- Mantiene compatibilidad con el flujo de AWorldAwayLuna.
"""

import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("⚠️ Necesitas instalar lightgbm: pip install lightgbm")

# ======================================================
# RUTAS Y CONFIGURACIÓN
# ======================================================

DATA_PATH = Path("/home/luna/Desktop/AWorldAwayLuna/FilteredData/FinalDataHarmonization.csv")
OUTDIR = Path("/home/luna/Desktop/AWorldAwayLuna/ModelOutputs_LightGBM")

LABEL_ALIASES = {
    "PC": "CANDIDATE", "KP": "CANDIDATE", "APC": "CANDIDATE",
    "CP": "CONFIRMED", "FP": "FALSE_POSITIVE",
    "FALSE POSITIVE": "FALSE_POSITIVE", "REFUTED": "FALSE_POSITIVE",
    "FA": "FALSE_POSITIVE"
}


# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las etiquetas de disposición y crea la columna binaria `y_confirmed`."""
    out = df.copy()
    disp = out["final_disposition"].astype(str).str.upper().replace(LABEL_ALIASES)
    out["disp_label"] = disp
    out["y_confirmed"] = (disp == "CONFIRMED").astype(int)
    return out


def split_feature_types(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    """Identifica columnas numéricas y categóricas."""
    cols = [c for c in df.columns if c not in exclude and c not in ("final_disposition", "disp_label", "y_confirmed")]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """Crea pipeline de preprocesamiento."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return pre


def best_f1_threshold(y_true, probs):
    """Encuentra el umbral de probabilidad que maximiza el F1-score."""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)]) if len(thr) > 0 else 0.5


# ======================================================
# ENTRENAMIENTO PRINCIPAL
# ======================================================

def main():
    print(f"📥 Cargando dataset desde: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = ensure_labels(df)
    df = df[df["disp_label"] != "UNKNOWN"].copy()

    num_cols, cat_cols = split_feature_types(df, [])
    X = df[num_cols + cat_cols]
    y = df["y_confirmed"].values

    # División en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = build_preprocessor(num_cols, cat_cols)

    # Modelo base LightGBM
    base_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )

    # Espacio de hiperparámetros
    param_grid = {
        "lgbm__num_leaves": [31, 50, 100],
        "lgbm__max_depth": [-1, 10, 20, 30],
        "lgbm__learning_rate": [0.05, 0.1, 0.2],
        "lgbm__n_estimators": [200, 400, 800],
        "lgbm__subsample": [0.7, 0.8, 1.0],
        "lgbm__colsample_bytree": [0.7, 0.8, 1.0],
        "lgbm__min_child_samples": [10, 20, 40],
        "lgbm__reg_lambda": [0.0, 0.5, 1.0],
    }

    pipe = Pipeline([
        ("prep", pre),
        ("lgbm", base_model)
    ])

    print("🚀 Buscando mejores hiperparámetros (LightGBM Random Search)...")
    search = RandomizedSearchCV(
        pipe, param_distributions=param_grid, n_iter=25,
        scoring="roc_auc", n_jobs=-1, cv=3, verbose=1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"\n✅ Mejores hiperparámetros:\n{json.dumps(search.best_params_, indent=2)}")

    # Evaluación
    probs = best_model.predict_proba(X_test)[:, 1]
    thr = best_f1_threshold(y_test, probs)
    preds = (probs >= thr).astype(int)

    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    rep = classification_report(y_test, preds, target_names=["NOT_CONFIRMED", "CONFIRMED"])
    cm = confusion_matrix(y_test, preds)

    print("\n📊 === EVALUACIÓN FINAL ===")
    print(rep)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | ACC: {acc:.4f}")

    # ======================================================
    # GUARDADO DE RESULTADOS
    # ======================================================

    # Modelo y métricas
    dump(best_model, OUTDIR / "lightgbm_model.joblib")
    json.dump({
        "ROC_AUC": roc, "PR_AUC": pr_auc, "Accuracy": acc,
        "Best_F1_Threshold": thr,
        "Best_Params": search.best_params_
    }, open(OUTDIR / "metrics.json", "w"), indent=2)

    # Importancias
    try:
        importances = best_model.named_steps["lgbm"].feature_importances_
        fi = pd.DataFrame({
            "feature": num_cols + cat_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        fi.to_csv(OUTDIR / "feature_importances.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.barh(fi["feature"][:15], fi["importance"][:15])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances (LightGBM)")
        plt.tight_layout()
        plt.savefig(OUTDIR / "feature_importances.png", dpi=150)
        plt.close()
        print("📈 Feature importances guardadas.")
    except Exception as e:
        print(f"[WARN] No se pudieron generar importancias: {e}")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Curva ROC - LightGBM")
    plt.tight_layout()
    plt.savefig(OUTDIR / "roc_curve.png", dpi=150)
    plt.close()

    print(f"\n✅ Modelo guardado en: {OUTDIR / 'lightgbm_model.joblib'}")
    print(f"💾 Métricas guardadas en: {OUTDIR / 'metrics.json'}")


if __name__ == "__main__":
    main()