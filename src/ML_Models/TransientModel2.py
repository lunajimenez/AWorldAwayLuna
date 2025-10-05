#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransientsModel_Clean.py
------------------------
Modelo robusto de clasificación para señales de tránsito (CONFIRMED / CANDIDATE / FALSE_POSITIVE)
basado en el dataset armonizado FinalDataHarmonization.csv.

- Limpia y valida la columna 'final_disposition'
- Imputa valores nulos
- Entrena y evalúa con LightGBM (o RandomForest si LightGBM no está disponible)
- Genera métricas, gráficas y modelo guardado
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# === CONFIGURACIÓN ===
DATA_PATH = "../../FilteredData/FinalDataHarmonization.csv"
OUTDIR = "outputs_clean"
os.makedirs(OUTDIR, exist_ok=True)
os.makeDATA_PATH = "../../FilteredData/FinalDataHarmonization.csv"
os.makedirs(f"{OUTDIR}/figs", exist_ok=True)
os.makedirs(f"{OUTDIR}/reports", exist_ok=True) 

# === 1. CARGA Y LIMPIEZA DE DATOS ===
print(f"📥 Cargando dataset desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset leído: {df.shape[0]} filas × {df.shape[1]} columnas\n")

# Revisar las etiquetas del target
print("Valores únicos en final_disposition (antes de limpieza):")
print(df["final_disposition"].value_counts(dropna=False), "\n")

# Normalizar etiquetas
df["final_disposition"] = (
    df["final_disposition"].astype(str).str.strip().str.upper()
)

# Reemplazar NaN o desconocidos por "UNKNOWN"
df["final_disposition"] = df["final_disposition"].replace(
    {"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"}
)

# Filtrar solo las clases válidas
valid_classes = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
df = df[df["final_disposition"].isin(valid_classes)]

print("Valores únicos en final_disposition (después de limpieza):")
print(df["final_disposition"].value_counts(), "\n")

# === 2. SELECCIÓN DE FEATURES ===
target = "final_disposition"
drop_cols = ["source_mission"]
features = [c for c in df.columns if c not in [target] + drop_cols]

X = df[features].copy()
y = df[target].copy()

print(f"Características seleccionadas: {features}\n")

# === 3. IMPUTACIÓN DE DATOS NULOS ===
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# === 4. ENCODING DEL TARGET ===
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = list(le.classes_)

print(f"Clases detectadas: {class_names}\n")

# === 5. DIVISIÓN TRAIN / TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# === 6. CONSTRUCCIÓN DEL MODELO ===
if _HAS_LGBM:
    print("⚡ Usando LightGBM como modelo base.\n")
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
    )
else:
    print("🌲 Usando RandomForest (LightGBM no disponible).\n")
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

# === 7. ENTRENAMIENTO ===
print("🧠 Entrenando modelo...")
model.fit(X_train, y_train)
print("✅ Entrenamiento completado.\n")

# === 8. EVALUACIÓN ===
y_pred = model.predict(X_test)

rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print("📊 === Evaluación del modelo ===")
print(rep)
print(f"Balanced Accuracy: {bal_acc:.4f}")

# === 9. GUARDAR REPORTE ===
report_path = f"{OUTDIR}/reports/metrics.txt"
ax.set_yticklabels(class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/figs/confusion_matrix.png", dpi=150)
plt.close()
with open(report_path, "w", encoding="utf-8") as f:
    f.write(rep)
    f.write(f"\nBalanced Accuracy: {bal_acc:.4f}\n")
print(f"[INFO] Reporte guardado en: {report_path}\n")

# === 10. MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Matriz de confusión")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticklabels(class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/figs/confusion_matrix.png", dpi=150)
plt.close()
print(f"[INFO] Matriz de confusión guardada en: {OUTDIR}/figs/confusion_matrix.png")

# === 11. IMPORTANCIAS ===
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([features[i] for i in indices], rotation=90)
    ax.set_title("Importancias de características")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/figs/feature_importances.png", dpi=150)
    plt.close()
    print(f"[INFO] Importancias de características guardadas en: {OUTDIR}/figs/feature_importances.png")

# === 12. GUARDAR MODELO ===
bundle = {
    "model": model,
    "imputer": imputer,
    "label_encoder": le,
    "features": features,
    "class_names": class_names,
    "balanced_accuracy": bal_acc,
}
joblib.dump(bundle, f"{OUTDIR}/transients_model_clean.joblib")
print(f"✅ Modelo guardado en: {OUTDIR}/transients_model_clean.joblib")