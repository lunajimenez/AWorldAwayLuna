# -*- coding: utf-8 -*-
"""
ML_LightGBM_Lite.py
--------------------------------------
Versión liviana y optimizada del modelo ML para predecir exoplanetas transientes.
Usa LightGBM con parámetros robustos y ejecuta rápido (<2 min).

Autor: Luna 💫
"""

import warnings
from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    accuracy_score,
    confusion_matrix
)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ==== CONFIGURACIÓN ====
DATA_PATH = Path("/home/luna/Desktop/AWorldAwayLuna/FilteredData/FinalDataHarmonization.csv")
OUTDIR = Path("/home/luna/Desktop/AWorldAwayLuna/ModelOutputs/LightGBM_Lite")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2

print(f"📥 Leyendo dataset desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ==== PREPARAR TARGET ====
df["final_disposition"] = df["final_disposition"].str.upper()
df["final_disposition"] = df["final_disposition"].replace({
    "CP": "CONFIRMED", "KP": "CONFIRMED",
    "APC": "CANDIDATE", "PC": "CANDIDATE",
    "FP": "FALSE_POSITIVE", "FA": "FALSE_POSITIVE"
})
df = df.dropna(subset=["final_disposition"])

# Binario: Confirmado (1) vs No confirmado (0)
df["y_confirmed"] = (df["final_disposition"] == "CONFIRMED").astype(int)

# ==== SPLIT FEATURES ====
exclude = {"final_disposition", "source_mission", "y_confirmed"}
features = [c for c in df.columns if c not in exclude]

num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]
print(f"🔢 Numéricas: {len(num_cols)}, Categóricas: {len(cat_cols)}")

# ==== PREPROCESAMIENTO ====
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
], remainder="drop")

# ==== MODELO LIGHTGBM ====
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    n_jobs=2,
    random_state=SEED,
)

pipe = Pipeline([
    ("prep", pre),
    ("lgbm", model)
])

# ==== TRAIN / TEST SPLIT ====
X = df[num_cols + cat_cols]
y = df["y_confirmed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# ==== ENTRENAMIENTO ====
print("🚀 Entrenando modelo LightGBM...")
pipe.fit(X_train, y_train)

# ==== EVALUACIÓN ====
probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

roc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)
acc = accuracy_score(y_test, preds)
rep = classification_report(y_test, preds, digits=4)
cm = confusion_matrix(y_test, preds)

print("\n📊 === Resultados LightGBM_Lite ===")
print(rep)
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"ACC:     {acc:.4f}")

# ==== GUARDAR MÉTRICAS ====
metrics = {
    "ROC_AUC": roc,
    "PR_AUC": pr_auc,
    "Accuracy": acc,
    "Confusion_Matrix": cm.tolist(),
    "N_numeric": len(num_cols),
    "N_categorical": len(cat_cols),
    "Rows": len(df),
}
(OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# ==== GUARDAR FEATURE IMPORTANCES ====
importances = model.feature_importances_
fi = pd.DataFrame({
    "feature": num_cols + cat_cols,
    "importance": importances[:len(num_cols) + len(cat_cols)]
}).sort_values("importance", ascending=False)
fi.to_csv(OUTDIR / "feature_importances.csv", index=False)

# ==== GUARDAR MODELO ====
dump(pipe, OUTDIR / "lightgbm_lite_model.joblib")
print(f"\n💾 Modelo guardado en: {OUTDIR / 'lightgbm_lite_model.joblib'}")
print(f"📈 Métricas guardadas en: {OUTDIR / 'metrics.json'}")