# StrictRFModel.py
# -*- coding: utf-8 -*-
"""
Entrena un modelo RandomForest robusto usando tu dataset armonizado final.
Entrada:  /home/luna/Desktop/AWorldAwayLuna/FilteredData/FinalDataHarmonization.csv
Salida:    /home/luna/Desktop/AWorldAwayLuna/ModelOutputs/
"""
import json
import warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================
# CONFIGURACIÓN LOCAL
# ===========================
DATA_PATH = Path("/home/luna/Desktop/AWorldAwayLuna/FilteredData/FinalDataHarmonization.csv")
OUTDIR = Path("/home/luna/Desktop/AWorldAwayLuna/ModelOutputs")

LABEL_ALIASES = {
    "PC": "CANDIDATE",
    "KP": "CANDIDATE",
    "APC": "CANDIDATE",
    "CP": "CONFIRMED",
    "FP": "FALSE_POSITIVE",
    "FALSE POSITIVE": "FALSE_POSITIVE",
    "REFUTED": "FALSE_POSITIVE",
    "FA": "FALSE_POSITIVE",
}

EXCLUDE_ALWAYS = {
    "final_disposition", "final_disposition_raw", "disp_label", "y_confirmed",
    "rowid", "id", "idx", "index"
}


# --------------------------
# FUNCIONES AUXILIARES
# --------------------------
def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "final_disposition" in out.columns:
        disp = out["final_disposition"].astype(str).str.upper()
    elif "final_disposition_raw" in out.columns:
        disp = out["final_disposition_raw"].astype(str).str.upper().replace(LABEL_ALIASES)
    else:
        disp = pd.Series(["UNKNOWN"] * len(out), index=out.index)
    out["disp_label"] = disp
    out["y_confirmed"] = (disp == "CONFIRMED").astype(int)
    return out


def split_feature_types(df: pd.DataFrame, exclude_cols: List[str], auto_drop_ids: bool = True):
    cols = [c for c in df.columns if c not in exclude_cols and c not in EXCLUDE_ALWAYS]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    dropped_cols = []

    if auto_drop_ids and len(df) > 0:
        to_drop = []
        for c in cat_cols:
            n_unique = df[c].nunique(dropna=True)
            if n_unique / max(1, len(df)) >= 0.9:
                to_drop.append(c)
        cat_cols = [c for c in cat_cols if c not in to_drop]
        dropped_cols.extend(to_drop)

    def drop_too_missing(candidates, threshold=0.98):
        keep, drop = [], []
        na_frac = df[candidates].isna().mean()
        for c in candidates:
            if na_frac[c] < threshold:
                keep.append(c)
            else:
                drop.append(c)
        return keep, drop

    num_cols, drop_n = drop_too_missing(num_cols)
    cat_cols, drop_c = drop_too_missing(cat_cols)
    dropped_cols.extend(drop_n + drop_c)

    def drop_constants(candidates):
        keep, drop = [], []
        for c in candidates:
            vals = df[c].dropna().unique()
            if len(vals) > 1:
                keep.append(c)
            else:
                drop.append(c)
        return keep, drop

    num_cols, drop_n2 = drop_constants(num_cols)
    cat_cols, drop_c2 = drop_constants(cat_cols)
    dropped_cols.extend(drop_n2 + drop_c2)
    return num_cols, cat_cols, dropped_cols


def build_preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return pre


def compute_best_f1_threshold(y_true, probs):
    prec, rec, thr = precision_recall_curve(y_true, probs)
    if len(thr) == 0:
        return 0.5
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1s)])


# --------------------------
# ENTRENAMIENTO PRINCIPAL
# --------------------------
def main():
    print(f"📥 Leyendo datos desde: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = ensure_labels(df)
    df.to_csv(OUTDIR / "dataset_used.csv", index=False)
    labeled = df[df["disp_label"] != "UNKNOWN"].copy()

    num_cols, cat_cols, dropped = split_feature_types(labeled, exclude_cols=[], auto_drop_ids=True)
    print(f"🔢 Numéricas: {len(num_cols)}, Categóricas: {len(cat_cols)}, Excluidas: {len(dropped)}")

    pre = build_preprocessor(num_cols, cat_cols)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("rf", model)])

    X = labeled[num_cols + cat_cols]
    y = labeled["y_confirmed"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    best_thr = compute_best_f1_threshold(y_test, probs)
    preds = (probs >= best_thr).astype(int)

    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"\n✅ Resultados del modelo:")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ACC:     {acc:.4f}")

    dump(pipe, OUTDIR / "strict_rf_model.joblib")
    print(f"\n💾 Modelo guardado en: {OUTDIR}/strict_rf_model.joblib")

    (OUTDIR / "metrics.json").write_text(json.dumps({
        "ROC_AUC": roc,
        "PR_AUC": pr_auc,
        "Accuracy": acc,
        "Threshold": best_thr
    }, indent=2), encoding="utf-8")

    print(f"\n Métricas guardadas en: {OUTDIR}/metrics.json")


if __name__ == "__main__":
    main()