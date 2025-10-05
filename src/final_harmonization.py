# final_harmonization.py
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/luna/Desktop/AWorldAwayLuna/FilteredData')
KOI_FP = BASE / 'KOI_All_Filtrated.csv'
K2_FP  = BASE / 'K2_All_Filtrated.csv'
TOI_FP = BASE / 'TOI_All_Filtrated.csv'
OUT_FP = BASE / 'FinalDataHarmonization.csv'

print("=== INICIO DEL PROCESO DE ARMONIZACIÓN ===\n")
print("Archivos utilizados:")
print(f"KOI: {KOI_FP}")
print(f"K2 : {K2_FP}")
print(f"TOI: {TOI_FP}\n")

# Leer los datasets
koi_raw = pd.read_csv(KOI_FP)
k2_raw  = pd.read_csv(K2_FP)
toi_raw = pd.read_csv(TOI_FP)

# --- Mapeos de columnas ---
koi_map = {
    'koi_period': 'orbital_period_days',
    'koi_duration': 'transit_duration_hours',
    'koi_depth': 'transit_depth_ppm',
    'koi_prad': 'planet_radius_earth',
    'koi_teq': 'equilibrium_temperature_K',
    'koi_insol': 'insolation_flux_Earth',
    'koi_srad': 'stellar_radius_solar',
    'koi_steff': 'stellar_temperature_K',
    'koi_disposition': 'final_disposition'
}

k2_map = {
    'pl_orbper': 'orbital_period_days',
    'pl_trandur': 'transit_duration_hours',
    'pl_trandep': 'transit_depth_ppm',
    'pl_rade': 'planet_radius_earth',
    'pl_eqt': 'equilibrium_temperature_K',
    'pl_insol': 'insolation_flux_Earth',
    'st_rad': 'stellar_radius_solar',
    'st_teff': 'stellar_temperature_K',
    'disposition': 'final_disposition'
}

toi_map = {
    'pl_orbper': 'orbital_period_days',
    'pl_trandurh': 'transit_duration_hours',
    'pl_trandep': 'transit_depth_ppm',
    'pl_rade': 'planet_radius_earth',
    'pl_eqt': 'equilibrium_temperature_K',
    'pl_insol': 'insolation_flux_Earth',
    'st_rad': 'stellar_radius_solar',
    'st_teff': 'stellar_temperature_K',
    'tfopwg_disp': 'final_disposition'
}

# --- Función general de preparación ---
def prepare_df(df_raw, mapping, mission):
    df = df_raw.rename(columns={k: v for k, v in mapping.items() if k in df_raw.columns}).copy()
    for col in mapping.values():
        if col in df.columns and col != 'final_disposition':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if mission == 'K2' and 'transit_depth_ppm' in df.columns:
        mx = df['transit_depth_ppm'].max(skipna=True)
        if pd.notna(mx) and mx < 1e4:
            print("  [K2] Convertido transit_depth de % a ppm (x1e4)")
            df['transit_depth_ppm'] *= 1e4
    return df

# --- Aplicar mapeos ---
koi_p = prepare_df(koi_raw, koi_map, 'KOI')
k2_p  = prepare_df(k2_raw, k2_map, 'K2')
toi_p = prepare_df(toi_raw, toi_map, 'TOI')

# --- Normalizar final_disposition ---
def unify_disposition(val, source=None):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper().replace('-', ' ').replace('_',' ')
    if source == 'TOI':
        if s in ('CP','KP','CONFIRMED'): return 'CONFIRMED'
        if s in ('PC','APC','CANDIDATE'): return 'CANDIDATE'
        if s in ('FP','FA','FALSE POSITIVE'): return 'FALSE_POSITIVE'
    if 'CONFIR' in s: return 'CONFIRMED'
    if 'CAND' in s: return 'CANDIDATE'
    if 'FALSE' in s or s in ('FP','FA'): return 'FALSE_POSITIVE'
    return np.nan

for df, src in [(koi_p,'KOI'),(k2_p,'K2'),(toi_p,'TOI')]:
    if 'final_disposition' in df.columns:
        df['final_disposition'] = df['final_disposition'].apply(lambda v: unify_disposition(v, src))

# --- Filtrado esencial ---
essential = [
    'orbital_period_days',
    'transit_duration_hours',
    'transit_depth_ppm',
    'planet_radius_earth',
    'equilibrium_temperature_K',
    'insolation_flux_Earth',
    'stellar_radius_solar',
    'stellar_temperature_K',
    'final_disposition'
]

def filter_essential(df, mission):
    before = df.shape[0]
    df_clean = df.dropna(subset=essential)
    after = df_clean.shape[0]
    print(f"[{mission}] Filtrado esencial: {before} → {after} filas (eliminadas {before - after})")
    return df_clean

koi_clean = filter_essential(koi_p, 'KOI')
k2_clean  = filter_essential(k2_p, 'K2')
toi_clean = filter_essential(toi_p, 'TOI')

# --- Añadir columna fuente ---
koi_clean['source_mission'] = 'KEPLER'
k2_clean['source_mission']  = 'K2'
toi_clean['source_mission'] = 'TESS'

# --- Selección final de columnas ---
cols_global = [
    'orbital_period_days','transit_duration_hours','transit_depth_ppm',
    'planet_radius_earth','equilibrium_temperature_K','insolation_flux_Earth',
    'stellar_radius_solar','stellar_temperature_K','final_disposition','source_mission'
]

def ensure_cols(df):
    for c in cols_global:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols_global]

koi_final = ensure_cols(koi_clean)
k2_final  = ensure_cols(k2_clean)
toi_final = ensure_cols(toi_clean)

# --- Concatenar ---
harm = pd.concat([koi_final, k2_final, toi_final], ignore_index=True)
print("\n✅ Dataset armonizado creado correctamente.")
print("Dimensiones finales:", harm.shape)
print("\nDistribución de final_disposition:")
print(harm['final_disposition'].value_counts())

# --- Guardar ---
harm.to_csv(OUT_FP, index=False)
print(f"\n💾 Archivo final guardado en: {OUT_FP}")