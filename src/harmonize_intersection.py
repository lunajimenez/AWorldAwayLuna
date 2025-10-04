# diagnostics_and_harmonize.py
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/luna/Desktop/AWorldAwayLuna/FilteredData')
KOI_FP = BASE / 'KOI_All_Filtrated.csv'
K2_FP  = BASE / 'K2_All_Filtrated.csv'
TOI_FP = BASE / 'TOI_All_Filtrated.csv'
OUT_FP = BASE / 'Harmonized_Intersection.csv'

print("Rutas usadas:")
print(KOI_FP)
print(K2_FP)
print(TOI_FP)
print()

# --- leer sin forzar tipos ---
koi_raw = pd.read_csv(KOI_FP)
k2_raw  = pd.read_csv(K2_FP)
toi_raw = pd.read_csv(TOI_FP)

print("Resumen de columnas leídas (KOI):", koi_raw.shape, "\n", list(koi_raw.columns)[:50])
print("Resumen de columnas leídas (K2):", k2_raw.shape, "\n", list(k2_raw.columns)[:50])
print("Resumen de columnas leídas (TOI):", toi_raw.shape, "\n", list(toi_raw.columns)[:50])
print("\n--- Diagnóstico específico K2 antes de renombrar ---")
print(k2_raw.head(5).T)         # ver primeros 5 registros (transpuesto para leer mejor)
print("\nK2 - percent NaN por columna (raw):")
print((k2_raw.isna().mean() * 100).sort_values(ascending=False).round(3))

# show some value samples for suspected columns
for col in ['pl_orbper','pl_trandur','pl_trandep','pl_rade','pl_eqt','pl_insol','st_rad','st_teff','disposition']:
    if col in k2_raw.columns:
        print(f"\nK2 sample values for '{col}': unique (up to 20):")
        print(k2_raw[col].astype(str).replace('nan','<NA>').unique()[:20])
    else:
        print(f"\nK2: column '{col}' NOT FOUND in file. Available columns: {', '.join(list(k2_raw.columns)[:20])}")

# --- Ahora aplicamos el mapeo/renombrado conocido y conversión con logs --- 
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

# función robusta para renombrar y convertir columnas
def prepare_df(df_raw, mapping, mission_name):
    df = df_raw.copy()
    present = [c for c in mapping.keys() if c in df.columns]
    missing = [c for c in mapping.keys() if c not in df.columns]
    print(f"\n[{mission_name}] mapping: {len(present)} found, {len(missing)} missing.")
    if missing:
        print(f"  Missing columns: {missing}")
    df = df.rename(columns={c:mapping[c] for c in present})
    # print sample of renamed columns
    print(f"[{mission_name}] Columns after rename (sample): {list(df.columns)[:30]}")
    # convert numeric columns except disposition
    for col in mapping.values():
        if col == 'final_disposition': 
            continue
        if col in df.columns:
            # strip whitespace and commas in numbers
            df[col] = df[col].astype(str).str.strip().str.replace(',', '').replace({'': np.nan, 'nan': np.nan, 'NA': np.nan, 'None': np.nan})
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # special handling: K2 pl_trandep might be percent -> ppm
    if mission_name == 'K2' and 'transit_depth_ppm' in df.columns:
        mx = df['transit_depth_ppm'].max(skipna=True)
        if pd.notna(mx) and mx < 1e4:
            print("  [K2] converting transit_depth from % -> ppm (x1e4)")
            df['transit_depth_ppm'] = df['transit_depth_ppm'] * 1e4
    # unify disposition strings (but keep original for diag)
    if 'final_disposition' in df.columns:
        df['final_disposition_raw'] = df['final_disposition'].astype(str)
    return df

koi_p = prepare_df(koi_raw, koi_map, 'KOI')
k2_p  = prepare_df(k2_raw, k2_map, 'K2')
toi_p = prepare_df(toi_raw, toi_map, 'TOI')

# show NaN% after conversion
print("\nNaN % after conversions (KOI):")
print((koi_p.isna().mean()*100).sort_values(ascending=False).round(3))
print("\nNaN % after conversions (K2):")
print((k2_p.isna().mean()*100).sort_values(ascending=False).round(3))
print("\nNaN % after conversions (TOI):")
print((toi_p.isna().mean()*100).sort_values(ascending=False).round(3))

# Inspect unique disposition values (raw) for K2 to see mapping issues
if 'final_disposition_raw' in k2_p.columns:
    print("\nK2 unique dispositions (raw) sample:")
    print(pd.Series(k2_p['final_disposition_raw'].unique()[:50]))

# --- Unify disposition function (robusta) ---
def unify_disposition(val, source=None):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    # normalize common forms
    s = s.replace('-', ' ').replace('_',' ').strip()
    # TOI special cases
    if source == 'TOI':
        if s in ('CP','KP','CONFIRMED'): return 'CONFIRMED'
        if s in ('PC','APC','CANDIDATE'): return 'CANDIDATE'
        if s in ('FP','FA','FALSE POSITIVE','FALSE_POSITIVE'): return 'FALSE_POSITIVE'
    # generic terms
    if 'CONFIR' in s or s == 'CP' or s == 'KP': return 'CONFIRMED'
    if 'CAND' in s or s == 'PC' or s == 'APC': return 'CANDIDATE'
    if 'FALSE' in s or 'REFUT' in s or s in ('FP','FA'): return 'FALSE_POSITIVE'
    # if it's already one of the target labels
    if s in ('CONFIRMED','CANDIDATE','FALSE_POSITIVE'): return s
    return np.nan

# apply unify
if 'final_disposition' in koi_p.columns:
    koi_p['final_disposition'] = koi_p['final_disposition'].apply(lambda v: unify_disposition(v, source='KOI'))
if 'final_disposition' in k2_p.columns:
    k2_p['final_disposition'] = k2_p['final_disposition'].apply(lambda v: unify_disposition(v, source='K2'))
if 'final_disposition' in toi_p.columns:
    toi_p['final_disposition'] = toi_p['final_disposition'].apply(lambda v: unify_disposition(v, source='TOI'))

# show counts after unify
print("\nCounts final_disposition (KOI):")
print(koi_p['final_disposition'].value_counts(dropna=False))
print("\nCounts final_disposition (K2):")
print(k2_p['final_disposition'].value_counts(dropna=False))
print("\nCounts final_disposition (TOI):")
print(toi_p['final_disposition'].value_counts(dropna=False))

# --- Now apply conservative filtering: require only essential observed features + label
essential = [
    'orbital_period_days',
    'transit_duration_hours',
    'transit_depth_ppm',
    'planet_radius_earth',
    'stellar_radius_solar',
    'stellar_temperature_K',
    'final_disposition'
]

def filter_essential(df, mission):
    before = df.shape[0]
    # keep rows that have non-null for ALL essential columns
    present = [c for c in essential if c in df.columns]
    df_clean = df.dropna(subset=present)
    after = df_clean.shape[0]
    print(f"[{mission}] filter essential: {before} -> {after} rows (dropped {before-after})")
    return df_clean

koi_clean = filter_essential(koi_p, 'KOI')
k2_clean  = filter_essential(k2_p, 'K2')
toi_clean = filter_essential(toi_p, 'TOI')

# add source
koi_clean['source_mission'] = 'KEPLER'
k2_clean['source_mission']  = 'K2'
toi_clean['source_mission'] = 'TESS'

# select only global columns + source
cols_global = ['orbital_period_days','transit_duration_hours','transit_depth_ppm',
               'planet_radius_earth','equilibrium_temperature_K','insolation_flux_Earth',
               'stellar_radius_solar','stellar_temperature_K','final_disposition','source_mission']

def ensure_cols(df):
    for c in cols_global:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols_global]

koi_final = ensure_cols(koi_clean)
k2_final  = ensure_cols(k2_clean)
toi_final = ensure_cols(toi_clean)

harm = pd.concat([koi_final, k2_final, toi_final], ignore_index=True)
print("\nHarmonized shape:", harm.shape)
print(harm['final_disposition'].value_counts(dropna=False))
harm.to_csv(OUT_FP, index=False)
print("Saved harmonized file to:", OUT_FP)