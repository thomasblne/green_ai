import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def clean_dataframe(df: pd.DataFrame, nan_threshold: float = 50.0) -> pd.DataFrame:
    
    # --- 1. Détection des colonnes avec trop de NaN ---
    missing_percent = df.isnull().mean() * 100
    # Colonnes à supprimer (> seuil de NaN)
    cols_to_drop_nan = missing_percent[missing_percent > nan_threshold].index.tolist()
    
    # --- 2. Détection des colonnes constantes (variance nulle) ---
    nunique = df.nunique(dropna=False) 
    cols_constantes = nunique[nunique == 1].index.tolist()
    
    # --- 3. Combinaison des colonnes à supprimer ---
    all_cols_to_drop = list(set(cols_to_drop_nan + cols_constantes))
    
    # --- 4. Suppression ---
    print(f"Colonnes avec plus de {nan_threshold}% de NaN à supprimer : {len(cols_to_drop_nan)}")
    print(f"Colonnes constantes à supprimer : {len(cols_constantes)}")
    print(f"Total de colonnes supprimées (Step 1) : {len(all_cols_to_drop)}")
    
    df_cleaned = df.drop(columns=all_cols_to_drop, errors='ignore')
    
    return df_cleaned

def handle_aberrant_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df_clean = df.copy()
    total_float_err = 0
    total_codes_err = 0
    total_topo_err = 0
    
    print("--- DÉBUT DU NETTOYAGE DÉTAILLÉ (handle_aberrant_values) ---")
    
    # --- Identification des colonnes FLOAT ---
    cols_float = df_clean.select_dtypes(include=['float']).columns
    print(f"   -> {len(cols_float)} colonnes float vérifiées.")

    # -------------------------------------------------------------
    # --- 1. ERREURS TECHNIQUES (FLOAT MIN) ---
    for col in cols_float:
        mask = df_clean[col] < -1e30
        count = mask.sum()
        if count > 0:
            df_clean.loc[mask, col] = np.nan
            total_float_err += count

    if total_float_err > 0:
        print(f"   -> {total_float_err} valeurs techniques (-1e30) remplacées par NaN.")

    # -------------------------------------------------------------
    # --- 2. CODES D'ERREUR NÉGATIFS (< -900) ---
    # On garde Lat/Long
    cols_excluded = ['LATITUDE', 'LONGITUDE']
    cols_to_check = [c for c in cols_float if c not in cols_excluded]
    
    for col in cols_to_check:
        mask = df_clean[col] < -900
        count = mask.sum()
        if count > 0:
            df_clean.loc[mask, col] = np.nan
            total_codes_err += count

    if total_codes_err > 0:
         print(f"   -> {total_codes_err} codes d'erreur (< -900) remplacés par NaN.")

    # -------------------------------------------------------------
    # --- 3. ERREUR TOPOGRAPHIQUE (32767) ---
    cols_topo = ['FRG', 'Elevation', 'Aspect', 'Slope']
    for col in cols_topo:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            mask = df_clean[col] >= 32767
            count = mask.sum()
            if count > 0:
                df_clean.loc[mask, col] = np.nan
                total_topo_err += count

    if total_topo_err > 0:
        print(f"   -> {total_topo_err} erreurs topo (32767) remplacées par NaN.")

    print(f"Total des cellules corrigées : {total_float_err + total_codes_err + total_topo_err}")

    return df_clean

def impute_nan_and_drop_manual_cols(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    print(f"Dimensions avant traitement des NaN : {df_clean.shape}")

    # --- ÉTAPE 0 : TRAITEMENT SPÉCIAL DE L'HEURE (DISCOVERY_TIME) ---
    if 'DISCOVERY_TIME' in df_clean.columns:
        # Conversion forcée en numérique
        d_time = pd.to_numeric(df_clean['DISCOVERY_TIME'], errors='coerce')
        d_time = d_time.fillna(d_time.median())
        hours = d_time // 100
        minutes = d_time % 100
        df_clean['Discovery_Time_Hours'] = hours + (minutes / 60.0)
        df_clean = df_clean.drop(columns=['DISCOVERY_TIME'])
        print("   -> 'DISCOVERY_TIME' transformée en 'Discovery_Time_Hours'.")

    # --- ÉTAPE 1 : SUPPRESSION DES COLONNES "FUITE" ET "INUTILES" ---
    cols_a_bannir = [
        'CONT_DATE', 'CONT_TIME', 'CONT_DOY',
        'FIRE_NAME', 'LOCAL_INCIDENT_ID', 
        'FIPS_NAME', 'FIPS_CODE', 'COUNTY'
    ]
    cols_a_supprimer = [c for c in cols_a_bannir if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_a_supprimer, errors='ignore')
    print(f"-> Colonnes supprimées (Leakage/IDs) : {len(cols_a_supprimer)}")

    # --- ÉTAPE 1.5 : SAUVETAGE DES COLONNES CRITIQUES (Végétation & Topo) ---
    # Ces colonnes sont souvent vides mais cruciales. On remplace NaN par -1 au lieu de supprimer.
    critical_features = ['EVH_1km', 'EVT_1km', 'EVC_1km', 'CH_1km', 'Land_Cover_1km']
    
    print(f"-> Sauvetage des colonnes critiques : {critical_features}")
    for col in critical_features:
        if col in df_clean.columns:
            # On remplit les NaN avec -1 (Flag technique pour les arbres de décision)
            nb_nan = df_clean[col].isna().sum()
            if nb_nan > 0:
                df_clean[col] = df_clean[col].fillna(-1.0)
                print(f"   -> {col} : {nb_nan} NaN remplacés par -1.0 (Sauvegardée).")
        else:
            # Optionnel : Créer la colonne avec -1 si elle n'existe pas du tout
            # df_clean[col] = -1.0 
            pass

    # --- ÉTAPE 2 : IMPUTATION (REMPLISSAGE) ---
    print("-> Remplissage des valeurs manquantes (Médiane)...")
    
    # Identifier les colonnes numériques STRICTES
    cols_num = df_clean.select_dtypes(include=['number']).columns 

    # Vérification des colonnes ENCORE 100% vides (Celles qui ne sont PAS critiques)
    empty_cols = df_clean[cols_num].columns[df_clean[cols_num].isnull().all()]
    
    if len(empty_cols) > 0:
        print(f"   ALERTE : {len(empty_cols)} colonnes non-critiques sont entièrement vides.")
        print(f"   (Exemples: {empty_cols[:3].tolist()}) -> Suppression.")
        df_clean = df_clean.drop(columns=empty_cols)
        # Mise à jour de la liste des colonnes numériques
        cols_num = df_clean.select_dtypes(include=['number']).columns

    # Imputation par la médiane pour le reste
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(df_clean[cols_num])
    
    # Réaffectation
    df_clean[cols_num] = pd.DataFrame(imputed_data, columns=cols_num, index=df_clean.index)

    # --- VÉRIFICATION ---
    restant = df_clean.select_dtypes(include=['number']).isnull().sum().sum()
    print(f"-> NaN restants dans les colonnes numériques : {restant}")

    return df_clean

def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les catégories, gère les dates et supprime les colonnes inutiles.
    Remplissage des NaN catégoriels par 'Missing' (Optimal pour CatBoost).
    """
    df_clean = df.copy() 

    # --- 1. GESTION DE LA DATE (DISCOVERY_DATE) ---
    if 'DISCOVERY_DATE' in df_clean.columns:
        df_clean['DISCOVERY_DATE'] = pd.to_datetime(df_clean['DISCOVERY_DATE'], errors='coerce')
        
        # Extraction des features temporelles
        df_clean['Discovery_Month'] = df_clean['DISCOVERY_DATE'].dt.month
        df_clean['Discovery_DayOfYear'] = df_clean['DISCOVERY_DATE'].dt.dayofyear
        
        print("-> Features temporelles (Mois, Jour) extraites.")
        
        # Remplissage des NaN créés
        df_clean['Discovery_Month'] = df_clean['Discovery_Month'].fillna(df_clean['Discovery_Month'].median())
        df_clean['Discovery_DayOfYear'] = df_clean['Discovery_DayOfYear'].fillna(df_clean['Discovery_DayOfYear'].median())
    
    # --- 2. CONVERSION DES "FAUX TEXTES" EN NUMÉRIQUES ---
    cols_to_numeric = [
        'NDVI_max', 'NDVI_mean', 'NDVI_min', 'EVT_1km',
        'MOD_NDVI_12m', 'MOD_EVI_12m', 'Land_Cover_1km',
        'EVH_1km', 'EVC_1km', 'FRG_1km'
    ]
    print("-> Conversion des colonnes 'faux-textes' en numériques...")
    for col in cols_to_numeric:
        if col in df_clean.columns: 
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # --- 3. SUPPRESSION CIBLÉE ---
    cols_to_drop = [
        # IDs TRES HAUTE CARDINALITÉ
        'FPA_ID', 'geometry', 'FIRE_NAME', 'LOCAL_INCIDENT_ID', 
        'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME',
        'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME',
        'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM','NAME',
        # Fuite
        'CONT_DATE',
        # Géographie trop précise ou redondante (Gardons LatLong State pour CatBoost éventuellement, sinon drop)
        'LatLong_County', 
        'Ecoregion_US_L4CODE', 
        'Ecoregion_NA_L3CODE', 
        # Dates brutes
        'DISCOVERY_DATE'
    ]

    unique_cols_to_drop = list(set(cols_to_drop))
    cols_present_to_drop = [c for c in unique_cols_to_drop if c in df_clean.columns]
    
    df_clean = df_clean.drop(columns=cols_present_to_drop, errors='ignore')
    print(f"-> Colonnes catégorielles supprimées (Manuelles) : {len(cols_present_to_drop)}")
    
    # --- 4. GESTION DES NaN POUR LES VARIABLES CATÉGORIELLES ---
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    # A. Suppression des colonnes 100% vides
    empty_cat_cols = [c for c in cat_cols if df_clean[c].isnull().all()]
    if empty_cat_cols:
        print(f"-> Suppression de {len(empty_cat_cols)} colonnes catégorielles 100% vides.")
        df_clean = df_clean.drop(columns=empty_cat_cols)
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    # B. Remplissage par "Missing"
    if len(cat_cols) > 0:
        # On cast en string d'abord pour éviter les erreurs de catégorie
        for col in cat_cols:
             df_clean[col] = df_clean[col].astype(str).replace('nan', 'Missing')
             df_clean[col] = df_clean[col].fillna('Missing')

        print(f"-> Imputation des NaN catégoriels par 'Missing' sur {len(cat_cols)} colonnes.")

    return df_clean

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features d'interaction physique et ajoute un clustering spatial.
    """
    df_feat = df.copy()
    print("-> Création des features avancées (Physique & Clustering)...")
    
    # --- 1. FEATURES D'INTERACTION PHYSIQUE ---
    if 'vs' in df_feat.columns and 'vpd' in df_feat.columns:
        df_feat['Wind_x_Dryness'] = df_feat['vs'] * df_feat['vpd']
    
    if 'vs' in df_feat.columns and 'erc' in df_feat.columns:
        df_feat['Wind_x_Potential'] = df_feat['vs'] * df_feat['erc']

    if 'Slope' in df_feat.columns and 'vs' in df_feat.columns:
        df_feat['Slope_x_Wind'] = df_feat['Slope'] * df_feat['vs']
        
    if 'Aridity_index' in df_feat.columns:
        df_feat['Dryness_Inverse'] = 1 / (df_feat['Aridity_index'] + 0.001)

    print("   -> Interactions physiques ajoutées.")

    # --- 2. CLUSTERING SPATIAL ---
    if 'LATITUDE' in df_feat.columns and 'LONGITUDE' in df_feat.columns:
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        coords = df_feat[['LATITUDE', 'LONGITUDE']]
        # Le cluster devient une catégorie (0, 1, 2... 49)
        df_feat['Region_Cluster'] = kmeans.fit_predict(coords)
        
        print("   -> Clustering Spatial terminé (50 régions créées).")
    
    return df_feat

def prepare_types_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Au lieu de One-Hot Encoding, on convertit explicitement les colonnes
    catégorielles en dtype 'category'. CatBoost les détectera automatiquement
    ou via cat_features.
    """
    df_out = df.copy()
    
    # On détecte les colonnes object (strings) et les entiers qui devraient être des catégories
    # Ici on se concentre sur les objets/strings restants
    cat_cols = df_out.select_dtypes(include=['object']).columns.tolist()
    
    # On ajoute aussi Region_Cluster si présent (c'est un int mais conceptuellement une catégorie)
    if 'Region_Cluster' in df_out.columns:
        cat_cols.append('Region_Cluster')

    # PROTECTION DE LA CIBLE (si c'est de la classification)
    if 'FIRE_SIZE_CLASS' in cat_cols:
        # Généralement la target on la laisse telle quelle ou LabelEncode, 
        # mais ici on s'assure juste de ne pas la casser.
        pass

    if cat_cols:
        print(f"-> Conversion de {len(cat_cols)} colonnes en type 'category' pour CatBoost.")
        for col in cat_cols:
            df_out[col] = df_out[col].astype('category')
    else:
        print("-> Aucune colonne catégorielle détectée pour conversion.")

    return df_out

def without_encode(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()
    print(f"Pipeline de Nettoyage (Mode CatBoost) - Démarrage. Dimensions : {df_temp.shape}")
    print("=" * 70)
    
    # 1. Nettoyage Grossier
    df_temp = clean_dataframe(df_temp)
    
    # 2. Valeurs Aberrantes
    df_temp = handle_aberrant_values(df_temp)

    # 3. Conversion Type, Dates et Suppression IDs
    df_temp = clean_categorical_features(df_temp)

    # 4. Imputation Numérique et Gestion Time
    df_temp = impute_nan_and_drop_manual_cols(df_temp)
    
    # 5. Features Avancées (Interactions + Clustering)
    df_temp = add_advanced_features(df_temp)
    
    # 6. Préparation des Types (PAS DE ONE-HOT, juste cast en 'category')
    df_final = prepare_types_for_catboost(df_temp)
    
    print("=" * 70)
    print(f"Pipeline terminé. Dimensions finales : {df_final.shape}")
    print("Note : Les colonnes catégorielles sont préservées (dtype='category').")
    return df_final