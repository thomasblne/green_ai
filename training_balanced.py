import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# Import de votre module de nettoyage
try:
    from cleaning import full_data_cleaning_pipeline
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'cleaning.py' est introuvable dans ce dossier.")
    exit()

# ==============================================================================
# 0. CONFIGURATION : LA LISTE D'OR (TOP 50)
# ==============================================================================
# Liste issue de votre Feature Selection (Test 100 vs 50)
SELECTED_FEATURES = [
    'LONGITUDE', 'EVH', 'EVC', 'Population', 'Popo_1km', 'GDP', 'EBLR_PFS', 
    'No_FireStation_5.0km', 'GHM', 'Ecoregion_US_L3CODE', 'Ecoregion_NA_L2CODE', 
    'Ecoregion_NA_L1CODE', 'Wind_x_Dryness', 'NWCG_REPORTING_AGENCY_FWS', 
    'NWCG_CAUSE_CLASSIFICATION_Natural', 'NWCG_GENERAL_CAUSE_Natural', 
    'NWCG_GENERAL_CAUSE_Recreation and ceremony', 'OWNER_DESCR_MUNICIPAL/LOCAL', 
    'OWNER_DESCR_PRIVATE', 'OWNER_DESCR_UNDEFINED FEDERAL', 'STATE_AL', 
    'STATE_AZ', 'STATE_KS', 'STATE_KY', 'STATE_NC', 'STATE_ND', 'STATE_OK', 
    'STATE_SC', 'STATE_UT', 'STATE_WA', 'Mang_Type_TRIB', 'Mang_Type_UNK', 
    'Des_Tp_NF', 'NAME_Blue Mountains', 'NAME_Dissected Appalachian Plateau', 
    'NAME_Flint Hills', 'NAME_High Lava Plateau and Semiarid Hills', 
    'NAME_Middle Rockies (Townsend-Elkhorn)', 'NAME_Northern Cross Timbers and Lower Canadian Hills', 
    'NAME_Snake River Plain', 'NAME_South Central California Foothills and Coastal Mountains', 
    'NAME_Western Corn-Belt and Central Irregular Plains', 'fm100_Percentile_70-90%', 
    'vpd_Percentile_5-10%', 'Region_Cluster_2', 'Region_Cluster_3', 
    'Region_Cluster_12', 'Region_Cluster_28', 'Region_Cluster_41', 'Region_Cluster_44'
]

# ==============================================================================
# 1. FONCTIONS UTILITAIRES
# ==============================================================================

def regroup_target_classes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Regroupe les 7 classes en 3 classes logiques (0: Petit, 1: Moyen, 2: Grand)."""
    mapping = {
        'A': 0, 'B': 0, 
        'C': 1, 
        'D': 2, 'E': 2, 'F': 2, 'G': 2
    }
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str).map(mapping)
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
    return df

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les caractères spéciaux pour XGBoost."""
    new_columns = {}
    pattern = re.compile(r'[\[\]<]')
    for col in df.columns:
        new_name = pattern.sub('_', col).strip()
        if new_name != col:
            new_columns[col] = new_name
    return df.rename(columns=new_columns)

def filter_top_features(df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    Ne conserve que les 50 meilleures variables + la cible.
    Utilise reindex pour éviter les crashs si une colonne spécifique manque.
    """
    # On s'assure de garder la cible
    cols_to_keep = features_list + ['FIRE_SIZE_CLASS']
    
    # On filtre. fill_value=0 remplit par 0 si une colonne (ex: un État) est absente du fichier
    df_filtered = df.reindex(columns=cols_to_keep, fill_value=0)
    
    # On remet la cible au propre (si reindex a créé des 0.0 au lieu de 0)
    df_filtered['FIRE_SIZE_CLASS'] = df['FIRE_SIZE_CLASS'] 
    
    return df_filtered

# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION
# ==============================================================================
print("--- DÉMARRAGE DU PIPELINE FINAL ---")
filename = "2020_FPA_FOD_cons.csv"  # Modifiez si nécessaire

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"ERREUR : Le fichier '{filename}' est introuvable.")
    exit()

# A. Nettoyage complet (Cleaning.py)
df_final = full_data_cleaning_pipeline(df)

# B. Suppression IDs et Regroupement Cible
cols_to_banish = ['FOD_ID', 'ID', 'FIRE_SIZE', 'FIRE_SIZE_CLASS_A', 'FIRE_SIZE_CLASS_B', 
                  'FIRE_SIZE_CLASS_C', 'FIRE_SIZE_CLASS_D', 'FIRE_SIZE_CLASS_E', 
                  'FIRE_SIZE_CLASS_F', 'FIRE_SIZE_CLASS_G', 'FIRE_SIZE_CLASS_nan']
df_prep = df_final.drop(columns=[c for c in cols_to_banish if c in df_final.columns], errors='ignore')
df_prep = regroup_target_classes(df_prep, 'FIRE_SIZE_CLASS')

# C. Nettoyage des noms de colonnes
df_prep = clean_feature_names(df_prep)

# D. FILTRAGE : On ne garde que l'Elite (Top 50)
print(f"\n-> Application du filtre 'Top 50' features...")
print(f"   Dimensions avant : {df_prep.shape}")
df_optimized = filter_top_features(df_prep, SELECTED_FEATURES)
print(f"   Dimensions après : {df_optimized.shape}")

# ==============================================================================
# 3. SPLIT ET SOFT UNDERSAMPLING
# ==============================================================================
print("\n--- PRÉPARATION DES DONNÉES (TRAIN/TEST) ---")

X = df_optimized.drop(columns=['FIRE_SIZE_CLASS'])
y = df_optimized['FIRE_SIZE_CLASS']

# Split 80/20 (Stratifié pour garder la proportion de grands feux dans le Test)
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- SOFT UNDERSAMPLING (Ratio 4:2:1) ---
print("-> Application du Soft Undersampling (4:2:1)...")

# Recombinaison temporaire
train_data = pd.concat([X_train_raw, y_train_raw], axis=1)

df_0 = train_data[train_data.FIRE_SIZE_CLASS == 0] # Petits
df_1 = train_data[train_data.FIRE_SIZE_CLASS == 1] # Moyens
df_2 = train_data[train_data.FIRE_SIZE_CLASS == 2] # Grands (Minoritaire)

n_minority = len(df_2)
print(f"   Nombre de Grands Feux (Train) : {n_minority}")

# Définition des cibles (4x pour les petits, 2x pour les moyens)
target_0 = n_minority * 4
target_1 = n_minority * 2

# Resampling
df_0_down = resample(df_0, replace=False, n_samples=min(target_0, len(df_0)), random_state=42)
df_1_down = resample(df_1, replace=False, n_samples=min(target_1, len(df_1)), random_state=42)
df_2_down = df_2 # On garde tout

# Fusion et Mélange
df_train_balanced = pd.concat([df_0_down, df_1_down, df_2_down])
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42)

X_train = df_train_balanced.drop(columns=['FIRE_SIZE_CLASS'])
y_train = df_train_balanced['FIRE_SIZE_CLASS']

print(f"   Taille finale du Train Set : {X_train.shape}")
print(f"   (Ratio respecté pour optimiser le Recall sans tuer la Précision)")

# ==============================================================================
# 4. ENTRAÎNEMENT FINAL
# ==============================================================================
print("\n--- ENTRAÎNEMENT DU MODÈLE FINAL ---")

xgb_final = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=500,        # Augmenté pour la convergence fine
    learning_rate=0.03,      # Lent et précis
    max_depth=6,             # Profondeur moyenne pour éviter l'overfitting
    min_child_weight=3,      # Conservateur
    gamma=0.2,               # Réduction du bruit
    colsample_bytree=0.8,    # Diversité des colonnes
    subsample=0.8,           # Diversité des lignes
    random_state=42,
    n_jobs=-1
)

xgb_final.fit(X_train, y_train)
print("-> Entraînement terminé.")

# ==============================================================================
# 5. ÉVALUATION ET RÉSULTATS
# ==============================================================================
print("\n" + "="*60)
print(" RÉSULTATS FINAUX (Top 50 Features + Soft Undersampling)")
print("="*60)

y_pred = xgb_final.predict(X_test)

target_names = ['Petits (0)', 'Moyens (1)', 'Grands (2)']
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("\n--- Matrice de Confusion ---")
print(cm)

# Focus Spécifique Grands Feux
tp = cm[2, 2]
fn = cm[2, 0] + cm[2, 1]
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
prec = tp / (tp + cm[0, 2] + cm[1, 2]) if (tp + cm[0, 2] + cm[1, 2]) > 0 else 0

print("-" * 40)
print(f"FOCUS CLASSE 'GRANDS FEUX' (>100 acres) :")
print(f"   - Rappel (Recall)    : {rec:.2%}  (Capacité à détecter le danger)")
print(f"   - Précision          : {prec:.2%}  (Fiabilité de l'alerte)")
print("-" * 40)
print("Note : Ce modèle est un compromis optimisé. Il est beaucoup plus")
print("léger et rapide que la version initiale, et plus équilibré.")