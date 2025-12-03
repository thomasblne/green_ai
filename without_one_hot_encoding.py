import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# Import du nettoyage
try:
    from cleaning import full_data_cleaning_pipeline
except ImportError:
    print("ERREUR : 'cleaning.py' introuvable.")
    exit()

# ==============================================================================
# 1. PRÉPARATION "TOUTES COLONNES"
# ==============================================================================
def prepare_data_full():
    print("--- 1. CHARGEMENT ET NETTOYAGE (TOUTES COLONNES) ---")
    filename = "2020_FPA_FOD_cons.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERREUR : '{filename}' introuvable.")
        exit()

    df_final = full_data_cleaning_pipeline(df)

    # Gestion de la Cible
    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 2}
    if 'FIRE_SIZE_CLASS' in df_final.columns:
        df_final = df_final.dropna(subset=['FIRE_SIZE_CLASS'])
        sample_val = df_final['FIRE_SIZE_CLASS'].iloc[0]
        if isinstance(sample_val, str) and sample_val in mapping:
             df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].map(mapping)
        df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].astype(int)

    # CORRECTION CRITIQUE : Suppression de la cible (FIRE_SIZE_CLASS)
    # ET de la variable source (FIRE_SIZE) qui provoquait la fuite de données.
    cols_to_drop = ['FIRE_SIZE_CLASS', 'FIRE_SIZE']
    
    X = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], errors='ignore')
    y = df_final['FIRE_SIZE_CLASS']

    # DÉTECTION AUTOMATIQUE DES CATÉGORIES
    cat_features_indices = np.where(
        (X.dtypes == 'object') | (X.dtypes == 'category')
    )[0]
    
    cat_features_names = X.columns[cat_features_indices].tolist()
    
    print(f"-> Nombre total de features : {X.shape[1]}")
    print(f"-> Dont features catégorielles détectées : {len(cat_features_names)}")

    return X, y, cat_features_names

# ==============================================================================
# 2. SPLIT & ENTRAÎNEMENT
# ==============================================================================
X, y, cat_features = prepare_data_full()

print("\n--- 2. ENTRAÎNEMENT CATBOOST (FULL FEATURES) CORRIGÉ ---")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Équilibrage (Undersampling des petits feux)
train_data = pd.concat([X_train_raw, y_train_raw], axis=1)
df_0 = train_data[train_data.FIRE_SIZE_CLASS == 0]
df_1 = train_data[train_data.FIRE_SIZE_CLASS == 1]
df_2 = train_data[train_data.FIRE_SIZE_CLASS == 2]

n_minority = len(df_2)
df_0_down = resample(df_0, replace=False, n_samples=min(len(df_0), n_minority * 4), random_state=42)
df_1_down = resample(df_1, replace=False, n_samples=min(len(df_1), n_minority * 2), random_state=42)

df_train_balanced = pd.concat([df_0_down, df_1_down, df_2]).sample(frac=1, random_state=42)

X_train = df_train_balanced.drop(columns=['FIRE_SIZE_CLASS'])
y_train = df_train_balanced['FIRE_SIZE_CLASS']

print(f"Distribution Train : {y_train.value_counts().to_dict()}")

# Configuration CatBoost
model = CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    cat_features=cat_features, # On passe la liste auto-détectée
    verbose=100,
    random_seed=42,
    allow_writing_files=False
)

model.fit(X_train, y_train)
print("-> Modèle entraîné (Résultats désormais fiables).")

# ==============================================================================
# 3. RÉSULTATS & IMPORTANCE DES FEATURES
# ==============================================================================
print("\n" + "="*80)
print(" RÉSULTATS (TOUTES COLONNES) - SANS FUITE DE DONNÉES")
print("="*80)

# 1. Score Global
probas = model.predict_proba(X_test)
risk_scores = probas[:, 2] 
y_test_binary = (y_test == 2).astype(int)
auc = roc_auc_score(y_test_binary, risk_scores)

print(f"SCORE GLOBAL (AUC) : {auc:.3f} / 1.000")
print("-" * 80)

# 2. FEATURE IMPORTANCE
feature_importances = model.get_feature_importance()
feature_names = X_train.columns

fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print("TOP 20 DES VARIABLES LES PLUS UTILES (APRÈS CORRECTION) :")
print(fi_df.head(20))