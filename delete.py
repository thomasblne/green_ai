import pandas as pd
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from scipy.stats import uniform, randint

# ==============================================================================
# 1. CONFIGURATION ET FONCTIONS UTILES
# ==============================================================================
FINAL_SELECTED_FEATURES = [
    'Popo_1km', 'Population', 'LONGITUDE', 'Ecoregion_NA_L2CODE', 'EVC', 'EVH', 
    'rpms_1km', 'FOD_ID', 'GHM', 'Wind_x_Dryness', 'EBLR_PFS', 'Discovery_Time_Hours', 
    'Land_Cover', 'EALR_PFS', 'Annual_etr', 'GDP', 'EVT', 'LATITUDE', 'EPL_MOBILE', 
    'Ecoregion_US_L3CODE', 'NDVI-1day', 'No_FireStation_20.0km', 'NWCG_GENERAL_CAUSE_Recreation and ceremony', 
    'TRACT', 'NWCG_CAUSE_CLASSIFICATION_Human', 'EPL_AGE17', 'sph_Normal', 'vs', 
    'PM25F_PFS', 'Annual_tempreture', 'OWNER_DESCR_PRIVATE', 'EPLR_PFS', 'fm1000', 
    'rmin_5D_min', 'Wind_x_Potential', 'NPL_PFS', 'No_FireStation_5.0km', 'SDI', 
    'GACC_New fire', 'Slope_x_Wind', 'GAP_Sts', 'Elevation_1km', 'LMI_PFS', 'rmax_Normal', 
    'DISCOVERY_DOY', 'MHVF_PFS', 'rmin', 'NWCG_GENERAL_CAUSE_Arson/incendiarism', 
    'Annual_precipitation', 'rmin_5D_mean'
]

try:
    from without_one_hot_encoding import full_data_cleaning_pipeline
except ImportError:
    print("ERREUR : 'cleaning_without_one_hot_encoding.py' introuvable.")
    exit()

def prepare_data_for_tuning():
    """Charge, nettoie, filtre et renvoie X_train/y_train non équilibré."""
    print("--- 1. PRÉPARATION DES DONNÉES POUR LE TUNING ---")
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

    # CORRECTION CRITIQUE (Fuite de données)
    cols_to_drop_leakage = ['FIRE_SIZE_CLASS', 'FIRE_SIZE']
    df_features = df_final.drop(columns=[c for c in cols_to_drop_leakage if c in df_final.columns], errors='ignore')
    y = df_final['FIRE_SIZE_CLASS']

    # FILTRATION : On ne garde que les 50 features optimales
    X = df_features[FINAL_SELECTED_FEATURES].copy()
    
    # Détection des catégories dans le TOP 50
    cat_features_indices = np.where(
        (X.dtypes == 'object') | (X.dtypes == 'category')
    )[0]
    final_cat_features_names = X.columns[cat_features_indices].tolist()

    # Split (unbalanced)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, y_train, final_cat_features_names

def calculate_class_weights(y):
    """Calcule les poids de classe pour le paramètre class_weights."""
    class_counts = y.value_counts()
    n_samples = len(y)
    # Inversement de la fréquence de chaque classe
    weights = n_samples / (len(class_counts) * class_counts)
    # Remise dans l'ordre (0, 1, 2)
    return [weights.get(0, 1.0), weights.get(1, 1.0), weights.get(2, 1.0)]


# ==============================================================================
# 2. ESPACE DE RECHERCHE ET TUNING
# ==============================================================================

# 1. Préparation des données et des poids
X_train, y_train, cat_features = prepare_data_for_tuning()
weights = calculate_class_weights(y_train)

print("-> Poids de classe calculés (pour gestion de l'imbalance) :", {0: round(weights[0], 2), 1: round(weights[1], 2), 2: round(weights[2], 2)})
print(f"-> Démarrage de la recherche aléatoire sur {len(X_train)} échantillons...")

# 2. Définition de l'espace de recherche (Distribution)
param_dist = {
    'learning_rate': uniform(0.01, 0.2),  # Taux d'apprentissage : entre 0.01 et 0.21
    'depth': randint(4, 10),              # Profondeur de l'arbre : entre 4 et 9
    'l2_leaf_reg': uniform(1, 10),        # Régularisation L2 : entre 1 et 10
    'min_data_in_leaf': randint(1, 100),  # Minimum de données par feuille
    'subsample': uniform(0.6, 0.4),       # Sous-échantillonnage (si utilisé) : entre 0.6 et 1.0
}

# 3. Modèle de base
cat_model = CatBoostClassifier(
    iterations=500, # On réduit pour accélérer la recherche, le meilleur set sera finalisé plus tard
    loss_function='MultiClass',
    eval_metric='AUC',
    cat_features=cat_features,
    class_weights=weights,
    random_seed=42,
    verbose=0, # Tuning doit être silencieux
    thread_count=-1,
    bootstrap_type='Bernoulli'
)

# Custom AUC Scorer (CatBoost a besoin d'une évaluation binaire pour l'AUC sur la classe 2)
def custom_auc_scorer(y_true, y_proba, **kwargs):
    y_true_binary = (y_true == 2).astype(int)
    # La probabilité est sur la classe 2 (indice 2 pour un problème multi-classe)
    return roc_auc_score(y_true_binary, y_proba[:, 2])

auc_scorer = make_scorer(custom_auc_scorer, needs_proba=True)


# 4. Configuration de la Recherche Aléatoire
random_search = RandomizedSearchCV(
    estimator=cat_model, 
    param_distributions=param_dist, 
    n_iter=50, # Nombre de combinaisons de paramètres à tester
    scoring=auc_scorer,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), # Validation croisée K=3
    verbose=1,
    random_state=42,
    n_jobs=-1 # Utiliser tous les cœurs
)

# 5. Lancement de la Recherche
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

# ==============================================================================
# 3. AFFICHAGE DES RÉSULTATS
# ==============================================================================
print("\n" + "="*80)
print(" RÉSULTATS DE L'OPTIMISATION DES HYPERPARAMÈTRES")
print("="*80)
print(f"Temps total de la recherche : {(end_time - start_time) / 60:.1f} minutes")
print("-" * 80)
print(f"MEILLEURE AUC TROUVÉE (CV Score) : {random_search.best_score_:.4f}")
print("MEILLEURS PARAMÈTRES :")
for param, value in random_search.best_params_.items():
    print(f"   -> {param}: {value}")
print("-" * 80)

print("\nProchaine étape : Utiliser ces paramètres pour entraîner le modèle final (1000+ iterations).")