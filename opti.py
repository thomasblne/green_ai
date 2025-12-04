import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample

# Import du nettoyage
try:
    from catclean import without_encode
except ImportError:
    print("ERREUR : 'catclean.py' introuvable.")
    exit()

# --- VOS 50 MEILLEURES FEATURES (Copiées depuis votre résultat) ---
BEST_FEATURES = [
    'Popo_1km', 'STATE', 'Population', 'EVH', 'EVC', 'rpms_1km', 'OWNER_DESCR', 
    'FOD_ID', 'Discovery_Time_Hours', 'NWCG_GENERAL_CAUSE', 'Region_Cluster', 
    'EBLR_PFS', 'EALR_PFS', 'No_FireStation_20.0km', 'Wind_x_Dryness', 'Land_Cover', 
    'EVT', 'EPL_AGE17', 'LATITUDE', 'GHM', 'NWCG_CAUSE_CLASSIFICATION', 'EPL_MOBILE', 
    'Ecoregion_US_L3CODE', 'NDVI-1day', 'Ecoregion_NA_L2CODE', 'Mang_Type', 'GDP', 
    'pr_Normal', 'EPL_NOHSDP', 'PM25F_PFS', 'bi_Percentile', 'rmin', 'sph', 
    'Annual_precipitation', 'HBF_PFS', 'vpd_Normal', 'Slope_x_Wind', 'sph_Normal', 
    'Slope_1km', 'rmax_Normal', 'bi_Normal', 'TRACT', 'Elevation', 'RPL_THEME4', 
    'EPLR_PFS', 'MHVF_PFS', 'RPL_THEME2', 'SDI', 'Wind_x_Potential', 'RMP_PFS'
]

def get_data_prepared(filename):
    print(f"--- Chargement et Nettoyage ({filename}) ---")
    df = pd.read_csv(filename)
    
    # 1. Nettoyage
    df_clean = without_encode(df)

    # 2. Cible (A/B->0, C->1, D-G->2)
    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 2}
    if 'FIRE_SIZE_CLASS' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['FIRE_SIZE_CLASS'])
        val_sample = df_clean['FIRE_SIZE_CLASS'].iloc[0]
        if isinstance(val_sample, str) and val_sample in mapping:
             df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].map(mapping)
        df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].astype(int)

    # 3. Filtrage : On ne garde QUE les BEST_FEATURES + la Target
    # On vérifie d'abord que toutes les best features sont bien là
    existing_features = [f for f in BEST_FEATURES if f in df_clean.columns]
    
    if len(existing_features) < len(BEST_FEATURES):
        missing = set(BEST_FEATURES) - set(existing_features)
        print(f"⚠️ Attention, certaines features manquent après nettoyage : {missing}")

    X = df_clean[existing_features]
    y = df_clean['FIRE_SIZE_CLASS']

    return X, y

def balance_train_set(X_train, y_train):
    """Même stratégie d'équilibrage que précédemment"""
    train_data = pd.concat([X_train, y_train], axis=1)
    
    df_0 = train_data[train_data.FIRE_SIZE_CLASS == 0]
    df_1 = train_data[train_data.FIRE_SIZE_CLASS == 1]
    df_2 = train_data[train_data.FIRE_SIZE_CLASS == 2]

    n_minority = len(df_2)
    
    # Ratio 4:2:1
    df_0_down = resample(df_0, replace=False, n_samples=min(len(df_0), n_minority * 4), random_state=42)
    df_1_down = resample(df_1, replace=False, n_samples=min(len(df_1), n_minority * 2), random_state=42)
    
    df_balanced = pd.concat([df_0_down, df_1_down, df_2]).sample(frac=1, random_state=42)
    
    return df_balanced.drop(columns=['FIRE_SIZE_CLASS']), df_balanced['FIRE_SIZE_CLASS']

def objective(trial, X_train, y_train, X_test, y_test, cat_features):
    """Fonction objectif pour Optuna"""
    
    # Espace de recherche des hyperparamètres
    params = {
        'iterations': 1000, # On met un grand nombre, l'early_stopping arrêtera avant
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        
        # Paramètres fixes
        'loss_function': 'MultiClass',
        'eval_metric': 'AUC', # On optimise pour l'AUC
        'cat_features': cat_features,
        'verbose': False,
        'random_seed': 42,
        'allow_writing_files': False,
        'task_type': 'CPU' # Mettre 'GPU' si vous avez une carte graphique NVIDIA et catboost gpu installé
    }

    model = CatBoostClassifier(**params)
    
    # Entraînement avec arrêt précoce (si pas d'amélioration pendant 50 itérations)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # On récupère le meilleur score (AUC)
    # CatBoost calcule l'AUC pour chaque classe, on prend souvent la moyenne ou le best_score_ du modèle
    # Ici, on va recalculer l'AUC weighted manuellement pour être sûr de la métrique
    probas = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
    
    return auc

def run_optimization():
    filename = "2020_FPA_FOD_cons.csv"
    X, y = get_data_prepared(filename)

    # Identification des features catégorielles PARMI les 50 retenues
    cat_features_indices = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Features catégorielles retenues ({len(cat_features_indices)}) : {cat_features_indices}")

    # Split
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance Train
    X_train, y_train = balance_train_set(X_train_raw, y_train_raw)
    
    print(f"Dataset prêt pour optimisation. Train: {X_train.shape}, Test: {X_test.shape}")
    print("--- Lancement d'Optuna (cela peut prendre quelques minutes) ---")

    # Création de l'étude
    study = optuna.create_study(direction='maximize') # On veut maximiser l'AUC
    
    # Lancement de l'optimisation (n_trials = nombre d'essais)
    # 20 essais donnent déjà un bon résultat. Mettez 50 ou 100 si vous avez le temps (café !)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, cat_features_indices), n_trials=30)

    print("\n" + "="*80)
    print("✅ OPTIMISATION TERMINÉE")
    print(f"Meilleur AUC trouvé : {study.best_value:.4f}")
    print("="*80)
    
    print("\nCOPIEZ CE DICTIONNAIRE FINAL POUR VOTRE MODÈLE DE PRODUCTION :")
    best_params = study.best_params
    # On ajoute les paramètres fixes qu'on n'a pas optimisés
    best_params['loss_function'] = 'MultiClass'
    best_params['eval_metric'] = 'AUC'
    best_params['cat_features'] = cat_features_indices
    best_params['random_seed'] = 42
    
    print(best_params)
    print("="*80)

    # Entraînement final rapide pour voir l'accuracy avec ces params
    print("\nTest final avec les meilleurs paramètres...")
    final_model = CatBoostClassifier(**best_params, iterations=1000, verbose=100)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    acc = accuracy_score(y_test, final_model.predict(X_test))
    print(f"Accuracy Finale sur Test : {acc:.4f}")

if __name__ == "__main__":
    run_optimization()