import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
# On n'a plus besoin de resample car on utilise les class_weights sur tout le dataset

# Import du nettoyage
try:
    from catclean import without_encode
except ImportError:
    print("ERREUR : 'catclean.py' introuvable.")
    exit()

# --- 1. CONFIGURATION (Les 50 Variables "Élues") ---
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

# --- 2. HYPERPARAMÈTRES DE BASE (Sans les poids) ---
BASE_PARAMS = {
    'iterations': 1000,          
    'learning_rate': 0.094012,   
    'depth': 7,                  
    'l2_leaf_reg': 7.827,        
    'bagging_temperature': 0.703,
    'random_strength': 9.302,    
    'border_count': 182,         
    'loss_function': 'MultiClass',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'allow_writing_files': False,
    'verbose': 100
}

def get_data_prepared(filename):
    print(f"--- 1. CHARGEMENT ET NETTOYAGE ({filename}) ---")
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERREUR : '{filename}' introuvable.")
        exit()

    # Pipeline de nettoyage
    df_clean = without_encode(df)

    # Création de la Cible (3 Classes)
    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 2}
    if 'FIRE_SIZE_CLASS' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['FIRE_SIZE_CLASS'])
        # Nettoyage si mélange str/int
        val_sample = df_clean['FIRE_SIZE_CLASS'].iloc[0]
        if isinstance(val_sample, str) and val_sample in mapping:
             df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].map(mapping)
        df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].astype(int)

    # Filtrage des colonnes
    existing_features = [f for f in BEST_FEATURES if f in df_clean.columns]
    X = df_clean[existing_features]
    y = df_clean['FIRE_SIZE_CLASS']

    return X, y

def find_best_risk_weights(X_train, y_train, X_test, y_test, cat_features):
    """
    Grid Search pour trouver le meilleur équilibrage des classes.
    Objectif : Maximiser le Recall des Grands Feux (Classe 2) sans détruire la précision.
    """
    # Scénarios de poids : [Petit, Moyen, Grand]
    scenarios = {
        "1. Naturel (Reference)": [1, 1, 1],
        "2. Doux": [1, 3, 10],
        "3. Agressif": [1, 5, 20],
        "4. Très Agressif": [1, 10, 50],
        "5. Paranoïaque": [1, 10, 100]
    }
    
    results = []
    print("\n--- RECHERCHE DU MEILLEUR ÉQUILIBRAGE (GRID SEARCH) ---")
    print(f"{'Stratégie':<25} | {'Recall Gds Feux':<15} | {'Précision Gds Feux':<18} | {'Global Acc':<10}")
    print("-" * 75)

    for name, weights in scenarios.items():
        # Modèle rapide (moins d'itérations) pour tester
        model = CatBoostClassifier(
            iterations=400,          # Suffisant pour voir la tendance
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            class_weights=weights,   # <--- On teste ce poids
            cat_features=cat_features,
            verbose=0,               # Silencieux
            random_seed=42,
            allow_writing_files=False
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métriques classe par classe
        recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
        precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        # Focus sur Classe 2 (Grand Feu)
        recall_c2 = recalls[2]
        prec_c2 = precisions[2]
        
        results.append({
            'name': name,
            'weights': weights,
            'recall_c2': recall_c2,
            'prec_c2': prec_c2,
            'acc': acc
        })

        print(f"{name:<25} | {recall_c2:.1%}          | {prec_c2:.1%}            | {acc:.1%}")

    print("-" * 75)
    return results

def print_calibration_table(df_results, class_id, class_name):
    """ Affiche la table de fiabilité pour une classe donnée """
    col_score = f'Prob_Class_{class_id}'
    
    # Création des tranches
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df_results['Tranche'] = pd.cut(df_results[col_score], bins=bins, labels=labels)

    print(f"\n>>> ANALYSE DE FIABILITÉ : {class_name.upper()} (Classe {class_id})")
    print(f"{'Tranche de Proba':<20} | {'Nb Feux':<10} | {'% Réellement ' + class_name}")
    print("-" * 65)

    counts = df_results['Tranche'].value_counts(sort=False)
    
    for label in labels:
        if label in counts.index and counts[label] > 0:
            subset = df_results[df_results['Tranche'] == label]
            real_percentage = (subset['True_Class'] == class_id).mean() * 100
            print(f"{label:<20} | {counts[label]:<10} | {real_percentage:.1f}%")
        else:
            print(f"{label:<20} | 0          | -")
    print("-" * 65)

# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
if __name__ == "__main__":
    filename = "2020_FPA_FOD_cons.csv"
    X, y = get_data_prepared(filename)

    # 1. Identification des features catégorielles
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Features catégorielles détectées : {len(cat_features)}")

    # 2. Split (Stratifié pour garder les proportions initiales)
    # Note : On ne fait plus de 'balance_train_set' manuel destructif.
    # On utilise tout le dataset d'entraînement et on gère le déséquilibre avec les poids.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

    # 3. RECHERCHE DES MEILLEURS POIDS (GRID SEARCH)
    search_results = find_best_risk_weights(X_train, y_train, X_test, y_test, cat_features)

    # --- Logique de sélection automatique ---
    # Règle : Prendre le meilleur Recall possible tant que la Précision est > 40%
    # Si aucune stratégie n'a > 40% de précision, on prend celle avec la meilleure précision globale.
    selected_weights = [1, 1, 1] # Fallback
    best_metric = -1
    best_name = "Default"

    valid_results = [r for r in search_results if r['prec_c2'] >= 0.40]
    
    if valid_results:
        # On maximise le Recall parmi les modèles "propres"
        best_scenario = max(valid_results, key=lambda x: x['recall_c2'])
    else:
        # Si tout est mauvais, on prend le "moins pire" (meilleure précision)
        print("ATTENTION : Aucune stratégie n'atteint 40% de précision. Sélection de sûreté.")
        best_scenario = max(search_results, key=lambda x: x['prec_c2'])

    selected_weights = best_scenario['weights']
    best_name = best_scenario['name']

    print(f"\n>>> STRATÉGIE GAGNANTE CHOISIE : {best_name}")
    print(f">>> Poids appliqués : {selected_weights}")

    # 4. Entraînement Final avec les POIDS GAGNANTS
    print("\n--- Entraînement Final du modèle CatBoost ---")
    
    # On fusionne les paramètres de base avec les poids choisis
    final_params = BASE_PARAMS.copy()
    final_params['class_weights'] = selected_weights
    
    model = CatBoostClassifier(**final_params, cat_features=cat_features)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    # 5. Prédictions & Scores
    print("\n" + "="*80)
    print(" RÉSULTATS DÉTAILLÉS MULTI-CLASSES (OPTIMISÉS)")
    print("="*80)
    
    probas = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Score Global
    auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
    acc = accuracy_score(y_test, preds)
    print(f"AUC Global (Weighted) : {auc:.4f}")
    print(f"Accuracy Globale      : {acc:.4f}")

    # Préparation du DataFrame d'analyse
    results = X_test.copy()
    results['True_Class'] = y_test
    results['Predicted_Class'] = preds
    results['Prob_Class_0'] = probas[:, 0]
    results['Prob_Class_1'] = probas[:, 1]
    results['Prob_Class_2'] = probas[:, 2]

    # --- PARTIE A : ALERTES GRANDS FEUX ---
    print("\n--- TOP 10 ALERTES : GRANDS FEUX (Classe 2) ---")
    top_alerts = results.sort_values(by='Prob_Class_2', ascending=False).head(10)
    
    print(f"{'RISQUE':<10} | {'VRAI':<6} | {'ÉTAT':<6} | {'CAUSE'}")
    for idx, row in top_alerts.iterrows():
        cause = str(row['NWCG_GENERAL_CAUSE'])[:20]
        state = str(row['STATE'])
        print(f"{row['Prob_Class_2']*100:.1f}%     | {row['True_Class']:<6} | {state:<6} | {cause}")

    # --- PARTIE B : ANALYSE DE FIABILITÉ ---
    print_calibration_table(results, 0, "Petit Feu")
    print_calibration_table(results, 1, "Moyen Feu")
    print_calibration_table(results, 2, "Grand Feu")