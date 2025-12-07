import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score

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

# --- 2. HYPERPARAMÈTRES DE BASE ---
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
        val_sample = df_clean['FIRE_SIZE_CLASS'].iloc[0]
        if isinstance(val_sample, str) and val_sample in mapping:
             df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].map(mapping)
        df_clean['FIRE_SIZE_CLASS'] = df_clean['FIRE_SIZE_CLASS'].astype(int)

    # Filtrage des colonnes
    existing_features = [f for f in BEST_FEATURES if f in df_clean.columns]
    X = df_clean[existing_features]
    y = df_clean['FIRE_SIZE_CLASS']

    return X, y

def print_calibration_table(df_results, class_id, class_name):
    """ Affiche la table de fiabilité pour une classe donnée """
    col_score = f'Prob_Class_{class_id}'
    
    # Création des tranches de probabilités
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

    # 3. DÉFINITION DES POIDS (STRATÉGIE 4 - TRÈS AGRESSIF)
    # [Petit, Moyen, Grand] -> On met un poids de 50 sur les Grands Feux
    selected_weights = [1, 5, 20]
    
    print(f"\n>>> STRATÉGIE DE POIDS APPLIQUÉE : [Petit: 1, Moyen: 5, Grand: 20]")

    # 4. Entraînement Final
    print("\n--- Entraînement Final du modèle CatBoost ---")
    
    # Fusion des paramètres
    final_params = BASE_PARAMS.copy()
    final_params['class_weights'] = selected_weights
    
    model = CatBoostClassifier(**final_params, cat_features=cat_features)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

    # 5. Prédictions & Scores
    print("\n" + "="*90)
    print(" RÉSULTATS DÉTAILLÉS MULTI-CLASSES (OPTIMISÉS SÉCURITÉ)")
    print("="*90)
    
    probas = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Scores
    auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
    acc = accuracy_score(y_test, preds)
    # Recall par classe (le [2] est celui du Grand Feu)
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    
    print(f"AUC Global (Weighted) : {auc:.4f}")
    print(f"Accuracy Globale      : {acc:.4f}")
    print(f"Recall Grand Feu      : {recalls[2]:.2%} (Priorité absolue)")

    # Préparation du DataFrame d'analyse
    results = X_test.copy()
    results['True_Class'] = y_test
    results['Predicted_Class'] = preds
    results['Prob_Class_0'] = probas[:, 0] # Petit
    results['Prob_Class_1'] = probas[:, 1] # Moyen
    results['Prob_Class_2'] = probas[:, 2] # Grand

    # --- PARTIE A : TOP 10 DES PLUS GROS RISQUES ---
    print("\n--- TOP 10 ALERTES (Triées par probabilité de Grand Feu) ---")
    top_alerts = results.sort_values(by='Prob_Class_2', ascending=False).head(10)
    
    # En-tête avec les 3 probabilités
    print(f"{'P(Petit)':<10} | {'P(Moyen)':<10} | {'P(Grand)':<10} | {'VRAI':<6} | {'ÉTAT':<6} | {'CAUSE'}")
    print("-" * 90)
    
    for idx, row in top_alerts.iterrows():
        p0 = f"{row['Prob_Class_0']*100:.1f}%"
        p1 = f"{row['Prob_Class_1']*100:.1f}%"
        p2 = f"{row['Prob_Class_2']*100:.1f}%"
        
        cause = str(row['NWCG_GENERAL_CAUSE'])[:20]
        state = str(row['STATE'])
        true_c = str(row['True_Class'])
        
        print(f"{p0:<10} | {p1:<10} | {p2:<10} | {true_c:<6} | {state:<6} | {cause}")

    # --- PARTIE B : ANALYSE DE FIABILITÉ ---
    print_calibration_table(results, 0, "Petit Feu")
    print_calibration_table(results, 1, "Moyen Feu")
    print_calibration_table(results, 2, "Grand Feu")

    # --- PARTIE C : ANALYSE DE RISQUE (SEUIL 50% GRAND FEU) ---
    print("\n" + "="*90)
    print(" ANALYSE DE RISQUE : PROBABILITÉ D'ÊTRE UN GRAND FEU > 50%")
    print("="*90)

    # 1. Pour les PETITS feux (True_Class == 0)
    subset_petit = results[results['True_Class'] == 0]
    nb_petit_alert = (subset_petit['Prob_Class_2'] > 0.5).sum()
    pct_petit_alert = (nb_petit_alert / len(subset_petit)) * 100 if len(subset_petit) > 0 else 0
    print(f"Petits Feux avec > 50% de risque Grand Feu : {pct_petit_alert:6.2f}%  ({nb_petit_alert}/{len(subset_petit)})")

    # 2. Pour les MOYENS feux (True_Class == 1)
    subset_moyen = results[results['True_Class'] == 1]
    nb_moyen_alert = (subset_moyen['Prob_Class_2'] > 0.5).sum()
    pct_moyen_alert = (nb_moyen_alert / len(subset_moyen)) * 100 if len(subset_moyen) > 0 else 0
    print(f"Moyens Feux avec > 50% de risque Grand Feu : {pct_moyen_alert:6.2f}%  ({nb_moyen_alert}/{len(subset_moyen)})")

    # 3. Pour les GRANDS feux (True_Class == 2) - C'est le Recall au seuil 0.5
    subset_grand = results[results['True_Class'] == 2]
    nb_grand_alert = (subset_grand['Prob_Class_2'] > 0.5).sum()
    pct_grand_alert = (nb_grand_alert / len(subset_grand)) * 100 if len(subset_grand) > 0 else 0
    print(f"Grands Feux avec > 50% de risque Grand Feu : {pct_grand_alert:6.2f}%  ({nb_grand_alert}/{len(subset_grand)})")
    print("-" * 90)