import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from typing import Tuple

# Import de votre module de nettoyage (qui contient la nouvelle gestion des dates)
from cleaning import *

# ==============================================================================
# 1. CHARGEMENT ET NETTOYAGE
# ==============================================================================
print("--- CHARGEMENT DES DONNÉES ---")
# J'ai remis le nom de fichier standard, modifiez-le si besoin
filename = "2020_FPA_FOD_cons.csv" 
# filename = "1992_FPA_FOD_cons.csv" # Décommentez si c'est votre fichier

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"ERREUR : Le fichier '{filename}' est introuvable.")
    exit()

# Exécution du pipeline de nettoyage (incluant l'extraction du Mois/Jour)
df_final = full_data_cleaning_pipeline(df)

# ==============================================================================
# 2. FONCTIONS DE PRÉPARATION (STRATÉGIE 3 CLASSES)
# ==============================================================================

def regroup_target_classes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Regroupe les 7 classes en 3 classes logiques.
    0: Petits (A, B)      -> Faciles à éteindre
    1: Moyens (C)         -> Demandent des moyens
    2: Grands (D, E, F, G)-> Feux majeurs (Cible prioritaire)
    """
    mapping = {
        'A': 0, 'B': 0, 
        'C': 1, 
        'D': 2, 'E': 2, 'F': 2, 'G': 2
    }
    
    if target_col in df.columns:
        # On passe en string pour le mapping
        df[target_col] = df[target_col].astype(str).map(mapping)
        # On supprime les lignes qui n'ont pas matché (NaN)
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
        
    return df

def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les IDs et prépare la cible."""
    df_prep = df.copy()
    
    # Suppression des colonnes techniques/fuites restantes
    cols_to_banish = [
        'FOD_ID', 'ID', 'FIRE_SIZE', 
        'FIRE_SIZE_CLASS_A', 'FIRE_SIZE_CLASS_B', 'FIRE_SIZE_CLASS_C', 
        'FIRE_SIZE_CLASS_D', 'FIRE_SIZE_CLASS_E', 'FIRE_SIZE_CLASS_F', 
        'FIRE_SIZE_CLASS_G', 'FIRE_SIZE_CLASS_nan' 
    ]
    df_prep = df_prep.drop(columns=[c for c in cols_to_banish if c in df_prep.columns], errors='ignore')
    
    # Application du regroupement
    df_prep = regroup_target_classes(df_prep, 'FIRE_SIZE_CLASS')
    
    return df_prep

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes pour XGBoost."""
    new_columns = {}
    pattern = re.compile(r'[\[\]<]')
    for col in df.columns:
        new_name = pattern.sub('_', col).strip()
        if new_name != col:
            new_columns[col] = new_name
    return df.rename(columns=new_columns)

# ==============================================================================
# 3. PRÉPARATION ET SPLIT
# ==============================================================================
print("\n--- PRÉPARATION AVANCÉE ---")

# A. Préparation
df_prepared = prepare_for_modeling(df_final)

# B. Séparation X / y
X = df_prepared.drop(columns=['FIRE_SIZE_CLASS'])
y = df_prepared['FIRE_SIZE_CLASS']

# C. Split Train / Test (80/20)
print("-> Séparation Train/Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# D. Nettoyage des noms de colonnes (Important pour XGBoost)
X_train_final = clean_feature_names(X_train)
X_test_final = clean_feature_names(X_test)

# ==============================================================================
# 4. MODÉLISATION XGBOOST (AVEC POIDS / WEIGHTS - TUNÉ)
# ==============================================================================
print("\n--- ENTRAÎNEMENT XGBOOST (Tuning Précision/Rappel) ---")

# Calcul des poids pour compenser le déséquilibre (MAINTENU)
classes_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)
print("-> Poids des classes calculés (Balanced).")

# Configuration avancée (AJUSTÉE)
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=400,        # AUGMENTÉ : Pour compenser le learning rate plus lent
    learning_rate=0.03,      # DIMINUÉ : Apprentissage plus lent et plus précis
    max_depth=8,             # MAIN_TENU
    # --- AJUSTEMENTS POUR RÉDUIRE LE BRUIT ET AUGMENTER LA PRÉCISION ---
    min_child_weight=5,      # AUGMENTÉ : Les feuilles doivent contenir au moins 5 échantillons (vs 3 avant)
    gamma=0.4,               # AUGMENTÉ : Rend le modèle plus exigeant sur le gain (plus conservateur)
    colsample_bytree=0.8,    
    random_state=42,
    n_jobs=-1
)

try:
    print("-> Démarrage du fit...")
    # ON UTILISE LES POIDS
    xgb_model.fit(
        X_train_final, 
        y_train,
        sample_weight=classes_weights 
    )
    print("-> Entraînement terminé avec succès.")

    # ... (La prédiction et l'évaluation sont inchangées) ...
    y_pred = xgb_model.predict(X_test_final)

    # RÉSULTATS
    print("\n" + "="*50)
    print(" RÉSULTATS FINAUX (Tuning conservateur)")
    print("="*50)
    
    target_names = ['Petits (A+B)', 'Moyens (C)', 'Grands (D+)']
    
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_test, y_pred)
    # ... (Le reste du code d'évaluation) ...
    print("\n--- Matrice de Confusion ---")
    print(cm)
    
    tp_grands = cm[2, 2]
    fn_grands = cm[2, 0] + cm[2, 1]
    total_grands = tp_grands + fn_grands
    recall_grands = tp_grands / total_grands if total_grands > 0 else 0
    
    print("-" * 30)
    print(f"FOCUS GRANDS FEUX (>100 acres) :")
    print(f"   - Détectés correctement : {tp_grands}")
    print(f"   - Manqués (Faux Négatifs): {fn_grands}")
    print(f"   - Rappel (Recall) : {recall_grands:.2%}")
    print("-" * 30)


except Exception as e:
    print(f"\nERREUR LORS DE L'ENTRAÎNEMENT : {e}")