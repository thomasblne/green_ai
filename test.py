import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.utils import resample

# Import du nettoyage
try:
    from cleaning import full_data_cleaning_pipeline
except ImportError:
    print("ERREUR : 'cleaning.py' introuvable.")
    exit()

# ==============================================================================
# 1. CONFIGURATION (Top 50 Features)
# ==============================================================================
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
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
def prepare_data():
    print("--- CHARGEMENT ET NETTOYAGE ---")
    filename = "2020_FPA_FOD_cons.csv" 
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERREUR : '{filename}' introuvable.")
        exit()

    # 1. Pipeline de nettoyage
    df_final = full_data_cleaning_pipeline(df)

    # 2. Cible (3 Classes)
    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 2}
    if 'FIRE_SIZE_CLASS' in df_final.columns:
        df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].astype(str).map(mapping)
        df_final = df_final.dropna(subset=['FIRE_SIZE_CLASS'])
        df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].astype(int)

    # 3. Nettoyage des noms de colonnes
    new_cols = {c: re.sub(r'[\[\]<]', '_', c).strip() for c in df_final.columns}
    df_final = df_final.rename(columns=new_cols)

    # 4. Filtrage (Top 50)
    # On ajoute la cible à la liste pour ne pas la perdre
    cols_to_keep = SELECTED_FEATURES + ['FIRE_SIZE_CLASS']
    df_optimized = df_final.reindex(columns=cols_to_keep, fill_value=0)
    # On s'assure que la cible est correcte (reindex peut mettre NaN ou 0.0)
    df_optimized['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS']

    return df_optimized

# ==============================================================================
# 3. SPLIT ET SOFT UNDERSAMPLING (4:2:1)
# ==============================================================================
df = prepare_data()
X = df.drop(columns=['FIRE_SIZE_CLASS'])
y = df['FIRE_SIZE_CLASS']

print("\n--- SPLIT ET ÉQUILIBRAGE (SOFT 4:2:1) ---")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Création du dataset équilibré pour l'entraînement
train_data = pd.concat([X_train_raw, y_train_raw], axis=1)
df_0 = train_data[train_data.FIRE_SIZE_CLASS == 0]
df_1 = train_data[train_data.FIRE_SIZE_CLASS == 1]
df_2 = train_data[train_data.FIRE_SIZE_CLASS == 2]

n_minority = len(df_2)
# Ratios : 4x Petits, 2x Moyens, 1x Grands
df_0_down = resample(df_0, replace=False, n_samples=min(len(df_0), n_minority * 4), random_state=42)
df_1_down = resample(df_1, replace=False, n_samples=min(len(df_1), n_minority * 2), random_state=42)
df_2_down = df_2 

df_train_balanced = pd.concat([df_0_down, df_1_down, df_2_down]).sample(frac=1, random_state=42)

X_train = df_train_balanced.drop(columns=['FIRE_SIZE_CLASS'])
y_train = df_train_balanced['FIRE_SIZE_CLASS']
print(f"-> Train Set équilibré : {X_train.shape} lignes.")

# ==============================================================================
# 4. ENTRAÎNEMENT
# ==============================================================================
print("\n--- ENTRAÎNEMENT XGBOOST ---")
model = xgb.XGBClassifier(
    objective='multi:softprob', # Important pour avoir les probabilités
    num_class=3,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("-> Modèle entraîné.")

# ==============================================================================
# 5. ANALYSE DES SEUILS (THRESHOLD MOVING)
# ==============================================================================
print("\n" + "="*70)
print(" ANALYSE D'IMPACT DU SEUIL DE DÉTECTION (GRANDS FEUX)")
print("="*70)
print("Le seuil standard est la probabilité majoritaire (souvent > 0.33 ou 0.50).")
print("Ici, on force la détection si Prob(Grand Feu) > Seuil choisi.\n")

# On récupère les probabilités brutes pour le jeu de test
# y_proba est une matrice (N_samples, 3 classes)
y_proba = model.predict_proba(X_test)
probs_grands_feux = y_proba[:, 2] # Colonne des grands feux

print(f"{'SEUIL (%)':<10} | {'RAPPEL (Grands)':<15} | {'PRÉCISION (Grands)':<18} | {'COMMENTAIRE'}")
print("-" * 70)

# Liste des seuils à tester
thresholds = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]

for thresh in thresholds:
    # 1. Prédiction par défaut (argmax)
    y_pred_custom = np.argmax(y_proba, axis=1)
    
    # 2. Application du seuil "Force"
    # Si la proba d'être un grand feu dépasse le seuil, on prédit "2" (Grand)
    # peu importe si la proba d'être "Petit" était plus élevée.
    y_pred_custom[probs_grands_feux >= thresh] = 2
    
    # 3. Calcul des scores pour la classe 2
    rec = recall_score(y_test, y_pred_custom, labels=[2], average=None)[0]
    prec = precision_score(y_test, y_pred_custom, labels=[2], average=None)[0]
    
    comment = ""
    if thresh == 0.50: comment = "(Classique)"
    if 0.20 <= thresh <= 0.30: comment = "<-- ZONE OPTIMALE ?"
    if thresh < 0.20: comment = "(Trop d'alertes)"
    
    print(f"{thresh:.2f}       | {rec:.2%}          | {prec:.2%}            | {comment}")

print("-" * 70)

# ==============================================================================
# 6. APPLICATION DU MEILLEUR SEUIL (Exemple : 0.25)
# ==============================================================================
BEST_THRESHOLD = 0.25  # <-- VOUS POUVEZ CHANGER CETTE VALEUR SELON LE TABLEAU CI-DESSUS

print(f"\n--- RÉSULTATS DÉTAILLÉS AVEC SEUIL OPTIMISÉ ({BEST_THRESHOLD}) ---")

# Application finale
y_pred_final = np.argmax(y_proba, axis=1)
y_pred_final[probs_grands_feux >= BEST_THRESHOLD] = 2

target_names = ['Petits (0)', 'Moyens (1)', 'Grands (2)']
print(classification_report(y_test, y_pred_final, target_names=target_names))

cm = confusion_matrix(y_test, y_pred_final)
print("Matrice de Confusion :")
print(cm)

tp = cm[2, 2]
fn = cm[2, 0] + cm[2, 1]
total_grands = tp + fn
print(f"\n-> Sur {total_grands} Grands Feux réels :")
print(f"   - {tp} ont été détectés.")
print(f"   - {fn} ont été manqués.")