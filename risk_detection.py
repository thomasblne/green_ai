import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils import resample

# Import du nettoyage
try:
    from cleaning import full_data_cleaning_pipeline
except ImportError:
    print("ERREUR : 'cleaning.py' introuvable.")
    exit()

# ==============================================================================
# 1. CONFIGURATION (Les 50 Variables "Élues")
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
# 2. PRÉPARATION (Identique)
# ==============================================================================
def prepare_data():
    print("--- 1. CHARGEMENT ET NETTOYAGE ---")
    filename = "2020_FPA_FOD_cons.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERREUR : '{filename}' introuvable.")
        exit()

    df_final = full_data_cleaning_pipeline(df)

    mapping = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 2}
    if 'FIRE_SIZE_CLASS' in df_final.columns:
        df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].astype(str).map(mapping)
        df_final = df_final.dropna(subset=['FIRE_SIZE_CLASS'])
        df_final['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS'].astype(int)

    new_cols = {c: re.sub(r'[\[\]<]', '_', c).strip() for c in df_final.columns}
    df_final = df_final.rename(columns=new_cols)

    cols_to_keep = SELECTED_FEATURES + ['FIRE_SIZE_CLASS']
    df_optimized = df_final.reindex(columns=cols_to_keep, fill_value=0)
    df_optimized['FIRE_SIZE_CLASS'] = df_final['FIRE_SIZE_CLASS']

    return df_optimized

# ==============================================================================
# 3. SPLIT & ENTRAÎNEMENT (Soft Undersampling)
# ==============================================================================
df = prepare_data()
X = df.drop(columns=['FIRE_SIZE_CLASS'])
y = df['FIRE_SIZE_CLASS']

print("--- 2. ENTRAÎNEMENT DU MODÈLE DE RISQUE ---")
# Split
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Équilibrage (4:2:1) pour bien apprendre les dangers
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

# Modèle
model = xgb.XGBClassifier(
    objective='multi:softprob', 
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
print("-> Modèle entraîné avec succès.")

# ==============================================================================
# 4. DÉTECTION ET SCORING (Le Cœur du sujet)
# ==============================================================================
print("\n" + "="*80)
print(" RÉSULTATS DE LA DÉTECTION DE RISQUE (GRANDS FEUX)")
print("="*80)

# 1. On récupère les probabilités (N_lignes x 3 classes)
probas = model.predict_proba(X_test)

# 2. On s'intéresse uniquement à la colonne 2 (Probabilité d'être un Grand Feu)
risk_scores = probas[:, 2] 

# Création d'un DataFrame de résultats pour analyse
results = X_test.copy()
results['Vrai_Classe'] = y_test
results['Score_Risque_GrandFeu'] = risk_scores # C'est le % de chance (0.0 à 1.0)
results['Score_Risque_Percent'] = (results['Score_Risque_GrandFeu'] * 100).round(1)

# ==============================================================================
# 5. AFFICHAGE DES RÉSULTATS CLÉS
# ==============================================================================

# A. Score Global du Modèle (AUC)
# L'AUC (Area Under Curve) mesure si le modèle met bien les grands feux "en haut" de la liste.
# 0.5 = Hasard, 1.0 = Parfait.
y_test_binary = (y_test == 2).astype(int) # On transforme en problème binaire (Grand vs Reste)
auc = roc_auc_score(y_test_binary, risk_scores)

print(f"SCORE DE PERFORMANCE GLOBAL (AUC) : {auc:.3f} / 1.000")
print(f"(Interprétation : Capacité du modèle à classer un Grand Feu plus risqué qu'un Petit Feu)")
print("-" * 80)

# B. Simulation : "Top Alertes"
# Imaginons qu'on reçoit ces feux en temps réel. On affiche ceux qui ont le plus gros score.
top_alerts = results.sort_values(by='Score_Risque_GrandFeu', ascending=False).head(15)

print("TOP 15 DES ALERTES DÉTECTÉES (Les feux les plus risqués selon le modèle) :")
print(f"{'RISQUE (%)':<12} | {'VRAIE CLASSE':<15} | {'LOCALISATION (State)':<20} | {'POPULATION'}")
print("-" * 80)

for index, row in top_alerts.iterrows():
    risk = row['Score_Risque_Percent']
    
    # Traduction classe
    classe_reelle = row['Vrai_Classe']
    label_classe = "PETIT (0)"
    if classe_reelle == 1: label_classe = "MOYEN (1)"
    if classe_reelle == 2: label_classe = "GRAND (2) [DANGER]"
    
    # On cherche l'état (One-hot decoding rapide pour l'affichage)
    state = "Unknown"
    for col in SELECTED_FEATURES:
        if col.startswith('STATE_') and row[col] == 1:
            state = col.replace('STATE_', '')
            break
            
    pop = row['Population']
    
    print(f"{risk}%       | {label_classe:<15} | {state:<20} | {pop:.0f}")

print("-" * 80)

# C. Analyse par tranche de probabilité
# Pour vérifier si le % est fiable.
print("\nFIABILITÉ DES PROBABILITÉS :")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
results['Tranche_Risque'] = pd.cut(results['Score_Risque_GrandFeu'], bins=bins, labels=labels)

# On calcule quel % de feux étaient VRAIMENT grands dans chaque tranche
stats = results.groupby('Tranche_Risque', observed=False)['Vrai_Classe'].apply(lambda x: (x==2).mean() * 100)
counts = results['Tranche_Risque'].value_counts(sort=False)

print(f"{'Tranche de Risque':<20} | {'Nb Feux':<10} | {'% Réel de Grands Feux'}")
for label in labels:
    if label in stats.index:
        print(f"{label:<20} | {counts[label]:<10} | {stats[label]:.1f}%")

print("\n(Si le modèle est bon, le '% Réel' doit augmenter avec la 'Tranche de Risque')")