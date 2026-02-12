import os
import sys
import pandas as pd
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
# Risaliamo a: space-invaders-ai/
project_root = os.path.dirname(current_dir)

# Aggiungiamo la root al path per importare i moduli
sys.path.append(project_root)

# Definiamo la cartella resources
resources_dir = os.path.join(project_root, "resources")

# IMPORTIAMO LE FUNZIONI CONDIVISE
from utils.game_utils import compute_features, get_feature_names

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join(resources_dir, 'dataset_heuristic.csv')
MODEL_FILE = os.path.join(resources_dir, 'xgboost_brain.json')

def train_and_evaluate():
    print("--- 1. CARICAMENTO DATI ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERRORE: Non trovo {INPUT_FILE}. Genera prima i dati!")
        return

    print(f"Righe originali: {len(df)}")
    
    # Rimozione duplicati per evitare Data Leakage
    df = df.drop_duplicates()
    print(f"Righe uniche (dopo pulizia): {len(df)}")
    
    # --- 2. FEATURE ENGINEERING (Tramite game_utils) ---
    print("Applicazione Feature Engineering...")
    
    # Applica la funzione compute_features riga per riga
    features_dicts = df.apply(compute_features, axis=1).tolist()
    X = pd.DataFrame(features_dicts)
    
    # Ordina le colonne rigorosamente
    X = X[get_feature_names()]
    y = df['action']
    
    print(f"Features finali: {X.columns.tolist()}")

    # --- 3. SPLIT TRAIN/TEST ---
    print("\nSeparazione Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. ADDESTRAMENTO (Con Timer) ---
    print("\n--- 2. TRAINING XGBOOST ---")
    model = xgb.XGBClassifier(
        n_estimators=200,      
        learning_rate=0.05,    
        max_depth=6,
        objective='multi:softmax',
        num_class=4,
        n_jobs=-1,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training completato in {training_time:.4f} secondi.")

    # --- 5. VALUTAZIONE COMPLETA ---
    print("\n--- 3. REPORT PRESTAZIONI ---")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    print("-" * 30)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("-" * 30)
    
    print("\nDettaglio per Azione:")
    print(classification_report(y_test, preds, target_names=['Fermo', 'SX', 'DX', 'Spara']))

    # --- 6. SALVATAGGIO ---
    os.makedirs(resources_dir, exist_ok=True)
    model.save_model(MODEL_FILE)
    print(f"Modello salvato correttamente in: {MODEL_FILE}")

if __name__ == "__main__":
    train_and_evaluate()