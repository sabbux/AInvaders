import os
import sys
import pandas as pd
import joblib # Per salvare modello e scaler
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
# Risaliamo a: space-invaders-ai/
project_root = os.path.dirname(current_dir)

# Aggiungiamo la root al path per importare i moduli
sys.path.append(project_root)

# Definiamo la cartella resources
resources_dir = os.path.join(project_root, "resources")

# Importiamo le funzioni condivise (FONDAMENTALE)
from utils.game_utils import compute_features, get_feature_names

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join(resources_dir, 'dataset_heuristic.csv')
MODEL_FILE = os.path.join(resources_dir, 'mlp_brain.pkl')
SCALER_FILE = os.path.join(resources_dir, 'mlp_scaler.pkl')

def mlp_train():
    print("--- 1. PREPARAZIONE DATI PER RETE NEURALE ---")
    
    # 1. Caricamento
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERRORE: Non trovo {INPUT_FILE}. Genera prima i dati!")
        return

    print(f"Righe Totali: {len(df)}")
    df = df.drop_duplicates()
    print(f"Righe Uniche: {len(df)}")
    
    # 2. Feature Engineering (Usando game_utils)
    print("Calcolo variabili matematiche...")
    # Applica la funzione condivisa a ogni riga
    features_list = df.apply(compute_features, axis=1).tolist()
    
    # Crea il DataFrame delle features X
    X_raw = pd.DataFrame(features_list)
    # Assicura l'ordine corretto delle colonne
    X_raw = X_raw[get_feature_names()]
    
    y = df['action']
    
    # 3. SCALING (Step Critico per MLP)
    # Le reti neurali falliscono se i numeri non sono tra -1 e 1 (o simili)
    print("Normalizzazione (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 4. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- 2. CONFIGURAZIONE RETE NEURALE (MLP) ---
    print("\n--- 2. ADDESTRAMENTO MODELLO ---")
    
    # Struttura:
    # - Input Layer: 11 neuroni (le tue feature)
    # - Hidden Layer 1: 128 neuroni (impara pattern complessi)
    # - Hidden Layer 2: 64 neuroni (raffina i pattern)
    # - Output Layer: 4 neuroni (le 4 azioni)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        activation='relu',       # Funzione di attivazione standard
        solver='adam',           # Ottimizzatore standard
        max_iter=500,            # Numero massimo di epoche
        random_state=42,
        early_stopping=True,     # Smette se non migliora più
        validation_fraction=0.1, # Usa il 10% del train per validare
        verbose=True             # Mostra la "Loss" che scende
    )
    
    # Cronometro Avvio
    start_time = time.time()
    
    mlp.fit(X_train, y_train)
    
    # Cronometro Stop
    end_time = time.time()
    training_time = end_time - start_time

    # --- 3. VALUTAZIONE COMPLETA ---
    print("\n--- 3. REPORT PRESTAZIONI ---")
    
    preds = mlp.predict(X_test)

    # Calcolo Metriche (Weighted gestisce lo sbilanciamento delle classi)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    print(f"Tempo di Training: {training_time:.4f} secondi")
    print("-" * 30)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("-" * 30)
    
    print("\nDettaglio per Azione:")
    print(classification_report(y_test, preds, target_names=['Fermo', 'SX', 'DX', 'Spara'], zero_division=0))
    
    # 4. Salvataggio
    os.makedirs(resources_dir, exist_ok=True)
    joblib.dump(mlp, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSalvati: {MODEL_FILE} e {SCALER_FILE}")

if __name__ == "__main__":
    mlp_train()