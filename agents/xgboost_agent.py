import sys
import os
import time
import numpy as np
import xgboost as xgb
import pygame

# Setup path per trovare environment e utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
resources_dir = os.path.join(project_root, "resources")

from environment.spaceinvaders import SpaceInvadersEnvironment
# IMPORTA LA LOGICA CONDIVISA
from utils.game_utils import compute_features, get_feature_names

# Configurazione Modello
MODEL_FILE = os.path.join(resources_dir, "xgboost_brain.json")

if not os.path.exists(MODEL_FILE):
    print(f"ERRORE: Non trovo il file {MODEL_FILE}. Esegui prima xgboost_train.py!")
    sys.exit()

# Carica il modello una volta sola all'avvio
model = xgb.Booster()
model.load_model(MODEL_FILE)

def get_ai_action(state):
    """
    Riceve lo stato grezzo dall'ambiente, calcola le feature matematiche
    e chiede a XGBoost cosa fare.
    """
    # 1. Convertiamo l'array numpy grezzo in un dizionario
    # state = [p_x, my_bullet, e_x, e_y, b_x, b_y, dir]
    raw_data = {
        'p_x': state[0], 
        'my_bullet': state[1],
        'e_x': state[2], 
        'e_y': state[3],
        'b_x': state[4], 
        'b_y': state[5], 
        'dir': state[6]
    }
    
    # 2. FEATURE ENGINEERING
    processed_data = compute_features(raw_data)
    
    # 3. Ordiniamo i dati esattamente come vuole XGBoost
    feature_order = get_feature_names()
    input_values = [processed_data[col] for col in feature_order]
    
    # 4. Creiamo la matrice DMatrix per la predizione
    # XGBoost si aspetta una matrice 2D, quindi mettiamo la lista dentro un'altra lista
    dmatrix = xgb.DMatrix(np.array([input_values]), feature_names=feature_order)
    
    # 5. Predizione
    prediction = model.predict(dmatrix)
    return int(prediction[0])

def run_ai_player():
    # collect_data=False -> Non registriamo, giochiamo solo
    env = SpaceInvadersEnvironment(collect_data=False)
    
    # Assicurati che in environment.py WATCH_MODE sia True per vedere la partita!
    print("--- AVVIO AGENTE XGBOOST (Premi CTRL+C per uscire) ---")
    
    episodes = 5 # Quante partite vuoi vedere
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"Inizio Episodio {episode + 1}...")
        
        while not done:
            # 1. Chiedi al cervello cosa fare
            action = get_ai_action(state)
            
            # 2. Esegui l'azione
            state, reward, done = env.step(action)
            total_reward += reward
            
        print(f"Episodio {episode+1} terminato. Livello Raggiunto: {env.level} | Reward Totale: {int(total_reward)}")
        print(f"Episodio {episode+1}: MORTO al Livello {env.level} | Score: {env.score}")
        time.sleep(1) # Pausa tra una partita e l'altra

if __name__ == "__main__":
    run_ai_player()