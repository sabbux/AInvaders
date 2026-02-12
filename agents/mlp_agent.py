import sys
import os
import joblib
import numpy as np
import pandas as pd  
import time

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
resources_dir = os.path.join(project_root, "resources")

from environment.spaceinvaders import SpaceInvadersEnvironment
from utils.game_utils import compute_features, get_feature_names

# CONFIGURAZIONE
MODEL_FILE = os.path.join(resources_dir, 'mlp_brain.pkl')
SCALER_FILE = os.path.join(resources_dir, 'mlp_scaler.pkl')

def run_mlp_agent():
    # Caricamento Modello e Scaler
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("ERRORE: Manca il modello o lo scaler. Esegui train_mlp.py!")
        return

    mlp = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Rete Neurale Caricata. Inizio Partita...")
    
    env = SpaceInvadersEnvironment(collect_data=False)
    
    # Assicurati che WATCH_MODE sia True in environment.py per vedere!
    episodes = 5
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Inizio Episodio {ep+1} ---")
        
        while not done:
            # 1. Prepara i dati grezzi
            raw_data = {
                'p_x': state[0], 'my_bullet': state[1],
                'e_x': state[2], 'e_y': state[3],
                'b_x': state[4], 'b_y': state[5], 'dir': state[6]
            }
            
            # 2. Calcola le features matematiche (Dizionario)
            features = compute_features(raw_data)
            
            # 3. CREIAMO UN DATAFRAME
            # Creiamo un DataFrame con una sola riga, usando le chiavi del dizionario come colonne
            input_df = pd.DataFrame([features])
            
            # 4. Ordiniamo le colonne rigorosamente come nel training
            input_df = input_df[get_feature_names()]
            
            # 5. SCALING (Ora riceve un DataFrame con i nomi corretti)
            scaled_features = scaler.transform(input_df)
            
            # 6. Predizione
            action = mlp.predict(scaled_features)[0]
            
            state, reward, done = env.step(action)
            total_reward += reward
                        
        print(f"Episodio {ep+1} terminato - Score: {env.score} (Reward: {total_reward})")
        time.sleep(1)

if __name__ == "__main__":
    run_mlp_agent()