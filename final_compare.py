import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from collections import deque  # <--- Serve per la memoria dei 4 frame
from stable_baselines3 import PPO


from spaceinvaders import SpaceInvadersEnvironment
from game_utils import compute_features, get_feature_names

# --- CONFIGURAZIONE PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.join(current_dir, "agents")

# FILE
XGB_FILE = os.path.join(current_dir, "xgboost_brain.json")
MLP_FILE = os.path.join(current_dir, "mlp_brain.pkl")
SCALER_FILE = os.path.join(current_dir, "mlp_scaler.pkl")
# Nota: Cerca il file con "_stacked"
PPO_FILE = os.path.join(agents_dir, "ppo_space_invaders_stacked.zip")

# --- CARICAMENTO MODELLI ---
print(f"--- CARICAMENTO AGENTI ---")

# 1. XGBoost
xgb_model = None
if os.path.exists(XGB_FILE):
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model(XGB_FILE)
        print(f"✅ XGBoost caricato")
    except: print(f"❌ XGBoost non trovato")

# 2. MLP
mlp_model = None
mlp_scaler = None
if os.path.exists(MLP_FILE):
    try:
        mlp_model = joblib.load(MLP_FILE)
        mlp_scaler = joblib.load(SCALER_FILE)
        print(f"✅ MLP caricato")
    except: print(f"❌ MLP non trovato")

# 3. PPO Stacked
ppo_model = None
if os.path.exists(PPO_FILE):
    try:
        ppo_model = PPO.load(PPO_FILE)
        print(f"✅ PPO Stacked caricato")
    except Exception as e:
        print(f"❌ Errore PPO: {e}")

ppo_memory = deque(maxlen=4)



def get_xgboost_action(state):
    raw_data = {
        'p_x': state[0], 'my_bullet': state[1], 'e_x': state[2], 
        'e_y': state[3], 'b_x': state[4], 'b_y': state[5], 'dir': state[6]
    }
    feats = compute_features(raw_data)
    dmatrix = xgb.DMatrix(np.array([[feats[col] for col in get_feature_names()]]), feature_names=get_feature_names())
    return int(xgb_model.predict(dmatrix)[0])

def get_mlp_action(state):
    raw_data = {
        'p_x': state[0], 'my_bullet': state[1], 'e_x': state[2], 
        'e_y': state[3], 'b_x': state[4], 'b_y': state[5], 'dir': state[6]
    }
    feats = compute_features(raw_data)
    df = pd.DataFrame([feats])[get_feature_names()]
    scaled = mlp_scaler.transform(df)
    return mlp_model.predict(scaled)[0]

def get_ppo_stacked_action(state, reset_memory=False):
    
    obs = np.array(state, dtype=np.float32)
    
    # Se è l'inizio della partita, riempiamo la memoria con 4 copie dello stesso stato
    if reset_memory:
        ppo_memory.clear()
        for _ in range(4):
            ppo_memory.append(obs)
    else:
        # Altrimenti aggiungiamo il nuovo stato (il più vecchio viene buttato fuori automaticamente)
        ppo_memory.append(obs)
    
    
    
    stacked_obs = np.concatenate(list(ppo_memory))
    
    
    final_input = stacked_obs.reshape(1, -1)
    
    action, _ = ppo_model.predict(final_input, deterministic=True)

# Estrae il valore scalare dall'array numpy
    return int(action.item())

# --- MOTORE DEL TORNEO ---

def run_tournament(name, action_function, episodes=5):
    if action_function is None: return 0
    
    
    env = SpaceInvadersEnvironment(collect_data=False)
    
    scores = []
    print(f"\n>>> INIZIO TURNO: {name} <<<")
    
    for i in range(episodes):
        state = env.reset()
        done = False
        start_time = time.time()
        
       
        if name == "PPO":
           
            pass 
            
       
        is_ppo_start = True

        while not done:
            try:
                if name == "PPO":
                    # Passiamo is_ppo_start per resettare la memoria al primo frame
                    action = get_ppo_stacked_action(state, reset_memory=is_ppo_start)
                    is_ppo_start = False # Dal prossimo frame in poi, usa la memoria normale
                elif name == "XGBoost":
                    action = get_xgboost_action(state)
                elif name == "MLP":
                    action = get_mlp_action(state)
                
                state, reward, done = env.step(action)
            except Exception as e:
                print(f"ERRORE: {e}")
                done = True
        
        final_score = env.score
        scores.append(final_score)
        duration = time.time() - start_time
        print(f"   Partita {i+1}: {int(final_score)} punti ({duration:.1f}s) - Lvl {env.level}")
        
    avg_score = sum(scores) / len(scores)
    print(f"   ---> MEDIA {name}: {avg_score:.1f}")
    return avg_score



results = {}

if xgb_model: results['XGBoost'] = run_tournament("XGBoost", get_xgboost_action)
if mlp_model: results['MLP'] = run_tournament("MLP", get_mlp_action)
if ppo_model: results['PPO'] = run_tournament("PPO", get_ppo_stacked_action)

print("\n\n================ CLASSIFICA FINALE ================")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(sorted_results):
    print(f"{rank+1}. {name}: {score:.1f} punti medi")
print("===================================================")