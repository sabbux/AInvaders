import os
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor 

# --- 1. SETUP PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Risaliamo alla cartella del progetto per trovare gym_env
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from environment.gym_env import SpaceInvadersGym

# --- 2. CONFIGURAZIONE CARTELLE ---
agents_dir = os.path.join(project_root, "agents")
log_dir = os.path.join(agents_dir, "logs")
checkpoints_dir = os.path.join(agents_dir, "checkpoints")

os.makedirs(agents_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

def make_env():
    """
    Funzione helper per creare l'ambiente con il Monitor.
    Serve a DummyVecEnv per inizializzare tutto correttamente.
    """
    env = SpaceInvadersGym()
    # Avvolgiamo l'ambiente nel Monitor.
    # Questo riabilita la stampa di ep_rew_mean e ep_len_mean!
    env = Monitor(env, log_dir) 
    return env

def train_rl():
    print("--- TRAINING PPO CON FRAME STACKING & MONITOR ---")
    print(f"I log verranno salvati in: {log_dir}")
    
    # 1. Crea l'ambiente Vettoriale
    # Passiamo la funzione make_env che contiene il Monitor
    env = DummyVecEnv([make_env])
    
    # 2. APPLICA IL FRAME STACKING
    # Ora l'IA vede 4 frame alla volta (velocità + direzione)
    env = VecFrameStack(env, n_stack=4)
    
    # 3. Configura PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003)
    
    # 4. Setup Salvataggi
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoints_dir, 
        name_prefix="ppo_stack"
    )
    
    # 5. ADDESTRAMENTO
    TIMESTEPS = 1000000 
    
    print(f"🚀 Inizio training ({TIMESTEPS} steps)...")
    start_time = time.time()
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    end_time = time.time()
    print(f"⏱️ Finito in {end_time - start_time:.2f} secondi.")
    
    # 6. Salvataggio Finale
    final_path = os.path.join(agents_dir, "ppo_space_invaders_stacked")
    model.save(final_path)
    print(f"✅ MODELLO COMPLETATO: {final_path}.zip")

if __name__ == "__main__":
    train_rl()