import csv
import numpy as np
import sys
import os

# --- AGGIUNTA PER TROVARE IL MODULO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------------------

from spaceinvaders import SpaceInvadersEnvironment

# --- CONFIGURAZIONE ---
EPISODES = 100
OUTPUT_FILE = 'dataset_heuristic.csv'

class HeuristicAgent:
    def __init__(self):
        pass

    def get_action(self, state):
        """
        state = [p_x, my_bullet, e_x, e_y, b_x, b_y, dir]
        """
        
        # ============================================================
        # STRATEGIA 1: IL MAESTRO UBRIACO (Hardcore Version)
        # ============================================================
        # Ora il 33% delle volte (1 su 3) fa una mossa a caso.
        # L'agente morirà molto spesso, ma genererà dati di "recupero estremo".
        if np.random.rand() < 0.33:
            return np.random.randint(0, 4)
        # ============================================================

        player_x = state[0]
        my_bullet_flying = (state[1] == 1.0)
        enemy_x = state[2]
        # enemy_y = state[3]
        bullet_x = state[4]
        bullet_y = state[5] 

        # --- PARAMETRI DI COMPORTAMENTO ---
        DANGER_DIST = 0.1
        AIM_TOLERANCE = 0.02

        # --- 1. MODULO SOPRAVVIVENZA (Priorità Assoluta) ---
        if bullet_y > 0 and abs(player_x - bullet_x) < DANGER_DIST:
            if player_x > bullet_x:
                if player_x < 0.95:
                    return 2 # Destra
                else:
                    return 1 # Sinistra (angolo)
            else:
                if player_x > 0.05:
                    return 1 # Sinistra
                else:
                    return 2 # Destra

        # --- 2. MODULO ATTACCO ---
        dist_enemy = player_x - enemy_x

        if abs(dist_enemy) < AIM_TOLERANCE:
            if not my_bullet_flying:
                return 3 # SPARA
            else:
                return 0 # FERMO
        elif dist_enemy < 0:
            return 2 # Insegui a Destra
        else:
            return 1 # Insegui a Sinistra

def run_data_collection():
    # collect_data=False perché gestiamo il salvataggio qui
    env = SpaceInvadersEnvironment(collect_data=False)
    agent = HeuristicAgent()

    print(f"--- INIZIO RACCOLTA SMART (33% Caos) ---")

    final_levels = []
    final_scores = []

    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["p_x", "my_bullet", "e_x", "e_y", "b_x", "b_y", "dir", "action"])

        total_rows = 0
        skipped_rows = 0

        for episode in range(1, EPISODES + 1):
            state = env.reset()
            done = False
            last_saved_state = None

            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)

                # --- FILTRO ANTI-DOPPIONI ---
                should_save = True
                if last_saved_state is not None:
                    if np.allclose(state, last_saved_state, atol=1e-4):
                        should_save = False

                if should_save:
                    # ============================================================
                    # STRATEGIA 2: UNDERSAMPLING AGGRESSIVO
                    # ============================================================
                    # Scartiamo l'80% delle volte in cui sta fermo.
                    # Vogliamo un'IA che si muova sempre.
                    if action == 0 and np.random.rand() > 0.20: # Salviamo solo il 20% degli "0"
                        skipped_rows += 1
                    else:
                        row = list(state) + [action]
                        writer.writerow(row)
                        last_saved_state = state 
                        total_rows += 1
                else:
                    skipped_rows += 1

                state = next_state

            # Feedback visuale
            print(f"Episodio {episode}: MORTO al Livello {env.level} | Score: {env.score}")
            final_levels.append(env.level)
            final_scores.append(env.score)

            if episode % 10 == 0:
                print(f"   >>> Righe Totali: {total_rows} (Scartate: {skipped_rows})")

    print("\n--- FINE ---")
    print(f"Livello Massimo Raggiunto: {max(final_levels)}")
    print(f"Livello Medio: {sum(final_levels) / len(final_levels):.2f}")
    print(f"Righe Utili: {total_rows}")
    print(f"Righe Scartate: {skipped_rows}")

if __name__ == "__main__":
    run_data_collection()