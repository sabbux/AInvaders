import csv
import numpy as np
import sys
import os

# --- AGGIUNTA PER TROVARE IL MODULO ---
current_dir = os.path.dirname(os.path.abspath(_file_))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------------------

from spaceinvaders import SpaceInvadersEnvironment

# --- CONFIGURAZIONE ---
EPISODES = 100
OUTPUT_FILE = 'dataset_heuristic.csv'

class HeuristicAgent:
    def _init_(self):
        pass

    def get_action(self, state):
        """
        state = [p_x, my_bullet, e_x, e_y, b_x, b_y, dir]
        """
        player_x = state[0]
        my_bullet_flying = (state[1] == 1.0)
        enemy_x = state[2]
        # enemy_y = state[3]
        bullet_x = state[4]
        bullet_y = state[5] # 0.0 se non ci sono proiettili, >0 se esistono

        # --- PARAMETRI DI COMPORTAMENTO ---
        # Distanza di sicurezza: se il proiettile è più vicino di così, SCAPPA.
        # 0.1 è circa il doppio della larghezza della nave.
        DANGER_DIST = 0.1

        # Tolleranza di mira: se la differenza col nemico è minore di questa, SPARA.
        # Aumentare questo valore riduce il "tremolio".
        AIM_TOLERANCE = 0.02

        # --- 1. MODULO SOPRAVVIVENZA (Priorità Assoluta) ---
        # Si attiva se c'è un proiettile (b_y > 0) E se è vicino orizzontalmente
        if bullet_y > 0 and abs(player_x - bullet_x) < DANGER_DIST:

            # Strategia di fuga: Vai nella direzione opposta al proiettile
            # Ma controlla i bordi per non restare incastrato!

            if player_x > bullet_x:
                # Il proiettile è a sinistra, vado a DESTRA
                # (Se non sono già al bordo destro 0.95)
                if player_x < 0.95:
                    return 2
                else:
                    return 1 # Se sono all'angolo, prega (o stai fermo)

            else:
                # Il proiettile è a destra (o sopra), vado a SINISTRA
                if player_x > 0.05:
                    return 1
                else:
                    return 2

        # --- 2. MODULO ATTACCO ---
        dist_enemy = player_x - enemy_x

        # Se sono allineato "abbastanza bene" (dentro la tolleranza)
        if abs(dist_enemy) < AIM_TOLERANCE:
            if not my_bullet_flying:
                return 3 # SPARA!
            else:
                return 0 # STAI FERMO

       # Se non sono allineato, correggo la posizione
        elif dist_enemy < 0:
            return 2 # Il nemico è a destra -> Vado a destra
        else:
            return 1 # Il nemico è a sinistra -> Vado a sinistra

def run_data_collection():
    # collect_data=False perché gestiamo il salvataggio qui
    env = SpaceInvadersEnvironment(collect_data=False)
    agent = HeuristicAgent()

    print(f"--- INIZIO RACCOLTA SMART ---")

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

            # Variabile per ricordare lo stato precedente
            last_saved_state = None

            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)

                # --- FILTRO ANTI-DOPPIONI ---
                # Salviamo la riga SOLO se lo stato è cambiato significativamente rispetto all'ultimo salvato.
                # Usiamo np.allclose per confrontare i numeri float con una piccola tolleranza.
                should_save = True
                if last_saved_state is not None:
                    # Se lo stato è quasi identico al precedente (es. nemici fermi e noi fermi)
                    if np.allclose(state, last_saved_state, atol=1e-4):
                        should_save = False

                if should_save:
                    row = list(state) + [action]
                    writer.writerow(row)
                    last_saved_state = state # Aggiorna l'ultimo salvato
                    total_rows += 1
                else:
                    skipped_rows += 1

                state = next_state

            print(f"Episodio {episode}: MORTO al Livello {env.level} | Score: {env.score}")
            final_levels.append(env.level)
            final_scores.append(env.score)

            if episode % 2 == 0:
                print(f"Episodio {episode}/{EPISODES} - Righe: {total_rows} (Saltate: {skipped_rows})")

    print("\n--- FINE ---")
    print(f"Livello Massimo Raggiunto: {max(final_levels)}")
    print(f"Livello Medio: {sum(final_levels) / len(final_levels):.2f}")
    print(f"Score Medio: {sum(final_scores) / len(final_scores):.1f}")
    print(f"Righe Utili: {total_rows}")
    print(f"Righe Inutili (Duplicate) rimosse: {skipped_rows}")

if _name_ == "_main_":
    run_data_collection()