import numpy as np

def compute_features(row):
    """
    Riceve un dizionario o una riga (con chiavi p_x, e_x, b_x...)
    Restituisce un dizionario con TUTTE le feature (originali + calcolate).
    """
    # 1. Estrai variabili base
    p_x = row['p_x']
    e_x = row['e_x']
    b_x = row['b_x']
    
    # 2. Calcoli matematici (Feature Engineering)
    # Distanza relativa X (Nemico - Player)
    delta_x = e_x - p_x
    
    # Pericolo Proiettile (Distanza X dal proiettile)
    bullet_danger_x = b_x - p_x
    
    # Booleani (convertiti in int 0/1 per XGBoost)
    # Sotto tiro: se il proiettile è allineato verticalmente (tolleranza 0.05)
    under_fire = 1 if abs(bullet_danger_x) < 0.05 else 0
    
    # Allineato per sparare: se il nemico è davanti a noi
    aim_locked = 1 if abs(delta_x) < 0.05 else 0
    
    # 3. Restituisci tutto
    # Nota: copiamo anche le altre variabili originali per averle nel dizionario finale
    return {
        'p_x': p_x,
        'my_bullet': row['my_bullet'],
        'e_x': e_x,
        'e_y': row['e_y'],
        'b_x': b_x,
        'b_y': row['b_y'],
        'dir': row['dir'],
        'delta_x': delta_x,
        'bullet_danger_x': bullet_danger_x,
        'under_fire': under_fire,
        'aim_locked': aim_locked
    }

def get_feature_names():
    """
    Restituisce l'elenco ORDINATO delle colonne.
    È vitale che Training e Agente usino lo stesso identico ordine.
    """
    return [
        'p_x', 'my_bullet', 'e_x', 'e_y', 'b_x', 'b_y', 'dir',  # Originali
        'delta_x', 'bullet_danger_x', 'under_fire', 'aim_locked' # Calcolate
    ]