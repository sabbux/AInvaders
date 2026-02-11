import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Importa il tuo gioco originale (Assicurati che environment.py esista!)
from spaceinvaders import SpaceInvadersEnvironment

class SpaceInvadersGym(gym.Env):
    """
    Wrapper per rendere SpaceInvaders compatibile con Stable Baselines3 (PPO)
    """
    def __init__(self):
        super(SpaceInvadersGym, self).__init__()

        # WATCH_MODE=False è fondamentale per la velocità
        self.game = SpaceInvadersEnvironment(collect_data=False)

        # 4 Azioni: 0=Fermo, 1=SX, 2=DX, 3=Spara
        self.action_space = spaces.Discrete(4)

        # Osservazione: 7 numeri float (coordinate, distanze, ecc.)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_state = self.game.reset()
        # Converte in float32 per PPO
        return np.array(raw_state, dtype=np.float32), {}

    def step(self, action):
        next_state, reward, done = self.game.step(action)

        # PPO vuole float32
        observation = np.array(next_state, dtype=np.float32)

        # Gymnasium distingue tra 'terminated' (morto/vinto) e 'truncated' (tempo scaduto)
        terminated = bool(done)
        truncated = False

        return observation, reward, terminated, truncated, {}

    def render(self):
        pass # Il rendering lo fa Pygame se WATCH_MODE=True