import numpy as np
from gymnasium import spaces

from src.environment.dam import TestEnvWrapper
from src.environment.TestEnv import HydroElectric_Test


class TestEnvWrapperSmall(TestEnvWrapper):
    def __init__(self, env: HydroElectric_Test):
        super().__init__(env)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3,))

    def _preprocess_state(self, state: np.ndarray):
        # we only care about the first three features from the state
        processed_state = state[:3].copy()

        processed_state[0] /= self.env.max_volume
        processed_state[1] /= 200
        processed_state[2] /= 24

        return processed_state
