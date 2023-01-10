import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class DiscreteDamEnv(gym.Env):
    """Dam Environment that follows gym interface"""

    def __init__(self, data:pd.DataFrame):
        # static properties
        self.max_stored_energy = 100000 * 1000 * 9.81 * 30 # U = mgh
        self.min_stored_energy = 0
        self.max_flow_rate = 5 * 3600 * 9.81 * 30 # 5 m^3/s
        self.price_bins = 10

        self.data = data

        self.reset()

        # 0 = do nothing
        # 1 = empty
        # 2 = fill
        self.action_space = spaces.Discrete(3)

        # state is (hour, electricity price (bins))
        self.observation_space = spaces.MultiDiscrete([24, self.price_bins])

    def step(self, action):
        # do nothing
        if action == 0:
            pass

        # empty
        elif action == 1:
            flow = self._action_empty(self.max_flow_rate)

        # fill
        elif action == 2:
            flow = self._action_fill(self.max_flow_rate)

        self._next_hour()
        
        return 
        
    def _action_fill(self, rate:float):
        # don't allow reservoir to fill above max
        difference_to_max = self.max_stored_energy - self.stored_energy
        if difference_to_max < rate:
            rate = difference_to_max

        self.stored_energy += rate
        return rate

    def _action_empty(self, rate:float):
        # don't allow reservoir to empty below min
        difference_to_min = self.stored_energy - self.min_stored_energy
        if difference_to_min < rate:
            rate = difference_to_min

        self.stored_energy -= rate
        return rate

    def _next_hour(self):
        self.hour += 1
        if self.hour == 24:
            self.hour = 0

        self.energy_price = self._get_energy_price(self.hour)

    def reset(self):
        self.stored_energy = self.max_stored_energy
        self.hour = 0

