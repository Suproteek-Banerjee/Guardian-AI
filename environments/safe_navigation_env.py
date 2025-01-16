import gym
from gym import spaces
import numpy as np

class SafeNavigationEnv(gym.Env):
    def __init__(self):
        super(SafeNavigationEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Actions: Left, Stay, Right
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = 0.0
        self.goal = 1.0

    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if action == 0:  # Move Left
            self.state -= 0.1
        elif action == 2:  # Move Right
            self.state += 0.1

        reward = -1 if abs(self.state - self.goal) > 0.1 else 1
        done = abs(self.state - self.goal) <= 0.1
        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        print(f"State: {self.state:.2f}")
