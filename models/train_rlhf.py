import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environments.safe_navigation_env import SafeNavigationEnv
from utils.feedback_handler import FeedbackHandler

from gym.envs.registration import register

register(
    id='SafeNavigation-v0',
    entry_point='environments.safe_navigation_env:SafeNavigationEnv',
    max_episode_steps=100,
)
from gym.envs.registration import register

register(
    id='SafeNavigation-v0',
    entry_point='environments.safe_navigation_env:SafeNavigationEnv',
    max_episode_steps=100,
)


# Load the custom environment
env = gym.make('SafeNavigation-v0')

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Feedback handler
feedback_handler = FeedbackHandler()

# Training with feedback loop
def train_with_feedback(total_timesteps=10000):
    obs = env.reset()
    for t in range(total_timesteps):
        action, _states = model.predict(obs)
        feedback = feedback_handler.collect_feedback(obs, action)
        feedback_handler.log_feedback(obs, action, feedback)

        # Apply feedback to reward
        modified_reward = feedback
        obs, _, done, _ = env.step(action)

        # Update model reward signal
        model.policy.optimizer.zero_grad()
        if feedback == 1:
            model.policy.log_probabilities += modified_reward
        elif feedback == -1:
            model.policy.log_probabilities -= modified_reward
        model.policy.optimizer.step()

        if done:
            obs = env.reset()

    # Save the model
    model.save("ppo_safe_navigation_with_feedback")
    print("Training complete. Model saved!")

# Train the model with human feedback
train_with_feedback()

# Test the model
def test_model():
    model = PPO.load("ppo_safe_navigation_with_feedback")
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            break

# Test the model
test_model()
