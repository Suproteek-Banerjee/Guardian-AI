from gym.envs.registration import register

register(
    id='SafeNavigation-v0',
    entry_point='environments.safe_navigation_env:SafeNavigationEnv',
    max_episode_steps=100,
)
