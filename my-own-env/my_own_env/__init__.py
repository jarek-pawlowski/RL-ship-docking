from gym.envs.registration import register

register(
    id='Poll2D-v0',
    entry_point='my_own_env.envs:Poll2DEnv'
)