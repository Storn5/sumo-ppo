from gymnasium.envs.registration import register

register(
    id='Lights-Sumo-v1',
    entry_point='sumo_env.lights_env:LightsEnv',
)