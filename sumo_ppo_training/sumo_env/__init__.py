from gymnasium.envs.registration import register

register(
    id='Lights-Sumo-v1',
    entry_point='sumo_env.lights_env:LightsEnv',
)

register(
    id='Vehicle-Sumo-v1',
    entry_point='sumo_env.vehicle_env:VehicleEnv',
)
