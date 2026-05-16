"""
Script for demonstrating the learned behavior of a PPO agent to control an autonomous vehicle at an unsignalized intersection in SUMO
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import sumo_env

# Env setup
MODEL_STEPS_LIMIT = 128
SIMULATION_STEPS_LIMIT = 5 * MODEL_STEPS_LIMIT # Steps per simulation (how many tenths of a second), but we only make an action every 5 steps (every 0.5 seconds)
MAX_TRAFFIC = 20 # Max traffic for curriculum learning
COLLISION_COEF = 100.0
TIMEOUT_COEF = 100.0
SPEED_COEF = 0.0
SUCCESS_COEF = 100.0
PROXIMITY_COEF = 5.0

EPISODES_TO_RUN = 10 # How many episodes to demo

model_to_load = 'models/vehicle_ppo_lr0_0002_ec0_05_g0_99_l0_95_2026_05_16T18_02_33/720896.zip'

if __name__ == '__main__':
  vec_env = make_vec_env('Vehicle-Sumo-v1', n_envs=8, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'max_traffic': MAX_TRAFFIC,
    'collision_coef': COLLISION_COEF,
    'timeout_coef': TIMEOUT_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'proximity_coef': PROXIMITY_COEF,
    'render_mode': 'human',
    'sumo_config_file': 'sumo_env/sumo_files/intersection_vehicle.sumocfg',
  })

  model = PPO.load(model_to_load)
  model.set_env(vec_env)
  obs = vec_env.reset()

  for _ in range(EPISODES_TO_RUN):
    while True:
      action, _state = model.predict(obs, deterministic=False)
      obs, reward, done, info = vec_env.step(action)
      if done.any():
        print(f'Episode over, reward: {reward}')
        break

  vec_env.close()
