"""
Script for demonstrating the learned behavior of a PPO agent to control a traffic light at an intersection in SUMO
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import sumo_env

# Env setup
MODEL_STEPS_LIMIT = 150
SIMULATION_STEPS_LIMIT = MODEL_STEPS_LIMIT * 60 # Steps per simulation (how many tenths of a second), but we only make an action every 60 steps (every 6 seconds)
MAX_TRAFFIC = 50 # Max traffic for curriculum learning
AWT_COEF = 0.05
AQL_COEF = 0.05
SPEED_COEF = 0.15
SUCCESS_COEF = 0.005

EPISODES_TO_RUN = 10 # How many episodes to demo
NUM_CPUS = 8

model_to_load = 'models/lights5_ppo_lrdecay_lr0_0005_ec0_01_g0_99_l0_95_2026_05_18T18_10_42/_345600_steps.zip'

if __name__ == '__main__':
  vec_env = make_vec_env('Lights-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'max_traffic': MAX_TRAFFIC,
    'awt_coef': AWT_COEF,
    'aql_coef': AQL_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'sumo_config_file': 'sumo_env/sumo_files/intersection.sumocfg',
    'render_mode': 'human',
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
