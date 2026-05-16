"""
Script for continuing the training of a PPO agent to control traffic lights in the SUMO intersection environment
from a saved model state
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import sumo_env
from sumo_env.lights_env import LightsEnv

# Env setup
MODEL_STEPS_LIMIT = 150
SIMULATION_STEPS_LIMIT = MODEL_STEPS_LIMIT * 60 # Steps per simulation (how many tenths of a second), but we only make an action every 60 steps (every 6 seconds)
AWT_COEF = 0.05
AQL_COEF = 0.05
SPEED_COEF = 0.15
SUCCESS_COEF = 0.005

EPISODES_PER_MINIBATCH = 2 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 150 # How many updates to run in total for training
OFFSET_BATCHES = 50 # How many batches were already run

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 32 # How many steps in each "minibatch" that PPO performs

model_to_load = 'models/ppo_lr0_001_ec0_04_g0_99_l0_95_2026_05_12T14_26_17/120000.zip'
model_name = '_'.join(model_to_load.split('/')[1].split('_')[:9])
model_name = f'{model_name}_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}'
models_dir = f'models/{model_name}'
logs_dir = f'logs/{model_name}'

if __name__ == '__main__':
  for folder in (models_dir, logs_dir):
    if not os.path.exists(folder):
      os.makedirs(folder)

  vec_env = make_vec_env('Lights-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'awt_coef': AWT_COEF,
    'aql_coef': AQL_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'render_mode': None,
  }, monitor_dir=logs_dir, monitor_kwargs={
    'info_keywords': ('total_arrived', 'episode_mean_waiting_time', 'episode_mean_queue_length', 'episode_mean_speed')
  })

  logger = configure(logs_dir, ['csv'])

  model = PPO.load(model_to_load)
  model.set_env(vec_env)
  model.set_logger(logger)

  for i in range(OFFSET_BATCHES + 1, TOTAL_BATCHES + OFFSET_BATCHES + 1):
    model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
  vec_env.close()
