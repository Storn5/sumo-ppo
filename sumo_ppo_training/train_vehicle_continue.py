"""
Script for continuing the training of a PPO agent to control an autonomous vehicle at an unsignalized intersection
in the SUMO intersection environment from a saved model state
"""

import os
import re
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

import sumo_env
from sumo_env.vehicle_env import VehicleEnv
from ppo_levy_flight import LevyPPO
from ppo_decay_schedule import linear_schedule

# These parameters need to change between runs
MAX_TRAFFIC = 5 # Max traffic for curriculum learning
OFFSET_BATCHES = 200 # How many batches were already run
model_to_load = 'models/ppo_lr0_001_ec0_04_g0_99_l0_95_2026_05_12T14_26_17/120000.zip'
# Hyperparameters setup
ENT_COEF_DECAY = 0.0
CLIP_RANGE_DECAY = False
LR_DECAY = False
USE_LEVY = False

# Env setup
MODEL_STEPS_LIMIT = 128
SIMULATION_STEPS_LIMIT = 5 * MODEL_STEPS_LIMIT # Steps per simulation (how many tenths of a second), but we only make an action every 5 steps (every 0.5 seconds)
COLLISION_COEF = 100.0
TIMEOUT_COEF = 100.0
SPEED_COEF = 5.0
SUCCESS_COEF = 50.0
PROXIMITY_COEF = 5.0

EPISODES_PER_MINIBATCH = 4 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 200 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 16 # How many steps in each "minibatch" that PPO performs

model_name = '_'.join(model_to_load.split('/')[1].split('_')[:-5])
model_name = re.sub('vehicle(\\d)+_', f'vehicle{MAX_TRAFFIC}_', model_name)
model_name = f'{model_name}_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}'
models_dir = f'models/{model_name}'
logs_dir = f'logs/{model_name}'

if __name__ == '__main__':
  for folder in (models_dir, logs_dir):
    if not os.path.exists(folder):
      os.makedirs(folder)

  vec_env = make_vec_env('Vehicle-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'max_traffic': MAX_TRAFFIC,
    'collision_coef': COLLISION_COEF,
    'timeout_coef': TIMEOUT_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'proximity_coef': PROXIMITY_COEF,
    'render_mode': None,
    'sumo_config_file': 'sumo_env/sumo_files/intersection_vehicle.sumocfg',
  }, monitor_dir=logs_dir, monitor_kwargs={
    'info_keywords': ('episode_mean_speed', 'success')
  })

  logger = configure(logs_dir, ['csv'])

  model = None
  if USE_LEVY:
    model = LevyPPO.load(model_to_load, env=vec_env, device='cpu',  verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  else:
    model = PPO.load(model_to_load, env=vec_env, device='cpu',  verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)

  model.set_logger(logger)

  # Handle saving every batch
  checkpoint_callback = CheckpointCallback(
    save_freq=STEPS_PER_BATCH, 
    save_path=models_dir,
    name_prefix=''
  )

  if ENT_COEF_DECAY:
    for i in range(OFFSET_BATCHES + 1, TOTAL_BATCHES + OFFSET_BATCHES + 1):
      model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
      model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
      model.ent_coef *= ENT_COEF_DECAY
  else:
    model.learn(total_timesteps=STEPS_PER_BATCH * TOTAL_BATCHES, callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)

  vec_env.close()
