"""
Script for training a PPO agent to control an autonomous vehicle at an unsignalized intersection in the SUMO intersection environment
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import sumo_env
from sumo_env.vehicle_env import VehicleEnv
from ppo_levy_flight import LevyPPO

# Env setup
MODEL_STEPS_LIMIT = 128
SIMULATION_STEPS_LIMIT = 5 * MODEL_STEPS_LIMIT # Steps per simulation (how many tenths of a second), but we only make an action every 5 steps (every 0.5 seconds)
MAX_TRAFFIC = 15 # Max traffic for curriculum learning
COLLISION_COEF = 100.0
TIMEOUT_COEF = 100.0
SPEED_COEF = 0.1
SUCCESS_COEF = 50.0
PROXIMITY_COEF = 5.0

# Hyperparameters setup
LEARNING_RATE = 0.0002
ENTROPY_COEF = 0.05
GAMMA = 0.99
LAMBDA = 0.95

ENT_COEF_DECAY = 0.99
CLIP_RANGE_DECAY = 0.0
LR_DECAY = 0.0
USE_LEVY = False

EPISODES_PER_MINIBATCH = 4 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 200 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 16 # How many steps in each "minibatch" that PPO performs

model_name = f'vehicle_ppo_lr{LEARNING_RATE}_ec{ENTROPY_COEF}_g{GAMMA}_l{LAMBDA}'.replace('.', '_')
if ENT_COEF_DECAY:
  model_name = model_name.replace('ppo', f'ppo_ecdecay_{ENT_COEF_DECAY}'.replace('.', '_'))
if CLIP_RANGE_DECAY:
  model_name = model_name.replace('ppo', f'ppo_crdecay_{CLIP_RANGE_DECAY}'.replace('.', '_'))
if LR_DECAY:
  model_name = model_name.replace('ppo', f'ppo_lrdecay_{LR_DECAY}'.replace('.', '_'))
if USE_LEVY:
  model_name = model_name.replace('ppo', 'levyppo')
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

  model = PPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE, batch_size=MINIBATCH_SIZE,
    learning_rate=LEARNING_RATE, ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  if USE_LEVY:
    model = LevyPPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE, batch_size=MINIBATCH_SIZE,
      learning_rate=LEARNING_RATE, ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  model.set_logger(logger)

  for i in range(1, TOTAL_BATCHES + 1):
    model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
    if ENT_COEF_DECAY:
      model.ent_coef *= ENT_COEF_DECAY
    if CLIP_RANGE_DECAY:
      model.clip_range *= CLIP_RANGE_DECAY
    if LR_DECAY:
      model.learning_rate *= LR_DECAY

  vec_env.close()
