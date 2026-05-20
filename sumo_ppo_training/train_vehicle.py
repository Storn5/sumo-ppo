"""
Script for training a PPO agent to control an autonomous vehicle at an unsignalized intersection in the SUMO intersection environment
"""

import os
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

# Env setup
MODEL_STEPS_LIMIT = 128
SIMULATION_STEPS_LIMIT = 5 * MODEL_STEPS_LIMIT # Steps per simulation (how many tenths of a second), but we only make an action every 5 steps (every 0.5 seconds)
MAX_TRAFFIC = 1 # Max traffic for curriculum learning
COLLISION_COEF = 100.0
TIMEOUT_COEF = 100.0
SPEED_COEF = 5.0
SUCCESS_COEF = 50.0
PROXIMITY_COEF = 5.0

# Hyperparameters setup
LEARNING_RATE = 0.0003
ENTROPY_COEF = 0.1
GAMMA = 0.99
LAMBDA = 0.95

ENT_COEF_DECAY = 0.0
CLIP_RANGE_DECAY = False
LR_DECAY = False
USE_LEVY = False

EPISODES_PER_MINIBATCH = 4 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 100 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 16 # How many steps in each "minibatch" that PPO performs

model_name = f'vehicle{MAX_TRAFFIC}_ppo_lr{LEARNING_RATE}_ec{ENTROPY_COEF}_g{GAMMA}_l{LAMBDA}'.replace('.', '_')
if ENT_COEF_DECAY:
  model_name = model_name.replace('ppo', f'ppo_ecdecay_{ENT_COEF_DECAY}'.replace('.', '_'))
if CLIP_RANGE_DECAY:
  model_name = model_name.replace('ppo', f'ppo_crdecay'.replace('.', '_'))
if LR_DECAY:
  model_name = model_name.replace('ppo', f'ppo_lrdecay'.replace('.', '_'))
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
  })

  logger = configure(logs_dir, ['csv'])

  model = None
  if USE_LEVY:
    model = LevyPPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), gamma=GAMMA, gae_lambda=LAMBDA)
  else:
    model = PPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), gamma=GAMMA, gae_lambda=LAMBDA)
  model.set_logger(logger)


  # Handle saving every batch
  checkpoint_callback = CheckpointCallback(
    save_freq=STEPS_PER_BATCH, 
    save_path=models_dir,
    name_prefix=''
  )

  # Curriculum
  for (curriculum_i, max_traffic) in enumerate([1, 3, 10]):
    vec_env.close()
    if not os.path.exists(f'{logs_dir}/traffic{max_traffic}'):
      os.makedirs(f'{logs_dir}/traffic{max_traffic}')
    vec_env = make_vec_env('Vehicle-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
      'steps_limit': SIMULATION_STEPS_LIMIT,
      'max_traffic': max_traffic,
      'collision_coef': COLLISION_COEF,
      'timeout_coef': TIMEOUT_COEF,
      'speed_coef': SPEED_COEF,
      'success_coef': SUCCESS_COEF,
      'proximity_coef': PROXIMITY_COEF,
      'render_mode': None,
      'sumo_config_file': 'sumo_env/sumo_files/intersection_vehicle.sumocfg',
    }, monitor_dir=f'{logs_dir}/traffic{max_traffic}', monitor_kwargs={
      'info_keywords': ('episode_mean_speed', 'success')
    })
    model.set_env(vec_env)

    if ENT_COEF_DECAY:
      for i in range(1 + (curriculum_i * TOTAL_BATCHES), TOTAL_BATCHES + 1 + (curriculum_i * TOTAL_BATCHES)):
        model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
        model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
        if ENT_COEF_DECAY:
          model.ent_coef *= ENT_COEF_DECAY
    else:
      model.learn(total_timesteps=STEPS_PER_BATCH * TOTAL_BATCHES, callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)

  vec_env.close()
