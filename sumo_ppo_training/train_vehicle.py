"""
Script for training a PPO agent to control an autonomous vehicle at an unsignalized intersection in the SUMO intersection environment
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import sumo_env
from sumo_env.vehicle_env import VehicleEnv

# Env setup
SIMULATION_STEPS_LIMIT = 500 # Steps per simulation (how many tenths of a second), but we only make an action every 5 steps (every 0.5 seconds)
MODEL_STEPS_LIMIT = SIMULATION_STEPS_LIMIT // 5
COLLISION_COEF = 10.0
TIMEOUT_COEF = 10.0
SPEED_COEF = 0.1
SUCCESS_COEF = 10.0

# Hyperparameters setup
LEARNING_RATE = 0.0005
ENTROPY_COEF = 0.01
GAMMA = 0.99
LAMBDA = 0.95

EPISODES_PER_MINIBATCH = 5 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 50 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 32 # How many steps in each "minibatch" that PPO performs

model_name = 'vehicle_ppo_lr0_0005_ec0_01_g99_l95'
model_name = f'{model_name}_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}'
models_dir = f'models/{model_name}'
logs_dir = f'logs/{model_name}'

if __name__ == '__main__':
  for folder in (models_dir, logs_dir):
    if not os.path.exists(folder):
      os.makedirs(folder)

  # DummyVecEnv could be faster, because it creates only 1 process w/ multiple envs
  vec_env = make_vec_env('Vehicle-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'collision_coef': COLLISION_COEF,
    'timeout_coef': TIMEOUT_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'render_mode': None,
  }, monitor_dir=logs_dir, monitor_kwargs={
    'info_keywords': ('total_departed', 'episode_mean_waiting_time', 'episode_mean_queue_length', 'episode_mean_speed')
  })

  logger = configure(logs_dir, ['csv'])

  model = PPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE, batch_size=MINIBATCH_SIZE,
    learning_rate=LEARNING_RATE, ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  model.set_logger(logger)

  for i in range(1, TOTAL_BATCHES + 1):
    model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
  vec_env.close()
