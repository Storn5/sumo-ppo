"""
Script for training a PPO agent to control traffic lights in the SUMO intersection environment
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

import sumo_env
from sumo_env.lights_env import LightsEnv
from ppo_levy_flight import LevyPPO
from ppo_decay_schedule import linear_schedule

# Env setup
MODEL_STEPS_LIMIT = 150
SIMULATION_STEPS_LIMIT = MODEL_STEPS_LIMIT * 60 # Steps per simulation (how many tenths of a second), but we only make an action every 60 steps (every 6 seconds)
MAX_TRAFFIC = 5 # Max traffic for curriculum learning
AWT_COEF = 0.05
AQL_COEF = 0.05
SPEED_COEF = 0.15
SUCCESS_COEF = 0.005

# Hyperparameters setup
LEARNING_RATE = 0.0005
ENTROPY_COEF = 0.01
GAMMA = 0.99
LAMBDA = 0.95

ENT_COEF_DECAY = 0.0
CLIP_RANGE_DECAY = False
LR_DECAY = False
USE_LEVY = False
BETA_LEVY = 0.005
ALPHA_LEVY = 1.5

EPISODES_PER_MINIBATCH = 2 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 50 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 32 # How many steps in each "minibatch" that PPO performs

model_name = f'lights{MAX_TRAFFIC}_ppo_lr{LEARNING_RATE}_ec{ENTROPY_COEF}_g{GAMMA}_l{LAMBDA}'.replace('.', '_')
if ENT_COEF_DECAY:
  model_name = model_name.replace('ppo', f'ppo_ecdecay_{ENT_COEF_DECAY}'.replace('.', '_'))
if CLIP_RANGE_DECAY:
  model_name = model_name.replace('ppo', 'ppo_crdecay')
if LR_DECAY:
  model_name = model_name.replace('ppo', 'ppo_lrdecay')
if USE_LEVY:
  model_name = model_name.replace('ppo', f'levyppo_beta{BETA_LEVY}_alpha{ALPHA_LEVY}'.replace('.', '_'))
model_name = f'{model_name}_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}'
models_dir = f'models/{model_name}'
logs_dir = f'logs/{model_name}'

if __name__ == '__main__':
  for folder in (models_dir, logs_dir):
    if not os.path.exists(folder):
      os.makedirs(folder)

  vec_env = make_vec_env('Lights-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'max_traffic': MAX_TRAFFIC,
    'awt_coef': AWT_COEF,
    'aql_coef': AQL_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'sumo_config_file': 'sumo_env/sumo_files/intersection.sumocfg',
    'render_mode': None,
  })

  logger = configure(logs_dir, ['csv'])

  model = None
  if USE_LEVY:
    model = LevyPPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), gamma=GAMMA, gae_lambda=LAMBDA,
      beta_levy=BETA_LEVY, alpha_levy=ALPHA_LEVY)
  else:
    model = PPO('MlpPolicy', vec_env, device='cpu', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE,
      batch_size=MINIBATCH_SIZE, learning_rate=(linear_schedule(LEARNING_RATE) if LR_DECAY else LEARNING_RATE),
      clip_range=(linear_schedule(0.2) if CLIP_RANGE_DECAY else 0.2), ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  model.set_logger(logger)

  # Handle saving every batch
  checkpoint_callback = CheckpointCallback(
    save_freq=STEPS_PER_BATCH, 
    save_path=models_dir,
    name_prefix=''
  )

  # Curriculum
  for (curriculum_i, max_traffic) in enumerate([5, 20, 50]):
    vec_env.close()
    if not os.path.exists(f'{logs_dir}/traffic{max_traffic}'):
      os.makedirs(f'{logs_dir}/traffic{max_traffic}')
    vec_env = make_vec_env('Lights-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
      'steps_limit': SIMULATION_STEPS_LIMIT,
      'max_traffic': max_traffic,
      'awt_coef': AWT_COEF,
      'aql_coef': AQL_COEF,
      'speed_coef': SPEED_COEF,
      'success_coef': SUCCESS_COEF,
      'sumo_config_file': 'sumo_env/sumo_files/intersection.sumocfg',
      'render_mode': None,
    }, monitor_dir=f'{logs_dir}/traffic{max_traffic}', monitor_kwargs={
      'info_keywords': ('total_arrived', 'episode_mean_waiting_time', 'episode_mean_queue_length', 'episode_mean_speed')
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
