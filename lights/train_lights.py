import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import sumo_env
from sumo_env.lights_env import LightsEnv

# Env setup
SIMULATION_STEPS_LIMIT = 9_000 # Steps per simulation (how many tenths of a second), but we only make an action every 60 steps (every 6 seconds)
MODEL_STEPS_LIMIT = SIMULATION_STEPS_LIMIT // 60
AWT_COEF = 0
AQL_COEF = 0
SPEED_COEF = 0.25
SUCCESS_COEF = 0

# Hyperparameters setup
LEARNING_RATE = 0.001
ENTROPY_COEF = 0.04
GAMMA = 0.99
LAMBDA = 0.95

EPISODES_PER_MINIBATCH = 2 # How many episodes until each update
NUM_CPUS = 8 # How many envs are trained in parallel
TOTAL_BATCHES = 50 # How many updates to run in total for training

STEPS_PER_UPDATE = MODEL_STEPS_LIMIT * EPISODES_PER_MINIBATCH
STEPS_PER_BATCH = NUM_CPUS * STEPS_PER_UPDATE # Real batch size (STEPS_PER_BATCH) has to be divisible by MINIBATCH_SIZE
MINIBATCH_SIZE = STEPS_PER_BATCH // 32 # How many steps in each "minibatch" that PPO performs

model_name = 'ppo_lr0_001_ec0_04_g99_l95'
model_name = f'{model_name}_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}'
models_dir = f'models/{model_name}'
logs_dir = f'logs/{model_name}'

if __name__ == '__main__':
  for folder in (models_dir, logs_dir):
    if not os.path.exists(folder):
      os.makedirs(folder)

  # DummyVecEnv could be faster, because it creates only 1 process w/ multiple envs
  vec_env = make_vec_env('Lights-Sumo-v1', n_envs=NUM_CPUS, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={
    'steps_limit': SIMULATION_STEPS_LIMIT,
    'awt_coef': AWT_COEF,
    'aql_coef': AQL_COEF,
    'speed_coef': SPEED_COEF,
    'success_coef': SUCCESS_COEF,
    'sumo_config_file': 'sumo_env/sumo_files/intersection.sumocfg',
    'render_mode': None,
  }, monitor_dir=logs_dir, monitor_kwargs={
    'info_keywords': ('total_departed', 'episode_mean_waiting_time', 'episode_mean_queue_length', 'episode_mean_speed')
  })

  logger = configure(logs_dir, ['csv'])

  model = PPO('MlpPolicy', vec_env, device='cuda', verbose=0, tensorboard_log=logs_dir, n_steps=STEPS_PER_UPDATE, batch_size=MINIBATCH_SIZE,
    learning_rate=LEARNING_RATE, ent_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=LAMBDA)
  model.set_logger(logger)

  for i in range(1, TOTAL_BATCHES + 1):
    model.learn(total_timesteps=STEPS_PER_BATCH, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    model.save(f'{models_dir}/{STEPS_PER_BATCH * i}')
  vec_env.close()
