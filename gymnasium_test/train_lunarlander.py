import numpy as np
import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# def make_env(env_id: str, rank: int, render_mode, seed: int = 0):
#     def _init():
#         env = gym.make(env_id, render_mode=render_mode)
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

models_dir = 'models/ppo'
logs_dir = 'logs'

if __name__ == '__main__':
    for folder in (models_dir, logs_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)

    test_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    print('Checking env')
    check_env(env=test_env)
    test_env.close()

    num_cpu = 8
    #envs = [make_env('LunarLander-v3', i, 'rgb_array') for i in range(num_cpu)]
    #vec_env = SubprocVecEnv(envs)
    # DummyVecEnv should be faster, because it creates only 1 process w/ multiple envs
    vec_env = make_vec_env('LunarLander-v3', n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, monitor_dir=logs_dir)

    logger = configure(logs_dir, ['csv'])

    # Iteration steps = num_cpu * n_steps = 10000
    # num_cpu * n_steps (real batch size) has to be divisible by batch_size (minibatch size)
    model_ppo = PPO('MlpPolicy', vec_env, verbose=0, tensorboard_log=logs_dir, n_steps=1250, batch_size=50)
    model_ppo.set_logger(logger)

    for i in range(1, 31):
        model_ppo.learn(total_timesteps=10_000, reset_num_timesteps=False, tb_log_name='PPO', progress_bar=True)
        model_ppo.save(f'{models_dir}/{10_000 * i}')
    vec_env.close()
