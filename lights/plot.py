import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy

logs_dir = 'logs'

def moving_average(values, window):
    '''
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    '''
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve, Smoothed'):
    '''
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    '''
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.show()

plot_results(logs_dir)

def plot_episode_stats(log_folder, title='Episode Stats'):
    progress = pd.read_csv(f'{logs_dir}/progress.csv')
    x = progress['time/total_timesteps']

    fig = plt.figure(title)
    plt.plot(x, progress['rollout/ep_len_mean'], label='Episode Length')
    plt.plot(x, progress['rollout/ep_rew_mean'], label='Episode Reward')
    plt.plot(x, progress['train/loss'], label='Loss')
    plt.xlabel('Number of Timesteps')
    plt.legend()
    plt.title(title)
    plt.show()

def plot_kl_divergence(log_folder, title='KL Divergence'):
    progress = pd.read_csv(f'{logs_dir}/progress.csv')
    x = progress['time/total_timesteps']

    fig = plt.figure(title)
    plt.plot(x, progress['train/approx_kl'], label='KL Divergence')
    plt.xlabel('Number of Timesteps')
    plt.legend()
    plt.title(title)
    plt.show()

plot_episode_stats(logs_dir)
plot_kl_divergence(logs_dir)
