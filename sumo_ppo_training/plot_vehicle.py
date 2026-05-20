import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy

logs_dir = 'logs'

def plot_learning_curves(log_folder, title='Learning Curves'):
  metrics = [
    ('rollout/ep_rew_mean', 'Episode Reward'),
    ('train/loss', 'Loss'),
    ('train/approx_kl', 'KL Divergence'),
    ('train/entropy_loss', 'Entropy Loss'),
    ('train/policy_gradient_loss', 'Policy Gradient Loss'),
    ('train/explained_variance', 'Explained Variance'),
  ]

  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True)
  axes = axes.flatten()
  fig.suptitle(title, fontsize=16)

  for path in os.listdir(logs_dir):
    if not os.path.isfile(os.path.join(logs_dir, path)) and path.startswith('vehicle'):
      progress = pd.read_csv(f'{os.path.join(logs_dir, path)}/progress.csv')
      x = progress['time/total_timesteps'] // 1000

      for i, (column, label) in enumerate(metrics):
        ax = axes[i]
        line, = ax.plot(x, progress[column], label=label)
        line.set_label(path)
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_ylabel(label)
        ax.legend()

        # Only the bottom plots need the X-label if sharing X-axis
        if i >= 3:
          ax.set_xlabel('Sim Steps, thousands')

  plt.show()

def plot_episode_metrics(log_folder, title='Episode Metrics'):
  metrics = [
    ('episode_mean_speed', 'Average Speed, m/s'),
    ('success', 'Success Rate'),
  ]

  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
  axes = axes.flatten()
  fig.suptitle(title, fontsize=16)

  for path in os.listdir(logs_dir):
    if not os.path.isfile(os.path.join(logs_dir, path)) and path.startswith('vehicle'):
      scenario_monitors = []
      for index, max_traffic in enumerate([1, 3, 10]):
        monitors = []
        for i in range(8):
          monitors.append(pd.read_csv(f'{os.path.join(logs_dir, path)}/traffic{max_traffic}/{i}.monitor.csv', skiprows=1))

        monitor = pd.concat(monitors)
        monitor = monitor.groupby(monitor.index).mean()
        monitor.index += len(monitor) * index
        scenario_monitors.append(monitor)
      monitor = pd.concat(scenario_monitors)
      monitor = monitor.rolling(10, center=True).mean()
      x = (monitor.index + 1) * 2400 // 2000

      for i, (column, label) in enumerate(metrics):
        ax = axes[i]
        line, = ax.plot(x, monitor[column], label=label)
        line.set_label(path)
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_ylabel(label)
        ax.legend()

        # Only the bottom plots need the X-label if sharing X-axis
        if i >= 1:
          ax.set_xlabel('Sim Steps, thousands')

  plt.show()

plot_learning_curves(logs_dir)
plot_episode_metrics(logs_dir)
