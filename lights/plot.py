import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy

logs_dir = 'logs/ppo_lr0_001_ec0_02_g99_l95_2026_05_11T15_34_32'

def plot_episode_stats(log_folder, title='Episode Stats'):
  progress = pd.read_csv(f'{logs_dir}/progress.csv')
  x = progress['time/total_timesteps'] // 1000

  metrics = [
    ('rollout/ep_rew_mean', 'Episode Reward', 'green'),
    ('train/loss', 'Loss', 'red'),
    ('train/approx_kl', 'KL Divergence', 'purple'),
    ('train/entropy_loss', 'Entropy Loss', 'blue'),
    ('train/policy_gradient_loss', 'Policy Gradient Loss', 'cyan'),
    ('train/explained_variance', 'Explained Variance', 'orange'),
  ]

  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True)
  axes = axes.flatten()
  fig.suptitle(title, fontsize=16)

  for i, (column, label, color) in enumerate(metrics):
    ax = axes[i]
    ax.plot(x, progress[column], label=label, color=color)
    ax.set_title(label)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylabel(label)
    
    # Only the bottom plots need the X-label if sharing X-axis
    if i >= 3:
      ax.set_xlabel('Sim Steps, thousands')

  plt.show()

plot_episode_stats(logs_dir)
