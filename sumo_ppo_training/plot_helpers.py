import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_label_from_path(path):
  # Remove env name and datetime from path
  path = path.split('_')[1:-5]
  label = 'PPO'

  if 'levy' in path[0]:
    label += ' with Levy Flight'
    if 'beta0' in path:
      beta_index = list('beta0' in x for x in path).index(True)
      label += f', Beta=0.{path[beta_index + 1]}'
    if 'alpha1' in path:
      alpha_index = list('alpha1' in x for x in path).index(True)
      label += f', Alpha=0.{path[alpha_index + 1]}'
  if 'lrdecay' in path:
    label += ', LR decay'
  if 'crdecay' in path:
    label += ', Clip decay'
  if 'ecdecay' in path:
    label += ', Ent. decay'

  lr_index = list('lr0' in x for x in path).index(True)
  label += f', LR=0.{path[lr_index + 1]}'
  ec_index = list('ec0' in x for x in path).index(True)
  label += f', Ent.=0.{path[ec_index + 1]}'
  g_index = list('g0' in x for x in path).index(True)
  label += f', Gamma=0.{path[g_index + 1]}'
  l_index = list('l0' in x for x in path).index(True)
  label += f', Lambda=0.{path[l_index + 1]}'
  label += ', Clip=0.2'

  return label

def plot_learning_curves(metrics, scenario_boundary_steps, pathstart, title='Learning Curves', logs_dir='logs'):
  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
  axes = axes.flatten()
  fig.suptitle(title, fontsize=16)

  # Iterate over every algorithm
  for path in os.listdir(logs_dir):
    if not os.path.isfile(os.path.join(logs_dir, path)) and path.startswith(pathstart):
      progress = pd.read_csv(f'{os.path.join(logs_dir, path)}/progress.csv')
      x = progress['time/total_timesteps'] // 1000

      for i, (column, label) in enumerate(metrics):
        ax = axes[i]
        line, = ax.plot(x, progress[column], label=label)
        line.set_label(extract_label_from_path(path))
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_ylabel(label)
        ax.legend()

        for boundary in scenario_boundary_steps:
          ax.axvline(boundary // 1000)

        # Only the bottom plots need the X-label if sharing X-axis
        if i >= 1:
          ax.set_xlabel('Sim Steps, thousands')

  plt.show()

def plot_episode_metrics(metrics, traffic_scenarios, scenario_boundary_steps, pathstart, episode_steps_limit, title='Episode Metrics', logs_dir='logs', average_envs=True):
  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
  axes = axes.flatten()
  fig.suptitle(title, fontsize=16)

  # Iterate over every algorithm
  for path in os.listdir(logs_dir):
    if not os.path.isfile(os.path.join(logs_dir, path)) and path.startswith(pathstart):
      scenario_monitors = []
      # Iterate over every scenario and concatenate them
      for index, max_traffic in enumerate(traffic_scenarios):
        monitors = []
        if average_envs:
          for i in range(8):
            monitors.append(pd.read_csv(f'{os.path.join(logs_dir, path)}/traffic{max_traffic}/{i}.monitor.csv', skiprows=1))
        else:
          monitors = [pd.read_csv(f'{os.path.join(logs_dir, path)}/traffic{max_traffic}/0.monitor.csv', skiprows=1),]

        monitor = pd.concat(monitors)
        if average_envs:
          monitor = monitor.groupby(monitor.index).mean()
        monitor.index += scenario_monitors[index-1].index[-1] + 1 if index > 0 else 0
        scenario_monitors.append(monitor)
      monitor = pd.concat(scenario_monitors)
      #monitor = monitor.rolling(10, center=True).mean()
      x = (monitor.index + 1) * 8 * episode_steps_limit // 1000
      if not average_envs:
        monitor['l_cumsum'] = monitor['l'].astype(np.float32).cumsum()
        print(monitor)
        x = monitor['l_cumsum'] * 8 // 1000
        monitor = monitor.rolling(100, center=True).mean()
      else:
        monitor = monitor.rolling(5, center=True).mean()

      for i, (column, label) in enumerate(metrics):
        ax = axes[i]
        line, = ax.plot(x, monitor[column], label=label)
        line.set_label(extract_label_from_path(path))
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_ylabel(label)
        ax.legend()

        for boundary in scenario_boundary_steps:
          ax.axvline(boundary // 1000)

        # Only the bottom plots need the X-label if sharing X-axis
        if i >= 1:
          ax.set_xlabel('Sim Steps, thousands')

  plt.show()
