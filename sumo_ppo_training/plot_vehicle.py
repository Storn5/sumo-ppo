from plot_helpers import plot_learning_curves, plot_episode_metrics

traffic_scenarios = [1, 3, 10]
scenario_boundary_steps = [409_600, 819_200]
episode_steps_limit = 128

metrics = [
  ('rollout/ep_rew_mean', 'Episode Reward'),
  ('train/loss', 'Loss'),
  ('train/approx_kl', 'KL Divergence'),
  ('train/entropy_loss', 'Entropy Loss'),
  ('train/policy_gradient_loss', 'Policy Gradient Loss'),
  ('train/explained_variance', 'Explained Variance'),
]

plot_learning_curves(metrics[:2], scenario_boundary_steps, 'vehicle', 'Vehicle Learning Curves: Reward and Loss')
plot_learning_curves(metrics[2:4], scenario_boundary_steps, 'vehicle', 'Vehicle Learning Curves: KL Divergence and Ent. Loss')
plot_learning_curves(metrics[4:], scenario_boundary_steps, 'vehicle', 'Vehicle Learning Curves: PG Loss and Explained Variance')

metrics = [
  ('episode_mean_speed', 'Average Speed, m/s'),
  ('success', 'Success Rate'),
]

plot_episode_metrics(metrics, traffic_scenarios, scenario_boundary_steps, 'vehicle', episode_steps_limit, 'Vehicle Episode Metrics: Success Rate and Speed', average_envs=False)
