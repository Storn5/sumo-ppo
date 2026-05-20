from plot_helpers import plot_learning_curves, plot_episode_metrics

traffic_scenarios = [5, 20, 50]
scenario_boundary_steps = [120_000, 240_000]
episode_steps_limit = 150

metrics = [
  ('rollout/ep_rew_mean', 'Episode Reward'),
  ('train/loss', 'Loss'),
  ('train/approx_kl', 'KL Divergence'),
  ('train/entropy_loss', 'Entropy Loss'),
  ('train/policy_gradient_loss', 'Policy Gradient Loss'),
  ('train/explained_variance', 'Explained Variance'),
]

plot_learning_curves(metrics[:2], scenario_boundary_steps, 'lights', 'Traffic Light Learning Curves: Reward and Loss')
plot_learning_curves(metrics[2:4], scenario_boundary_steps, 'lights', 'Traffic Light Learning Curves: KL Divergence and Ent. Loss')
plot_learning_curves(metrics[4:], scenario_boundary_steps, 'lights', 'Traffic Light Learning Curves: PG Loss and Explained Variance')

metrics = [
  ('episode_mean_speed', 'Average Speed, m/s'),
  ('total_arrived', 'Total Successful Vehicles'),
  ('episode_mean_waiting_time', 'Average Waiting Time, s'),
  ('episode_mean_queue_length', 'Average Queue Length'),
]

plot_episode_metrics(metrics[:2], traffic_scenarios, scenario_boundary_steps, 'lights', episode_steps_limit, 'Traffic Light Episode Metrics: Speed and Success Rate')
plot_episode_metrics(metrics[2:], traffic_scenarios, scenario_boundary_steps, 'lights', episode_steps_limit, 'Traffic Light Episode Metrics: AWT and AQL')
