import torch
import numpy as np
from scipy.stats import levy, levy_stable
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from gymnasium import spaces

class LevyPPO(PPO):
  def __init__(self, *args, beta_levy=0.01, alpha_levy=1.5, **kwargs):
    super().__init__(*args, **kwargs)
    self.beta_levy = beta_levy               # β coefficient from the paper formula
    self.alpha_levy = alpha_levy   # Stability parameter for Lévy distribution (usually 1.5)

  def train(self) -> None:
    """
    Custom PPO training loop that injects Lévy Flight noise into 
    the gradient step, matching the formula: 
    θ_{t+1} = θ_t + α·∇L_CLIP(θ) + β·L(s; λ)
    """
    # Set model to training mode
    self.policy.set_training_mode(True)
    self.policy_class = self.policy.__class__

    # Update learning rate schedules
    self._update_learning_rate(self.policy.optimizer)

    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    # Re-initialize PPO data storage variables
    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    continue_training = True
    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      # Do a complete pass on the rollout buffer
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
          actions = rollout_data.actions.long().flatten()

        # Evaluate actions
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        # Normalize advantages
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate Policy Gradient (CLIP) Loss
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the difference between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = rollout_data.old_values + torch.clamp(
            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
          )

        # Calculate Value Loss
        value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -torch.mean(-log_prob)
        else:
          entropy_loss = -torch.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
          log_ratio = log_prob - rollout_data.old_log_prob
          approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
          break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()

        # Clip gradients if necessary
        if self.max_grad_norm is not None:
          torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

        # Inject Lévy flight noise directly into Actor Gradients
        with torch.no_grad():
          # Gather all parameters belonging strictly to the Actor (Policy) side of the network
          actor_parameters = list(self.policy.mlp_extractor.policy_net.parameters()) + \
                             list(self.policy.action_net.parameters())

          for param in actor_parameters:
            if param.requires_grad and param.grad is not None:
              # Sample from heavy-tailed Lévy distribution matching the parameter's shape
              # levy_stable.rvs arguments:
              # alpha = self.alpha_levy (1.5 determines the tail thickness)
              # beta = 0 (ensures a symmetric distribution so lights can adjust both directions)
              levy_step = levy_stable.rvs(
                alpha=self.alpha_levy, 
                beta=0, 
                loc=0, 
                scale=1.0, 
                size=param.grad.shape
              )
              levy_tensor = torch.as_tensor(levy_step, device=self.device).float()

              # Update gradient buffer: Grad = Grad - (Beta * Lévy Step)
              # (Subtracted because Adam/SGD subtracts gradients to minimize loss,
              # meaning subtracting from the gradient results in addition to the weights: W_new = W_old + noise)
              param.grad.sub_(self.beta_levy * levy_tensor)

        # Step the optimizer with the modified gradients
        self.policy.optimizer.step()

      self._n_updates += 1
      if not continue_training:
          break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)
