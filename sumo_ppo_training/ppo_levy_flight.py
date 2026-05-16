import torch
import numpy as np
from stable_baselines3 import PPO
from scipy.stats import levy

class LevyPPO(PPO):
  def __init__(self, *args, beta=0.01, **kwargs):
    super().__init__(*args, **kwargs)
    self.beta = beta # The scale of the random walk

  def train(self) -> None:
    # Run the standard PPO update first
    super().train()

    # Apply Lévy Flight noise to the policy parameters
    # We do this after the standard gradient updates
    with torch.no_grad():
      for param in self.policy.parameters():
        if param.requires_grad:
          # Generate Lévy noise
          # alpha=1.5 is common for Lévy flights (heavy-tailed)
          noise = levy.rvs(loc=0, scale=self.beta, size=param.shape)
          noise_tensor = torch.as_tensor(noise, device=self.device).float()

          # Update weights: W = W + noise
          param.add_(noise_tensor)
