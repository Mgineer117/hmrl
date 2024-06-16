import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional

from copy import deepcopy

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

class TRPOActor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        latent_dim,
        output_dim,
        mean_range: tuple = (-2., 2.),
        logstd_range: tuple = (-2., 5.),
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.mu = nn.Linear(latent_dim, output_dim).to(device)
        self.sigma = nn.Linear(latent_dim, output_dim).to(device)

        nn.init.uniform_(self.mu.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.mu.bias, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.sigma.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.sigma.bias, a=-1e-3, b=1e-3)

        self._mean_min, self._mean_max = mean_range
        self._logstd_min, self._logstd_max = logstd_range
        self._eps = 1e-6

    def forward(self, obs: Union[np.ndarray, torch.Tensor]):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)

        mu = self.mu(logits)
        mu = torch.clamp(mu, self._mean_min, self._mean_max)
        
        logstd = self.sigma(logits)
        logstd = torch.clamp(logstd, self._logstd_min, self._logstd_max)
        std = torch.exp(logstd)

        dist = torch.distributions.Normal(mu, std)
        sampled_action = dist.rsample()
        logprob = dist.log_prob(sampled_action)

        return mu, sampled_action, logstd, std, logprob
    
    def get_log_prob(self, x, actions):
        action_mean, _, action_logstd, action_std, _ = self.forward(x)
        return normal_log_density(actions, action_mean, action_logstd, action_std)
    
    def get_kl(self, x):
        mean1, _, log_std1, std1, _ = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist

    def get_kl(self, x):
        dist = self.forward(x)

        mean1 = dist.mode()
        log_std1 = torch.log(dist.stddev)
        std1 = dist.stddev

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    
# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions
    
# for Optidice
class DiceActor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        latent_dim,
        output_dim,
        max_action: float = 1.,
        mean_range: tuple = (-7., 7.),
        logstd_range: tuple = (-2., 5.),
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.mu = nn.Linear(latent_dim, output_dim).to(device)
        self.sigma = nn.Linear(latent_dim, output_dim).to(device)

        nn.init.uniform_(self.mu.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.mu.bias, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.sigma.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.sigma.bias, a=-1e-3, b=1e-3)

        self._max = max_action
        self._mean_min, self._mean_max = mean_range
        self._logstd_min, self._logstd_max = logstd_range
        self._eps = 1e-6

    def forward(self, obs: Union[np.ndarray, torch.Tensor]):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)

        mu = self.mu(logits)
        mu = torch.clamp(mu, self._mean_min, self._mean_max)
        logstd = self.sigma(logits)
        logstd = torch.clamp(logstd, self._logstd_min, self._logstd_max)
        std = torch.exp(logstd)

        pretanh_action_dist = torch.distributions.MultivariateNormal(mu, torch.diag_embed(std))
        pretanh_action = pretanh_action_dist.rsample()
        action = self._max * torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist
    
    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = self._max * torch.tanh(pretanh_action)
        else:
            squashed_action = action / self._max
            pretanh_action = torch.atanh(torch.clamp(squashed_action, -1 + self._eps, 1 - self._eps))
        
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - torch.sum(torch.log(1 - action ** 2 + self._eps), axis=-1)

        return log_prob, pretanh_log_prob
    
    def deterministic_action(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        logits = self.backbone(obs)
        mu = self.mu(logits)
        mu = torch.clamp(mu, self._mean_min, self._mean_max)
        action = self._max * torch.tanh(mu)

        return action