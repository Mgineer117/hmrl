import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        linear_layer = nn.Linear(latent_dim, 1).to(device)
        nn.init.uniform_(linear_layer.weight, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
        nn.init.uniform_(linear_layer.bias, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
        self.last = linear_layer

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values
    
class DistCritic(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 logstd_range: tuple = (-2., 5.),
                 device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")

        self.mu = nn.Linear(latent_dim, 1).to(device)
        nn.init.uniform_(self.mu.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.mu.bias, a=-1e-3, b=1e-3)
        
        sigma_layer = nn.Linear(latent_dim, 1).to(device)
        nn.init.uniform_(sigma_layer.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(sigma_layer.bias, a=-1e-3, b=1e-3)

        sigma_modlue = [sigma_layer, nn.Dropout(p=0.25)]
        self.sigma = nn.Sequential(*sigma_modlue)

        self._eps = 1e-6 # for numerical stability
        self._logstd_min, self._logstd_max = logstd_range

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)

        mu = self.mu(logits)
        logstd = self.sigma(logits) #torch.exp()
        logstd = torch.clamp(logstd, self._logstd_min , self._logstd_max)
        std = torch.exp(logstd)

        #distrib = torch.distributions.Normal(mu + self._eps, std + self._eps)
        #values = distrib.rsample()

        return mu, std