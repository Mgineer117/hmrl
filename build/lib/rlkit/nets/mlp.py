import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        mlp_initialization = True,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if mlp_initialization:
                nn.init.xavier_uniform_(linear_layer.weight)
                linear_layer.bias.data.fill_(0.01)
            model += [linear_layer, activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            nn.init.uniform_(linear_layer.weight, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
            nn.init.uniform_(linear_layer.bias, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
            model += [linear_layer]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class BaseEncoder():
    def __init__(
            self,
            device="cpu"
    ):
        self.encoder_type = 'none'
        self.device = torch.device(device)

    def __call__(self, obs, env_idx=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return obs
    
class OneHotEncoder(BaseEncoder):
    def __init__(
            self,
            embed_dim:int,
            eval_env_idx:int = 0,
            device="cpu"
    ):
        self.embed_dim = embed_dim
        self.eval_env_idx = eval_env_idx
        self.encoder_type = 'onehot'
        self.device = torch.device(device)

    def __call__(self, obs, env_idx=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if len(obs.shape) == 1:
            obs = obs[None, :]
        embedding = torch.zeros((obs.shape[0], self.embed_dim)).to(self.device)
        if env_idx is not None:
            embedding[:, env_idx] = 1
        else:
            embedding[:, self.eval_env_idx] = 1
        #embedded_obs = torch.concatenate((embedding, obs), axis=-1)
        return embedding.squeeze()
    
