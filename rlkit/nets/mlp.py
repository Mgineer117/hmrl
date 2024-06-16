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
        initialization = True,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
        device = torch.device('cpu')
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization:
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
        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)