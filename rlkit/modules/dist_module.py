import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalWrapper(torch.distributions.MultivariateNormal):
    '''returning with 2d-array'''
    def log_prob(self, actions):
        return super().log_prob(actions).unsqueeze(-1)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

    def mode(self):
        return self.mean#.unsqueeze(-1)

    def std(self):
        return self.stddev.unsqueeze(-1)
        
    def logstd(self):
        return torch.log(self.stddev).unsqueeze(-1)

class TanhNormalWrapper(torch.distributions.Normal):
    def __init__(self, loc, scale, max_action):
        super().__init__(loc, scale)
        self._max_action = max_action
        self._eps = 1e-10

    def log_prob(self, action, raw_action=None):
        squashed_action = action/self._max_action
        if raw_action is None:
            raw_action = self.arctanh(squashed_action)
        
        pretanh_log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        log_prob = pretanh_log_prob - torch.log(1 - action**2 + self._eps).sum(-1, keepdim=True)
        return log_prob, pretanh_log_prob

    def mode(self):
        raw_action = self.mean
        action = self._max_action * torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = self._max_action * torch.tanh(raw_action)
        return action, raw_action

class TanhMixtureNormalWrapper(torch.distributions.multivariate_normal.MultivariateNormal):
    def __init__(self, loc, scale, component_logits, max_action, n_components):
        super().__init__(loc, torch.diag_embed(scale))
        self.component_dist = torch.distributions.Categorical(logits=component_logits)
        self._max = max_action
        self._n_components = n_components

    def log_prob(self, action, raw_action=None):
        squashed_action = action/self._max
        if raw_action is None:
            raw_action = self.arctanh(squashed_action)
        
        component_logits = self.component_dist.logits
        component_log_prob = component_logits - torch.logsumexp(component_logits, dim=-1, keepdim=True)

        raw_actions = raw_action[:, None, :].expand(-1, self._n_components, -1)  # (batch_size, num_components, action_dim)

        pretanh_log_prob = torch.logsumexp(component_log_prob + super().log_prob(raw_actions), dim=1)[:, None]
        log_prob = pretanh_log_prob - torch.sum(torch.log(1 - action ** 2 + 1e-3), dim=-1)[:, None]

        return log_prob, pretanh_log_prob

    def mode(self):
        raw_action = self.mean
        action = self._max * torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_actions = super().rsample()
        component = self.component_dist.sample()
        raw_action = raw_actions[torch.arange(raw_actions.size(0)), component]
        action = self._max * torch.tanh(raw_action)
        
        return action, raw_action
    
class DiagGaussian(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__()

        mu_layer = nn.Linear(latent_dim, output_dim)
        nn.init.uniform_(mu_layer.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(mu_layer.bias, a=-1e-3, b=1e-3)
        self.mu = mu_layer
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            sigma_layer = nn.Linear(latent_dim, output_dim)
            nn.init.uniform_(sigma_layer.weight, a=-1e-3, b=1e-3)
            nn.init.uniform_(sigma_layer.bias, a=-1e-3, b=1e-3)
            self.sigma = sigma_layer
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        cov_mat = torch.diag_embed(sigma)
        return NormalWrapper(mu, cov_mat)


class TanhDiagGaussian(DiagGaussian):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            max_mu=max_mu,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = torch.clamp(mu, min=-self._max, max=self._max)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max)#logstd
            sigma = torch.exp(sigma)
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma, self._max)
    


class TanhMixDiagGaussian(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_sizes, num_components=2,
                 max_mu=1.0, logstd_range=(-5., 2.), eps=1e-6, mdn_temperature=1.0):
        super(TanhMixDiagGaussian, self).__init__()
        
        self.input_size = obs_shape[0]
        self.action_dim = action_dim
        self.num_components = num_components
        self._max = max_mu
        self.mdn_temperature = mdn_temperature
        
        self.fc_layers = nn.ModuleList()
        prev_size = self.input_size
        for size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, size))
            prev_size = size
            
        self.fc_means = nn.Linear(hidden_sizes[-1], num_components * action_dim)
        self.fc_logstds = nn.Linear(hidden_sizes[-1], num_components * action_dim)
        self.fc_logits = nn.Linear(hidden_sizes[-1], num_components)
        
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, logits):
        means = self.fc_means(logits)
        means = torch.clamp(means, -self._max, self._max)
        means = means.view(-1, self.num_components, self.action_dim)
        
        logstds = self.fc_logstds(logits)
        logstds = torch.clamp(logstds, self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.action_dim)
        stds = torch.exp(logstds)
        
        component_logits = self.fc_logits(logits) / self.mdn_temperature

        return TanhMixtureNormalWrapper(means, stds, component_logits, self._max, self.num_components)