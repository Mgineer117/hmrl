import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=(256, 256), output_activation_fn=None, output_dim=None):
        super(ValueNetwork, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim

        self.fc_layers = nn.ModuleList()
        last_dim = input_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size
        self.last_layer = nn.Linear(last_dim, output_dim or 1)

    def forward(self, inputs):
        h = torch.cat(inputs, dim=-1)
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        h = self.last_layer(h)

        if self.output_dim is None:
            h = h.view(-1)

        return h

class TanhNormalPolicy(nn.Module):
    def __init__(self, input_size, action_dim, hidden_sizes, mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=1e-6):
        super(TanhNormalPolicy, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim

        self.fc_layers = nn.ModuleList()
        last_dim = input_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size
        self.fc_mean = nn.Linear(last_dim, action_dim)
        self.fc_logstd = nn.Linear(last_dim, action_dim)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs):
        h = torch.cat(inputs, dim=-1)
        for layer in self.fc_layers:
            h = F.tanh(layer(h))

        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)
        pretanh_action_dist = td.Normal(mean, std)
        pretanh_action = pretanh_action_dist.sample()
        action = torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = torch.atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        h = torch.cat(inputs, dim=-1)
        for layer in self.fc_layers:
            h = F.tanh(layer(h))

        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        action = torch.tanh(mean)

        return action

class TanhMixtureNormalPolicy(nn.Module):
    def __init__(self, input_size, action_dim, hidden_sizes, num_components=2, mean_range=(-9., 9.), logstd_range=(-5., 2.), eps=1e-6, mdn_temperature=1.0):
        super(TanhMixtureNormalPolicy, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.num_components = num_components
        self.mdn_temperature = mdn_temperature

        self.fc_layers = nn.ModuleList()
        last_dim = input_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size
        self.fc_means = nn.Linear(last_dim, num_components * action_dim)
        self.fc_logstds = nn.Linear(last_dim, num_components * action_dim)
        self.fc_logits = nn.Linear(last_dim, num_components)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs):
        h = torch.cat(inputs, dim=-1)
        for layer in self.fc_layers:
            h = F.relu(layer(h))

        means = self.fc_means(h)
        means = torch.clamp(means, self.mean_min, self.mean_max)
        means = means.view(-1, self.num_components, self.action_dim)
        logstds = self.fc_logstds(h)
        logstds = torch.clamp(logstds, self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.action_dim)
        stds = torch.exp(logstds)

        component_logits = self.fc_logits(h) / self.mdn_temperature

        pretanh_actions_dist = td.Normal(means, stds)
        component_dist = td.Categorical(logits=component_logits)

        pretanh_actions = pretanh_actions_dist.sample()  # (batch_size, num_components, action_dim)
        component = component_dist.sample()  # (batch_size)

        batch_idx = torch.arange(inputs[0].shape[0])
        pretanh_action = pretanh_actions[batch_idx, component]
        action = torch.tanh(pretanh_action)

        log_prob, pretanh_log_prob = self.log_prob((component_dist, pretanh_actions_dist), pretanh_action, is_pretanh_action=True)

        return action, pretanh_action, log_prob, pretanh_log_prob, (component_dist, pretanh_actions_dist)

    def log_prob(self, dists, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = torch.atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        component_dist, pretanh_actions_dist = dists
        component_logits = component_dist.logits / self.mdn_temperature
        component_log_prob = component_logits - torch.logsumexp(component_logits, dim=-1, keepdim=True)

        pretanh_actions = pretanh_action.unsqueeze(1).repeat(1, self.num_components, 1)

        pretanh_log_prob = torch.logsumexp(component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions), dim=1)
        log_prob = pretanh_log_prob - torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        return log_prob, pretanh_log_prob
