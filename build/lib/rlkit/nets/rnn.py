import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from rlkit.nets.mlp import MLP


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def identity(x):
    return x

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_initialization = False,
        hidden_dims=[200, 200, 200, 200],
        rnn_num_layers=3,
        dropout_rate=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = torch.device(device)

        self.activation = Swish()
        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=rnn_num_layers,
            batch_first=True
        )
        module_list = []
        self.input_layer = ResBlock(input_dim, hidden_dims[0], dropout=dropout_rate, with_residual=False)
        dims = list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)
        self.merge_layer = nn.Linear(dims[0] + dims[-1], hidden_dims[0])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.to(self.device)

    def forward(self, input, h_state=None):
        batch_size, num_timesteps, _ = input.shape
        input = torch.as_tensor(input, dtype=torch.float32).to(self.device)
        rnn_output, h_state = self.rnn_layer(input, h_state)
        rnn_output = rnn_output.reshape(-1, self.hidden_dims[0])
        input = input.view(-1, self.input_dim)
        output = self.input_layer(input)
        output = torch.cat([output, rnn_output], dim=-1)
        output = self.activation(self.merge_layer(output))
        for layer in self.backbones:
            output = layer(output)
        output = self.output_layer(output)
        output = output.view(batch_size, num_timesteps, -1)
        return output, h_state

class RecurrentEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size:int,
            output_size: int,
            obs_dim: int,
            action_dim: int,
            masking_dim: int,
            rnn_initialization: bool = True,
            output_activation=identity,
            device="cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn_hidden_dim = hidden_size
        self.embed_dim = output_size
        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.masking_dim=masking_dim

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.input_size, self.rnn_hidden_dim, num_layers=1, batch_first=True).to(device)
        
        self.last_layer = nn.Linear(self.rnn_hidden_dim, output_size).to(device)
        if rnn_initialization:
            nn.init.uniform_(self.last_layer.weight, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
            nn.init.uniform_(self.last_layer.bias, a=-3e-3, b=3e-3)  # Set weights using uniform initialization

        self.output_activation = output_activation

        self.state_decoder = MLP(
            input_dim=self.embed_dim + self.obs_dim - self.masking_dim + self.action_dim,
            hidden_dims=(64, 64, 32),
            output_dim=self.obs_dim,
        )

        self.reward_decoder = MLP(
            input_dim=self.embed_dim + self.obs_dim - self.masking_dim + self.action_dim + self.obs_dim,
            hidden_dims=(64, 64, 32),
            output_dim=self.embed_dim,
        )

        self.loss_fn = torch.nn.MSELoss()
        
        self.encoder_type = 'recurrent'
        self.device = torch.device(device)

    def forward(self, input, do_reset, is_batch=False):
        # prepare for batch update
        if is_batch:
            input, lengths = self.pack4rnn(input)
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        trj, seq, fea = input.shape
        # reset the LSTM
        if do_reset:
            self.hn = torch.zeros(1, trj, self.rnn_hidden_dim).to(self.device)
            self.cn = torch.zeros(1, trj, self.rnn_hidden_dim).to(self.device)
        
        if is_batch:
            # pass into LSTM with allowing automatic initialization for each trajectory
            out, (hn, cn) = self.lstm(input, (self.hn, self.cn))
            output = torch.zeros((sum(lengths), fea)).to(self.device)
            last_length = 0
            for i, length in enumerate(lengths):
                output[last_length:last_length+length, :] = out[i, :length, :]
                last_length += length
            out = output
        else:
            # pass into LSTM
            out, (hn, cn) = self.lstm(input, (self.hn, self.cn))
            self.hn = hn # update LSTM
            self.cn = cn # update LSTM
            out = torch.squeeze(out) # to match the dimension

        # output layer for Tanh activation
        out = self.last_layer(out)
        out = self.output_activation(out)

        embedding = out
        return embedding.squeeze()
    
    def pack4rnn(self, tuple):
        obss, actions, next_obss, rewards, masks = tuple
        trajs = []
        lengths = []
        prev_i = 0
        for i, mask in enumerate(masks):
            if mask == 0:
                trajs.append(torch.concatenate((obss[prev_i:i+1, :], actions[prev_i:i+1, :], next_obss[prev_i:i+1, :], rewards[prev_i:i+1, :]), axis=-1))
                lengths.append(i+1 - prev_i)
                prev_i = i + 1    
        
        # pad the data
        largest_length = max(lengths)
        mdp_dim = trajs[0].shape[-1]
        padded_data = torch.zeros((len(lengths), largest_length, mdp_dim))

        for i, traj in enumerate(trajs):
            padded_data[i, :lengths[i], :] = traj
        
        return padded_data, lengths
    
    def decode(self, mdp_tuple, embedding, maksed_obs):
        _, actions, next_obs, rewards, _ = mdp_tuple

        state_decoder_input = torch.concatenate((embedding, maksed_obs, actions), axis=-1)
        reward_decoder_input = torch.concatenate((embedding, maksed_obs, actions, next_obs), axis=-1)

        next_obs_pred = self.state_decoder(state_decoder_input)
        decomposed_rewards_pred = self.reward_decoder(reward_decoder_input)

        rewards_pred = torch.sum(decomposed_rewards_pred * embedding, axis=-1, keepdim=True)

        decoder_loss = self.loss_fn(next_obs, next_obs_pred) + self.loss_fn(rewards, rewards_pred)

        return decoder_loss
    
    
class RecurrentOfflineEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size:int,
            output_size: int,
            obs_dim: int,
            action_dim: int,
            rnn_initialization: bool = True,
            output_activation=identity,
            device="cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn_hidden_dim = hidden_size
        self.embed_dim = output_size
        self.obs_dim=obs_dim
        self.action_dim=action_dim

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.input_size, self.rnn_hidden_dim, num_layers=1, batch_first=True).to(device)
        
        self.last_layer = nn.Linear(self.rnn_hidden_dim, output_size).to(device)
        if rnn_initialization:
            nn.init.uniform_(self.last_layer.weight, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
            nn.init.uniform_(self.last_layer.bias, a=-3e-3, b=3e-3)  # Set weights using uniform initialization

        self.output_activation = output_activation

        self.state_decoder = MLP(
            input_dim=self.embed_dim + self.obs_dim + self.action_dim,
            hidden_dims=(128, 128, 64, 64),
            output_dim=self.obs_dim,
        )

        self.reward_decoder = MLP(
            input_dim=self.embed_dim + self.obs_dim + self.action_dim + self.obs_dim,
            hidden_dims=(128, 128, 64, 64),
            output_dim=1,
        )

        self.loss_fn = torch.nn.MSELoss()
        
        self.encoder_type = 'recurrent'
        self.device = torch.device(device)

        # initialize LSTM hidden state with predefined trajectory
        self.traj_num = 100
        self.hn = torch.zeros((1, self.traj_num, self.rnn_hidden_dim)).to(self.device)
        self.cn = torch.zeros((1, self.traj_num, self.rnn_hidden_dim)).to(self.device)

    def forward(self, input):
        # prepare for batch update
        input, lengths = self.pack4rnn(input)
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        trj, seq, fea = input.shape
        assert trj == self.traj_num

        # reset the LSTM cell state
        self.cn = torch.zeros(self.cn.shape).to(self.device)
        
        # pass into LSTM with allowing automatic initialization for each trajectory
        out, (hn, cn) = self.lstm(input, (self.hn, self.cn))
        self.hn = hn # update only hn

        output = torch.zeros((sum(lengths), fea)).to(self.device)
        last_length = 0
        for i, length in enumerate(lengths):
            output[last_length:last_length+length, :] = out[i, :length, :]
            last_length += length
        out = output

        # output layer for Tanh activation
        out = self.last_layer(out)
        out = self.output_activation(out)

        embedding = out
        return embedding.squeeze()

    def pack4rnn(self, tuple):
        obss, actions, next_obss, rewards, masks = tuple
        trajs = []
        lengths = []
        prev_i = 0
        for i, mask in enumerate(masks):
            if mask == 0:
                trajs.append(torch.concatenate((obss[prev_i:i+1, :], actions[prev_i:i+1, :], next_obss[prev_i:i+1, :], rewards[prev_i:i+1, :]), axis=-1))
                lengths.append(i+1 - prev_i)
                prev_i = i + 1    
        
        # pad the data
        largest_length = max(lengths)
        mdp_dim = trajs[0].shape[-1]
        padded_data = torch.zeros((len(lengths), largest_length, mdp_dim))

        for i, traj in enumerate(trajs):
            padded_data[i, :lengths[i], :] = traj

        return padded_data, lengths
    
    def decode(self, mdp_tuple, embedding):
        obs, actions, next_obs, rewards, masks = mdp_tuple
        state_decoder_input = torch.concatenate((embedding, obs, actions))
        reward_decoder_input = torch.concatenate((embedding, obs, actions, next_obs))

        next_obs_pred = self.state_decoder(state_decoder_input)
        rewards_pred = self.reward_decoder(reward_decoder_input)

        decoder_loss = self.loss_fn(next_obs, next_obs_pred) + self.loss_fn(rewards, rewards_pred)

        return decoder_loss


    
if __name__ == "__main__":
    model = RNNModel(14, 12)
    x = torch.randn(64, 20, 14)
    y, _ = model(x)
    print(y.shape)