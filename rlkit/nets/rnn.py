import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from rlkit.nets.mlp import MLP

def identity(x):
    return x

class RecurrentEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size:int,
            device="cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn_hidden_dim = hidden_size

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.input_size, self.rnn_hidden_dim, num_layers=1, batch_first=True).to(device)

        self.encoder_type = 'recurrent'
        self.device = torch.device(device)

    def init_hidden_info(self):
        self.hn = torch.zeros(1, 1, self.rnn_hidden_dim).to(self.device)
        self.cn = torch.zeros(1, 1, self.rnn_hidden_dim).to(self.device)

    def forward(self, mdp_and_lengths, is_batch=False):
        # prepare for batch update
        mdp, lengths = mdp_and_lengths
        mdp = torch.as_tensor(mdp, device=self.device, dtype=torch.float32)
        trj, seq, fea = mdp.shape
        
        if is_batch:
            # pass into LSTM with allowing automatic initialization for each trajectory
            out, _ = self.lstm(mdp)
            output = torch.zeros((sum(lengths), fea)).to(self.device)
            last_length = 0
            for i, length in enumerate(lengths):
                output[last_length:last_length+length, :] = out[i, :length, :]
                last_length += length
            out = output
        else:
            # pass into LSTM
            out, (hn, cn) = self.lstm(mdp, (self.hn, self.cn))
            self.hn = hn # update LSTM
            self.cn = cn # update LSTM

        return out.squeeze()
    
class RecurrentDecoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size:int,
            device="cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn_hidden_dim = hidden_size

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.input_size, self.rnn_hidden_dim, num_layers=1, batch_first=True).to(device)

        self.encoder_type = 'recurrent'
        self.device = torch.device(device)

    def forward(self, input_tuple, z):
        # prepare for batch update
        input, lengths = input_tuple
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        trj, seq, fea = input.shape
        
        # reset the LSTM
        self.hn = torch.zeros(1, trj, self.rnn_hidden_dim).to(self.device)
        self.cn = torch.zeros(1, trj, self.rnn_hidden_dim).to(self.device)
        
        # pass into LSTM with allowing automatic initialization for each trajectory
        # The output is padded
        out, _ = self.lstm(input, (self.hn, self.cn))
        
        # Convert to the non-padded 2-D arrays
        output = torch.zeros((trj, fea)).to(self.device) # trj, fea since we want to keep the last dim of the output
        for i in range(trj):
            # shape : trj, 1, fea
            output[i, :] = out[i, -1, :] 
        
        out = output

        return out.squeeze()