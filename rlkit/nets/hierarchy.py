import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

from rlkit.nets.rnn import RecurrentEncoder
from rlkit.nets.mlp import MLP
from rlkit.modules import ActorProb, Critic, DiagGaussian

class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim, device):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim).to(device)
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.device = device

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y.squeeze()
    
class LLmodel(nn.Module):
    def __init__(
            self, 
            actor_hidden_dim: tuple,
            critic_hidden_dim:tuple,
            state_dim: int,
            action_dim: int,
            latent_dim: int,
            masking_indices: List,
            max_action: int = 1.0,
            device = torch.device("cpu")
            ):
        super(LLmodel, self).__init__() #- len(masking_indices)
        actor_backbone = MLP(input_dim=latent_dim + state_dim, hidden_dims=actor_hidden_dim, 
                             activation=torch.nn.Tanh)
        critic_backbone = MLP(input_dim=latent_dim + state_dim, hidden_dims=critic_hidden_dim, 
                              activation=torch.nn.Tanh)
        
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=False,
            conditioned_sigma=True,
            max_mu=max_action,
            sigma_min=-3.0,
            sigma_max=2.0
        )

        actor = ActorProb(actor_backbone,
                          dist_net=dist,
                          device=device)   
                
        critic = Critic(critic_backbone, 
                        device=device)

        self.actor = actor
        self.critic = critic   

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device

    def change_device_info(self, device):
        self.actor.device = device
        self.critic.device = device
        self.device = device

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        logprob = dist.log_prob(action)
        return action, logprob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, logprob = self.actforward(obs, deterministic)
        return action.cpu().numpy(), logprob.cpu().numpy()
    
class ILmodel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        masking_indices: List,
        device = torch.device("cpu")
    ) -> None:
        super(ILmodel, self).__init__()
        # save parameter first
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.masking_indices = masking_indices
        self.masking_length = len(self.masking_indices)

        self.device = device

        '''Define model'''
        # embedding has tanh as an activation function while encoder and decoder have ReLU
        self.embed = MLP(
            input_dim=state_dim+latent_dim-self.masking_length ,
            hidden_dims=(64, 64),
            output_dim=state_dim+latent_dim-self.masking_length,
            activation=nn.Tanh,
            device=device
        )

        self.encoder = MLP(
            input_dim=state_dim+latent_dim-self.masking_length ,
            hidden_dims=(64, 64, 32),
            output_dim=latent_dim,
            device=device
        )

        self.mu_network = nn.Linear(latent_dim, latent_dim).to(device)
        self.logstd_network = nn.Linear(latent_dim, latent_dim).to(device)

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=(64, 64, 32),
            output_dim=state_dim,
            dropout_rate=0.7,
            device=device
        )        

        self.to(device=self.device)

    def change_device_info(self, device):
        self.embed.device = device
        self.mu_network.device = device
        self.logstd_network.device = device
        self.encoder.device = device
        self.decoder.device = device
        self.device = device

    def forward(
        self,
        obs: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # conversion
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)

        ego_state = self.torch_delete(obs, self.masking_indices, axis=-1)
        input = torch.concatenate((ego_state, y), axis=-1)

        # embedding network
        input = self.embed(input)

        # encoder
        z = self.encoder(input)

        z_mu = F.tanh(torch.clamp(self.mu_network(z), -7.24, 7.24)) # tanh to match to N(0, I) of prior distribution
        z_std = torch.exp(torch.clamp(self.logstd_network(z), -5, 2)) # clamping b/w -5 and 2

        dist = torch.distributions.multivariate_normal.MultivariateNormal(z_mu, torch.diag_embed(z_std))
        z = dist.rsample()

        return z, z_mu, z_std

    def decode(self, next_states: torch.Tensor, z: Optional[torch.Tensor], z_mu: torch.Tensor, z_std: torch.Tensor) -> torch.Tensor:
        next_state_pred = self.decoder(z)

        state_pred_loss = F.mse_loss(next_state_pred, next_states)
        kl_loss = - 0.5 * torch.sum(1 + torch.log(z_std.pow(2)) - z_mu.pow(2) - z_std.pow(2))

        ELBO_loss = state_pred_loss + kl_loss

        return ELBO_loss

    def torch_delete(self, tensor, indices, axis=None):
        tensor = tensor.cpu().numpy()
        tensor = np.delete(tensor, indices, axis=axis)
        tensor = torch.tensor(tensor).to(self.device)
        return tensor

# MetaWorld Gaussian Mixture Variational Auto-Encoder 
class HLmodel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim:int,
        latent_dim: int,
        device = torch.device("cpu")
    ) -> None:
        super(HLmodel, self).__init__()
        '''parameter save to the class'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

        feature_dim = state_dim+action_dim+state_dim+1

        '''Pre-embedding'''
        self.state_embed = MLP(
            input_dim=state_dim,
            hidden_dims=(64, 64),
            output_dim=state_dim,
            initialization=True,
            activation=nn.Tanh,
            device=device
        )
        self.action_embed = MLP(
            input_dim=action_dim,
            hidden_dims=(32, 32),
            output_dim=action_dim,
            initialization=True,
            activation=nn.Tanh,
            device=device
        )
        self.reward_embed = MLP(
            input_dim=1,
            hidden_dims=(16, 16),
            output_dim=1,
            initialization=True,
            activation=nn.Tanh,
            device=device
        )

        '''Encoder definitions'''
        self.encoder = RecurrentEncoder(
            input_size=feature_dim,
            hidden_size=feature_dim,
            device=device
        )
        # cat(h) -> y -> en(h, y) 
        self.cat_layer = MLP(
            input_dim=feature_dim,
            hidden_dims=(512, 512), # hidden includes the relu activation
            dropout_rate=0.7,
            device=device
        )

        self.Gumbel_layer = GumbelSoftmax(512, 
                                          self.latent_dim, 
                                          device)

        self.to(device=self.device)

    def change_device_info(self, device):
        self.state_embed.device = device
        self.state_embed.device = device
        self.state_embed.device = device
        self.encoder.device = device
        self.cat_layer.device = device
        self.Gumbel_layer.device = device
        self.device = device

    def pack4rnn(self, input_tuple, is_batch):
        '''
        Input: tuple of s, a, ns, r, m
        Return: padded_data (batch, seq, fea) and legnths for each trj
        =============================================
        1. find the maximum length of the given traj
        2. create a initialized batch with that max traj length
        3. put the values in
        4. return the padded data and its corresponding length for later usage.
        '''
        obss, actions, next_obss, rewards, masks = input_tuple
        if is_batch:
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
            data_dim = trajs[0].shape[-1]
            padded_data = torch.zeros((len(lengths), largest_length, data_dim))

            for i, traj in enumerate(trajs):
                padded_data[i, :lengths[i], :] = traj
            
            return (padded_data, lengths)
        else:
            states, actions, next_states, rewards, masks = input_tuple
            mdp = torch.concatenate((states, actions, next_states, rewards), axis=-1)
            # convert to 3d aray for rnn
            mdp = mdp[None, None, :]
            return (mdp, None)
    
    def forward(
        self,
        input_tuple: tuple,
        is_batch: bool = False,
    ) -> Tuple[torch.Tensor]:
        states, actions, next_states, rewards, masks = input_tuple

        # conversion
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, device=self.device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.int32)

        #print(states.shape, actions.shape, next_states.shape, rewards.shape, masks.shape)
        '''Embedding'''
        states = self.state_embed(states)
        actions = self.action_embed(actions)
        next_states = self.state_embed(next_states)

        input_tuple = (states, actions, next_states, rewards, masks)        

        mdp_and_lengths = self.pack4rnn(input_tuple, is_batch)
        out = self.encoder(mdp_and_lengths, is_batch)

        # categorical
        out = self.cat_layer(out)
        logits, prob, y = self.Gumbel_layer(out)

        return states, y # this pair directly goes to IL
    