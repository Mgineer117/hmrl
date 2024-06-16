import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import math
import time
from copy import deepcopy

from typing import Dict, Union, Tuple
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder
from rlkit.utils.utils  import estimate_advantages, estimate_episodic_value

class PPOPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            critic: nn.Module,  
            encoder: BaseEncoder,
            optimizer: torch.optim.Optimizer,
            masking_indices: list = [],
            decoder_masking_indices: list = [],
            embed_loss: str = None,
            tau: float = 0.95,
            gamma: float = 0.99,
            K_epochs: int = 3,
            eps_clip: float = 0.2,
            l2_reg: float = 1e-4,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.optimizer = optimizer            

        self.loss_fn = torch.nn.MSELoss()

        self.masking_indices = masking_indices
        self.decoder_masking_indices = decoder_masking_indices
        self.embed_loss = embed_loss

        self._gamma = gamma
        self._tau = tau
        self._K_epochs = K_epochs
        self._eps_clip = eps_clip
        self._l2_reg = l2_reg

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device
    
    def train(self) -> None:
        self.actor.train()
        if self.encoder.encoder_type == 'recurrent':
            self.encoder.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        if self.encoder.encoder_type == 'recurrent':
            self.encoder.eval()
        self.critic.eval()

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
    
    def encode_obs(self, mdp_tuple, env_idx = None, reset=False):
        '''
        Input: mdp = (s, a, s', r, mask)
        Return: s, s', (s + embedding), (s' + embeding)
          since it should include the information of reward and transition dynamics
        '''
        obs, actions, next_obs, rewards, masks = mdp_tuple
        # check dimension
        is_batch = len(obs.shape) > 1

        # transform to tensor
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.int32)

        if self.encoder.encoder_type == 'none':
            # skip embedding
            embedding, embedded_obs, embedded_next_obs = None, obs, next_obs
        elif self.encoder.encoder_type == 'onehot':
            obs_embedding = self.encoder(obs, env_idx)
            next_obs_embedding = self.encoder(next_obs, env_idx)
            embedded_obs = torch.concatenate((obs_embedding, obs), axis=-1) 
            embedded_next_obs = torch.concatenate((next_obs_embedding, next_obs), axis=-1)
        elif self.encoder.encoder_type == 'recurrent':
            if is_batch:
                t_obs = torch.concatenate((obs[0][None, :], obs), axis=0)
                t_actions = torch.concatenate((actions[0][None, :], actions), axis=0)
                t_next_obs = torch.concatenate((obs[0][None, :], next_obs), axis=0)
                t_rewards = torch.concatenate((torch.tensor([0.0]).to(self.device)[None, :], rewards), axis=0)
                t_masks = torch.concatenate((torch.tensor([1.0]).to(self.device)[None, :], masks), axis=0)

                mdp = (t_obs, t_actions, t_next_obs, t_rewards, t_masks)
                embedding = self.encoder(mdp, do_reset=reset, is_batch=is_batch)
                
                embedded_obs = torch.concatenate((embedding[:-1], self.mask_obs(obs, self.masking_indices, dim=-1)), axis=-1)
                embedded_next_obs = torch.concatenate((embedding[1:], self.mask_obs(next_obs, self.masking_indices, dim=-1)), axis=-1)
            else:
                mdp = torch.concatenate((obs, actions, next_obs, rewards), axis=-1)
                mdp = mdp[None, None, :]
                embedding = self.encoder(mdp, do_reset=reset)
                embedded_next_obs = torch.concatenate((embedding, self.mask_obs(next_obs, self.masking_indices, dim=-1)), axis=-1)
                embedded_obs = embedded_next_obs # we are not using this            
        else:
            NotImplementedError
        return obs, next_obs, embedded_obs, embedded_next_obs, embedding
    
    def learn(self, batch):
        obss = torch.from_numpy(batch['observations']).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)
        next_obss = torch.from_numpy(batch['next_observations']).to(self.device)
        rewards = torch.from_numpy(batch['rewards']).to(self.device)
        masks = torch.from_numpy(batch['masks']).to(self.device)
        env_idxs = torch.from_numpy(batch['env_idxs']).to(self.device)
        logprobs = torch.from_numpy(batch['logprobs']).to(self.device)
        successes = torch.from_numpy(batch['successes']).to(self.device)
        
        mdp_tuple = (obss, actions, next_obss, rewards, masks)
        
        with torch.no_grad():
            _, _, embedded_obss, _, _ = self.encode_obs(mdp_tuple, env_idx=env_idxs, reset=True)
            values = self.critic(embedded_obss)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self._gamma, self._tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)
        advantages = torch.squeeze(advantages)

        '''Update the parameters'''
        for _ in range(self._K_epochs):    
            _, _, embedded_obss, _, embedding = self.encode_obs(mdp_tuple, env_idx=env_idxs, reset=True)
            embedded_obss = torch.as_tensor(embedded_obss, device=self.device, dtype=torch.float32)

            '''get network output'''
            if self.embed_loss == 'action':
                dist = self.actor(embedded_obss) # detaching the gradient to focus encoder learning with critic reward maximization
                r_pred = self.critic(embedded_obss.detach())
                decoder_loss = torch.tensor(0)
            elif self.embed_loss == 'reward':
                dist = self.actor(embedded_obss.detach()) # detaching the gradient to focus encoder learning with critic reward maximization
                r_pred = self.critic(embedded_obss)
                decoder_loss = torch.tensor(0)
            elif self.embed_loss == 'decoder':
                dist = self.actor(embedded_obss.detach()) # detaching the gradient to focus encoder learning with critic reward maximization
                r_pred = self.critic(embedded_obss.detach())
                decoder_loss = self.encoder.decode(mdp_tuple, embedding[:-1], self.mask_obs(obss, self.decoder_masking_indices,dim=-1))
            else:
                dist = self.actor(embedded_obss.detach()) # detaching the gradient to focus encoder learning with critic reward maximization
                r_pred = self.critic(embedded_obss.detach())
                decoder_loss = torch.tensor(0)

            new_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            '''get value loss'''
            v_loss = self.loss_fn(r_pred, returns)

            '''get policy loss'''
            ratios = torch.exp(new_logprobs - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self._eps_clip, 1+self._eps_clip) * advantages

            loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * v_loss - 0.01 * dist_entropy) + decoder_loss

            '''Update agents'''
            self.optimizer.zero_grad()
            loss.backward(); 
            self.optimizer.step()

        result = {
            'loss/critic_loss': v_loss.item(),
            'loss/actor_loss': loss.item(),
            'loss/decoder_loss': decoder_loss.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/success': successes.mean().item()
        }
        
        return result 
    
    def mask_obs(self, obs: torch.Tensor, ind: list, dim: int) -> torch.Tensor:
        obs = obs.cpu().numpy()
        obs = np.delete(obs, ind, axis=dim)
        obs = torch.from_numpy(obs).to(self.device)
        return obs
    
    def save_model(self, logdir, epoch, running_state=None, is_best=False):
        self.actor, self.critic = self.actor.cpu(), self.critic.cpu()
        if self.encoder.encoder_type == 'recurrent':
            self.encoder = self.encoder.cpu()
        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump((self.actor, self.critic, self.encoder), open(path, 'wb'))
        if running_state is not None:
            pickle.dump((self.actor, self.critic, self.encoder, running_state), open(path, 'wb'))
        self.actor, self.critic = self.actor.to(self.device), self.critic.to(self.device)
        if self.encoder.encoder_type == 'recurrent':
            self.encoder = self.encoder.to(self.device)