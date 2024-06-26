import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

from rlkit.utils.buffer import TrajectoryBuffer


# Managing High-Task Complexity: Hierarchical Meta-RL via Skill Representational Learning 
# HiMeta (Hierarchical Meta-Reinforcement Learning)
class HiMeta(nn.Module):
    def __init__(self,
                 HLmodel: nn.Module,
                 ILmodel: nn.Module,
                 LLmodel: nn.Module,
                 traj_buffer: TrajectoryBuffer,
                 HL_lr: float = 1e-3, # HL and critic
                 IL_lr: float = 5e-4, # VAE lr
                 LL_lr: float = 3e-4, # this is PPO policy agent
                 ###params###
                 tau: float = 0.95,
                 gamma: float = 0.99,
                 K_epochs: int = 3,
                 eps_clip: float = 0.2,
                 l2_reg: float = 1e-4,
                 device='cpu'):
        super(HiMeta, self).__init__()
        self.HLmodel = HLmodel
        self.ILmodel = ILmodel
        self.LLmodel = LLmodel
        self.traj_buffer = traj_buffer

        self.loss_fn = torch.nn.MSELoss()

        optim_params = [{'params': self.HLmodel.parameters(), 'lr': HL_lr},
                        {'params': self.ILmodel.parameters(), 'lr': IL_lr},
                        {'params': self.LLmodel.actor.parameters(), 'lr': LL_lr},
                        {'params': self.LLmodel.critic.parameters(), 'lr': HL_lr}]

        self.optimizers = torch.optim.AdamW(optim_params)

        self.tau = tau
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.l2_reg = l2_reg
        self.device = torch.device(device)

    def train(self) -> None:
        self.HLmodel.train()
        self.ILmodel.train()
        self.LLmodel.train()

    def eval(self) -> None:
        self.HLmodel.eval()
        self.ILmodel.eval()
        self.LLmodel.eval()

    def to_device(self, device=torch.device('cpu')):
        self.device = device
        self.HLmodel.change_device_info(device)
        self.ILmodel.change_device_info(device)
        self.LLmodel.change_device_info(device)
        self.to(device)
    
    def init_encoder_hidden_info(self):
        self.HLmodel.encoder.init_hidden_info()

    def forward(self, input_tuple, deterministic=False):
        '''decision making framework using all hierarchy'''
        '''Input: tuple or tuple batch dim: 1 x (s, a, ns, r, m) or batch x (s, a, ...)'''

        '''HL-model first'''
        
        with torch.no_grad():
            # obs and its corresponding categorical inference
            obs, y = self.HLmodel(input_tuple) 

            '''IL-model'''        
            z, z_mu, z_std = self.ILmodel(obs.clone().detach(), y.detach())

            '''LL-model'''
            obs = torch.concatenate((obs, z), axis=-1)
            action, logprob = self.LLmodel.select_action(obs.clone().detach(), deterministic=deterministic)
        
        return action, logprob, (y, z)
        
    def embed(self, input_tuple):
        '''
        Used during the update (learn), since it does not need to 
        make an action but encodded obs or embedding itself.
        '''
        # HL
        states, y = self.HLmodel(input_tuple, is_batch=True)

        # IL
        z, z_mu, z_std = self.ILmodel(states.clone().detach(), y.clone().detach())

        y_embedded_states = torch.concatenate((states, y), axis=-1)
        z_embedded_states = torch.concatenate((states, z), axis=-1)

        return states, y_embedded_states, z_embedded_states, (z, z_mu, z_std)
    
    def learn(self, batch):
        from rlkit.utils.utils import estimate_advantages, estimate_episodic_value
        '''
        # Update the buffer
        self.traj_buffer.push(batch)
        traj_mdp_tuple = self.traj_buffer.sample(40)
        traj_states, traj_actions, traj_next_states, traj_rewards, traj_masks = traj_mdp_tuple
        
        traj_states = torch.tensor(traj_states).to(self.device)
        traj_actions = torch.tensor(traj_actions).to(self.device)
        traj_next_states = torch.tensor(traj_next_states).to(self.device)
        traj_rewards = torch.tensor(traj_rewards).to(self.device)
        traj_masks = torch.tensor(traj_masks).to(self.device)
        '''
        states = torch.tensor(batch['states']).to(self.device)
        actions = torch.tensor(batch['actions']).to(self.device)
        next_states = torch.tensor(batch['next_states']).to(self.device)
        rewards = torch.tensor(batch['rewards']).to(self.device)
        masks = torch.tensor(batch['masks']).to(self.device)
        logprobs = torch.tensor(batch['logprobs']).to(self.device)
        successes = torch.tensor(batch['successes']).to(self.device)

        #traj_mdp_tuple = (traj_states, traj_actions, traj_next_states, traj_rewards, traj_masks)
        mdp_tuple = (states, actions, next_states, rewards, masks)
        
        '''COMPUTE ADVANTAGES FOR POLICY UPDATE'''
        with torch.no_grad():
            _, y_embedded_states, _, _ = self.embed(mdp_tuple) # for LL actor
            values = self.LLmodel.critic(y_embedded_states)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)

        #use_batch = False

        '''Update the parameters'''
        for _ in range(self.K_epochs):    
            '''HL and IL batch traj sampler update'''
            '''
            if use_batch:
                _, traj_y_embedded_states, _, (traj_z, traj_z_mu, traj_z_std) = self.embed(traj_mdp_tuple) # for HL, IL, and LL critic
                _, _, z_embedded_states, _ = self.embed(mdp_tuple) # for LL actor

                # HL grad in critic to train HLmodel
                r_pred = self.LLmodel.critic(traj_y_embedded_states) # y embedding b/c it is sub-task info while z is action-task info; this is not action-value fn

                # HL and IL loss
                _, traj_returns = estimate_advantages(traj_rewards, traj_masks, r_pred, self.gamma, self.tau, self.device)
                value_loss = self.loss_fn(r_pred, traj_returns)
                #(decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(next_states, z, z_mu, z_std)
                (decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(traj_states, traj_next_states, traj_z, traj_z_mu, traj_z_std)
            else:
            '''
            _, y_embedded_states, z_embedded_states, (z, z_mu, z_std) = self.embed(mdp_tuple) # for LL actor

            '''HL and IL loss'''
            # HL grad in critic to train HLmodel
            r_pred = self.LLmodel.critic(y_embedded_states) # y embedding b/c it is sub-task info while z is action-task info; this is not action-value fn

            value_loss = self.loss_fn(r_pred, returns)
            (decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(states, next_states, z, z_mu, z_std)
            
            '''LL loss'''
            dist = self.LLmodel.actor(z_embedded_states.detach())

            new_logprobs = dist.log_prob(actions)            
            dist_entropy = dist.entropy()
    
            ratios = torch.exp(new_logprobs - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            #print(new_logprobs.shape, logprobs.shape, ratios.shape, surr1.shape, surr2.shape, advantages.shape, dist_entropy.shape, value_loss.shape, decoder_loss.shape)

            '''Combination loss''' 
            # this is fine since they have own grad for each loss term 
            # (surr1, surr2, dist_entropy)-> LL actor
            # value_loss -> LL critic & HL model
            # decoder_loss = IL model
            actor_loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * value_loss - 0.01 * dist_entropy)
            loss = actor_loss + decoder_loss 

            '''Update agents'''
            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()

        result = {
            'loss/critic_loss': value_loss.item(),
            'loss/actor_loss': actor_loss.item(),
            'loss/decoder_loss': decoder_loss.item(),
            'loss/decoder_pred_loss': state_pred_loss.item(),
            'loss/decoder_kl_loss': kl_loss.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/success': successes.mean().item()
        }
        
        return result 
    
    def save_model(self, logdir, epoch, is_best=False):
        self.actor, self.critic = self.LLmodel.actor.cpu(), self.LLmodel.critic.cpu()
        self.ILmodel = self.ILmodel.cpu()
        self.HLmodel = self.HLmodel.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump((self.actor, self.critic, self.ILmodel, self.HLmodel), open(path, 'wb'))

        self.actor, self.critic = self.actor.to(self.device), self.critic.to(self.device)
        self.ILmodel = self.ILmodel.to(self.device)
        self.HLmodel = self.HLmodel.to(self.device)