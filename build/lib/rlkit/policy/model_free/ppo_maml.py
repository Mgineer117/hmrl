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
from rlkit.utils.utils  import estimate_advantages, estimate_episodic_value, get_flat_params_from

class PPOMAMLPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            critic: nn.Module,  
            optimizer: torch.optim.Optimizer,
            encoder: BaseEncoder = BaseEncoder(),
            tau: float = 0.95,
            gamma: float = 0.99,
            K_epochs: int = 3,
            eps_clip: float = 0.2,
            l2_reg: float = 1e-4,
            actor_lr: float = 1e-4,
            critic_lr: float = 3e-4,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.optimizer = optimizer            

        self.loss_fn = torch.nn.MSELoss()

        self._gamma = gamma
        self._tau = tau
        self._K_epochs = K_epochs
        self._eps_clip = eps_clip
        self._l2_reg = l2_reg

        self._actor_lr = actor_lr
        self._critic_lr = critic_lr

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device
    
    def initialize_optimizer(self):
        # re-initialize the optimizer's referencing parameters
        # since deepcopy method does not copy its parameter reference
        # i.e., deepcopy copies everything but optimizer's parameter referencing
        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': self._actor_lr},
                        {'params': self.critic.parameters(), 'lr': self._critic_lr}
                    ])

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
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
        Given mdp = (s, a, s', r, mask)
        return embedding, embedded_next_obs = embedding is attached to the next_ob
          since it should include the information of reward and transition dynamics
        It should handle both tensor and numpy since some do not have network embedding but some do.
        All encoders take input in tensors
        Hence, we transform to tensor for all cases but return as tensor if is_batch else as numpy
        '''
        obs, actions, next_obs, rewards, masks = mdp_tuple
        # check dimension
        is_batch = True if len(obs.shape) > 1 else False

        # transform to tensor
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.int32)

        if self.encoder.encoder_type == 'none':
            # skip embedding
            embedding = None
            embedded_obs = obs
            embedded_next_obs = next_obs
            return obs, next_obs, embedded_obs, embedded_next_obs
        elif self.encoder.encoder_type == 'onehot':
            obs_embedding = self.encoder(obs, env_idx)
            next_obs_embedding = self.encoder(next_obs, env_idx)
            embedded_obs = torch.concatenate((obs_embedding, obs), axis=-1) 
            embedded_next_obs = torch.concatenate((next_obs_embedding, next_obs), axis=-1)
            return obs, next_obs, embedded_obs, embedded_next_obs
        elif self.encoder.encoder_type == 'recurrent':
            if is_batch:
                t_obs = torch.concatenate((obs[0][None, :], obs), axis=0)
                t_actions = torch.concatenate((actions[0][None, :], actions), axis=0)
                t_next_obs = torch.concatenate((obs[0][None, :], next_obs), axis=0)
                t_rewards = torch.concatenate((torch.tensor([0.0]).to(self.device)[None, :], rewards), axis=0)
                t_masks = torch.concatenate((torch.tensor([1.0]).to(self.device)[None, :], masks), axis=0)

                mdp = (t_obs, t_actions, t_next_obs, t_rewards, t_masks)
                embedding = self.encoder(mdp, do_reset=reset, is_batch=is_batch)
                embedded_obs = torch.concatenate((embedding[:-1], obs), axis=-1)
                embedded_next_obs = torch.concatenate((embedding[1:], next_obs), axis=-1)
                return obs, next_obs, embedded_obs, embedded_next_obs
            else:
                mdp = torch.concatenate((obs, actions, next_obs, rewards), axis=-1)
                mdp = mdp[None, None, :]
                embedding = self.encoder(mdp, do_reset=reset)
                embedded_next_obs = torch.concatenate((embedding, next_obs), axis=-1)
                embedded_obs = embedded_next_obs
                return obs, next_obs, embedded_obs, embedded_next_obs
        else:
            NotImplementedError

    def learn(self, batch, compute_param_grad=True):
        self.initialize_optimizer()

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
            _, _, embedded_obss, _ = self.encode_obs(mdp_tuple, env_idx=env_idxs, reset=True)
            values = self.critic(embedded_obss)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self._gamma, self._tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)
        advantages = torch.squeeze(advantages)

        # because of K updates, gradients need to be accumulated
        second_loss_grads = torch.ones(self.param_size)

        '''Update the parameters'''
        for k in range(self._K_epochs):    
            _, _, embedded_obss, _ = self.encode_obs(mdp_tuple, env_idx=env_idxs, reset=True)
            embedded_obss = torch.as_tensor(embedded_obss, device=self.device, dtype=torch.float32)

            '''get policy output'''
            dist = self.actor(embedded_obss.detach()) # detaching the gradient to focus encoder learning with critic reward maximization
            new_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            '''get value loss'''
            r_pred = self.critic(embedded_obss)
            v_loss = self.loss_fn(r_pred, returns)

            '''get policy loss'''
            ratios = torch.exp(new_logprobs - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self._eps_clip, 1+self._eps_clip) * advantages
    
            loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * v_loss - 0.01 * dist_entropy)
            
            '''Update agents'''
            self.optimizer.zero_grad()

            loss_grads = torch.autograd.grad(loss, self.actor.parameters(), retain_graph=True, create_graph=True)
            loss_critic_grads = torch.autograd.grad(loss, self.critic.parameters())

            for grad, param in zip(loss_grads, self.actor.parameters()):
                param.grad = grad
            for grad, param in zip(loss_critic_grads, self.critic.parameters()):
                param.grad = grad
            
            loss_grads = torch.cat([grad.view(-1) for grad in loss_grads])
            #print(loss_grads)

            if compute_param_grad or not k == (self._K_epochs - 1):    
                for i in range(self.param_size):
                    grads = torch.autograd.grad(loss_grads[i], self.actor.parameters(), retain_graph=True) 
                    second_loss_grads[i] *= torch.cat([grad.view(-1) for grad in grads])[i] # collect diagonal element of Hessian
                
                second_loss_grads = second_loss_grads.detach()
            
            loss_grads = loss_grads.detach() * second_loss_grads

            self.optimizer.step()
            #selected_params_after = self.actor.state_dict()['dist_net.mu.weight'].clone()
            #print(selected_params_after - selected_params_before)

        result = {
            'loss/critic_loss': v_loss.item(),
            'loss/actor_loss': loss.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/success': successes.mean().item()
        }
        
        actor_param = get_flat_params_from(self.actor).detach()
        critic_param = get_flat_params_from(self.critic).detach()
        if self.encoder.encoder_type =='recurrent':
            encoder_param = get_flat_params_from(self.encoder).detach()
        else:
            encoder_param = None


        return result, (loss_grads, second_loss_grads), (actor_param, critic_param, encoder_param)

    def learn_with_grad(self, memory, grad):
        self.initialize_optimizer()
        self.optimizer.zero_grad()
        
        # prepare the actor parameter's grad in array
        offset = 0
        for param in self.actor.parameters():
            num_elements = param.numel()
            param_grad = grad[offset:offset+num_elements].view(param.shape)
            param.grad = param_grad
            offset += num_elements
        
        # prepare the critic parameter's grad ready
        mdp_tuple = (memory['observations'], memory['actions'], memory['next_observations'], memory['rewards'], memory['masks'])
        _, _, embedded_obss, _ = self.encode_obs(mdp_tuple, env_idx=memory['env_idxs'], reset=True)

        values = self.critic(embedded_obss)
        _, returns = estimate_advantages(memory['rewards'], memory['masks'], values.detach(), self._gamma, self._tau, self.device)
        
        v_loss = self.loss_fn(values, returns)
        v_loss.backward()

        self.optimizer.step()

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