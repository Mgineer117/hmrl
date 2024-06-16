import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import math
import time

from typing import Dict, Union, Tuple
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder
from rlkit.utils.utils  import estimate_advantages, get_flat_params_from, set_flat_params_to, normal_log_density, estimate_episodic_value

def conjugate_gradients(Avp, b, nsteps, device, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


class TRPOPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            critic: nn.Module,  
            encoder: BaseEncoder,
            optimizer: torch.optim.Optimizer,
            masking_indices: list = None,
            tau: float = 0.95,
            gamma: float  = 0.99,
            max_kl: float = 1e-3,
            damping: float = 1e-2,
            l2_reg: float = 1e-6,
            grad_norm: bool = False,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.optimizer = optimizer

        self.masking_indices = masking_indices

        self.loss_fn = torch.nn.MSELoss()

        self._gamma = gamma
        self._tau = tau
        self._max_kl = max_kl
        self._damping = damping
        self._l2_reg = l2_reg
        self.grad_norm = grad_norm

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
        deterministic: bool = False
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
        successes = torch.from_numpy(batch['successes']).to(self.device)

        mdp_tuple = (obss, actions, next_obss, rewards, masks)
        _, _, embedded_obss, _ , _= self.encode_obs(mdp_tuple, env_idx=env_idxs, reset=True)

        values = self.critic(embedded_obss)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values.detach(), self._gamma, self._tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)

        """update critic"""
        self.optimizer.zero_grad()
        v_loss = self.loss_fn(values, returns)
        v_loss.backward()
        self.optimizer.step()

        """update policy"""
        with torch.no_grad():
            dist = self.actor(embedded_obss)
            fixed_log_probs = normal_log_density(actions, dist.mode(), dist.logstd(), dist.std())

        def get_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                dist = self.actor(embedded_obss)
                #log_probs = dist.log_prob(actions)
                log_probs = normal_log_density(actions, dist.mode(), dist.logstd(), dist.std())
                action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
            
        """directly compute Hessian*vector from KL"""
        def Fvp_direct(v):
            kl = self.actor.get_kl(embedded_obss)
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

            return flat_grad_grad_kl + v * self._damping

        Fvp = Fvp_direct
        
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        if self.grad_norm:
            loss_grad = loss_grad/torch.norm(loss_grad)
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10, device=self.device)

        shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
        lm = math.sqrt(self._max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = get_flat_params_from(self.actor)
        ln_sch_success, new_params = line_search(self.actor, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.actor, new_params)

        result = {
            'loss/critic_loss': v_loss.item(),
            'loss/actor_loss': loss.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/success': successes.mean().item(),
            'train/line_search': int(ln_sch_success)
        }
        
        return result 
    
    def mask_obs(self, obs: torch.Tensor, ind: list, dim: int) -> torch.Tensor:
        obs = obs.cpu().numpy()
        if ind is not None:
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