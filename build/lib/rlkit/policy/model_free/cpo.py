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
from rlkit.utils.utils  import estimate_advantages, estimate_episodic_value, get_flat_params_from, set_flat_params_to, normal_log_density

def conjugate_gradients(Avp, b, nsteps, device, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
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

class CPOPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            r_critic: nn.Module,  
            c_critic: nn.Module,  
            critic_optimizer: torch.optim.Optimizer,
            encoder: BaseEncoder = None,
            encoder_optim: torch.optim.Optimizer = None,
            tau: float = 0.95,
            gamma: float  = 0.99,
            max_kl: float = 1e-3,
            damping: float = 1e-2,
            l2_reg: float = 1e-4,
            d_k: float = 10.0,
            grad_norm: bool = True,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.r_critic = r_critic
        self.c_critic = c_critic
        self.critic_optimizer = critic_optimizer

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = BaseEncoder(device=device)
        self.encoder_optim = encoder_optim

        self.loss_fn = torch.nn.MSELoss()

        self._gamma = gamma
        self._tau = tau
        self._max_kl = max_kl
        self._damping = damping
        self._l2_reg = l2_reg
        self._d_k = d_k
        self.grad_norm = grad_norm

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device
    
    def train(self) -> None:
        self.actor.train()
        self.r_critic.train()
        self.c_critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.r_critic.eval()
        self.c_critic.eval()

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

    def encode_obs(self, mdp_tuple, env_idx = None, reset=True):
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
                embedding = self.encoder(mdp, do_pad=True)
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
    
    def learn(self, batch):
        obss = torch.from_numpy(batch['observations']).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)
        next_obss = torch.from_numpy(batch['next_observations']).to(self.device)
        rewards = torch.from_numpy(batch['rewards']).to(self.device)
        costs = torch.from_numpy(batch['costs']).to(self.device)
        masks = torch.from_numpy(batch['masks']).to(self.device)
        env_idxs = torch.from_numpy(batch['env_idxs']).to(self.device)
        successes = torch.from_numpy(batch['successes']).to(self.device)

        mdp_tuple = (obss, actions, next_obss, rewards, masks)
        _, _, embedded_obss, _ = self.encode_obs(mdp_tuple, env_idx=env_idxs)

        with torch.no_grad():
            reward_values = self.r_critic(embedded_obss)
            cost_values = self.c_critic(embedded_obss)

        """get advantage estimation from the trajectories"""
        reward_advantages, reward_returns = estimate_advantages(rewards, masks, reward_values, self._gamma, self._tau, self.device)
        cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values, self._gamma, self._tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)
        episodic_cost = estimate_episodic_value(costs, masks, 1.0, self.device)
        # Match the dimension
        #reward_advantages = torch.squeeze(reward_advantages)
        #cost_advantages = torch.squeeze(cost_advantages)

        """update critic"""
        r_pred = self.r_critic(embedded_obss)
        c_pred = self.c_critic(embedded_obss)
        r_v_loss = self.loss_fn(r_pred, reward_returns)
        c_v_loss = self.loss_fn(c_pred, cost_returns)

        self.critic_optimizer.zero_grad()
        r_v_loss.backward(); c_v_loss.backward()
        self.critic_optimizer.step()

        """update policy"""
        with torch.no_grad():
            dist = self.actor(embedded_obss)
            #fixed_log_probs = dist.log_prob(actions)
            fixed_log_probs = normal_log_density(actions, dist.mode(), dist.logstd(), dist.std())

        def get_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                dist = self.actor(embedded_obss)
                #log_probs = dist.log_prob(actions)
                log_probs = normal_log_density(actions, dist.mode(), dist.logstd(), dist.std())
                action_loss = -reward_advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
        
        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                dist = self.actor(embedded_obss)
                #log_probs = dist.log_prob(actions)
                log_probs = normal_log_density(actions, dist.mode(), dist.logstd(), dist.std())
                action_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
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

        def f_a_lambda(lamda):
            a = ((r**2)/s - q)/(2*lamda)
            b = lamda*((cc**2)/s - self._max_kl)/2
            c = - (r*cc)/s
            return a+b+c
    
        def f_b_lambda(lamda):
            a = -(q/lamda + lamda*self._max_kl)/2
            return a  
    
        Fvp = Fvp_direct
        
        '''reward grad'''
        reward_loss = get_loss()
        reward_grads = torch.autograd.grad(reward_loss, self.actor.parameters())
        reward_loss_grad = torch.cat([grad.view(-1) for grad in reward_grads]).detach()
        if self.grad_norm:
            reward_loss_grad = reward_loss_grad/torch.norm(reward_loss_grad)
        reward_stepdir = conjugate_gradients(Fvp, -reward_loss_grad, 10, device=self.device)
        if self.grad_norm:
            reward_stepdir = reward_stepdir/torch.norm(reward_stepdir)

        '''cost grad'''
        cost_loss = get_cost_loss()
        cost_grads = torch.autograd.grad(cost_loss, self.actor.parameters(), allow_unused=True)
        cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #a
        if self.grad_norm:
            cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad)
        cost_stepdir = conjugate_gradients(Fvp, -cost_loss_grad, 10, device=self.device) #(H^-1)*a
        if self.grad_norm:
            cost_stepdir = cost_stepdir/torch.norm(cost_stepdir)

        '''Define q, r, s'''
        p = -cost_loss_grad.dot(reward_stepdir) #a^T.H^-1.g
        q = -reward_loss_grad.dot(reward_stepdir) #g^T.H^-1.g
        r = reward_loss_grad.dot(cost_stepdir) #g^T.H^-1.a
        s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 

        d_k = torch.as_tensor(self._d_k, dtype=episodic_cost.dtype, device=self.device)
        cc = episodic_cost - d_k # c would be positive for most part of the training
        lamda = 2*self._max_kl

        #find optimal lambda_a and lambda_b
        A = torch.sqrt((q - (r**2)/s)/(self._max_kl - (cc**2)/s))
        B = torch.sqrt(q/self._max_kl)
        if cc>0:
            opt_lam_a = torch.max(r/cc,A)
            opt_lam_b = torch.max(0*A,torch.min(B,r/cc))
        else: 
            opt_lam_b = torch.max(r/cc,B)
            opt_lam_a = torch.max(0*A,torch.min(A,r/cc))
        
        #find values of optimal lambdas 
        opt_f_a = f_a_lambda(opt_lam_a)
        opt_f_b = f_b_lambda(opt_lam_b)
        
        if opt_f_a > opt_f_b:
            opt_lambda = opt_lam_a
        else:
            opt_lambda = opt_lam_b
                
        #find optimal nu
        nu = (opt_lambda*cc - r)/s
        if nu>0:
            opt_nu = nu 
        else:
            opt_nu = 0

        """ find optimal step direction """
        # check for feasibility
        if ((cc**2)/s - self._max_kl) > 0 and cc>0:
            opt_stepdir = torch.sqrt(2*self._max_kl/s)*Fvp(cost_stepdir)
        else: 
            opt_stepdir = (reward_stepdir - opt_nu*cost_stepdir)/(opt_lambda + 1e-7)
        
        # perform with line search
        prev_params = get_flat_params_from(self.actor)
        fullstep = opt_stepdir
        expected_improve = -reward_loss_grad.dot(fullstep)
        ln_sch_success, new_params = line_search(self.actor, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.actor, new_params)
        
        '''
        # perform w/o line search
        prev_params = get_flat_params_from(self.actor)
        new_params = prev_params + opt_stepdir
        set_flat_params_to(self.actor, new_params)
        ln_sch_success = 1
        '''

        result = {
            'loss/value_loss': r_v_loss.item(),
            'loss/cost_loss': c_v_loss.item(),
            'loss/actor_reward_loss': reward_loss.item(),
            'loss/actor_cost_loss': cost_loss.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/episodic_cost': episodic_cost.item(),
            'train/success': successes.mean().item(),
            'train/line_search': int(ln_sch_success)
        }
        
        return result 
    
    def save_model(self, logdir, epoch, running_state=None, is_best=False):
        self.actor, self.r_critic, self.c_critic = self.actor.cpu(), self.r_critic.cpu(), self.c_critic.cpu()
        if self.encoder.encoder_type == 'recurrent':
            self.encoder = self.encoder.cpu()
        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump((self.actor, self.r_critic, self.c_critic), open(path, 'wb'))
        if running_state is not None:
            pickle.dump((self.actor, self.r_critic, self.c_critic, running_state), open(path, 'wb'))
        self.actor, self.r_critic, self.c_critic = self.actor.to(self.device), self.r_critic.to(self.device), self.c_critic.to(self.device)
        if self.encoder.encoder_type == 'recurrent':
            self.encoder = self.encoder.to(self.device)