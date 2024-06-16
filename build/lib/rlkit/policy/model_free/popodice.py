import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, Union, Tuple
from rlkit.modules import TanhMixtureNormalPolicy, TanhNormalPolicy, ValueNetwork

np.set_printoptions(precision=3, suppress=True)
sns.set_theme()

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

class PPDPolicy(nn.Module):
    """Offline policy Optimization via Stationary DIstribution Correction Estimation (OptiDICE)"""

    def __init__(
            self, 
            actor,
            data_actor,
            v_network,
            e_network,
            phi_network,
            v_network_optim,
            e_network_optim,
            phi_network_optim,
            args
            ):
        super(PPDPolicy, self).__init__()
        self._gamma = args.gamma
        self._policy_extraction = args.policy_extraction
        self._use_policy_entropy_constraint = args.use_policy_entropy_constraint
        self._use_data_policy_entropy_constraint = args.use_data_policy_entropy_constraint
        self._target_entropy = args.target_entropy
        self._alpha = args.alpha
        self._f = args.f
        self._gendice_v = args.gendice_v
        self._gendice_e = args.gendice_e
        self._gendice_loss_type = args.gendice_loss_type
        self._lr = args.actor_lr
        self._e_loss_type = args.e_loss_type
        self._v_l2_reg = args.v_l2_reg
        self._e_l2_reg = args.e_l2_reg
        self._lamb_scale = args.lamb_scale
        self._reward_scale = args.reward_scale
        self._eps = 1e-10

        self._iteration = torch.tensor(0, dtype=torch.int64, requires_grad=False)
        self._optimizers = dict()

        # create networks / variables for DICE-RL
        self._v_network = v_network
        self._optimizers['v'] = v_network_optim

        self._e_network = e_network
        self._optimizers['e'] = e_network_optim

        self._phi_network = phi_network
        self._optimizers['phi'] = phi_network_optim

        self.args = args

        # GenDICE regularization, i.e., E[w] = 1.
        if self._gendice_v:
            self._lamb_v = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_v'] = optim.Adam([self._lamb_v], lr=self._lr)
        else:
            self._lamb_v = 0

        if self._gendice_e:
            self._lamb_e = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_e'] = optim.Adam([self._lamb_e], lr=self._lr)
        else:
            self._lamb_e = 0

        # f-divergence functions
        # NOTE: g(x) = f(ReLU((f')^{-1}(x)))
        # NOTE: r(x) = ReLU((f')^{-1}(x)
        if self._f == 'chisquare':
            self._f_fn = lambda x: 0.5 * (x - 1) ** 2
            self._f_prime_inv_fn = lambda x: x + 1
            self._g_fn = lambda x: 0.5 * (F.relu(x + 1) - 1) ** 2
            self._r_fn = lambda x: F.relu(self._f_prime_inv_fn(x))
            self._log_r_fn = lambda x: torch.where(x < 0, torch.log(1e-10), torch.log(torch.max(x, torch.tensor(0.0)) + 1))
        elif self._f == 'kl':
            self._f_fn = lambda x: x * torch.log(x + 1e-10)
            self._f_prime_inv_fn = lambda x: torch.exp(x - 1)
            self._g_fn = lambda x: torch.exp(x - 1) * (x - 1)
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: x - 1
        elif self._f == 'elu':
            self._f_fn = lambda x: torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2)
            self._f_prime_inv_fn = lambda x: torch.where(x < 0, torch.exp(torch.minimum(x, torch.tensor(0.0))), x + 1)
            self._g_fn = lambda x: torch.where(x < 0, torch.exp(torch.minimum(x, torch.tensor(0.0))) * (torch.minimum(x, torch.tensor(0.0)) - 1) + 1, 0.5 * x ** 2)
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: torch.where(x < 0, x, torch.log(torch.maximum(x, torch.tensor(0.0)) + 1))
        else:
            raise NotImplementedError()

        self._bonus = lambda x, y: torch.sqrt(x / (0.25 * torch.mean(torch.abs(y))))
        self._std_fn = lambda x, y: torch.where(y > 0, self._bonus(x, y),  1 / self._bonus(x, y))
        self._std_penalty = lambda x, y: torch.clamp(self._std_fn(x, y), 0, 1.5) 

        # policy
        self._policy_network = actor
        self._optimizers['policy'] = optim.Adam(self._policy_network.parameters(), lr=self._lr)

        if self._use_policy_entropy_constraint:
            self._log_ent_coeff = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['ent_coeff'] = optim.Adam([self._log_ent_coeff], lr=self._lr)

        # data policy
        if self._policy_extraction == 'iproj':
            if self.args.data_policy == 'tanh_normal':
                self._data_policy_network = data_actor
            elif self.args.data_policy == 'tanh_mdn':
                if self.args.data_policy_num_mdn_components == 1:
                    self._data_policy_network = data_actor
                else:
                    self._data_policy_network = data_actor
            self._optimizers['data_policy'] = optim.Adam(self._data_policy_network.parameters(), lr=self._lr)

            if self._use_data_policy_entropy_constraint:
                self._data_log_ent_coeff = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                self._optimizers['data_ent_coeff'] = optim.Adam([self._data_log_ent_coeff], lr=self._lr)

    def v_loss(self, initial_v_values, e_v, w_v, f_w_v, lamb_v, result={}):
        # Compute v loss
        v_loss0 = (1 - self._gamma) * torch.mean(initial_v_values)
        v_loss1 = torch.mean(- self._alpha * f_w_v)
        if self._gendice_loss_type == 'gendice':
            v_loss2 = torch.mean(w_v * (e_v - self._lamb_scale * lamb_v))
            v_loss3 = self._lamb_scale * (lamb_v + lamb_v ** 2 / 2)
        elif self._gendice_loss_type == 'bestdice':
            v_loss2 = torch.mean(w_v * (e_v - lamb_v))
            v_loss3 = lamb_v
        else:
            raise NotImplementedError
        v_loss = v_loss0 + v_loss1 + v_loss2 + v_loss3 #+ std_sum

        v_ob_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._v_network.parameters()]))

        if self._v_l2_reg is not None:
            v_loss += self._v_l2_reg * v_ob_l2_norm

        result.update({
            'v_loss0': v_loss0,
            'v_loss1': v_loss1,
            'v_loss2': v_loss2,
            'v_loss3': v_loss3,
            'v_loss': v_loss,
            'v_l2_norm': v_ob_l2_norm
        })

        return result

    def lamb_v_loss(self, e_v, w_v, f_w_v, lamb_v, result={}):
        # GenDICE regularization: E_D[w(s,a)] = 1
        if self._gendice_loss_type == 'gendice':
            lamb_v_loss = torch.mean(- self._alpha * f_w_v + w_v * (e_v - self._lamb_scale * lamb_v)
                                         + self._lamb_scale * (lamb_v + lamb_v ** 2 / 2))
        elif self._gendice_loss_type == 'bestdice':
            lamb_v_loss = torch.mean(- self._alpha * f_w_v + w_v * (e_v - self._lamb_scale * lamb_v) + lamb_v)
        else:
            raise NotImplementedError

        result.update({
            'lamb_v_loss': lamb_v_loss,
            'lamb_v': lamb_v,
        })

        return result

    def e_loss(self, e_v, e_values, w_e, f_w_e, lamb_e, result={}):

        # Compute e loss
        if self._e_loss_type == 'minimax':
            e_loss = torch.mean(self._alpha * f_w_e - w_e * (e_v - self._lamb_scale * lamb_e)) # 
        elif self._e_loss_type == 'mse':
            e_loss = torch.mean((e_v - e_values) ** 2)
        else:
            raise NotImplementedError

        e_ob_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._e_network.parameters()]))

        if self._e_l2_reg is not None:
            e_loss += self._e_l2_reg * e_ob_l2_norm

        result.update({
            'e_loss': e_loss,
            'e_l2_norm': e_ob_l2_norm,
        })

        return result

    def lamb_e_loss(self, e_v, w_e, f_w_e, lamb_e, result={}):
        # GenDICE regularization: E_D[w(s,a)] = 1
        if self._gendice_loss_type == 'gendice':
            lamb_e_loss = torch.mean(- self._alpha * f_w_e + w_e * (e_v - self._lamb_scale * lamb_e)
                                         + self._lamb_scale * (lamb_e + lamb_e ** 2 / 2))
        elif self._gendice_loss_type == 'bestdice':
            lamb_e_loss = torch.mean(- self._alpha * f_w_e + w_e * (e_v - self._lamb_scale * lamb_e) + lamb_e)
        else:
            raise NotImplementedError

        result.update({
            'lamb_e_loss': lamb_e_loss,
            'lamb_e': lamb_e,
        })

        return result

    def policy_loss(self, observation, action, w_e, result={}):
        # Compute policy loss
        sampled_action, sampled_pretanh_action, sampled_action_log_prob, sampled_pretanh_action_log_prob, pretanh_action_dist \
            = self._policy_network(observation)
        # Entropy is estimated on newly sampled action.
        negative_entropy_loss = torch.mean(sampled_action_log_prob)
        positive_entropy_loss = -torch.mean(sampled_action_log_prob)

        policy_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._policy_network.parameters()]))

        if self._policy_extraction == 'wbc':
            # Weighted BC
            action_log_prob, _ = self._policy_network.log_prob(pretanh_action_dist, action, is_pretanh_action=False)
            policy_loss = - torch.mean(w_e * action_log_prob)

        elif self._policy_extraction == 'iproj':
            # Information projection
            _, _, _, _, data_pretanh_action_dist = self._data_policy_network(observation.detach())

            sampled_e_values = self._e_network(observation.detach(), sampled_action) # policy advantage

            if self._gendice_loss_type == 'gendice':
                sampled_log_w_e = self._log_r_fn((sampled_e_values - self._lamb_scale * self._lamb_e.detach()) / self._alpha)
            elif self._gendice_loss_type == 'bestdice':
                sampled_log_w_e = self._log_r_fn((sampled_e_values - self._lamb_e.detach()) / self._alpha)
            else:
                raise NotImplementedError()

            _, sampled_pretanh_action_data_log_prob = self._data_policy_network.log_prob(data_pretanh_action_dist, sampled_pretanh_action)
            kl = sampled_pretanh_action_log_prob - sampled_pretanh_action_data_log_prob

            policy_loss = - torch.mean(sampled_log_w_e - kl)

            result.update({'kl': torch.mean(kl)})

        else:
            raise NotImplementedError()

        if self._use_policy_entropy_constraint:
            ent_coeff = torch.exp(self._log_ent_coeff).detach()
            policy_loss += ent_coeff * negative_entropy_loss

            ent_coeff_loss = - self._log_ent_coeff * (sampled_action_log_prob.detach() + self._target_entropy)
            
            result.update({
                'ent_coeff_loss': torch.mean(ent_coeff_loss),
                'ent_coeff': ent_coeff,
            })

        result.update({
            'policy_loss': policy_loss,
            'policy_l2_norm': policy_l2_norm,
            'q_loss': positive_entropy_loss,
            'sampled_action_log_prob': torch.mean(sampled_action_log_prob),
            'negative_entropy_loss': negative_entropy_loss
        })

        return result

    def data_policy_loss(self, observation, action, result={}):
        # Compute data policy loss
        _, _, data_sampled_action_log_prob, _, data_policy_dists = self._data_policy_network(observation)

        data_action_log_prob, _ = self._data_policy_network.log_prob(data_policy_dists, action, is_pretanh_action=False)
        data_policy_loss = - torch.mean(data_action_log_prob)

        # Entropy is estimated on newly sampled action.
        data_negative_entropy_loss = torch.mean(data_sampled_action_log_prob)

        if self._use_data_policy_entropy_constraint:
            data_ent_coeff = torch.exp(self._data_log_ent_coeff)
            data_policy_loss += data_ent_coeff * data_negative_entropy_loss

            data_ent_coeff_loss = - self._data_log_ent_coeff * (data_sampled_action_log_prob.detach() + self._target_entropy)

            result.update({
                'data_ent_coeff_loss': torch.mean(data_ent_coeff_loss),
                'data_ent_coeff': data_ent_coeff,
            })

        result.update({
            'data_negative_entropy_loss': data_negative_entropy_loss,
            'data_action_log_prob': torch.mean(data_action_log_prob),
            'data_policy_loss': data_policy_loss
        })

        return result

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        with torch.no_grad():
            if deterministic:
                action = self._policy_network.deterministic_action(obs)
            else:
                NotImplementedError
        return action.cpu().numpy()
        
    def learn(self, batch):
        observation, action, next_observation, initial_observation, reward, terminal = \
            batch["observations"], batch["actions"],  batch["next_observations"], \
                batch["initial_observations"],  batch["rewards"], batch["terminals"]
        
        reward = reward * self._reward_scale
        ####################################################################################################################
        # Preprocess the data to concat with latent variable
        indices = [torch.where(terminal == 1)] + [len(terminal)]

        context = torch.concatenate((observation, action, next_observation, reward), axis=1)
        context = [context[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]

        z = self._phi_network(observation, context)

        '''divider for Ss and Sv'''
        initial_observation = torch.concatenate((initial_observation, z[torch.where(terminal == 1)[0]].reshape(-1,1)), axis=1)
        observation = torch.concatenate((observation[:-1], z[:-1].reshape(-1,1)), axis=1)
        next_observation = torch.concatenate((next_observation[:-1], z[1:].reshape(-1,1)), axis=1)
        terminal = terminal[:-1]
        reward = reward[:-1]
        action = action[:-1]

        ####################################################################################################################
        # Shared network values
        initial_v_values = self._v_network(initial_observation.detach())
        v_values = self._v_network(observation.detach())
        next_v_values = self._v_network(next_observation.detach())

        e_v = reward + (1 - terminal) * self._gamma * next_v_values - v_values
        preactivation_v = (e_v - self._lamb_scale * self._lamb_v) / self._alpha
        w_v = self._r_fn(preactivation_v)
        f_w_v = self._g_fn(preactivation_v)

        if self._gendice_v:
            preactivation_v_lamb = (e_v.detach() - self._lamb_scale * self._lamb_v) / self._alpha
            w_v_ob_lamb = self._r_fn(preactivation_v_lamb)
            f_w_v_ob_lamb = self._g_fn(preactivation_v_lamb)

        ####################################################################################################################

        e_values = self._e_network(observation.detach(), action)
        preactivation_e = (e_values - self._lamb_scale * self._lamb_e) / self._alpha
        w_e = self._r_fn(preactivation_e)
        f_w_e = self._g_fn(preactivation_e)
        
        if self._gendice_e:
            preactivation_e_lamb = (e_values.detach() - self._lamb_scale * self._lamb_e) / self._alpha 
            w_e_lamb = self._r_fn(preactivation_e_lamb)
            f_w_e_lamb = self._g_fn(preactivation_e_lamb)
         
        ####################################################################################################################
        # Compute loss and optimize
        loss_result = self.v_loss(initial_v_values, e_v, w_v, f_w_v, self._lamb_v, result={})

        self._optimizers['v'].zero_grad()
        v_loss = loss_result['v_loss']; v_loss.backward()
        self._optimizers['v'].step()

        if self._gendice_v:
            loss_result.update(self.lamb_v_loss(e_v.detach(), w_v_ob_lamb, f_w_v_ob_lamb, self._lamb_v))

            self._optimizers['lamb_v'].zero_grad()
            lamb_v_loss = loss_result['lamb_v_loss']; lamb_v_loss.backward()
            self._optimizers['lamb_v'].step()
        
        ####################################################################################################################

        loss_result.update(self.e_loss(e_v.detach(), e_values, w_e, f_w_e, self._lamb_e))

        self._optimizers['e'].zero_grad(); 
        e_loss = loss_result['e_loss']; e_loss.backward()
        self._optimizers['e'].step(); self._optimizers['phi'].step()

        if self._gendice_e:
            loss_result.update(self.lamb_e_loss(e_v.detach(), w_e_lamb, f_w_e_lamb, self._lamb_e))
            self._optimizers['lamb_e'].zero_grad()
            lamb_e_loss = loss_result['lamb_e_loss']; lamb_e_loss.backward()
            self._optimizers['lamb_e'].step()

        ####################################################################################################################
        ####################################################################################################################

        loss_result.update(self.policy_loss(observation, action, w_e.detach()))
        
        self._optimizers['policy'].zero_grad(); self._optimizers['phi'].zero_grad()
        
        policy_loss = loss_result['policy_loss']; policy_loss.backward(retain_graph=True)
        self._optimizers['policy'].step()

        q_loss = loss_result['q_loss']; q_loss.backward()
        self._optimizers['phi'].step()

        if self._use_policy_entropy_constraint: 
            self._optimizers['ent_coeff'].zero_grad()
            ent_coeff_loss = loss_result['ent_coeff_loss']; ent_coeff_loss.backward()
            self._optimizers['ent_coeff'].step()

        if self._policy_extraction == 'iproj':
            loss_result.update(self.data_policy_loss(observation.detach(), action))
            
            self._optimizers['data_policy'].zero_grad()
            data_policy_loss = loss_result['data_policy_loss']; data_policy_loss.backward()
            self._optimizers['data_policy'].step()

            if self._use_data_policy_entropy_constraint:
                self._optimizers['data_ent_coeff'].zero_grad()
                data_ent_coeff_loss = loss_result['data_ent_coeff_loss']; data_ent_coeff_loss.backward()
                self._optimizers['data_ent_coeff'].step()

        # Update iteration
        self._iteration += 1

        loss_dict = {}
        for key, value in loss_result.items():
            if isinstance(value, torch.Tensor):
                loss_dict[key] = value.item()
            else:
                loss_dict[key] = value
        
        return loss_dict

    def get_loss_info(self):
        loss_info = {
            'iteration': self._iteration.item()
        }
        return loss_info