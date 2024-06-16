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
            v_ob_network,
            v_st_network,
            e_ob_network,
            e_st_network,
            v_ob_network_optim,
            e_ob_network_optim,
            v_st_network_optim,
            e_st_network_optim,
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
        self._e_min_ratio, self._e_max_ratio = args.e_range
        self._e_penalty_coeff = args.e_penalty_coeff

        self._iteration = torch.tensor(0, dtype=torch.int64, requires_grad=False)
        self._optimizers = dict()

        # create networks / variables for DICE-RL
        self._v_ob_network = v_ob_network
        self._optimizers['v_ob'] = v_ob_network_optim
        self._v_st_network = v_st_network
        self._optimizers['v_st'] = v_st_network_optim

        self._e_ob_network = e_ob_network
        self._optimizers['e_ob'] = e_ob_network_optim
        self._e_st_network = e_st_network
        self._optimizers['e_st'] = e_st_network_optim

        self.args = args

        # GenDICE regularization, i.e., E[w] = 1.
        if self._gendice_v:
            self._lamb_v_ob = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_v_ob'] = optim.Adam([self._lamb_v_ob], lr=self._lr)
            self._lamb_v_st = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_v_st'] = optim.Adam([self._lamb_v_st], lr=self._lr)
        else:
            self._lamb_v_ob = 0
            self._lamb_v_st = 0

        if self._gendice_e:
            self._lamb_e_ob = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_e_ob'] = optim.Adam([self._lamb_e_ob], lr=self._lr)
            self._lamb_e_st = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            self._optimizers['lamb_e_st'] = optim.Adam([self._lamb_e_st], lr=self._lr)
        else:
            self._lamb_e_ob = 0
            self._lamb_e_st = 0

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

        v_ob_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._v_ob_network.parameters()]))

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

        e_ob_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._e_ob_network.parameters()]))

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

    def policy_loss(self, observation, state, action, w_e, result={}):
        # Compute policy loss
        sampled_action, sampled_pretanh_action, sampled_action_log_prob, sampled_pretanh_action_log_prob, pretanh_action_dist \
            = self._policy_network(observation)
        # Entropy is estimated on newly sampled action.
        negative_entropy_loss = torch.mean(sampled_action_log_prob)

        policy_l2_norm = torch.linalg.norm(torch.cat([p.flatten() for p in self._policy_network.parameters()]))

        if self._policy_extraction == 'wbc':
            # Weighted BC
            action_log_prob, _ = self._policy_network.log_prob(pretanh_action_dist, action, is_pretanh_action=False)
            policy_loss = - torch.mean(w_e * action_log_prob)

        elif self._policy_extraction == 'iproj':
            # Information projection
            _, _, _, _, data_pretanh_action_dist = self._data_policy_network(state)

            _, sampled_e_ob_mu, _ = self._e_ob_network(observation, sampled_action) # policy advantage
            _, sampled_e_st_mu, _ = self._e_st_network(state, sampled_action) # policy advantage
            
            if self._iteration % 500 == 0:
                e_ob_value = sampled_e_ob_mu.clone().detach().cpu().numpy()
                e_st_value = sampled_e_st_mu.clone().detach().cpu().numpy()
                plt.plot(smooth(e_st_value, 0.95), label='e_st')
                plt.plot(smooth(e_ob_value, 0.95), label='e_ob')
                plt.legend()
                plt.savefig(f'error/iteration = {self._iteration}.png')
                plt.close()

            if self._gendice_loss_type == 'gendice':
                sampled_log_w_e = self._log_r_fn((sampled_e_ob_mu - self._lamb_scale * self._lamb_e_ob.detach()) / self._alpha)
            elif self._gendice_loss_type == 'bestdice':
                sampled_log_w_e = self._log_r_fn((sampled_e_st_mu - self._lamb_e_st.detach()) / self._alpha)
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
            'sampled_action_log_prob': torch.mean(sampled_action_log_prob),
            'negative_entropy_loss': negative_entropy_loss
        })

        return result

    def data_policy_loss(self, observation, state, action, result={}):
        # Compute data policy loss
        _, _, data_sampled_action_log_prob, _, data_policy_dists = self._data_policy_network(state)

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
        observation, state, action, next_observation, next_state, initial_observation, initial_state, reward, terminal = \
            batch["observations"], batch["states"], batch["actions"],  batch["next_observations"], batch["next_states"], \
                batch["initial_observations"], batch["initial_states"], batch["rewards"], batch["terminals"]
        
        reward = reward * self._reward_scale

        ####################################################################################################################
        # Shared network values
        ob_initial_v_values = self._v_ob_network(initial_observation)#; initial_v_values = initial_v_values * l_hat_init_v
        ob_v_values = self._v_ob_network(observation)#; v_values = v_values * l_hat_v
        ob_next_v_values  = self._v_ob_network(next_observation)#; next_v_values = next_v_values * l_hat_next_v

        st_initial_v_values = self._v_st_network(initial_state)#; initial_v_values = initial_v_values * l_hat_init_v
        st_v_values = self._v_st_network(state)#; v_values = v_values * l_hat_v
        st_next_v_values  = self._v_st_network(next_state)#; next_v_values = next_v_values * l_hat_next_v

        e_v_ob = reward + (1 - terminal) * self._gamma * ob_next_v_values - ob_v_values
        preactivation_v_ob = (e_v_ob - self._lamb_scale * self._lamb_v_ob) / self._alpha
        w_v_ob = self._r_fn(preactivation_v_ob)
        f_w_v_ob = self._g_fn(preactivation_v_ob)

        e_v_st = reward + (1 - terminal) * self._gamma * st_next_v_values - st_v_values
        preactivation_v_st = (e_v_st - self._lamb_scale * self._lamb_v_st) / self._alpha
        w_v_st = self._r_fn(preactivation_v_st)
        f_w_v_st = self._g_fn(preactivation_v_st)

        if self._gendice_v:
            preactivation_v_ob_lamb = (e_v_ob.detach() - self._lamb_scale * self._lamb_v_ob) / self._alpha
            w_v_ob_lamb = self._r_fn(preactivation_v_ob_lamb)
            f_w_v_ob_lamb = self._g_fn(preactivation_v_ob_lamb)

            preactivation_v_st_lamb = (e_v_st.detach() - self._lamb_scale * self._lamb_v_st) / self._alpha
            w_v_st_lamb = self._r_fn(preactivation_v_st_lamb)
            f_w_v_st_lamb = self._g_fn(preactivation_v_st_lamb)
        
        ####################################################################################################################

        e_ob_values, _, _ = self._e_ob_network(observation, action)#; e_values = e_values * l_hat_e
        preactivation_e_ob = (e_ob_values - self._lamb_scale * self._lamb_e_ob) / self._alpha #* self._std_penalty(l_hat_e.detach(), e_mu.detach())
        w_e_ob = self._r_fn(preactivation_e_ob)
        f_w_e_ob = self._g_fn(preactivation_e_ob)

        e_st_values, _, _ = self._e_st_network(state, action)#; e_values = e_values * l_hat_e
        preactivation_e_st = (e_st_values - self._lamb_scale * self._lamb_e_st) / self._alpha #* self._std_penalty(l_hat_e.detach(), e_mu.detach())
        w_e_st = self._r_fn(preactivation_e_st)
        f_w_e_st = self._g_fn(preactivation_e_st)
        
        if self._gendice_e:
            preactivation_e_ob_lamb = (e_ob_values.detach() - self._lamb_scale * self._lamb_e_ob) / self._alpha # * self._std_penalty(l_hat_e.detach(), e_mu.detach())
            w_e_ob_lamb = self._r_fn(preactivation_e_ob_lamb)
            f_w_e_ob_lamb = self._g_fn(preactivation_e_ob_lamb)
            
            preactivation_e_st_lamb = (e_st_values.detach() - self._lamb_scale * self._lamb_e_st) / self._alpha # * self._std_penalty(l_hat_e.detach(), e_mu.detach())
            w_e_st_lamb = self._r_fn(preactivation_e_st_lamb)
            f_w_e_st_lamb = self._g_fn(preactivation_e_st_lamb)
        
        ####################################################################################################################
        ####################################################################################################################
        # Compute loss and optimize (observation first)
        loss_result = self.v_loss(ob_initial_v_values, e_v_ob, w_v_ob, f_w_v_ob, self._lamb_v_ob, result={})

        self._optimizers['v_ob'].zero_grad()
        v_loss = loss_result['v_loss']; v_loss.backward()
        self._optimizers['v_ob'].step()

        if self._gendice_v:
            loss_result.update(self.lamb_v_loss(e_v_ob.detach(), w_v_ob_lamb, f_w_v_ob_lamb, self._lamb_v_ob))

            self._optimizers['lamb_v_ob'].zero_grad()
            lamb_v_loss = loss_result['lamb_v_loss']; lamb_v_loss.backward()
            self._optimizers['lamb_v_ob'].step()
        
        ####################################################################################################################

        loss_result.update(self.e_loss(e_v_ob.detach(), e_ob_values, w_e_ob, f_w_e_ob, self._lamb_e_ob))

        self._optimizers['e_ob'].zero_grad()
        e_loss = loss_result['e_loss']; e_loss.backward()
        self._optimizers['e_ob'].step()

        if self._gendice_e:
            loss_result.update(self.lamb_e_loss(e_v_ob.detach(), w_e_ob_lamb, f_w_e_ob_lamb, self._lamb_e_ob))
            self._optimizers['lamb_e_ob'].zero_grad()
            lamb_e_loss = loss_result['lamb_e_loss']; lamb_e_loss.backward()
            self._optimizers['lamb_e_ob'].step()

        ####################################################################################################################
        ####################################################################################################################
        # Compute loss and optimize (state)
        loss_result = self.v_loss(st_initial_v_values, e_v_st, w_v_st, f_w_v_st, self._lamb_v_st, result={})

        self._optimizers['v_st'].zero_grad()
        v_loss = loss_result['v_loss']; v_loss.backward()
        self._optimizers['v_st'].step()

        if self._gendice_v:
            loss_result.update(self.lamb_v_loss(e_v_st.detach(), w_v_st_lamb, f_w_v_st_lamb, self._lamb_v_st))

            self._optimizers['lamb_v_st'].zero_grad()
            lamb_v_loss = loss_result['lamb_v_loss']; lamb_v_loss.backward()
            self._optimizers['lamb_v_st'].step()
        
        ####################################################################################################################

        loss_result.update(self.e_loss(e_v_st.detach(), e_st_values, w_e_st, f_w_e_st, self._lamb_e_st))

        self._optimizers['e_st'].zero_grad()
        e_loss = loss_result['e_loss']; e_loss.backward()
        self._optimizers['e_st'].step()

        if self._gendice_e:
            loss_result.update(self.lamb_e_loss(e_v_st.detach(), w_e_st_lamb, f_w_e_st_lamb, self._lamb_e_st))
            self._optimizers['lamb_e_st'].zero_grad()
            lamb_e_loss = loss_result['lamb_e_loss']; lamb_e_loss.backward()
            self._optimizers['lamb_e_st'].step()

        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################

        loss_result.update(self.policy_loss(observation, state, action, w_e_st.detach()))
        
        self._optimizers['policy'].zero_grad()
        policy_loss = loss_result['policy_loss']; policy_loss.backward()
        self._optimizers['policy'].step()

        if self._use_policy_entropy_constraint: 
            self._optimizers['ent_coeff'].zero_grad()
            ent_coeff_loss = loss_result['ent_coeff_loss']; ent_coeff_loss.backward()
            self._optimizers['ent_coeff'].step()


        if self._policy_extraction == 'iproj':
            loss_result.update(self.data_policy_loss(observation, state, action))
            
            self._optimizers['data_policy'].zero_grad()
            data_policy_loss = loss_result['data_policy_loss']; data_policy_loss.backward()
            self._optimizers['data_policy'].step()

            if self._use_data_policy_entropy_constraint:
                self._optimizers['data_ent_coeff'].zero_grad()
                data_ent_coeff_loss = loss_result['data_ent_coeff_loss']; data_ent_coeff_loss.backward()
                self._optimizers['data_ent_coeff'].step()

        ####################################################################################################################
        loss_result.update({"e_values error": torch.mean(e_ob_values - e_st_values)})
        #loss_result.update({"e_st_values": torch.mean(l_hat_e)})
        #loss_result.update({"std/e_95": torch.quantile(l_hat_e, 0.95, interpolation='linear')})
        #loss_result.update({"std/e_max": torch.max(l_hat_e)})
        #loss_result.update({"std/std": torch.mean(self._std_fn(l_hat_e.detach(), e_mu.detach()))})
        #loss_result.update({"std/penalty": torch.mean(self._std_penalty(l_hat_e.detach(), e_mu.detach()))})
        #loss_result.update({"l_hat_e": torch.quantile(l_hat_e, 0.8, interpolation='linear')})

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

