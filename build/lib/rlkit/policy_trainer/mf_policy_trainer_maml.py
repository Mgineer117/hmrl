import time
import os

import gym
import cv2
import numpy as np
import torch
import gym
import wandb
from copy import deepcopy
import torch.multiprocessing as multiprocessing

import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from rlkit.buffer import ReplayBuffer, OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder
from rlkit.utils.utils import set_flat_params_to

# model-free policy trainer
class MFMAMLPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        eval_env_idx: int,
        logger: WandbLogger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        local_steps: int = 3,
        batch_size: int = 256,
        num_trj: int = 0,
        eval_episodes: int = 10,
        rendering: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_buffers: ReplayBuffer = None,
        testing_buffer: ReplayBuffer = None,
        sampler: OnlineSampler = None,
        eval_sampler: OnlineSampler = None,
        obs_dim: int = None,
        action_dim: int = None,
        cost_limit: float = 0.0,
        device=None,
    ) -> None:
        self.policy = policy
        self.training_buffers = training_buffers
        self.testing_buffer = testing_buffer
        self.sampler = sampler
        self.eval_sampler = eval_sampler
        self.eval_env = eval_env
        self.eval_env_idx = eval_env_idx
        self.logger = logger

        self.param_size = sum(p.numel() for p in self.policy.actor.parameters())
        
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._local_steps = local_steps
        self._batch_size = batch_size
        self._num_trj = num_trj
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.obs_dim = obs_dim
        self.action_dim = action_dim        

        self.device = device
        
        self.last_max_reward = 0.0
        self.cost_limit = cost_limit

        self.current_epoch = 0
        self.log_interval = 20
        self.rendering = rendering
        self.recorded_frames = []

        self.draw_maml = False
        if self.draw_maml:
            self.meta_loc = [0., 0.]
            self.local_loc = torch.zeros((len(self.sampler), self._local_steps + 2, 2))
            plt.xscale('log')
            plt.yscale('log')
    
    def maml_train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()

            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                local_policy_list = [deepcopy(self.policy) for _ in range(len(self.sampler))]
                param_gradients = [torch.ones(self.param_size) for _ in range(len(self.sampler))]
                queue = multiprocessing.Manager().Queue()

                '''Begin K+1 update loop'''
                for k in trange(self._local_steps, desc=f"Local-update", leave=False):
                    '''Sample multiple batches'''
                    batches = [None] * len(self.training_buffers)
                    for i, buffer in enumerate(self.training_buffers):
                        batches[i] = buffer.sample(batch_size=self._batch_size, num_trj=self._num_trj)

                    '''Gradient local computing''' # it updates local_policy and param_gradients
                    _ = self.update_rule(local_policy_list, queue, batches, param_gradients, k)
                
                '''Sample multiple batches'''
                batches = [None] * len(self.training_buffers)
                for i, buffer in enumerate(self.training_buffers):
                    batches[i] = buffer.sample(batch_size=self._batch_size, num_trj=self._num_trj)

                '''Gradient local computing''' # it updates local_policy and param_gradients
                loss = self.update_rule(local_policy_list, queue, batches, param_gradients, k)

                average_meta_grad = torch.mean(torch.stack(param_gradients), axis=0)
                self.meta_update(batches, average_meta_grad)

                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate(seed)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_success_mean, ep_success_std = np.mean(eval_info["eval/episode_success"]), np.std(eval_info["eval/episode_success"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            
            eval_data = {"eval/episode_reward": ep_reward_mean,
                         "eval/episode_reward_std": ep_reward_std,
                         "eval/ep_success_mean": ep_success_mean,
                         "eval/ep_success_std": ep_success_std,
                         "eval/episode_length": ep_length_mean,
                         "eval/episode_length_std": ep_length_std
                         }
            try:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                norm_data = {
                    "eval/normalized_episode_reward_std": norm_ep_rew_std,
                    "eval/normalized_episode_reward": norm_ep_rew_mean,
                }
                eval_data.update(norm_data)
            except:
                last_10_performance.append(ep_reward_mean)
            self.logger.store(**eval_data)        
            self.logger.write(int(e*self._step_per_epoch + it), display=False)
            
            # save checkpoint
            if self.current_epoch % self.log_interval == 0:
                self.policy.save_model(self.logger.checkpoint_dir, e, self.sampler.running_state)
            # save the best model
            if np.mean(last_10_performance) >= self.last_max_reward:
                self.policy.save_model(self.logger.log_dir, e, self.sampler.running_state, is_best=True)
                self.last_max_reward = np.mean(last_10_performance)

        '''
        meta-test
        '''

        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.log_dir, "policy.pth"))
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def maml_online_train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()
            # one iteration
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                local_policy_list = [deepcopy(self.policy) for _ in range(len(self.sampler))]
                param_gradients = [torch.ones(self.param_size) for _ in range(len(self.sampler))]
                queue = multiprocessing.Manager().Queue()

                '''Begin K+1 update loop'''
                total_sample_time = 0
                for k in trange(self._local_steps, desc=f"Local-update", leave=False):
                    '''Sample multiple batches'''
                    batches, sample_time = self.collect_multiple_batches(local_policy_list, queue, seed)
                    total_sample_time += sample_time

                    '''Gradient local computing''' # it updates local_policy and param_gradients
                    _ = self.update_rule(local_policy_list, queue, batches, param_gradients, k)
                    
                '''Last k+1 update to find the update gradient'''
                batches, sample_time = self.collect_multiple_batches(local_policy_list, queue, seed)
                total_sample_time += sample_time

                '''Gradient local computing'''
                loss = self.update_rule(local_policy_list, queue, batches, param_gradients, k+1) # is_true multiplies first grad
                
                '''Meta-parameter update'''
                average_meta_grad = torch.mean(torch.stack(param_gradients), axis=0)
                self.meta_update(batches, average_meta_grad)

                if self.draw_maml:
                    meta_vec = self.do_PCA(average_meta_grad)
                    self.draw_on_figure(meta_vec, int(e*self._step_per_epoch + it))
                
                '''Logging'''
                loss['total_sample_time'] = total_sample_time
                loss['sample_time_per_k'] = total_sample_time / (self._local_steps + 1)
                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate(seed)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_success_mean, ep_success_std = np.mean(eval_info["eval/episode_success"]), np.std(eval_info["eval/episode_success"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            
            eval_data = {"eval/episode_reward": ep_reward_mean,
                         "eval/episode_reward_std": ep_reward_std,
                         "eval/ep_success_mean": ep_success_mean,
                         "eval/ep_success_std": ep_success_std,
                         "eval/episode_length": ep_length_mean,
                         "eval/episode_length_std": ep_length_std
                         }
            try:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                norm_data = {
                    "eval/normalized_episode_reward_std": norm_ep_rew_std,
                    "eval/normalized_episode_reward": norm_ep_rew_mean,
                }
                eval_data.update(norm_data)
            except:
                last_10_performance.append(ep_reward_mean)
            self.logger.store(**eval_data)        
            self.logger.write(int(e*self._step_per_epoch + it), display=False)

            # save checkpoint
            if self.current_epoch % self.log_interval == 0:
                self.policy.save_model(self.logger.checkpoint_dir, e)
            # save the best model
            if np.mean(last_10_performance) >= self.last_max_reward:
                self.policy.save_model(self.logger.log_dir, e, is_best=True)
                self.last_max_reward = np.mean(last_10_performance)

        '''meta-test'''
        self.meta_test_update(self.eval_sampler, self.policy, seed)

        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.log_dir, "policy.pth"))
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def collect_multiple_batches(self, local_policy_list, queue, seed):
        # local k updates
        total_sample_time = 0
        batches = [None] * len(self.sampler)
        processes = []
        # collect the corresponding batch for each local policy
        pid = 0
        for sampler, local_policy in zip(self.sampler, local_policy_list):
            worker_args = (local_policy, seed, False, pid, queue)
            p = multiprocessing.Process(target=sampler.collect_samples, args=worker_args)
            processes.append(p)
            p.start()
            pid += 1
        for p in processes:
            p.join()    
        # saving the result
        for _ in range(len(self.sampler)):
            pid, memory, sample_time = queue.get()
            batches[pid] = memory
            total_sample_time += sample_time
        return batches, total_sample_time

    def update_rule(self, local_policy_list, queue, batches, param_gradients, k):
        processes = []
        meta_loss = []
        # local parameter; one by one
        for i in range(len(self.sampler)): 
            worker_args = (i, queue, batches[i], local_policy_list[i])
            p = multiprocessing.Process(target=self.local_update, args=worker_args)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # saving the result
        for _ in range(len(self.sampler)):
            pid, fist_grad, second_grad, actor_param, critic_param, encoder_param, loss = queue.get()
            if k == self._local_steps:
                param_gradients[pid] *= fist_grad # 1 + second_grad = param_grad
            else:
                param_gradients[pid] *= 1 - second_grad # 1 + second_grad = param_grad
                set_flat_params_to(local_policy_list[pid].actor, actor_param)
                set_flat_params_to(local_policy_list[pid].critic, critic_param)
                if encoder_param is not None:
                    set_flat_params_to(local_policy_list[pid].encoder, encoder_param)
            meta_loss.append(loss)
            if self.draw_maml:
                # draw grad MAML
                projected_grad = self.do_PCA(fist_grad)
                self.local_loc[pid, k+1:, :] += projected_grad
        return self.average_dict(meta_loss)
            
    def local_update(self, pid, queue, batch, local_policy, compute_param_grad=True):
        loss, (first_grad, second_grad), (actor_param, critic_param, encoder_param) = local_policy.learn(batch, compute_param_grad)
        if queue is not None:
            queue.put([pid, first_grad, second_grad, actor_param, critic_param, encoder_param, loss])
        else:
            return first_grad, second_grad, loss
    
    def meta_update(self, batches, meta_grad):
        memory = self.aggregate_batches(batches)
        self.policy.learn_with_grad(memory, meta_grad)

    def meta_test_update(self, sampler, policy, seed):
        for e in trange(100, desc=f"Meta-test"):
            batch, sample_time = sampler.collect_samples(policy, seed)
            loss, _ = policy.learn(batch); loss['sample_time'] = sample_time
            loss = {f"meta_test/{key}": value for key, value in loss.items()}
            self.logger.store(**loss)        
            self.logger.write(int(e), display=False)
    
    def aggregate_batches(self, batches):
        memory = dict()
        for batch in batches:
            for key, value in batch.items():
                tensor = torch.from_numpy(value).to(self.device)
                if key in memory:
                    memory[key] = torch.cat((memory[key], tensor), dim=0)
                else:
                    memory[key] = tensor
        return memory

    def normalize_obs(self, obs):
        if self.eval_sampler.running_state is not None: # check if there is running state enabled
            self.eval_sampler.running_state.fix = True
            obs = self.eval_sampler.running_state(obs)
            self.eval_sampler.running_state.fix = False
        elif self.testing_buffer is not None: # check if it is offline training
            if self.testing_buffer._obs_normalized: # check if obs is normalized
                obs = (obs - self.testing_buffer.obs_mean) / (self.testing_buffer.obs_std + 1e-10)
        return obs
    
    def average_dict(self, dict_list):
        sums = {}
        counts = {}
        for d in dict_list:
            for key, value in d.items():
                if key in sums:
                    sums[key] += value
                    counts[key] += 1
                else:
                    sums[key] = value
                    counts[key] = 1
        averages = {key: sums[key] / counts[key] for key in sums}
        return averages
     
    def do_PCA(self, meta_grad):
        '''perform PCA'''
        meta_grad = meta_grad[None, :]
        mean = torch.mean(meta_grad, dim=1, keepdim=True)

        cov_matrix = torch.matmul((meta_grad - mean).T, (meta_grad - mean)) / (meta_grad.size(1) - 1)

        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        eigenvalues = torch.real(eigenvalues)
        eigenvectors = torch.real(eigenvectors)

        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        principal_component = sorted_eigenvectors[:, 0:2]

        projected_grad = torch.matmul(meta_grad, principal_component)
        return projected_grad

    def draw_on_figure(self, meta_vec, e):
        plt.arrow(self.meta_loc[0], self.meta_loc[1], meta_vec[0, 0], meta_vec[0, 1], 
          head_width=0.05, head_length=0.1, fc='black', ec='black')
        
        for i in range(len(self.sampler)):
            for j in range(self._local_steps):
                plt.arrow(self.local_loc[i, j, 0], self.local_loc[i, j, 1], self.local_loc[i,j+1,0]-self.local_loc[i,j,0], self.local_loc[i,j+1,1]-self.local_loc[i,j,1], 
                    head_width=0.05, head_length=0.1, fc='blue', ec='blue')

        self.meta_loc[0] += meta_vec[0, 0]
        self.meta_loc[1] += meta_vec[0, 1]
        self.local_loc[:, :, 0] = self.meta_loc[0]
        self.local_loc[:, :, 1] = self.meta_loc[1]

        plt.savefig(str(e) + '.png')

    def _evaluate(self, seed) -> Dict[str, List[float]]:
        self.policy.eval()
        num_episodes = 0

        while num_episodes < self._eval_episodes:
            try:
                s, _ = self.eval_env.reset(seed=seed)
            except:
                s = self.eval_env.reset(seed=seed)
            s = self.normalize_obs(s)
            a = np.zeros((self.action_dim, ))
            ns = s # initialization
            
            mdp = (s, a, ns, np.array([0]), np.array([1]))
            with torch.no_grad():
                s, _, e_s, _ = self.policy.encode_obs(mdp, env_idx=self.eval_env_idx, reset=True)

            eval_ep_info_buffer = []
            episode_reward, episode_cost, episode_length, episode_success = 0, 0, 0, 0

            done = False
            while not done:
                with torch.no_grad():
                    a, _ = self.policy.select_action(e_s, deterministic=True) #(obs).reshape(1,-1)
                try:
                    ns, rew, trunc, term, infos = self.eval_env.step(a.flatten()); cost = 0.0
                    done = term or trunc
                except:
                    ns, rew, term, infos = self.eval_env.step(a.flatten()); cost = 0.0
                    done = term
                #print(ns[8])
                try:
                    success = infos['success']
                except:
                    success = 0.0
                
                mask = 0 if done else 1
                
                if self.current_epoch % self.log_interval == 0:
                    if self.rendering and num_episodes == 0:
                        self.recorded_frames.append(self.eval_env.render())
                
                episode_reward += rew
                episode_cost += cost
                episode_success += success
                episode_length += 1
                
                # state encoding
                ns = self.normalize_obs(ns)

                mdp = (s, a, ns, np.array([rew]), np.array([mask]))
                with torch.no_grad():
                    _, ns, _, e_ns = self.policy.encode_obs(mdp, env_idx=self.eval_env_idx)

                s = ns
                e_s = e_ns

                if done:
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_cost": episode_cost, "episode_length": episode_length, "episode_success":episode_success/episode_length}
                    )
                    num_episodes +=1
                    episode_reward, episode_cost, episode_length = 0, 0, 0

        if self.current_epoch % self.log_interval == 0:
            if self.rendering:
                self.save_rendering(self.logger.checkpoint_dir)
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_cost": [ep_info["episode_cost"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_success": [ep_info["episode_success"] for ep_info in eval_ep_info_buffer],
        }
    
    def save_rendering(self, path):
        directory = os.path.join(path, 'video')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = 'rendering' + str(self.current_epoch*self._step_per_epoch) +'.avi'
        output_file = os.path.join(directory, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
        fps = 120
        width = 480
        height = 480
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        self.recorded_frames = []


