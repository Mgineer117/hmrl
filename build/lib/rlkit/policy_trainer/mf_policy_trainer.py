import time
import os

import gym
import cv2
import numpy as np
import torch
import gym
import wandb
from copy import deepcopy

import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from rlkit.buffer import ReplayBuffer, OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        eval_env_idx: int,
        logger: WandbLogger,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        init_step_per_epoch: int = 0,
        local_steps: int = 3,
        batch_size: int = 256,
        num_trj: int = 0,
        eval_episodes: int = 10,
        rendering: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_buffers: ReplayBuffer = None,
        testing_buffer: ReplayBuffer = None,
        sampler: OnlineSampler = None,
        obs_dim: int = None,
        action_dim: int = None,
        embed_dim: int = None,
        cost_limit: float = 0.0,
        log_interval: int = 20,
        visualize_latent_space:bool = False,
        device=None,
    ) -> None:
        self.policy = policy
        self.training_buffers = training_buffers
        self.testing_buffer = testing_buffer
        self.sampler = sampler
        self.eval_env = eval_env
        self.eval_env_idx = eval_env_idx
        self.logger = logger

        self._epoch = epoch
        self._init_epoch = init_epoch
        self._step_per_epoch = step_per_epoch
        self._init_step_per_epoch = init_step_per_epoch
        self._local_steps = local_steps
        self._batch_size = batch_size
        self._num_trj = num_trj
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.obs_dim = obs_dim
        self.action_dim = action_dim        
        self.embed_dim = embed_dim

        self._device = device
        
        self.last_max_reward = 0.0
        self.cost_limit = cost_limit

        self.current_epoch = 0
        self.log_interval = log_interval
        self.rendering = rendering
        self.visualize_latent_space = visualize_latent_space
        self.latent_path = None
        if self.visualize_latent_space:
            directory = os.path.join(self.logger.checkpoint_dir, 'latent')
            os.makedirs(directory)
        self.recorded_frames = []

    def train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()

            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batches = []
                for buffer in self.training_buffers:
                    batches.append(buffer.sample(self._batch_size, self._num_trj))
                batches = self.aggregate_batches(batches)
                loss = self.policy.learn(batches)
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

        self.logger.store(**eval_data)        
        self.logger.write(int(e*self._step_per_epoch + it), display=False)
        
        # save checkpoint
        if self.current_epoch % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dir, e, self.sampler.running_state)
        # save the best model
        if np.mean(last_10_performance) >= self.last_max_reward and np.mean(last_10_performance) <= self.cost_limit:
            self.policy.save_model(self.logger.log_dir, e, self.sampler.running_state, is_best=True)
            self.last_max_reward = np.mean(last_10_performance)
        
        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def online_train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        last_10_reward_performance = deque(maxlen=10)
        last_10_cost_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._init_epoch, self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()
            
            for it in trange(self._init_step_per_epoch, self._step_per_epoch, desc=f"Training", leave=False):
                if self.visualize_latent_space and self.embed_dim > 0:
                    self.save_latent_space(e, it)

                batch, sample_time = self.sampler.collect_samples(self.policy, seed, latent_path=self.latent_path)
                loss = self.policy.learn(batch); loss['sample_time'] = sample_time
                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate(seed)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_cost_mean, ep_cost_std = np.mean(eval_info["eval/episode_cost"]), np.std(eval_info["eval/episode_cost"])
            ep_success_rate_mean, ep_success_rate_std = np.mean(eval_info["eval/episode_success_rate"]), np.std(eval_info["eval/episode_success_rate"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

            eval_data = {"eval/episode_reward": ep_reward_mean,
                         "eval/episode_reward_std": ep_reward_std,
                         "eval/episode_cost": ep_cost_mean,
                         "eval/episode_cost_std": ep_cost_std,
                         "eval/ep_success_mean": ep_success_rate_mean,
                         "eval/ep_success_std": ep_success_rate_std,
                         "eval/episode_length": ep_length_mean,
                         "eval/episode_length_std": ep_length_std
                         }
            try:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                norm_ep_cost_mean = self.eval_env.get_normalized_score(ep_cost_mean) * 100
                norm_ep_cost_std = self.eval_env.get_normalized_score(ep_cost_std) * 100

                last_10_reward_performance.append(norm_ep_rew_mean)
                last_10_cost_performance.append(norm_ep_cost_mean)
                norm_data = {
                    "eval/normalized_episode_reward_std": norm_ep_rew_std,
                    "eval/normalized_episode_reward": norm_ep_rew_mean,
                    "eval/normalized_episode_cost_std": norm_ep_cost_mean,
                    "eval/normalized_episode_cost": norm_ep_cost_std,
                }
                eval_data.update(norm_data)
            except:
                last_10_reward_performance.append(ep_reward_mean)
                last_10_cost_performance.append(ep_cost_mean)
            self.logger.store(**eval_data)        
            self.logger.write(int(e*self._step_per_epoch + it), display=False)
            
            # save checkpoint
            if self.current_epoch % self.log_interval == 0:
                self.policy.save_model(self.logger.checkpoint_dir, e, self.sampler.running_state)
            # save the best model
            if np.mean(last_10_reward_performance) >= self.last_max_reward and np.mean(last_10_cost_performance) <= self.cost_limit:
                self.policy.save_model(self.logger.log_dir, e, self.sampler.running_state, is_best=True)
                self.last_max_reward = np.mean(last_10_reward_performance)
        
        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        return {"last_10_reward_performance": np.mean(last_10_reward_performance),
                "last_10_cost_performance": np.mean(last_10_cost_performance)}
    
    def normalize_obs(self, obs):
        if self.sampler.running_state is not None: # check if there is running state enabled
            self.sampler.running_state.fix = True
            obs = self.sampler.running_state(obs)
            self.sampler.running_state.fix = False
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
                s, _, e_s, _, _ = self.policy.encode_obs(mdp, env_idx=self.eval_env_idx, reset=True)

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
                success = infos['success']
                
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
                    _, ns, _, e_ns, _ = self.policy.encode_obs(mdp, env_idx=self.eval_env_idx)
                
                s = ns
                e_s = e_ns

                if done:
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_cost": episode_cost, "episode_length": episode_length, "episode_success_rate":episode_success/episode_length}
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
            "eval/episode_success_rate": [ep_info["episode_success_rate"] for ep_info in eval_ep_info_buffer],
        }
    
    def aggregate_batches(self, batches):
        memory = dict()
        for batch in batches:
            for key, value in batch.items():
                if key in memory:
                    memory[key] = torch.cat((memory[key], value), dim=0)
                else:
                    memory[key] = value
        return memory
    
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

    def save_latent_space(self, e, it):
        if e % self.log_interval == 0 and it == 0:
            self.latent_path = os.path.join(self.logger.checkpoint_dir, 'latent', str(self.current_epoch*self._step_per_epoch) +'.png')
        else:
            self.latent_path = None
