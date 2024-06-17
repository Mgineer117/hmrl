import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple, Dict

class TrajectoryBuffer:
    def __init__(
        self,
        max_num_trj: int,
        max_trajectory_length: int,
        state_shape: Tuple[int],
        action_shape: Tuple[int],
        device: str = "cpu"
    ) -> None:
        self.max_num_trj = max_num_trj
        self.max_trajectory_length = max_trajectory_length
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = torch.device(device)
        
        # Preallocate arrays for trajectories
        self.states = np.zeros((max_num_trj, max_trajectory_length) + state_shape, dtype=np.float32)
        self.actions = np.zeros((max_num_trj, max_trajectory_length) + action_shape, dtype=np.float32)
        self.next_states = np.zeros((max_num_trj, max_trajectory_length) + state_shape, dtype=np.float32)
        self.rewards = np.zeros((max_num_trj, max_trajectory_length, 1), dtype=np.float32)
        self.masks = np.zeros((max_num_trj, max_trajectory_length, 1), dtype=np.int32)
        
        # Track current number of trajectories and their lengths
        self.num_trj = 0
        self.updating_trj_idx = 0
    
    def decompose(self, states, actions, next_states, rewards, masks):
        trajs = []
        prev_i = 0
        for i, mask in enumerate(masks.squeeze()):
            if mask == 0:
                data = {
                    "states": states[prev_i:i+1, :],#.numpy()
                    "actions": actions[prev_i:i+1, :],#.numpy()
                    "next_states": next_states[prev_i:i+1, :],#.numpy()
                    "rewards": rewards[prev_i:i+1, :],#.numpy()
                    "masks": masks[prev_i:i+1, :],#.numpy()
                }
                trajs.append(data)
                prev_i = i + 1
        return trajs

    def push(
        self,
        batch: dict
    ) -> None:
        state, action, next_state, reward, mask = \
            batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['masks']
        trajs = self.decompose(state, action, next_state, reward, mask)

        for traj in trajs:
            if self.updating_trj_idx >= self.max_num_trj:
                self.updating_trj_idx = 0
            
            traj_length = traj['states'].shape[0]
            self.states[self.updating_trj_idx, :traj_length, :] = traj['states']
            self.actions[self.updating_trj_idx, :traj_length, :] = traj['actions']
            self.next_states[self.updating_trj_idx, :traj_length, :] = traj['next_states']
            self.rewards[self.updating_trj_idx, :traj_length, :] = traj['rewards']
            self.masks[self.updating_trj_idx, :traj_length, :] = traj['masks']
            self.updating_trj_idx += 1
            
            if self.num_trj < self.max_num_trj:
                self.num_trj += 1

    def sample(
        self,
        num_traj: int
    ) -> Dict[str, torch.Tensor]:
        if num_traj > self.num_trj:
            num_traj = self.num_trj
        
        # Sample random trajectories
        sampled_trajs = np.random.choice(self.num_trj, num_traj, replace=False)
        
        # Collect sampled data
        sampled_states = self.states[sampled_trajs, :self.max_trajectory_length]
        sampled_actions = self.actions[sampled_trajs, :self.max_trajectory_length]
        sampled_next_states = self.next_states[sampled_trajs, :self.max_trajectory_length]
        sampled_rewards = self.rewards[sampled_trajs, :self.max_trajectory_length]
        sampled_masks = self.masks[sampled_trajs, :self.max_trajectory_length]

        '''refine trj (removing 0) and concatenate'''
        for i in range(len(sampled_trajs)):
            mask_idx = np.where(sampled_masks[i] == 0)[0][0]
            if i == 0:
                states = sampled_states[i, :mask_idx+1]
                actions = sampled_actions[i, :mask_idx+1]
                next_states = sampled_next_states[i, :mask_idx+1]
                rewards = sampled_rewards[i, :mask_idx+1]
                masks = sampled_masks[i, :mask_idx+1]
            else:
                states = np.concatenate((states, sampled_states[i, :mask_idx+1]), axis=0)
                actions = np.concatenate((actions, sampled_actions[i, :mask_idx+1]), axis=0)
                next_states = np.concatenate((next_states, sampled_next_states[i, :mask_idx+1]), axis=0)
                rewards = np.concatenate((rewards, sampled_rewards[i, :mask_idx+1]), axis=0)
                masks = np.concatenate((masks, sampled_masks[i, :mask_idx+1]), axis=0)
        
        # Convert to Torch tensors and move to device
        return (states, actions, next_states, rewards, masks)
