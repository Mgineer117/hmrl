import random
import time
import math
import h5py
import os
import torch.multiprocessing as multiprocessing

import torch
import numpy as np

from rlkit.utils.utils import visualize_latent_variable
from typing import Optional, Union, Tuple, Dict
from datetime import date
today = date.today()

def calculate_workers_and_rounds(environments, episodes_per_env, num_cores):
    if episodes_per_env == 1:
        num_worker_per_env = 1
    elif episodes_per_env >= 2:
        num_worker_per_env = episodes_per_env // 2
    
    # Calculate total number of workers
    total_num_workers = num_worker_per_env * len(environments)

    if total_num_workers > num_cores:
        rounds = math.ceil(total_num_workers / num_cores) 

        num_worker_per_round = []
        workers_remaining = total_num_workers
        for i in range(rounds):
            if workers_remaining >= num_cores:
                num_worker_per_round.append(num_cores)
                workers_remaining -= num_cores
            else:
                num_worker_per_round.append(workers_remaining)
                workers_remaining = 0
        num_env_per_round = [int(x / num_worker_per_env) for x in num_worker_per_round] #num_worker_per_round / num_worker_per_env
    else:
        rounds = 1
        num_worker_per_round = [total_num_workers]
        num_env_per_round = [len(environments)]
    
    episodes_per_worker = int(episodes_per_env * len(environments) / total_num_workers)
    return num_worker_per_round, num_env_per_round, episodes_per_worker, rounds

class OnlineSampler:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        episode_len: int,
        episode_num: int,
        training_envs: list,
        running_state = None,
        num_cores: int = None,
        data_num: int = None,

        device: str = "cpu"
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.episode_len = episode_len
        self.episode_num = episode_num
        self.training_envs = training_envs
        self.running_state = running_state
        self.data_num = data_num

        self.device = torch.device(device)

        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        self.num_cores = num_cores if num_cores is not None else multiprocessing.cpu_count() #torch.get_num_threads()
        num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = calculate_workers_and_rounds(self.training_envs, self.episode_num, self.num_cores)
        
        self.num_workers_per_round = num_workers_per_round
        self.num_env_per_round = num_env_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.num_worker_per_env = int(self.total_num_worker/len(self.training_envs))
        self.rounds = rounds

        print('Sampling Parameters:')
        print('--------------------')
        print(f'Core usage for this run           : {self.num_workers_per_round[0]}/{self.num_cores} | {multiprocessing.cpu_count()}')
        print(f'Number of Environments each Round : {self.num_env_per_round}')
        print(f'Total number of Worker            : {self.total_num_worker}')
        print(f'Episodes per Worker               : {self.episodes_per_worker}')
        torch.set_num_threads(1) # enforce one thread for each worker to avoid CPU overscription.

        if self.data_num is not None:
            # to create an enough batch..
            self.data_buffer = self.get_reset_data(2*self.data_num)
            self.buffer_last_idx = 0

    def save_buffer(self):
        for k in self.data_buffer:
            self.data_buffer[k] = self.data_buffer[k][:self.data_num]
        print('data saved!!')
        print('mean reward: ',np.mean(self.data_buffer['rewards']))
        print('mean cost: ',np.mean(self.data_buffer['costs']))
        hfile = h5py.File('data.h5py', 'w')
        for k in self.data_buffer:
            hfile.create_dataset(k, data=self.data_buffer[k], compression='gzip')
        hfile.close()

    def get_reset_data(self, batch_size):
        '''
        We create a initialization batch to avoid the daedlocking. 
        The remainder of zero arrays will be cut in the end.
        '''
        data = dict(
            states = np.zeros((batch_size, self.obs_dim)),
            next_states = np.zeros((batch_size, self.obs_dim)),
            actions = np.zeros((batch_size, self.action_dim)),
            embeddings = np.zeros((batch_size, self.embed_dim)),
            rewards = np.zeros((batch_size, 1)),
            terminals = np.zeros((batch_size, 1)),
            timeouts = np.zeros((batch_size, 1)),
            masks = np.zeros((batch_size, 1)),
            logprobs = np.zeros((batch_size, 1)),
            successes = np.zeros((batch_size, 1)),
        )
        return data
    
    def collect_trajectory(self, pid, queue, env, policy, thread_batch_size, episode_len,
                           episode_num, deterministic=False, seed=0):
        # estimate the batch size to hava a large batch
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)
        current_step = 0
        ep_num = 0
        while current_step < thread_batch_size:
            # break criteria
            if ep_num >= episode_num:
                break
            
            # var initialization
            _returns = 0
            t = 0 

            # env initialization
            s, _ = env.reset(seed=seed)

            # create mdp for encoding process. all element should have dimension (1,) than scaler
            a = np.zeros((self.action_dim, ))
            ns = s

            input_tuple = (s, a, ns, np.array([0]), np.array([1]))
            
            # begin the episodic loop by initializing the hidden info of LSTM
            policy.init_encoder_hidden_info()
            while t < episode_len:
                # sample action
                with torch.no_grad(): 
                    '''TO-DO // HOW WE RESET??'''
                    a, logprob, z = policy(input_tuple, deterministic=deterministic)
                    
                # env stepping
                ns, rew, term, trunc, infos = env.step(a)
                
                success = infos['success']
            
                done = trunc or term
                mask = 0 if done else 1

                # state encoding
                input_tuple = (s, a, ns, np.array([rew]), np.array([mask]))
                
                # saving the data
                data['states'][current_step+t, :] = s
                data['actions'][current_step+t, :] = a
                data['next_states'][current_step+t, :] = ns
                data['embeddings'][current_step+t, :] = z
                data['rewards'][current_step+t, :] = rew
                data['terminals'][current_step+t, :] = term
                data['timeouts'][current_step+t, :] = trunc
                data['masks'][current_step+t, :] = mask
                data['logprobs'][current_step+t, :] = logprob
                data['successes'][current_step+t, :] = success

                s = ns
                _returns += rew
                t += 1
    
                if done:        
                    # clear log
                    ep_num += 1
                    current_step += t
                    _returns = 0
                    break
                
        memory = dict(
            states=data['states'].astype(np.float32),
            actions=data['actions'].astype(np.float32),
            next_states=data['next_states'].astype(np.float32),
            embeddings=data['embeddings'].astype(np.float32),
            rewards=data['rewards'].astype(np.float32),
            terminals=data['terminals'].astype(np.int32),
            timeouts=data['timeouts'].astype(np.int32),
            masks=data['masks'].astype(np.int32),
            logprobs=data['logprobs'].astype(np.float32),
            successes=data['successes'].astype(np.float32),
        )
        if current_step < thread_batch_size:
            for k in memory:
                memory[k] = memory[k][:current_step]
        else:
            for k in memory:
                memory[k] = memory[k][:thread_batch_size]
        if queue is not None:
            queue.put([pid, memory])
        else:
            return memory

    def to_device(self, policy, device=torch.device('cpu')):
        policy = policy.to(device)
        policy.device = device
        policy.actor.device = device
        policy.encoder.device = device
        return policy

    def collect_samples(self, policy, seed, deterministic=False, pid=None, local_queue=None, latent_path=None):
        '''
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        '''
        t_start = time.time()
        policy.to_device()
        
        queue = multiprocessing.Manager().Queue()
        env_idx = 0
        worker_idx = 0

        for round_number in range(self.rounds):
            processes = []
            envs = self.training_envs[env_idx:env_idx+self.num_env_per_round[round_number]]
            for env in envs:
                workers_for_env = self.num_workers_per_round[round_number] // len(envs)
                for _ in range(workers_for_env):
                    if worker_idx == self.total_num_worker - 1:
                        '''Main thread process'''
                        memory = self.collect_trajectory(worker_idx, None, env, policy, self.thread_batch_size,
                                                         self.episode_len, self.episode_num, deterministic, seed)
                    else:
                        '''Sub-thread process'''
                        worker_args = (worker_idx, queue, env, policy, self.thread_batch_size, 
                                self.episode_len, self.episode_num, deterministic, seed)
                        p = multiprocessing.Process(target=self.collect_trajectory, args=worker_args)
                        processes.append(p)
                        p.start()
                    worker_idx += 1
                env_idx += 1
            for p in processes:
                p.join()        

        worker_memories = [None] * (worker_idx - 1)
        for _ in range(worker_idx - 1): 
            pid, worker_memory = queue.get()
            worker_memories[pid] = worker_memory
        
        if latent_path is not None:
            '''draw latent variable !!!'''
            latent_info = [worker_memories[i]['embeddings'] for i in range(self.num_worker_per_env-1, len(worker_memories), self.num_worker_per_env)]
            latent_info.append(memory['embeddings'])
            
            tasks_name = []
            for env in self.training_envs:
                try:
                    tasks_name.append(env.task_name)
                except:
                    tasks_name.append(env.unwrapped.spec.id)

            visualize_latent_variable(tasks_name, latent_info, latent_path)
            
        for worker_memory in worker_memories:
            for k in memory:
                memory[k] = np.concatenate((memory[k], worker_memory[k]), axis=0)
        if self.data_num is not None:
            memory_size = memory['observations'].shape[0]
            for k in memory:
                self.data_buffer[k][self.buffer_last_idx:self.buffer_last_idx+memory_size, :] = memory[k] 
            self.buffer_last_idx += memory_size
            if self.buffer_last_idx >= self.data_num:
                self.save_buffer()
                self.data_num = None

        policy.to_device(self.device)
        t_end = time.time()

        if local_queue is not None:
            return local_queue.put([pid, memory, t_end - t_start])
        else:
            return memory, t_end - t_start
