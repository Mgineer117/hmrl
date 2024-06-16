import numpy as np
import random
import h5py
import os
import gym
import d4rl

from rlkit.utils.load_env import load_metagym_env, load_gym_env
from rlkit.buffer import ReplayBuffer

METAGYM_ENV_TRAIN_NAME = [
    'basketball',
    'window-open',
    'door-open',
    'peg-insert-side',
    'sweep',
    'drawer-close',
    'pick-place',
    'reach',
    'button-press-topdown',
    'push',
]
METAGYM_ENV_TEST_NAME = [
    'door-close',
    'shelf-place',
    'drawer-open',
    'sweep-into',
    'lever-pull'
]

def collect_gym_buffers(args):
    buffers = []

    if args.normalize_obs or args.normalize_rewards:
        print('Warning: Normalization is not recomendded for multi-envs learning!!!')

    for i in range(args.num_task + 1):
        if i == args.num_task:
            # test data
            file_name = args.agent_type + '_test' + '.h5py'
            hfile_dir = os.path.join('data', args.env_type, args.agent_type, 'test', file_name)
        else:
            # train data
            file_name = args.agent_type + '_train_' + str(i) + '.h5py'
            hfile_dir = os.path.join('data', args.env_type, args.agent_type, 'train', file_name)
        
        with h5py.File(hfile_dir, 'r') as file:
            data = {}
            a_group_key = list(file.keys())
            for key in a_group_key:
                data[key] = file[key][...]

        args.obs_shape = (data['observations'].shape[-1],)
        args.action_dim = data['actions'].shape[-1]
        args.target_entropy = -args.action_dim # proposed by SAC (Haarnoja et al., 2018) (−dim(A) for each task).

        buffer = ReplayBuffer(
            buffer_size=len(data["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            obs_norm=args.normalize_obs,
            rew_norm=args.normalize_rewards,
            device=args.device
        )
        buffer.load_dataset(data)
        buffers.append(buffer)

    training_buffers = buffers[:-1]
    testing_buffer = buffers[-1]

    _, eval_env, eval_idx = load_gym_env(args.task)
    args.max_action = eval_env.action_space.high[0]
    
    return training_buffers, testing_buffer, eval_env, eval_idx

def collect_metagym_buffers(args):
    buffers = []
    args.task_num = int(args.agent_type[-2:])

    if args.normalize_obs or args.normalize_rewards:
        print('Warning: Normalization is not recomendded for multi-envs learning!!!')

    for i in range(args.task_num + 1):
        if i == args.task_num:
            # test data
            if args.agent_type == 'ML10':
                eval_idx = random.randint(0, len(METAGYM_ENV_TEST_NAME))
                args.test_task = METAGYM_ENV_TEST_NAME[eval_idx]
            elif args.agent_type == 'MT10':
                eval_idx = random.randint(0, len(METAGYM_ENV_TRAIN_NAME))
                args.test_task = METAGYM_ENV_TRAIN_NAME[eval_idx]
            file_name = args.test_task  + '.h5py'
        else:
            # train data
            file_name = METAGYM_ENV_TRAIN_NAME[i] + '.h5py'
        hfile_dir = os.path.join('data', args.env_type, file_name)
        
        with h5py.File(hfile_dir, 'r') as file:
            data = {}
            a_group_key = list(file.keys())
            for key in a_group_key:
                data[key] = file[key][...]

        args.obs_shape = (data['observations'].shape[-1],)
        args.action_dim = data['actions'].shape[-1]
        args.target_entropy = -args.action_dim # proposed by SAC (Haarnoja et al., 2018) (−dim(A) for each task).

        buffer = ReplayBuffer(
            buffer_size=len(data["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            obs_norm=args.normalize_obs,
            rew_norm=args.normalize_rewards,
            device=args.device
        )
        buffer.load_dataset(data)
        buffers.append(buffer)

    training_buffers = buffers[:-1]
    testing_buffer = buffers[-1]    

    test_task_address = '-'.join((args.env_type, 'MT1'))
    _, eval_env, _ = load_metagym_env(test_task_address, args.test_task, 1)
    args.max_action = eval_env.action_space.high[0]

    return training_buffers, testing_buffer, eval_env, eval_idx

def collect_d4rl_buffers(args):
    task_name = '-'.join((args.agent_type, args.task_name, 'v2')) 

    env = gym.make(task_name)
    dataset = d4rl.qlearning_dataset(env)
    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    args.target_entropy = -args.action_dim # proposed by SAC (Haarnoja et al., 2018) (−dim(A) for each task).
    
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        obs_norm=args.normalize_obs,
        rew_norm=args.normalize_rewards,
        device=args.device
    )
    buffer.load_dataset(dataset)

    eval_idx = None
    return [buffer], buffer, env, eval_idx