import numpy as np
import gym
import metaworld
import random

def load_gym_env(key, reward_fn=None, cost_fn=None, render_mode: str = 'rgb_array'):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """

    # Gym
    # Meta Gym
    '''
    Format: [MetaGym or Gym]-[Agent(MT1 or hopper)]
    optional(task) as an input
    '''
    if key == 'Gym-Ant':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Ant-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    if self._use_contact_forces:
                        penalties += self.contact_cost
                    reward = self.custom_reward_fn(info['x_velocity'], self.healthy_reward)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0
                
                return observation, reward, terminated, truncated, info       
    elif key == 'Gym-HalfCheetah':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('HalfCheetah-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    forward_reward_weight = 1.0
                    reward =  self.custom_reward_fn(info['x_velocity'], forward_reward_weight)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                if self.render_mode == "human":
                    self.render()
                return observation, reward, terminated, truncated, info         
    elif key == 'Gym-Hopper':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Hopper-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    forward_reward_weight = 1.0
                    reward = self.custom_reward_fn(info['x_velocity'], forward_reward_weight, self.healthy_reward)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info
    elif key == 'Gym-Humanoid-Standup':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('HumanoidStandup-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    reward =  self.custom_reward_fn(info['reward_linup'], info['reward_quadctrl'], info['reward_impact'])
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info          
    elif key == 'Gym-Humanoid':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Humanoid-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn
            
            def mass_center(self, model, data):
                mass = np.expand_dims(model.body_mass, axis=1)
                xpos = data.xipos
                return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()
            
            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    forward_reward_weight = 1.25
                    reward =  self.custom_reward_fn(info['x_velocity'], forward_reward_weight, self.healthy_reward)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info         
    elif key == 'Gym-InvertedDoublePendulum':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('InvertedDoublePendulum-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    reward =  self.custom_reward_fn(observation, reward)
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info          
    elif key == 'Gym-InvertedPendulum':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('InvertedPendulum-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    reward =  self.custom_reward_fn(observation, reward)
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info    
    elif key == 'Gym-Reacher':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Reacher-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    reward =  self.custom_reward_fn(info['reward_dist'], info['reward_ctrl'])
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info
    elif key == 'Gym-Swimmer':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Swimmer-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    forward_reward_weight = 1.0
                    reward =  self.custom_reward_fn(info['x_velocity'], forward_reward_weight)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info
    elif key == 'Gym-Walker':
        class Env(gym.Wrapper):
            def __init__(self, reward_fn=None, cost_fn=None, render_mode='rgb_array'):
                env = gym.make('Walker2d-v4', render_mode=render_mode)
                super().__init__(env)
                self.custom_reward_fn = reward_fn
                self.custom_cost_fn = cost_fn

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)

                if self.custom_reward_fn is not None:
                    penalties = self.control_cost(action)
                    forward_reward_weight = 1.0
                    reward =  self.custom_reward_fn(info['x_velocity'], forward_reward_weight, self.healthy_reward)
                    reward -= penalties
                cost = self.custom_cost_fn(observation, action) if self.custom_cost_fn is not None else 0.0

                info["cost"] = cost
                info["success"] = 0.0

                return observation, reward, terminated, truncated, info

    if isinstance(reward_fn, list):
        envs = []
        for rew_fn in reward_fn:
            envs.append(Env(rew_fn, cost_fn, render_mode)) 
        training_envs = envs[:-1]
        testing_envs = envs[-1]
    else:
        training_envs = [Env(reward_fn, cost_fn, render_mode)]
        testing_envs = training_envs[0]
    
    return training_envs, testing_envs, None

def load_metagym_env(key, task: str = None, task_num: int = None, render_mode: str = 'rgb_array'):
    if task is not None:
        task_name = '-'.join((task, 'v2'))

    if key == 'MetaGym-ML1':
        assert task is not None
        assert task_num is not None
        ml = metaworld.ML1(task_name)
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env.task_name = task.env_name
            training_envs.append(env)
        testing_envs = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env.task_name = task.env_name
            testing_envs.append(env)
        eval_env_idx = random.choice(range(len(testing_envs)))
        testing_envs = testing_envs[eval_env_idx]
    elif key == 'MetaGym-MT1':
        assert task is not None
        assert task_num is not None
        ml = metaworld.MT1(task_name)
        tasks = random.sample(ml.train_tasks, task_num)
        training_envs = []
        for task in tasks:
            env = ml.train_classes[task_name](render_mode=render_mode)  # Create an environment with task `pick_place`
            env.set_task(task)
            env.task_name = task.env_name
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]
    elif key == 'MetaGym-ML10':
        ml = metaworld.ML10()
        training_envs = []
        training_tasks = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env.task_name = task.env_name
            training_envs.append(env)
            training_tasks.append(task)
        testing_envs = []
        testing_tasks = []
        for name, env_cls in ml.test_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.test_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env.task_name = task.env_name
            testing_envs.append(env)
            testing_tasks.append(task)
        eval_env_idx = random.choice(range(len(testing_envs)))
        testing_envs = testing_envs[eval_env_idx]
    elif key == 'MetaGym-MT10':
        ml = metaworld.MT10()
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls(render_mode=render_mode)
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env.task_name = task.env_name
            training_envs.append(env)
        eval_env_idx = random.choice(range(len(training_envs)))
        testing_envs = training_envs[eval_env_idx]

    return training_envs, testing_envs, eval_env_idx