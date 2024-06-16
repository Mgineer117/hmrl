import numpy as np
            
def load_reward_fn(key, num_task=3):
    """
    Returns the reward function to penalize the agent with given state, action, and next state.
    This reward function is designed to reflect the realworld constraints to maintain the 
    robot's lifespan and to behave in rough terrain (e.g. ceiling). Currently only Gym is supported.

    Args:
        s: previous state
        a: action
        ns: next state

    Returns:
        reward functions with its own realistic constraints.
    """

    if key == 'Gym-Ant':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, healthy_reward):
                reward = 0.0
                reward += x_velocity if x_velocity < self.param else 0.0
                reward += healthy_reward
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-HalfCheetah':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, forward_weight):
                reward = 0.0
                reward += forward_weight * x_velocity if x_velocity < self.param else 0.0
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Hopper':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, forward_weight, healthy_reward):
                reward = 0.0
                penalty = 0.5 if x_velocity > self.param else 1.0
                reward += forward_weight * x_velocity * penalty
                reward += healthy_reward 
                return reward

        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(0.5, 1.5)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(2.5, 3.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Humanoid-Standup':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, uph_cost, quad_ctrl_cost, quad_impact_cost):
                reward = 0.0
                reward += uph_cost if uph_cost < self.param else 0.0
                reward -= quad_ctrl_cost
                reward -= quad_impact_cost
                reward += 1.0
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Humanoid':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, forward_weight, healthy_reward):
                reward = 0.0
                reward += forward_weight * x_velocity if x_velocity < self.param else 0.0
                reward += healthy_reward 
                return reward

        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-InvertedDoublePendulum':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, observation, reward):
                reward = self.param * reward
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-InvertedPendulum':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, observation, reward):
                reward = self.param * reward
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Reacher':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, reward_dist, reward_ctrl):
                reward = reward_dist + self.param * reward_ctrl
                return reward
        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Swimmer':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, forward_weight):
                reward = 0.0
                reward += forward_weight * x_velocity if x_velocity < self.param else 0.0
                return reward

        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    elif key == 'Gym-Walker':
        class reward_fn:
            def __init__(self, param):
                # dists include the vel_limit params
                self.param = param

            def __call__(self, x_velocity, forward_weight, healthy_reward):
                reward = 0.0
                reward += forward_weight * x_velocity if x_velocity < self.param else 0.0
                reward += healthy_reward 
                return reward

        reward_fn_list = []
        for _ in range(num_task):
            param = np.random.uniform(1.0, 2.0)
            reward_fn_list.append(reward_fn(param))
        param = np.random.uniform(3.0, 4.0)
        reward_fn_list.append(reward_fn(param))
    else:
        NotImplementedError
    
    return reward_fn_list