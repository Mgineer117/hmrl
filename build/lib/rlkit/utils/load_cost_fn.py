import numpy as np

def load_cost_fn(key):
    """
    Returns the cost function to penalize the agent with given state, action, and next state.
    This cost function is designed to reflect the realworld constraints to maintain the 
    robot's lifespan and to behave in rough terrain (e.g. ceiling). Currently only Gym is supported.

    Args:
        s: previous state
        a: action
        ns: next state

    Returns:
        Cost functions with its own realistic constraints.
    """

    if key == 'Gym-Ant':
        def cost_fn(s, a, ns):
            cost = 0.0
            # y-velocity constraint to fix the direction of travel
            if np.abs(ns[14]) > 0.1: 
                cost += 1.0
            return cost
    elif key == 'Gym-HalfCheetah':
        def cost_fn(s, a, ns):
            cost = 0.0
            # y-velocity constraint to fix the direction of travel
            if np.abs(ns[5]) > 0.1: 
                cost += 1.0
            return cost
    elif key == 'Gym-Hopper':
        def cost_fn(s, a, ns):
            cost = 0.0
            # jump height restriction (assuming there is a ceiling)
            if np.abs(ns[0]) > 1.0: 
                cost += 1.0
            return cost
    elif key == 'Gym-Humanoid-Standup':
        def cost_fn(s, a, ns):
            cost = 0.0
            # standing velocity restriction to prevent low blood pressure shock
            if np.abs(ns[24]) > 0.5: 
                cost += 1.0
            return cost
    elif key == 'Gym-Humanoid':
        def cost_fn(s, a, ns):
            cost = 0.0
            # y-velocity constraint to fix the direction of travel
            if np.abs(ns[23]) > 0.1: 
                cost += 1.0
            return cost
    elif key == 'Gym-InvertedDoublePendulum':
        def cost_fn(s, a, ns):
            cost = 0.0
            # positional constraint
            if np.abs(ns[0]) > 0.3:
                cost += 1.0
            return cost
    elif key == 'Gym-InvertedPendulum':
        def cost_fn(s, a, ns):
            cost = 0.0
            # positional constraint
            if np.abs(ns[0]) > 0.3:
                cost += 1.0
            return cost
    elif key == 'Gym-Reacher':
        def cost_fn(s, a, ns):
            cost = 0.0
            # angular velocity constraint (preventing drastic torque applied)
            if np.abs(ns[6]) > 0.5:
                cost += 0.5
            if np.abs(ns[7]) > 0.5:
                cost += 0.5
            return cost
    elif key == 'Gym-Swimmer':
        def cost_fn(s, a, ns):
            cost = 0.0
            # angular velocity constraint (preventing drastic torque applied)
            if np.abs(ns[6]) > 0.5:
                cost += 0.5
            if np.abs(ns[7]) > 0.5:
                cost += 0.5
            return cost
    elif key == 'Gym-Walker':
        def cost_fn(s, a, ns):
            cost = 0.0
            # jump height restriction (assuming there is a ceiling)
            if np.abs(ns[0]) > 1.0: 
                cost += 1.0
            return cost
    else:
        def cost_fn(s, a, ns):
            cost = 0.0
            return cost
    return cost_fn