import numpy as np
from lrl.data_stores import GeneralIterationData


CONVERGENCE_TOLERANCE = 0.000001


class BaseSolver:
    """Base class for solvers"""
    def __init__(self, env, gamma=0.9, tolerance=CONVERGENCE_TOLERANCE, policy_init_type='zeros',
                 value_function_initial_value=0.0):
        self.env = env
        self.tolerance = tolerance

        # Discount Factor
        self.gamma = gamma

        # Initialize policy and value data storage.
        # Use dictionaries that are indexed the same as the environment's transition matrix (P), which is indexed by
        # state denoted as either an index or a tuple
        self.policy_init_type = None
        self.policy = None
        self.init_policy(init_type=policy_init_type)
        self.value = {k: value_function_initial_value for k in self.env.P.keys()}

        # Storage for iteration metadata
        self.iteration = 0
        self.iteration_data = GeneralIterationData()

    def init_policy(self, init_type=None):
        """
        Initialize self.policy

        Args:
            init_type: Type of initialization.  If defined, will store to self.policy_init_type.  Can be any of:
                       None: Uses value in self.policy_init_type
                       zeros: Initialize policy to all 0's (first action)
                       random: Initialize policy to a random action (action indices are random integer from
                               [0, len(self.env.P[this_state])], where P is the transition matrix and P[state] is a list
                               of all actions available in the state)

        Returns:
            None
        """
        valid_init_types = ['zeros', 'random']
        if init_type in valid_init_types:
            self.policy_init_type = init_type
        elif init_type is None:
            pass
        else:
            raise ValueError(f"Invalid init_type {init_type} - must be one of {valid_init_types}")

        state_keys = self.env.P.keys()
        if self.policy_init_type == 'zeros':
            self.policy = {k: 0 for k in state_keys}
        elif self.policy_init_type == 'random':
            self.policy = {k: np.random.randint(low=0, high=len(self.env.P[k]), size=1, dtype=np.int)[0]
                           for k in state_keys}

    def iterate(self):
        """
        Perform the next iteration of the solver.

        This may be an iteration through all states in the environment (like in policy iteration) or obtaining and
        learning from a single experience (like in Q-Learning

        This method should update self.value but not have the side effect of updating self.policy

        # FEATURE: Should this be named differently?  Step feels intuitive, but can be confused with stepping in the env
        #          Could also do iteration, but doesn't imply that we're just doing a single iteration.

        Returns:
            None
        """
        pass

    def update_policy(self):
        """Update the policy to be greedy relative to the value function

        Side Effects:
            self.policy: Changed to reflect new policy

        Returns:
            None

        """
        pass

    def act_greedy(self, state):
        """Returns the greedy action in state given the current value function

        Ties are broken randomly.

        Args:
            state:

        Returns:
            int: Index of the greedy action

        """
        pass
