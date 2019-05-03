import numpy as np

from .base_solver import BaseSolver, q_from_outcomes, policy_evaluation
from lrl.utils.misc import Timer, count_dict_differences, dict_differences

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class PolicyIteration(BaseSolver):
    """Solver for policy iteration

    FEATURE: Improve this docstring.  Add refs
    """
    def __init__(self, env, max_iters_policy_evaluation=100, **kwargs):
        # FEATURE: Clean up the init arguments
        super().__init__(env, **kwargs)
        self.max_iters_policy_evaluation = max_iters_policy_evaluation

    def _policy_evaluation(self):
        """
        Compute an estimate of the value function for the current policy to within self.tolerance

        Side Effects:
            self.value: Updated to the newest estimate of the value function

        Returns:
            None
        """
        value_new = policy_evaluation(value_function=self.value, env=self.env,
                                                             policy=self.policy,
                                                             gamma=self.gamma, evaluation_type='on-policy',
                                                             tolerance=self.value_function_tolerance,
                                                             max_iters=self.max_iters_policy_evaluation)

        self.value = value_new

    def _policy_improvement(self, return_differences=True):
        """
        TODO: DOCSTRING.  Mention how value function is updated.  add to sideeffects
        Args:
            return_differences:

        Returns:

        """
        # Note: Strictly following the Policy Iteration from Sutton leads to us not saving this updated value function,
        # but that seems wasteful.  Instead we will save this value function as well
        value_new, policy_new = policy_evaluation(value_function=self.value, env=self.env,
                                                  gamma=self.gamma, evaluation_type='max')

        if return_differences:
            returned = count_dict_differences(policy_new, self.policy)
        else:
            returned = None

        self.value = value_new
        self.policy = policy_new
        return returned

    def iterate(self):
        """
        Perform a single iteration of policy iteration, updating self.value and storing metadata about the iteration.

        Side Effects:
            self.value: Updated to the newest estimate of the value function
            self.policy: Updated to the greedy policy according to the value function estimate
            self.iteration: Increment iteration counter by 1
            self.iteration_data: Add new record to iteration data store

        Returns:
            None
        """
        timer = Timer()

        logger.debug(f"Performing iteration {self.iteration} of policy iteration")
        value_old = self.value

        # Compute a value function for the current policy
        self._policy_evaluation()

        # Compute a new greedy policy based on the updated value function
        policy_changes = self._policy_improvement()

        delta_max, delta_mean = dict_differences(self.value, value_old)

        # Log metadata about iteration
        self.iteration_data.add({'iteration': self.iteration,
                                 'time': timer.elapsed(),
                                 'delta_mean': delta_mean,
                                 'delta_max': delta_max,
                                 'policy_changes': policy_changes,
                                 'converged': policy_changes == 0,
                                 })
        self.iteration += 1
