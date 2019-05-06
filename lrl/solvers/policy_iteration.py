from .base_solver import BaseSolver, policy_evaluation
from lrl.utils.misc import Timer, count_dict_differences, dict_differences

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MAX_POLICY_EVAL_ITERS_LAST_IMPROVEMENT = 1000


class PolicyIteration(BaseSolver):
    """Solver for policy iteration

    FEATURE: Improve this docstring.  Add refs
    """
    def __init__(self, env, max_policy_eval_iters_per_improvement=10, policy_evaluation_type='on-policy-iterative',
                 **kwargs):
        # FEATURE: Clean up the init arguments
        super().__init__(env, **kwargs)

        # Maximum number of policy evaluations invoked in one Evaluate-Improve iteration.  Note that this does not apply
        # if on the final Evaluate-Improve iteration (eg: if previous Evaluate-Improve iter found 0 policy changes)
        self.max_policy_eval_iters_per_improvement = max_policy_eval_iters_per_improvement
        self.policy_evaluation_type = policy_evaluation_type

    def _policy_evaluation(self, max_iters=None):
        """
        Compute an estimate of the value function for the current policy to within self.tolerance

        Side Effects:
            self.value: Updated to the newest estimate of the value function

        Returns:
            None
        """
        if max_iters is None:
            max_iters = self.max_policy_eval_iters_per_improvement
        value_new = policy_evaluation(value_function=self.value, env=self.env, policy=self.policy, gamma=self.gamma,
                                      evaluation_type=self.policy_evaluation_type,
                                      tolerance=self.value_function_tolerance,
                                      max_iters=max_iters)

        self.value = value_new

    def _policy_improvement(self, return_differences=True):
        """
        Update the policy to be greedy relative to the most recent value function

        Side Effects:
            self.policy: Updated to be greedy relative to self.value

        Args:
            return_differences: If True, return number of differences between old and new policies

        Returns:
            int: (if return_differences==True) Number of differences between the old and new policies
        """
        value_new, policy_new = policy_evaluation(value_function=self.value, env=self.env,
                                                  gamma=self.gamma, evaluation_type='max')

        if return_differences:
            returned = count_dict_differences(policy_new, self.policy)
        else:
            returned = None

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

        # If this iteration resulted in no policy changes but did not converge to the policy_evaluation convergence
        # tolerance, do one additional Evaluate-Iterate step to fully converge (this is done for productivity reasons.
        # Most iterations don't really need to be iterated down to a very small tolerance on the value function and thus
        # we have the iteration limit self.max_policy_eval_per_improvement.  But to ensure we haven't missed any policy
        # changes by making this simplification, we ensure whenever we see policy_changes == 0 (eg: we think PI has
        # converged) that we've fully converged the value function
        if policy_changes == 0 and delta_max > self.value_function_tolerance:
            self._policy_evaluation(max_iters=MAX_POLICY_EVAL_ITERS_LAST_IMPROVEMENT)

            # Recompute delta given fully converged solution
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
