from .base_solver import BaseSolver, policy_evaluation
from lrl.utils.misc import Timer, count_dict_differences, dict_differences


class ValueIteration(BaseSolver):
    """Solver for value iteration

    FEATURE: Improve this docstring.  Add refs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self):
        """
        Perform a single iteration of value iteration, updating self.value and storing metadata about the iteration.

        Side Effects:
            self.value: Updated to the newest estimate of the value function
            self.policy: Updated to the greedy policy according to the value function estimate
            self.iteration: Increment iteration counter by 1
            self.iteration_data: Add new record to iteration data store

        Returns:
            None
        """
        timer = Timer()

        value_new, policy_new = policy_evaluation(value_function=self.value, env=self.env, gamma=self.gamma,
                                                  evaluation_type='max', max_iters=1)

        delta_max, delta_mean = dict_differences(value_new, self.value)
        policy_changes = count_dict_differences(policy_new, self.policy)

        # Log metadata about iteration
        self.iteration_data.add({'iteration': self.iteration,
                                 'time': timer.elapsed(),
                                 'delta_mean': delta_mean,
                                 'delta_max': delta_max,
                                 'policy_changes': policy_changes,
                                 'converged': bool(delta_max <= self.value_function_tolerance),
                                 })

        # Store results and increment counter
        self.value = value_new
        self.policy = policy_new
        self.iteration += 1
