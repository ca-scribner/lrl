import numpy as np

from .base_solver import BaseSolver
from lrl.utils.misc import Timer, count_dict_differences


class ValueIteration(BaseSolver):
    """Solver for value iteration

    FEATURE: Improve this docstring.  Add refs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self):
        """
        Perform a single iteration of value iteration, updating self.value and storing metadata about the iteration.

        # FEATURE: This can be simplified if actions have to be integer indexed and not possibly tuple indexed

        Side Effects:
            self.value: Updated to the newest estimate of the value function
            self.policy: Updated to the greedy policy according to the value function estimate
            self.iteration: Increment iteration counter by 1
            self.iteration_data: Add new record to iteration data store

        Returns:
            None
        """
        timer = Timer()
        delta_max = 0.0
        delta_sum = 0.0

        value_new = self.value.copy()
        policy_new = self.policy.copy()

        for state in self.value:
            actions = self.env.P[state]

            # Actions can be a dict (indexed by tuples of action) or a list (indexed by action number)
            # Make numpy array for q values and a mapping to remember which q index relates to which action key
            try:
                i_to_key = {i: key for i, key in enumerate(actions.keys())}
            except AttributeError:
                i_to_key = {i: i for i in range(len(actions))}

            q_values = np.zeros(len(i_to_key))

            for i_a, key in i_to_key.items():
                action = actions[key]
                # Each action can have more than one result.  Results are in tuples of
                # (Probability, NextState (index or tuple), Reward for this action (float), IsTerminal (bool))
                # Sum contributions from all outcomes
                # FEATURE: Special handling of terminal state in value iteration?  Works itself out if they all point to
                #       themselves with 0 reward, but really we just don't need to compute them.  And if we don't zero
                #       their value somewhere, we've gotta wait for the discount factor to decay them to zero from the
                #       initialized value.
                for outcome in action:
                    probability, next_state, reward, is_terminal = outcome
                    # q_values[this_action] += Probability of Outcome * (Immediate Reward + Discounted Future Value)
                    q_values[i_a] += probability * (reward + self.gamma * self.value[next_state])

            # Choose between available actions for this state
            best_action_index = q_values.argmax()
            value_new[state] = q_values[best_action_index]
            best_action_key = i_to_key[best_action_index]
            policy_new[state] = best_action_key

            this_delta = abs(value_new[state] - self.value[state])
            delta_max = max(delta_max, this_delta)
            delta_sum += this_delta

        # Log metadata about iteration
        self.iteration_data.add({'iteration': self.iteration,
                                 'time': timer.elapsed,
                                 'delta_mean': delta_sum / len(self.value),
                                 'delta_max': delta_max,
                                 'policy_changes': count_dict_differences(policy_new, self.policy),
                                 })

        # Store results and increment counter
        self.value = value_new
        self.policy = policy_new
        self.iteration += 1
