import numpy as np
import math

from .base_solver import BaseSolver, q_from_outcomes, CONVERGENCE_TOLERANCE
from lrl.utils.misc import Timer, count_dict_differences, dict_differences
from lrl.data_stores import DictWithHistory

import logging
logger = logging.getLogger(__name__)


MAX_POLICY_EVAL_ITERS_LAST_IMPROVEMENT = 1000


class ValueIteration(BaseSolver):
    """Solver for value iteration

    FUTURE: Improve this docstring.  Add refs
    """
    def __init__(self, env, value_function_initial_value=0.0, **kwargs):
        super().__init__(env, **kwargs)

        # String description of convergence criteria
        self.convergence_desc = f"Max delta in value function < {self.value_function_tolerance}"

        self.value = DictWithHistory(timepoint_mode='explicit', tolernace=self.value_function_tolerance*0.1)
        for k in self.env.P.keys():
            self.value[k] = value_function_initial_value

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
        logger.debug(f"Performing iteration {self.iteration} of value iteration")

        timer = Timer()

        value_new, policy_new = policy_evaluation(value_function=self.value.to_dict(), env=self.env, gamma=self.gamma,
                                                  evaluation_type='max', max_iters=1,
                                                  tolerance=self.value_function_tolerance)

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
        # Use converged function to assess convergence and add that back into iteration_data
        # (prevents duplicating the convergence logic, at the expense of more complicated logging logic)
        self.iteration_data.get(-1)['converged'] = self.converged()
        logger.debug(f"Iteration {self.iteration} complete with d_max={delta_max}, policy_changes={policy_changes}")

        # Store results and increment counter
        self.value.update(value_new)
        self.value.increment_timepoint()
        self.policy.update(policy_new)
        self.policy.increment_timepoint()
        self.iteration += 1

    def converged(self):
        """
        Returns True if solver is converged.

        Returns:
            bool: Convergence status (True=converged)
        """
        if self.iteration < self.min_iters:
            logger.debug(f"Not converged: iteration ({self.iteration}) < min_iters ({self.min_iters})")
            return False
        else:
            try:
                return self.iteration_data.get(-1)['delta_max'] <= self.value_function_tolerance
            except IndexError:
                # Data store has no records and thus cannot be converged
                return False
            except KeyError:
                raise KeyError("Iteration Data has no delta_max entry - cannot determine convergence status")


class PolicyIteration(BaseSolver):
    """Solver for policy iteration

    FUTURE: Improve this docstring.  Add refs
    """
    def __init__(self, env, value_function_initial_value=0.0, max_policy_eval_iters_per_improvement=10,
                 policy_evaluation_type='on-policy-iterative', **kwargs):
        # FUTURE: Clean up the init arguments
        super().__init__(env, **kwargs)

        # Maximum number of policy evaluations invoked in one Evaluate-Improve iteration.  Note that this does not apply
        # if on the final Evaluate-Improve iteration (eg: if previous Evaluate-Improve iter found 0 policy changes)
        self.max_policy_eval_iters_per_improvement = max_policy_eval_iters_per_improvement
        self.policy_evaluation_type = policy_evaluation_type

        # String description of convergence criteria
        self.convergence_desc = "1 iteration without change in policy"

        self.value = DictWithHistory(timepoint_mode='explicit', tolernace=self.value_function_tolerance*0.1)
        for k in self.env.P.keys():
            self.value[k] = value_function_initial_value

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
        value_new = policy_evaluation(value_function=self.value.to_dict(), env=self.env, policy=self.policy, gamma=self.gamma,
                                      evaluation_type=self.policy_evaluation_type,
                                      tolerance=self.value_function_tolerance,
                                      max_iters=max_iters)

        self.value.update(value_new)
        self.value.increment_timepoint()

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
        value_new, policy_new = policy_evaluation(value_function=self.value.to_dict(), env=self.env,
                                                  gamma=self.gamma, evaluation_type='max',
                                                  tolerance=self.value_function_tolerance)

        if return_differences:
            returned = count_dict_differences(policy_new, self.policy)
        else:
            returned = None

        self.policy.update(policy_new)
        self.policy.increment_timepoint()
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
        value_old = self.value.to_dict()

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
        # Use converged function to assess convergence and add that back into iteration_data
        # (prevents duplicating the convergence logic, at the expense of more complicated logging logic)
        self.iteration_data.get(-1)['converged'] = self.converged()
        logger.debug(f"Iteration {self.iteration} complete with d_max={delta_max}, policy_changes={policy_changes}")

        self.iteration += 1

    def converged(self):
        """
        Returns True if solver is converged.

        Returns:
            bool: Convergence status (True=converged)
        """
        if self.iteration < self.min_iters:
            logger.debug(f"Not converged: iteration ({self.iteration}) < min_iters ({self.min_iters})")
            return False
        else:
            try:
                return self.iteration_data.get(-1)['policy_changes'] == 0
            except IndexError:
                # Data store has no records and thus cannot be converged
                return False
            except KeyError:
                raise KeyError("Iteration Data has no policy_changes entry - cannot determine convergence status")


# Helpers
def policy_evaluation(value_function, env, gamma, policy=None, evaluation_type='max', max_iters=1,
                      tolerance=CONVERGENCE_TOLERANCE):
    """
    Compute the value function of either a given policy of the best available policy (argmax)

    This function serves multiple purposes:
    -   With evaluation_type == max and max_iters > 1, we iterate on the value function in the typical Value Iteration
        algorithm fashion by iteratively computing:
            V_t+1(s) = argmax_a(sum_s'(P(s, a, s')*(Reward(s, a, s') + gamma * V_t(s'))))
        This returns both the value function and the greedy policy that was computed to generate it
    -   With evaluation_type == max and max_iters == 1, we compute the greedy value function and policy given our most
        recent estimate of the value function.  This is effectively the policy_improvement step in Policy Iteration, but
        also returns an value function as it must be computed anyway.
    -   With evaluation_type == on-policy, max_iters > 0, and a policy defined, we either iterate or solve directly for
        the value function in the typical Policy Evaluation style of the Policy Iteration algorithm:
            V_t+1(s) for policy = sum_s'(P(s, pi(s), s')*(Reward(s, pi(s), s') + gamma * V_t(s')))
        This returns the value function that indicates the value if following the given policy

    Args:
        value_function (TO BE UPDATED): The current estimate of the value function
        env (gym.Env.Discrete subclass): Environment describing the MDP to be planned
        gamma (float): Discount factor
        policy (TO BE UPDATED): (Required if evaluation_type=='on-policy') Policy to evaluate a value function for
        evaluation_type (str): One of:
            max: Compute the greedy value function (compute V using the greedy action for all actions)
            on-policy-iterative: Compute the value function given a fixed policy iteratively
            on-policy-direct: Compute the value function given a fixed policy by directly solving the system of
                              equations (typically much slower and higher memory than the iterative method)
        max_iters: Maximum number of iterations for any iterative method
        tolerance: Convergence tolerance on the value function for any iterative method (converged if maximum
                   elementwise difference between iterations is less than tolerance)

    Returns:
        (TO BE UPDATED): Value Function computed
        (TO BE UPDATED): (returned if evaluation_type == max) Greedy policy corresponding to returned Value Function
    """
    if evaluation_type == 'max' or evaluation_type == 'on-policy-iterative':
        if evaluation_type == 'on-policy-iterative':
            evaluation_type = 'on-policy'
        return policy_evaluation_iterative(value_function=value_function, env=env, gamma=gamma, policy=policy,
                                           evaluation_type=evaluation_type, max_iters=max_iters, tolerance=tolerance)
    elif evaluation_type == 'on-policy' or evaluation_type == 'on-policy-direct':
        return policy_evaluation_direct(env=env, gamma=gamma, policy=policy)
    else:
        raise ValueError(f'Invalid value for evaluation_type "{evaluation_type}"')


def policy_evaluation_iterative(value_function, env, gamma, policy=None, evaluation_type='max', max_iters=1,
                                tolerance=CONVERGENCE_TOLERANCE):
    """
    Compute the value function of either a given policy of the best available policy (argmax) iteratively

    See docstring for policy_evaluation() for more details.

    Returns:
        (TO BE UPDATED): Value Function computed
        (TO BE UPDATED): (returned if evaluation_type == max) Greedy policy corresponding to returned Value Function
    """
    logger.info(f"Computing policy_evaluation for evaluation_type == {evaluation_type}")

    delta_max = np.inf
    this_iter = 0
    policy_new = {}  # Only needed for evaluation_type == max, but this keeps IDE from flagging it later

    while this_iter < max_iters and delta_max > tolerance:
        value_new = value_function.copy()

        for state in value_function:
            if evaluation_type == 'max':
                actions = env.P[state]
            elif evaluation_type == 'on-policy':
                actions = {policy[state]: env.P[state][policy[state]]}
            else:
                raise ValueError(f"Invalid evaluation_type {evaluation_type}")

            # Actions are a dict keyed by tuples of action (eg: Racetrack) or integer action numbers (eg: FrozenLake)
            # Make numpy array for q values and a mapping to remember which q index relates to which action key
            # FUTURE: This could be done up front when initializing the env and then not repeated here to save cpu
            #          This would also be needed for decoding actions when plotting.
            i_to_action = {i: action for i, action in enumerate(actions.keys())}

            q_values = np.zeros(len(i_to_action))

            # For each action, compute the q-value of the probabilistic outcome (each action may have multiple possible
            # outcomes)
            for i_a, action in i_to_action.items():
                outcomes = actions[action]
                q_values[i_a] = q_from_outcomes(outcomes, gamma, value_function)

            if evaluation_type == 'max':
                # If evaluation_type == max, choose the action that had the best possible outcome.
                # Break ties by choosing the first tied action.
                # We also capture the updated greedy policy here, since we've done the work already.
                best_action_index = q_values.argmax()
                value_new[state] = q_values[best_action_index]
                best_action_key = i_to_action[best_action_index]
                policy_new[state] = best_action_key
            else:
                # Otherwise, there's just one value to choose from
                value_new[state] = q_values[0]

        if max_iters > 1:
            delta_max, delta_mean = dict_differences(value_new, value_function)
            logger.debug(f"Iter {this_iter} done with delta_max = {delta_max}")
        this_iter += 1
        value_function = value_new

    logger.info(f"policy_evaluation completed after {this_iter} iters")

    if evaluation_type == 'max':
        return value_function, policy_new
    else:
        return value_function


def policy_evaluation_direct(env, gamma, policy):
    """
    Compute the value function of either a given policy of the best available policy (argmax) by direct solution

    See docstring for policy_evaluation() for more details.

    NOTE: This method is almost always slower and more memory intensive than policy_evaluation_iterative due to the
          construction and solving of the matrix.  scipy's sparse matrix solver spsolve allowed for solution speeds that
          were comparable to the iterative method and maybe could be a bit faster, but it was not implemented here.
          Maybe something to try in future (to implement here, we need to refactor the construction of the A matrix
          below to incrementally build as a sparse matrix rather than build as dense and then convert)

    Returns:
        (TO BE UPDATED): Value Function computed
    """
    logger.debug(f"Computing policy_evaluation_direct")
    a = []
    b = []
    i_matrix_to_i_state = []
    value_new = {}

    # Build A and b matricies to solve this policy's value function.  Each row of A, b corresponds to a different
    # state's value.
    # 1)    Pass through all states and act whether they're terminal or not.
    #           If not terminal, build a row for A and b and keep track of which row in [A,b] maps to which state
    #           If terminal, do not build a row in A or b and keep track of which index we've skipped so we can
    #           remove the corresponding column after
    # 2)    Build numpy arrays for A and b, and remove all terminal columns from A
    # 3)    Solve Ax=b for x
    # 4)    Feed x back into value function
    for i_s in range(env.observation_space.n):
        state = env.index_to_state[i_s]
        action = policy[state]
        outcomes = env.P[state][action]

        # If this has a single outcome which points back to itself with is_terminal==True, this is terminal.
        # Else, make A, b
        if len(outcomes) == 1:
            probability, next_state, reward, is_terminal = outcomes[0]
            if math.isclose(probability, 1.0) and state == next_state and is_terminal:
                value_new[state] = 0.0
                continue

        this_row = np.zeros(env.observation_space.n, dtype=np.float)
        this_b = 0.0

        i_matrix_to_i_state.append(i_s)

        # Record this V in the matrix
        this_row[i_s] += 1.0

        # Record any transitions in A and b
        for outcome in outcomes:
            probability, next_state, reward, _ = outcome
            i_next_state = env.state_to_index[next_state]

            this_b += probability * reward
            this_row[i_next_state] -= probability * gamma

        a.append(this_row)
        b.append(this_b)

    a = np.asarray(a)
    b = np.asarray(b)

    # Keep columns that are for terminal (omitted from A) states, getting rid of the rest
    i_matrix_to_i_state = tuple(i_matrix_to_i_state)
    a = a[:, i_matrix_to_i_state]

    non_terminal_values = np.linalg.solve(a, b)
    for i, i_s in enumerate(i_matrix_to_i_state):
        value_new[env.index_to_state[i_s]] = non_terminal_values[i]
    return value_new
