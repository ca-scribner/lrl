import numpy as np
import math

from lrl.data_stores import GeneralIterationData
from lrl.utils.misc import dict_differences

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

CONVERGENCE_TOLERANCE = 0.000001
MAX_ITERATIONS = 50


class BaseSolver:
    """Base class for solvers"""
    def __init__(self, env, gamma=0.9, value_function_tolerance=CONVERGENCE_TOLERANCE, policy_init_type='zeros',
                 max_iters=MAX_ITERATIONS, value_function_initial_value=0.0):
        self.env = env
        self.value_function_tolerance = value_function_tolerance
        self.max_iters = max_iters

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

        state_keys = range(self.env.observation_space.n)
        # If state is indexed by something other than integer, convert to that
        try:
            state_keys = [self.env.index_to_state[k] for k in state_keys]
        except AttributeError:
            # If env.index_to_state does not exist, we will index by integer
            pass

        if self.policy_init_type == 'zeros':
            self.policy = {k: 0 for k in state_keys}
        elif self.policy_init_type == 'random':
            self.policy = {k: self.env.action_space.sample() for k in state_keys}

        # Try to convert index actions to their full representation, if used by env
        try:
            self.policy = {k: self.env.index_to_action[i_a] for k, i_a in self.policy.items()}
        except AttributeError:
            pass

    def iterate(self):
        """
        Perform the next iteration of the solver.

        This may be an iteration through all states in the environment (like in policy iteration) or obtaining and
        learning from a single experience (like in Q-Learning

        This method should update self.value and may update self.policy, and also commit iteration statistics to
        self.iteration_data.  Unless the subclass implements a custom self.converged, self.iteration_data should include
        a boolean entry for "converged", which is used by the default converged() function.

        # FEATURE: Should this be named differently?  Step feels intuitive, but can be confused with stepping in the env
        #          Could also do iteration, but doesn't imply that we're just doing a single iteration.

        Returns:
            None
        """
        pass

    def iterate_to_convergence(self):
        """
        Perform self.iterate repeatedly until convergence

        Returns:
            None
        """
        logger.debug(f"Solver iterating to convergence (delta<{self.value_function_tolerance} or iters>{self.max_iters})")
        # Binding to easily get most recent delta in readable way

        while (not self.converged()) and (self.iteration < self.max_iters):
            self.iterate()
            converged = self.iteration_data.get(i=-1)['converged']
            logger.debug(f'{self.iteration}: delta_max = {self.iteration_data.get(i=-1)["delta_max"]:.1e}, '
                         f'policy_changes = {self.iteration_data.get(i=-1)["policy_changes"]}, '
                         f'converged = {converged}')

    def converged(self):
        """
        Returns True if solver is converged.

        This may be custom for each solver, but as a default it checks whether the most recent iteration_data entry
        has converged==True

        Returns:
            bool: Convergence status (True=converged)
        """
        try:
            return self.iteration_data.get(i=-1)['converged']
        except IndexError:
            # Data store has no records and thus cannot be converged
            return False
        except AttributeError:
            raise AttributeError("Iteration Data has no converged entry - cannot determine convergence status")


# Helpers
def q_from_outcomes(outcomes, gamma, value_function):
    """
    Compute Q for this transition using the Bellman equation over a list of outcomes

    Args:
        outcomes (list): List of possible outcomes of this transition in format
                        [(Probability, Next State, Immediate Reward, Boolean denoting if transition is terminal), ...]
        gamma (float): Discount factor
        value_function (dict): Dictionary of current estimate of the value function

    Returns:
        float: Q value of this transition's outcomes
    """
    if len(outcomes) == 0:
        raise ValueError("Cannot compute Q value of empty list of outcomes")

    # Each action can have more than one result.  Results are in a list of tuples of
    # (Probability, NextState (index or tuple), Reward for this action (float), IsTerminal (bool))
    # Sum contributions from all outcomes
    # FEATURE: Special handling of terminal state in value iteration?  Works itself out if they all point to
    #       themselves with 0 reward, but really we just don't need to compute them.  And if we don't zero
    #       their value somewhere, we've gotta wait for the discount factor to decay them to zero from the
    #       initialized value.
    q_value = 0.0
    for outcome in outcomes:
        probability, next_state, reward, is_terminal = outcome
        # q_values[this_action] += Probability of Outcome * (Immediate Reward + Discounted Future Value)
        q_value += probability * (reward + gamma * value_function[next_state])
    return q_value


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
    delta_mean = 0.0
    iter = 0

    while iter < max_iters and delta_max > tolerance:
        value_new = value_function.copy()

        if evaluation_type == 'max':
            # If max, we will have to compute a greedy policy each iteration.  Might as well save and return it.
            policy_new = {}

        for state in value_function:
            if evaluation_type == 'max':
                actions = env.P[state]
            elif evaluation_type == 'on-policy':
                actions = {policy[state]: env.P[state][policy[state]]}
            else:
                raise ValueError(f"Invalid evaluation_type {evaluation_type}")

            # Actions are a dict keyed by tuples of action (eg: Racetrack) or integer action numbers (eg: FrozenLake)
            # Make numpy array for q values and a mapping to remember which q index relates to which action key
            # FEATURE: This could be done up front when initializing the env and then not repeated here to save cpu
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
            logger.debug(f"Iter {iter} done with delta_max = {delta_max}")
        iter += 1
        value_function = value_new

    logger.info(f"Policy evaluation completed after {iter} iters")

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
