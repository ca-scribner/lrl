from lrl.data_stores import GeneralIterationData, WalkStatistics, DictWithHistory

import logging
logger = logging.getLogger(__name__)

CONVERGENCE_TOLERANCE = 0.001
MAX_ITERATIONS = 500
MIN_ITERATIONS = 2
SOLVER_ITERATION_DATA_FIELDS = ['iteration', 'time', 'delta_max', 'delta_mean', 'policy_changes', 'converged']

DEFAULT_N_EVALS = 500
DEFAULT_TRAINS_PER_EVAL = 500
SCORING_SUMMARY_DATA_FIELDS = ['iteration', 'reward_mean', 'reward_median', 'reward_std', 'reward_min', 'reward_max',
                                   'steps_mean', 'steps_median', 'steps_std', 'steps_min', 'steps_max']
MAX_STEPS_PER_EPISODE = 100


class BaseSolver:
    """Base class for solvers

    FUTURE: Describe attributes.  Need to include things like policy is keyed by state tuple/index, etc.
    Dont forget value_function_tolerance can be for value or Q
    FUTURE: Explain score_while_training (true = 50 evals every 250 trains.  dict of n_evals, n_trains_per_eval can
    change that
    """
    def __init__(self, env, gamma=0.9, value_function_tolerance=CONVERGENCE_TOLERANCE, policy_init_type='zeros',
                 max_iters=MAX_ITERATIONS, min_iters=MIN_ITERATIONS, max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                 score_while_training=False, raise_if_not_converged=True):
        self.env = env
        self.value_function_tolerance = value_function_tolerance
        self.max_iters = max_iters
        self.min_iters = min_iters

        # Discount Factor
        self.gamma = gamma

        # Initialize policy and value data storage.
        # Use dictionaries that are indexed the same as the environment's transition matrix (P), which is indexed by
        # state denoted as either an index or a tuple
        self.policy_init_type = None
        self.policy = None
        self.init_policy(init_type=policy_init_type)

        # Storage for iteration metadata
        self.iteration = 0
        self.iteration_data = GeneralIterationData(columns=SOLVER_ITERATION_DATA_FIELDS)

        # String description of convergence criteria used here
        self.convergence_desc = "Criteria description not implemented"
        self.raise_if_not_converged = raise_if_not_converged

        if score_while_training is True:
            self.score_while_training = {
                'n_evals': DEFAULT_N_EVALS,
                'n_trains_per_eval': DEFAULT_TRAINS_PER_EVAL,
            }
        else:
            self.score_while_training = score_while_training

        self.scoring_summary = GeneralIterationData(columns=SCORING_SUMMARY_DATA_FIELDS)
        self.scoring_walk_statistics = {}
        self.max_steps_per_episode = max_steps_per_episode

    def init_policy(self, init_type=None):
        """
        Initialize self.policy, which is a dictionary-like DictWithHistory object for storing current and past policies

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

        # Index by whatever the environment indexes by (could be integers or some other object).
        state_keys = list(self.env.P.keys())

        self.policy = DictWithHistory(timepoint_mode='explicit')
        if self.policy_init_type == 'zeros':
            for k in state_keys:
                self.policy[k] = 0
        elif self.policy_init_type == 'random':
            for k in state_keys:
                self.policy[k] = self.env.action_space.sample()

        # Try to convert index actions to their full representation, if used by env
        try:
            for k in self.policy:
                self.policy[k] = self.env.index_to_action[self.policy[k]]
        except AttributeError:
            # If .index_to_action isn't there, keep policy as indices
            pass

    def iterate(self):
        """
        Perform the next iteration of the solver.

        This may be an iteration through all states in the environment (like in policy iteration) or obtaining and
        learning from a single experience (like in Q-Learning

        This method should update self.value and may update self.policy, and also commit iteration statistics to
        self.iteration_data.  Unless the subclass implements a custom self.converged, self.iteration_data should include
        a boolean entry for "converged", which is used by the default converged() function.

        # FUTURE: Should this be named differently?  Step feels intuitive, but can be confused with stepping in the env
        #          Could also do iteration, but doesn't imply that we're just doing a single iteration.

        Returns:
            None
        """
        pass

    def iterate_to_convergence(self, raise_if_not_converged=None, score_while_training=None):
        """
        Perform self.iterate repeatedly until convergence, optionally scoring the current policy periodically

        Side Effects:
            FUTURE: NEED TO BE ADDED

        Args:
            raise_if_not_converged (bool): If true, will raise an exception if convergence is not reached before hitting
                                           maximum number of iterations.  If None, uses self.raise_if_not_converged
            score_while_training (bool, dict, None): If None, use self.score_while_training.  Else, accepts inputs of
                                                     same format as accepted for score_while_training solver inputs

        Returns:
            None
        """
        if score_while_training is None:
            score_while_training = self.score_while_training

        if raise_if_not_converged is None:
            raise_if_not_converged = self.raise_if_not_converged

        logger.info(f"Solver iterating to convergence ({self.convergence_desc} or iters>{self.max_iters})")

        while (not self.converged()) and (self.iteration < self.max_iters):
            self.iterate()

            if score_while_training and (self.iteration % score_while_training['n_trains_per_eval'] == 0):
                logger.info(f'Current greedy policy being scored {score_while_training["n_evals"]} times at iteration '
                            f'{self.iteration}')
                self.scoring_walk_statistics[self.iteration] = self.score_policy(iters=score_while_training['n_evals'])

                data = {'iteration': self.iteration}
                data.update(self.scoring_walk_statistics[self.iteration].get_statistics())
                self.scoring_summary.add(data)
                logger.info(f'Current greedy policy achieved: '
                            f'r_mean = {data["reward_mean"]}, '
                            f'r_max = {data["reward_max"]}')
            converged = self.converged()
            logger.debug(f'{self.iteration}: delta_max = {self.iteration_data.get(i=-1)["delta_max"]:.1e}, '
                         f'policy_changes = {self.iteration_data.get(i=-1)["policy_changes"]}, '
                         f'converged = {converged}')
        if self.iteration >= self.max_iters and raise_if_not_converged:
            raise Exception(f"Max iterations ({self.max_iters}) reached - solver did not converge")

    def converged(self):
        """
        Returns True if solver is converged.

        This may be custom for each solver, but as a default it checks whether the most recent iteration_data entry
        has converged==True

        Returns:
            bool: Convergence status (True=converged)
        """
        if self.iteration < self.min_iters:
            logger.debug(f"Not converged: iteration ({self.iteration}) < min_iters ({self.min_iters})")
            return False
        else:
            try:
                return self.iteration_data.get(i=-1)['converged']
            except IndexError:
                # Data store has no records and thus cannot be converged
                return False
            except KeyError:
                raise KeyError("Iteration Data has no converged entry - cannot determine convergence status")

    def run_policy(self, max_steps=None, initial_state=None):
        """
        Perform a walk through the environment using the current policy

        Side Effects:
            self.env will be reset and optionally then forced into initial_state

        Args:
            max_steps: Maximum number of steps to be taken in the walk (step 0 is taken to be entering initial state)
                       If None, defaults to self.max_steps_per_episode
            initial_state: State for the environment to be placed in to start the walk (used to force a deterministic
                           start from anywhere in the environment rather than the typical start position)

        Returns:
            list of states encountered in the walk (including initial and final states)
            list of rewards obtained during the walk (rewards[0] == 0 as step 0 is simply starting the game)
            boolean indicating if the walk was terminal according to the environment
        """
        if max_steps is None:
            max_steps = self.max_steps_per_episode
        self.env.reset()
        if initial_state:
            # Override starting state
            self.env.s = initial_state
        states = [self.env.s]
        #
        # states = [self.env.reset()]
        rewards = [0.0]
        terminal = False

        for step in range(1, max_steps + 1):
            action = self.policy[states[-1]]
            new_state, reward, terminal, _ = self.env.step(action)
            logger.debug(f"Step {step}: From s={states[-1]} take a={action} --> s_prime={new_state} with r={reward}")

            states.append(new_state)
            rewards.append(float(reward))

            if terminal:
                break

        logger.debug(f"Walk completed in {len(states)} steps (terminal={terminal}), receiving total reward of {sum(rewards)}")
        return states, rewards, terminal

    def score_policy(self, iters=10, max_steps=None, initial_state=None):
        """
        Score the current policy by performing 'iters' greedy walks through the environment and returning statistics

        Side Effects:
            self.env will be reset

        Args:
            iters: Number of walks through the environment
            max_steps: Maximum number of steps allowed per walk.  If None, defaults to self.max_steps_per_episode
            initial_state: State for the environment to be placed in to start the walk (used to force a deterministic
                           start from anywhere in the environment rather than the typical start position)

        Returns:
            WalkStatistics: Object containing statistics about the walks (rewards, number of steps, etc.)
        """
        if max_steps is None:
            max_steps = self.max_steps_per_episode
        statistics = WalkStatistics()

        for iter in range(iters):
            # Walk through the environment following the current policy, up to max_steps total steps
            states, rewards, terminal = self.run_policy(max_steps=max_steps, initial_state=initial_state)
            statistics.add(reward=sum(rewards), walk=states, terminal=terminal)
            logger.debug(f"Policy scored {statistics.rewards[-1]} in {len(states)} steps (terminal={terminal})")

        return statistics


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
    # FUTURE: Special handling of terminal state in value iteration?  Works itself out if they all point to
    #       themselves with 0 reward, but really we just don't need to compute them.  And if we don't zero
    #       their value somewhere, we've gotta wait for the discount factor to decay them to zero from the
    #       initialized value.
    q_value = 0.0
    for outcome in outcomes:
        probability, next_state, reward, is_terminal = outcome
        if is_terminal:
            # q_values[this_action] += Probability of Outcome * (Immediate Reward)
            # (A terminal state has no discounted future value)
            q_value += probability * reward
        else:
            # q_values[this_action] += Probability of Outcome * (Immediate Reward + Discounted Future Value)
            q_value += probability * (reward + gamma * value_function[next_state])
    return q_value
