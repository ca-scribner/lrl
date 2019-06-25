import numpy as np
import numbers

from .base_solver import BaseSolver
from lrl.data_stores import WalkStatistics, DictWithHistory, GeneralIterationData
from lrl.utils.misc import Timer, count_dict_differences

import logging
logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2000
MIN_ITERATIONS = 250
NUM_EPISODES_FOR_CONVERGENCE = 20
SOLVER_ITERATION_DATA_FIELDS = ['iteration', 'time', 'delta_max', 'policy_changes', 'alpha', 'epsilon',
                                'converged']
CONVERGENCE_TOLERANCE = 0.1

# FUTURE: Fix documentation/inheritance after this sphinx bug is fixed:
#   https://github.com/sphinx-doc/sphinx/issues/741


class QLearning(BaseSolver):
    """
    Solver class for Q-Learning

    Notes:
        See also BaseSolver for additional attributes, members, and arguments (missing due here to Sphinx bug with
        inheritance in docs)

    Examples:
        See examples directory

    Args:
        alpha (float, dict): (OPTIONAL)

            * If None, default linear decay schedule applied, decaying from 0.1 at iter 0 to 0.025 at max iter
            * If float, interpreted as a constant alpha value
            * If dict, interpreted as specifications to a decay function as defined in decay_functions()

        epsilon (float, dict): (OPTIONAL)

            * If None, default linear decay schedule applied, decaying from 0.25 at iter 0 to 0.05 at max iter
            * If float, interpreted as a constant epsilon value
            * If dict, interpreted as specifications to a decay function as defined in decay_functions()

        num_episodes_for_convergence (int): Number of consecutive episodes with delta_Q < tolerance to say a solution is
            converged
        **kwargs: Other arguments passed to BaseSolver

    Returns:
        None
    """
    def __init__(self, env, value_function_tolerance=CONVERGENCE_TOLERANCE,
                 alpha=None, epsilon=None, max_iters=MAX_ITERATIONS, min_iters=MIN_ITERATIONS,
                 num_episodes_for_convergence=NUM_EPISODES_FOR_CONVERGENCE,
                 **kwargs):
        super().__init__(env, max_iters=max_iters, min_iters=min_iters,
                         value_function_tolerance=value_function_tolerance, **kwargs)

        # Interpret alpha and epsilon settings
        if alpha is None:
            # Default schedule
            self._alpha_settings = {
                'type': 'linear',
                'initial_value': 0.1,
                'initial_timestep': 0,
                'final_value': 0.025,
                'final_timestep': self._max_iters,
            }
        elif isinstance(alpha, dict):
            self._alpha_settings = alpha
        else:
            # Interpret alpha as a number and build a default settings dict
            self._alpha_settings = {
                'type': 'constant',
                'initial_value': float(alpha),
            }

        if epsilon is None:
            # Default schedule
            self._epsilon_settings = {
                'type': 'linear',
                'initial_value': 0.25,
                'initial_timestep': 0,
                'final_value': 0.05,
                'final_timestep': self._max_iters,
            }
        elif isinstance(epsilon, dict):
            self._epsilon_settings = epsilon
        else:
            # Interpret alpha as a number and build a default settings dict
            self._epsilon_settings = {
                'type': 'constant',
                'initial_value': float(epsilon),
            }

        #: int: Counter for number of transitions experienced during all learning
        self.transitions = 0

        # Estimate of Q, keyed by ((state), (action)) where state/action can be integers or qualified tuples
        #: DictWithHistory: Space-efficient dict-like storage of the current and all former q functions
        self.q = None
        self.init_q()

        #: GeneralIterationData: Data store for iteration data
        #:
        #: Overloads BaseSolver's iteration_data attribute with one that includes more fields
        self.iteration_data = GeneralIterationData(columns=SOLVER_ITERATION_DATA_FIELDS)

        #: WalkStatistics: Data store for statistics from training episodes
        self.walk_statistics = WalkStatistics()

        #: int: Number of consecutive episodes with delta_Q < tolerance to say a solution is converged
        self.num_episodes_for_convergence = num_episodes_for_convergence

        #: str: String description of convergence criteria
        self._convergence_desc = f"{self.num_episodes_for_convergence} episodes with max delta in Q function < " \
            f"{self._value_function_tolerance}"

    def _policy_improvement(self, states=None):
        """
        Update the policy to be greedy relative to the most recent q function

        Side Effects:
            self.policy: Updated to be greedy relative to self.q

        Args:
            states: List of states to update.  If None, all states will be updated

        Returns:
            None
        """
        if states is None:
            states = self.policy.keys()
        for state in states:
            self.policy[state] = self.choose_epsilon_greedy_action(state, epsilon=0)

    def step(self, count_transition=True):
        """
        Take and learn from a single step in the environment.

        Applies the typical Q-Learning approach to learn from the experienced transition

        Args:
            count_transition (bool): If True, increment transitions counter self.transitions.  Else, do not.

        Returns:
            (tuple): tuple containing:

            * **transition** (*tuple*): Tuple of (state, reward, next_state, is_terminal)
            * **delta_q** (*float*): The (absolute) change in q caused by this step
        """
        logger.debug(f'Taking and learning from a step in the environment (transition count = {self.transitions})')
        state = self.env.s

        # Make an epsilon-greedy choice
        action = self.choose_epsilon_greedy_action(state, self.epsilon)
        next_state, reward, is_terminal, _ = self.env.step(action)

        # If is_terminal, then the future value of the next action is 0 (game is ended so no future value available)
        if is_terminal:
            q_best_next_action = 0.0
        else:
            q_best_next_action = np.max(self.get_q_at_state(next_state))

        # Compute Q-Learning update (TD)
        # TODO: Is this eq different if this is a terminal step?  Memory says it should be different
        try:
            # This will work is q is indexed by integer state and action
            td = reward + self._gamma * q_best_next_action - self.q[state, action]
        except KeyError:
            # This will work if q is indexed by tuple state and action (merge the tuples for q index)
            td = reward + self._gamma * q_best_next_action - self.q[state + action]

        try:
            self.q[(int(state), int(action))] += self.alpha * td
        except (TypeError, KeyError) as e:
            self.q[state + action] += self.alpha * td

        delta_q = self.alpha * td

        if count_transition:
            self.transitions += 1
        logger.debug(f'Completed step from {state} -> {next_state} yielding {reward} (terminal={is_terminal})')

        return (state, reward, next_state, is_terminal), abs(delta_q)

    def iterate(self):
        """
        Perform and learn from a single episode in the environment (one walk from start to finish)

        Side Effects:

        * self.value: Updated to the newest estimate of the value function
        * self.policy: Updated to the greedy policy according to the value function estimate
        * self.iteration: Increment iteration counter by 1
        * self.iteration_data: Add new record to iteration data store
        * self.env: Reset and then walked through

        Returns:
            None
        """
        if self._iteration % 500 == 0:
            logger.info(f"Performing iteration (episode) {self._iteration} of Q-Learning")
        else:
            logger.debug(f"Performing iteration (episode) {self._iteration} of Q-Learning")
        timer = Timer()
        states = [self.env.reset()]
        rewards = [0.]
        delta_max = 0.0

        policy_old = self.policy.to_dict()

        # Perform a single episode, learning along the way (this implicitly updates self.q)
        for i_step in range(self._max_steps_per_episode):
            transition, this_delta_q = self.step()
            delta_max = max(delta_max, this_delta_q)
            state, reward, next_state, is_terminal = transition
            states.append(next_state)
            rewards.append(reward)

            if is_terminal:
                break
        logger.debug(f"Iteration {self._iteration} completed with r={sum(rewards)} in {len(states)} steps "
                     f"(terminal={is_terminal})")

        # Compute new greedy policy to compare to old policy
        # Only update states that were visited and thus have an updated Q and possibly a new best action/Q
        # Pass states as set to remove duplicates
        self._policy_improvement(set(states))

        # Log metadata about iteration
        # delta_max, delta_mean = dict_differences(self.q, q_old)
        policy_changes = count_dict_differences(self.policy, policy_old, keys=states)
        logger.debug(f"Walk resulted in delta_max = {delta_max}, and {policy_changes} policy "
                     f"changes")

        self.iteration_data.add({'iteration': self._iteration,
                                 'time': timer.elapsed(),
                                 'delta_max': delta_max,
                                 'steps': len(states),
                                 'policy_changes': policy_changes,
                                 'alpha': self.alpha,
                                 'epsilon': self.epsilon,
                                 })
        # Use converged function to assess convergence and add that back into iteration_data
        # (prevents duplicating the convergence logic, at the expense of more complicated logging logic)
        self.iteration_data.get(-1)['converged'] = self.converged()

        # Log more detailed metadata
        self.walk_statistics.add(reward=sum(rewards), walk=states, terminal=is_terminal)

        # Increment counters
        self.q.increment_timepoint()
        self.policy.increment_timepoint()
        self._iteration += 1

    def choose_epsilon_greedy_action(self, state, epsilon=None):
        """
        Return an action chosen by epsilon-greedy scheme based on the current estimate of Q

        Args:
            state (int, tuple): Descriptor of current state in environment
            epsilon: Optional.  If None, self.epsilon is used

        Returns:
            int or tuple: action chosen
        """
        if epsilon is None:
            epsilon = self.epsilon

        these_q = self.get_q_at_state(state)

        # Check if action is a tuple.  action=0 should be accessible if action is an integer
        return_action_as_tuple = False
        try:
            _ = self.q[(state, 0)]
        except KeyError:
            return_action_as_tuple = True

        # Try/except to handle other indexing here
        i_best_q = np.argmax(these_q)

        if epsilon > 0:
            # Evenly distribute an epsilon-chance of randomness
            weights = np.ones(len(these_q)) * epsilon / len(these_q)

            # Give the best choice the rest
            weights[i_best_q] += 1.0 - epsilon

            action = np.random.choice(len(these_q), p=weights)
        else:
            action = i_best_q

        if return_action_as_tuple:
            action = self.env.index_to_action[action]

        return action

    def converged(self):
        """
        Returns True if solver is converged.

        Returns:
            bool: Convergence status (True=converged)
        """
        logger.debug(f'Assessing convergence')
        # Try to use a previously memorized convergence result (converged field indicates whether this convergence
        # test was previously True/False for at this iteration)
        if self._iteration < self._min_iters:
            logger.debug(f"Not converged: iteration ({self._iteration}) < min_iters ({self._min_iters})")
            return False
        else:
            try:
                logger.debug(f'Returning memorized result ({self.iteration_data.get(-1)["converged"]})')
                return self.iteration_data.get(-1)['converged']
            except IndexError:
                # Data store has no records and thus cannot be converged
                logger.debug(f"Not converged: IndexError (not enough data)")
                return False
            except KeyError:
                # No converged field exists - convergence has not been previous assessed
                logger.debug(f'Found record without converged field - new convergence assessment required')
                pass

            # Check last self.num_episodes_for_convergence to ensure they have deltas lower than the convergence limit
            try:
                for i in range(1, self.num_episodes_for_convergence + 1):
                    if self.iteration_data.get(-i)['delta_max'] > self._value_function_tolerance:
                        logger.debug(f"Convergence failed - iter {self.iteration_data.get(-i)['iteration']} (now - {i-1}) "
                                     f"delta_max={self.iteration_data.get(-i)['delta_max']} "
                                     f"(> {self._value_function_tolerance})")
                        return False
                logger.debug(f'Convergence = True')
                return True
            except IndexError:
                # Data store does not have enough records to assess convergence and thus cannot be converged
                logger.debug(f"Not converged: IndexError (not enough data)")
                return False
            except KeyError:
                raise KeyError("Iteration Data has no delta_max field - cannot determine convergence status")

    def get_q_at_state(self, state):
        """
        Returns a numpy array of q values at the current state in the same order as the standard action indexing
        Args:
            state (int, tuple): Descriptor of current state in environment

        Returns:
            np.array: Numpy array of q for all actions
        """
        actions = list(range(self.env.action_space.n))
        try:
            # This will work if q is indexed by integers for action
            these_q = np.array([self.q[(state, action)] for action in actions])
        except KeyError:
            # Otherwise, try converting action index to tuple and then merging the tuples
            these_q = np.array([self.q[state + self.env.index_to_action[action]] for action in actions])

        return these_q

    def init_q(self, init_val=0.0):
        """
        Initialize self.q, a dict-like DictWithHistory object for storing the state-action value function q

        Args:
            init_val (float): Value to give all states in the initialized q

        Returns:
            None
        """
        # Index by whatever the environment indexes by
        state_keys = list(self.env.P.keys())
        action_keys = list(range(self.env.action_space.n))

        if isinstance(state_keys[0], int):
            state_action_keys = [(state_key, action_key) for state_key in state_keys for action_key in action_keys]
        else:
            action_keys = [self.env.index_to_action[action] for action in action_keys]
            state_action_keys = [state_key + action_key for state_key in state_keys for action_key in action_keys]

        self.q = DictWithHistory(timepoint_mode='explicit')
        for k in state_action_keys:
            self.q[k] = init_val

    @property
    def alpha(self):
        """Returns value of alpha at current iteration"""
        return decay_functions(self._alpha_settings)(self._iteration)

    @property
    def epsilon(self):
        """Returns value of epsilon at current iteration"""
        return decay_functions(self._epsilon_settings)(self._iteration)


def decay_functions(settings):
    """
    Returns a decay function that accepts timestep as argument and returns a value

    Essentially a pre-configured interpolator with different possible settings.  Return from this function is a
    lambda function of signature **function(timestep)** where timestep is the value to interpolate on, and function is
    preconfigured using a given interpolation schedule.

    Schedules supported:

    * constant: Returns a constant value, regardless of input timestep

      Args:
          initial_value (float): The value to be returned when the function is invoked
    * linear: Returns a linearly interpolated value, with extrapolation outside the given range being constant

      Args:
          initial_value (float): The value to be returned when the function is invoked with timestep <= initial_timestep

          initial_timestep (int): The timestep associated with the initial value

          final_value (float): The value to be returned when the function is invoked with timestep >= initial_timestep

          final_timestep (int): The timestep associated with the final value

      Return logic:
          If timestep <= initial_timestep: return initial_value

          elif initial_timestep < timestep < final_timestep: linearly interpolate between initial_value and final_value

          elif timestep >= final_timestep: return final_value

    FUTURE: Add exponential decay

    Args:
        settings (dict): Contains at least "type" and other values as described above based on chosen type

    Returns:
        function: Function with signature value_at_timestep = function(timestep)
    """
    if settings['type'] == 'constant':
        return lambda timestep: settings['initial_value']
    elif settings['type'] == 'linear':
        return lambda timepoint: np.interp(x=timepoint,
                                           xp=[settings['initial_timestep'], settings['final_timestep']],
                                           fp=[settings['initial_value'], settings['final_value']],
                                           )
    else:
        raise ValueError(f"Invalid function type {settings['type']}")
