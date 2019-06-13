import numpy as np

from .base_solver import BaseSolver
from lrl.data_stores import WalkStatistics, DictWithHistory
from lrl.utils.misc import Timer, count_dict_differences, dict_differences

import logging
logger = logging.getLogger(__name__)

MAX_STEPS_PER_EPISODE = 500
MAX_ITERATIONS = 2000
NUM_EPISODES_FOR_CONVERGENCE = 20


class QLearning(BaseSolver):
    """Solver class for Q-Learning

    FUTURE: Improve this docstring.  Add refs
    """
    def __init__(self, env, max_steps_per_episode=MAX_STEPS_PER_EPISODE, discount_factor=0.9,
                 alpha=0.1, epsilon=0.1, max_iters=MAX_ITERATIONS,
                 num_episodes_for_convergence=NUM_EPISODES_FOR_CONVERGENCE, **kwargs):
        # FUTURE: alpha, epsilon accepted as number (coerce to float) or dict of {initial, minimum, decay_type, decay_val}
        super().__init__(env, max_iters=max_iters, **kwargs)

        self._alpha_initial = alpha
        self._epsilon_initial = epsilon
        self.discount_factor = discount_factor
        self.transitions = 0
        self.max_steps_per_episode = max_steps_per_episode

        # Estimate of Q, keyed by ((state), (action)) where state/action can be integers or qualified tuples
        self.q = None
        self.init_q()

        # Additional statistics during solving
        self.walk_statistics = WalkStatistics()

        self.num_episodes_for_convergence = num_episodes_for_convergence

        # String description of convergence criteria
        self.convergence_desc = f"{self.num_episodes_for_convergence} episodes with max delta in Q function < " \
            f"{self.value_function_tolerance}"

        # TODO: Remove this!
        self.time_on_policy_improvement = 0.0

    @property
    def alpha(self):
        # FUTURE: Complete this for decay settings
        return self._alpha_initial

    @property
    def epsilon(self):
        # FUTURE: Complete this for decay settings
        return self._epsilon_initial

    def step(self, count_transition=True):
        """
        Take and learn from a single step in the environment.

        FUTURE: Improve docstring.

        Returns:
            tuple of transition: (state, reward, next_state, is_terminal)
        """
        logger.debug(f'Taking and learning from a step in the environment (transition count = {self.transitions})')
        state = self.env.s

        # Make an epsilon-greedy choice
        action = self.choose_epsilon_greedy_action(state, self.epsilon)
        next_state, reward, is_terminal, _ = self.env.step(action)

        q_best_next_action = np.max(self.get_q_at_state(next_state))

        # Compute Q-Learning update (TD)
        # TODO: Is this eq different if this is a terminal step?  Memory says it should be different
        try:
            # This will work is q is indexed by integer state and action
            td = reward + self.discount_factor * q_best_next_action - self.q[state, action]
            self.q[state, action] += self.alpha * td
        except KeyError:
            # This will work if q is indexed by tuple state and action (merge the tuples for q index)
            td = reward + self.discount_factor * q_best_next_action - self.q[state + action]
            self.q[state + action] += self.alpha * td

        if count_transition:
            self.transitions += 1
        logger.debug(f'Completed step from {state} -> {next_state} yielding {reward} (terminal={is_terminal})')

        return state, reward, next_state, is_terminal

    def get_q_at_state(self, state):
        """
        Returns a numpy array of q values at the current state in the same order as the standard action indexing
        Args:
            state:

        Returns:

        """
        actions = list(range(self.env.action_space.n))
        try:
            # This will work if q is indexed by integers for action
            these_q = np.array([self.q[(state, action)] for action in actions])
        except KeyError:
            # Otherwise, try converting action index to tuple and then merging the tuples
            these_q = np.array([self.q[state + self.env.index_to_action[action]] for action in actions])

        return these_q

    def choose_epsilon_greedy_action(self, state, epsilon=None):
        """
        Return an action chosen by epsilon-greedy scheme based on the current estimate of Q

        Args:
            state:
            epsilon: Optional.  If None, self.epsilon is used

        Returns:

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

    def iterate(self):
        """
        Perform and learn from a single episode in the environment (one walk from start to finish)

        Side Effects:
            self.value: Updated to the newest estimate of the value function
            self.policy: Updated to the greedy policy according to the value function estimate
            self.iteration: Increment iteration counter by 1
            self.iteration_data: Add new record to iteration data store
            self.env: Reset and then walked through

        Returns:

        """
        logger.debug(f"Performing iteration (episode) {self.iteration} of Q-Learning")
        timer = Timer()
        total_reward = 0.0
        states = [self.env.reset()]
        rewards = [0.]

        # Remember old q function for later
        q_old = self.q.to_dict()
        policy_old = self.policy.to_dict()

        # Perform a single episode, learning along the way (this implicitly updates self.q)
        for i_step in range(self.max_steps_per_episode):
            state, reward, next_state, is_terminal = self.step()
            states.append(next_state)
            rewards.append(reward)

            if is_terminal:
                break
        logger.debug(f"Iteration {self.iteration} completed with r={sum(rewards)} in {len(states)} steps "
                     f"(terminal={is_terminal})")

        # Compute new greedy policy to compare to old policy
        # TODO: How costly is this?  Feels like it might be costly.  If so, omit during training?
        # FUTURE: Remove this once you have a feel for it
        timer_policy_improvement = Timer()
        self._policy_improvement()
        logging.debug(f'Policy improvement took {timer_policy_improvement.elapsed()}s')
        self.time_on_policy_improvement += timer_policy_improvement.elapsed()

        # Log metadata about iteration
        delta_max, delta_mean = dict_differences(self.q, q_old)
        policy_changes = count_dict_differences(self.policy, policy_old)
        logger.debug(f"Walk resulted in delta_max = {delta_max}, delta_mean = {delta_mean}, and {policy_changes} policy "
                     f"changes")

        self.iteration_data.add({'iteration': self.iteration,
                                 'time': timer.elapsed(),
                                 'delta_mean': delta_mean,
                                 'delta_max': delta_max,
                                 'steps': len(states),
                                 'policy_changes': policy_changes,
                                 })
        # Use converged function to assess convergence and add that back into iteration_data
        # (prevents duplicating the convergence logic, at the expense of more complicated logging logic)
        self.iteration_data.get(-1)['converged'] = self.converged()

        # Log more detailed metadata
        self.walk_statistics.add(reward=sum(rewards), walk=states, terminal=is_terminal)

        # Increment counters
        self.q.increment_timepoint()
        self.policy.increment_timepoint()
        self.iteration += 1

    def converged(self):
        logger.debug(f'Assessing convergence')
        # Try to use a previously memorized convergence result (converged field indicates whether this convergence
        # test was previously True/False for at this iteration)
        try:
            logger.debug(f'Returning memorized result ({self.iteration_data.get(-1)["converged"]})')
            return self.iteration_data.get(-1)['converged']
        except IndexError:
            # Data store has no records and thus cannot be converged
            logger.debug(f"IndexError (not enough data)")
            return False
        except KeyError:
            # No converged field exists - convergence has not been previous assessed
            logger.debug(f'Found record without converged field - new convergence assessment required')
            pass

        # Check last self.num_episodes_for_convergence to ensure they have deltas lower than the convergence limit
        try:
            for i in range(1, self.num_episodes_for_convergence + 1):
                if self.iteration_data.get(-i)['delta_max'] > self.value_function_tolerance:
                    logger.debug(f"Convergence failed - iter {self.iteration_data.get(-i)['iteration']} (now - {i-1}) "
                                 f"delta_max={self.iteration_data.get(-i)['delta_max']} "
                                 f"(> {self.value_function_tolerance})")
                    return False
            logger.debug(f'Convergence = True')
            return True
        except IndexError:
            # Data store does not have enough records to assess convergence and thus cannot be converged
            logger.debug(f"IndexError (not enough data)")
            return False
        except KeyError:
            raise KeyError("Iteration Data has no delta_max field - cannot determine convergence status")

    def _policy_improvement(self):
        """
        Update the policy to be greedy relative to the most recent q function

        Side Effects:
            self.policy: Updated to be greedy relative to self.q

        Args:
            None

        Returns:
            None
        """
        for state in self.policy.keys():
            self.policy[state] = self.choose_epsilon_greedy_action(state, epsilon=0)

    def init_q(self, init_val=0.0):
        """
        Initialize self.q, a dict-like DictWithHistory object for storing the state-action value function q

        Args:
            init_val:

        Returns:
            None
        """
        # Index by whatever the environment indexes by
        state_keys = list(self.env.P.keys())
        action_keys = list(range(self.env.action_space.n))

        if isinstance(state_keys[0], int):
            state_action_keys = zip(state_keys, action_keys)
        else:
            action_keys = [self.env.index_to_action[action] for action in action_keys]
            state_action_keys = [state_key + action_key for state_key in state_keys for action_key in action_keys]

        self.q = DictWithHistory(timepoint_mode='explicit')
        for k in state_action_keys:
            self.q[k] = init_val
