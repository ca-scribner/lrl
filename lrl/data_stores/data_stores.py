import pandas as pd
import numpy as np
from collections.abc import MutableMapping


class GeneralIterationData:
    """Class to store data about solver iterations

    Data is stored as a list of dictionaries.  This is a placeholder for more advanced storage.  Class gives a minimal
    set of extra bindings for convenience.

    The present object has no checks to ensure consistency between added records (all have same fields, etc.).  If any
    columns are missing from an added record, outputting to a dataframe will result in Pandas treating these as missing
    columns from a record.

    Args:
        columns (list): An optional list of column names for the data (if specified, this sets the order of the
                        columns in any output Pandas DataFrame or csv)

    DOCTODO: Add example of usage
    """

    def __init__(self, columns=None):
        #: list: Column names used for data output.
        #:
        #: If specified, this sets the order of any columns being output to Pandas DataFrame or csv
        self.columns = columns

        #: list: List of dictionaries representing records.
        #:
        #: Intended to be internal in future, but public at present to give easy access to records for slicing
        self.data = []

    def add(self, d):
        """
        Add a dictionary record to the data structure.

        Args:
            d (dict): Dictionary of data to be stored

        Returns:
            None
        """
        self.data.append(d)

    def get(self, i=-1):
        """
        Return the ith entry in the data store (index of storage is in order in which data is committed to this object)

        Args:
            i (int): Index of data to return (can be any valid list index, including -1 and slices)

        Returns:
            dict: ith entry in the data store
        """
        return self.data[i]

    def to_dataframe(self):
        """
        Returns the data structure as a Pandas DataFrame

        Returns:
            dataframe: Pandas DataFrame of the data
        """
        # Add structure so everything is in same order and not random from dict
        df = pd.DataFrame(self.data, columns=self.columns)
        return df

    def to_csv(self, filename, **kwargs):
        """
        Write data structure to a csv via the Pandas DataFrame

        Args:
            filename (str): Filename or full path to output data to
            kwargs (dict): Optional arguments to  be passed to DataFrame.to_csv()

        Returns:
            None
        """
        self.to_dataframe().to_csv(filename, index=False, **kwargs)


class WalkStatistics(object):
    """
    Container for statistics about a set of independent walks through an environment, typically following one policy

    Statistics are lazily computed and memorized

    DOCTODO: Add example usage.  show plot_episodes
    """
    def __init__(self):
        #: list: List of the total reward for each walk (raw data)
        self.rewards = []

        #: list: List of all walks passed to the data object (raw data)
        self.walks = []

        #: list: List of the total steps taken for each walk (raw data)
        self.steps = []

        #: list: List of whether each input walk was terminal (raw data)
        self.terminals = []

        #: list: List of dicts of computed statistics
        self._statistics = []

        #: list: Column names/order used for outputting to dataframe
        self._statistics_columns = ['walk_index', 'reward', 'steps', 'terminal',
                                    'reward_mean', 'reward_median', 'reward_std', 'reward_min', 'reward_max',
                                    'steps_mean', 'steps_median', 'steps_std', 'steps_min', 'steps_max',
                                    'terminal_fraction']

    def get_statistic(self, statistic='reward_mean', index=-1):
        """
        Return a lazily computed and memorized statistic about the rewards from walks 0 to index

        If the statistic has not been previous computed, it will be computed and returned.  See .get_statistics() for
        list of statistics available

        Side Effects:
            self.statistics[index] will be computed using self.compute() if it has not been already

        Args:
            statistic (str): See .compute() for available statistics
            index (int): Walk index for requested statistic

        Notes:
            Statistics are computed for all walks up to and including the requested statistic.  For example if walks
            have rewards of [1, 3, 5, 10], get_statistic('reward_mean', index=2) returns 3 (mean of [1, 3, 5]).

        DOCTODO: Example usage (show getting some statistics)

        Returns:
            int, float: Value of the statistic requested
        """
        return self.get_statistics(index)[statistic]

    def get_statistics(self, index=-1):
        """
        Return a lazily computed and memorized dictionary of all statistics about walks 0 to index

        If the statistic has not been previous computed, it will be computed here.

        Side Effects:
            self.statistics[index] will be computed using self.compute() if it has not been already

        Args:
            index (int): Walk index for requested statistic

        Returns:
            dict: Details and statistics about this iteration, with keys:

            **Details about this iteration:**

            * **walk_index** (*int*): Index of episode
            * **terminal** (*bool*): Boolean of whether this episode was terminal
            * **reward** (*float*): This episode's reward (included to give easy access to per-iteration data)
            * **steps** (*int*): This episode's steps (included to give easy access to per-iteration data)

            **Statistics computed for all episodes up to and including this episode:**

            * **reward_mean** (*float*):
            * **reward_median** (*float*):
            * **reward_std** (*float*):
            * **reward_max** (*float*):
            * **reward_min** (*float*):
            * **steps_mean** (*float*):
            * **steps_median** (*float*):
            * **steps_std** (*float*):
            * **steps_max** (*float*):
            * **steps_min** (*float*):
            * **terminal_fraction** (*float*):
        """
        if self._statistics[index] is None:
            self.compute(index=index)
        return self._statistics[index]

    def add(self, reward, walk, terminal):
        """
        Add a walk to the data store

        Args:
            reward (float): Total reward from the walk
            walk (list): List of states encoutered in the walk, including the starting and final state
            terminal (bool): Boolean indicating if walk was terminal (did environment say walk has ended)

        Returns:
            None
        """
        self.rewards.append(reward)
        self.steps.append(len(walk))
        self.walks.append(walk)
        self.terminals.append(terminal)
        self._statistics.append(None)

    def compute(self, index=-1, force=False):
        """
        Compute and store statistics about rewards and steps for walks up to and including the indexth walk

        Side Effects:
            self.statistics[index] will be updated

        Args:
            index (int or 'all'): If integer, the index of the walk for which statistics are computed.  Eg: If index==3,
                compute the statistics (see get_statistics() for list) for the series of walks from
                0 up to and not including 3 (typical python indexing rules)
                If 'all', compute statistics for all indices, skipping any that have been previously
                computed unless force == True
            force (bool):
                If True, always recompute statistics even if they already exist.

                If False, only compute if no previous statistics exist.

        Returns:
            None
        """
        if index == 'all':
            # Compute all indices by calling this recursively
            for i in range(len(self._statistics)):
                self.compute(index=i, force=force)
        else:
            # Compute a single index
            # Interpret index, which could be negative
            if index < 0:
                index = len(self._statistics) + index

            # Get an inclusive slice_end index for grabbing rewards
            slice_end = index + 1

            if (self._statistics[index] is None) or force:
                reward_array = np.array(self.rewards[:slice_end])
                steps_array = np.array(self.steps[:slice_end])
                terminals_array = np.array(self.terminals[:slice_end])
                self._statistics[index] = {
                    'reward': reward_array[-1],
                    'reward_mean': np.mean(reward_array),
                    'reward_median': np.median(reward_array),
                    'reward_std': np.std(reward_array),
                    'reward_max': np.max(reward_array),
                    'reward_min': np.min(reward_array),
                    'steps': steps_array[-1],
                    'steps_mean': np.mean(steps_array),
                    'steps_median': np.median(steps_array),
                    'steps_std': np.std(steps_array),
                    'steps_max': np.max(steps_array),
                    'steps_min': np.min(steps_array),
                    'walk_index': index,
                    'terminal': terminals_array[-1],
                    'terminal_fraction': terminals_array.sum() / terminals_array.shape[0],
                }

    def to_dataframe(self, include_walks=False):
        """
        Return a Pandas DataFrame of the walk statistics

        See .get_statistics() for a definition of each column.  Order of columns is set through self.statistics_columns

        Args:
            include_walks (bool): If True, add column including the entire walk for each iteration

        Returns:
            Pandas DataFrame
        """
        # Ensure everything is computed
        self.compute('all')

        # Return as a DataFrame
        df = pd.DataFrame(self._statistics, columns=self._statistics_columns)
        if include_walks:
            df['walks'] = self.walks
        return df

    def to_csv(self, filename, **kwargs):
        """
        Write statistics to csv via the Pandas DataFrame

        See .get_statistics() for a definition of each column.  Order of columns is set through self.statistics_columns

        Args:
            filename (str): Filename or full path to output data to
            kwargs (dict): Optional arguments to be passed to DataFrame.to_csv()

        Returns:
            None
        """
        # Ensure everything is computed
        self.compute('all')
        self.to_dataframe().to_csv(filename, index=False, **kwargs)

    def __str__(self):
        self.compute()

        return 'walk: {}, reward: {}, reward_mean: {}, reward_std: {}, reward_max: {}, reward_min: {}, ' \
               'steps: {}, steps_mean: {}'.format(
                    self.get_statistic(statistic='walk_index', index=-1),
                    self.get_statistic(statistic='reward', index=-1),
                    self.get_statistic(statistic='reward_mean', index=-1),
                    self.get_statistic(statistic='reward_std', index=-1),
                    self.get_statistic(statistic='reward_max', index=-1),
                    self.get_statistic(statistic='reward_min', index=-1),
                    self.get_statistic(statistic='steps', index=-1),
                    self.get_statistic(statistic='steps_mean', index=-1),
               )


class DictWithHistory(MutableMapping):
    """
    Dictionary-like object that maintains a history of all changes, either incrementally or at set timepoints

    This object has access like a dictionary, but stores data internally such that the user can later recreate the state
    of the data from a past timepoint.

    The intended use of this object is to store large objects which are iterated on (such as value or policy functions)
    in a way that a history of changes can be reproduced without having to store a new copy of the object every time.
    For example, when doing 10000 episodes of Q-Learning in a grid world with 2500 states, we can retain the full
    policy history during convergence (eg: answer "what was my policy after episode 527") without keeping 10000 copies
    of a nearly-identical 2500 element numpy array or dict.  The cost for this is some computation, although this
    generally has not been seen to be too significant (~10's of seconds for a large Q-Learning problem in testing)

    Args:
        timepoint_mode (str): One of:

        * explicit: Timepoint incrementing is handled explicitly by the user (the timepoint only changes if the user
          invokes .update_timepoint()
        * implicit: Timepoint incrementing is automatic and occurs on every setting action, including redundant sets
          (setting a key to a value it already holds).  This is useful for a timehistory of all sets to the object

        tolerance (float): Absolute tolerance to test for when replacing values.  If a value to be set is less than
            tolerance different from the current value, the current value is not changed.

    Warnings:
        * Deletion of keys is not specifically supported.  Deletion likely works for the most recent timepoint, but the
          history does not handle deleted keys properly
        * Numeric data may work best due to how new values are compared to existing data, although tuples have also been
          tested.  See __setitem__ for more detail

    DOCTODO: Add example
    """
    def __init__(self, timepoint_mode='explicit', tolerance=1e-7):
        #: float: Tolerance applied when deciding whether new data is the same as current data
        self._absolute_tolerance = tolerance

        #: list: Internal data storage
        self._data = {}

        #: str: See Parameters for definition
        self.timepoint_mode = None

        if timepoint_mode in ['explicit', 'implicit']:
            self.timepoint_mode = timepoint_mode
        else:
            raise ValueError(f'Invalid value for timepoint_mode "{timepoint_mode}"')

        #: int: Timepoint that will be written to next
        self.current_timepoint = 0

    def __getitem__(self, key):
        """
        Return the most recent value for key

        Returns:
            Whatever is contained in ._data[key][-1][-1] (return only the value from the most recent timepoint, not the
            timepoint associated with it)
        """
        return self._data[key][-1][-1]

    def __setitem__(self, key, value):
        """
        Set the value at a key if it is different from the current data stored at key

        Data stored here is stored under the self.current_timepoint.

        Difference between new and current values is assessed by testing:

        * new_value == old_value
        * np.isclose(new_value, old_value)

        where if neither returns True, the new value is taken to be different from the current value

        Side Effects:
            If timepoint_mode == 'implicit', self.current_timepoint will be incremented after setting data

        Args:
            key (immutable): Key under which data is stored
            value: Value to store at key

        Returns:
            None
        """
        if key not in self._data:
            self._data[key] = [(self.current_timepoint, value)]
        else:
            # If value is close to the most recent entry, do nothing will apply to a single numeric or an array of
            # numerics
            matches = False
            if self._data[key][-1][1] == value:
                matches = True
            else:
                try:
                    matches = np.all(np.isclose(self._data[key][-1][1], value, atol=self._absolute_tolerance, rtol=0.0))
                except (ValueError, TypeError):
                    # ValueError in np.all means the values we're comparing dont broadcast together (might have
                    # different sizes, etc), so they don't match.
                    # TypeError means they're not easily coerced into the same type, so that's even more different...
                    pass
            if not matches:
                # If value if not close to the most recent entry...
                if self._data[key][-1][0] == self.current_timepoint:
                    # ...but we already have an entry for the current timepoint, replace it with value
                    self._data[key][-1] = (self.current_timepoint, value)
                else:
                    # ...and there is no entry for the current timepoint, append one
                    self._data[key].append((self.current_timepoint, value))

        if self.timepoint_mode == 'implicit':
            self.increment_timepoint()

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get_value_history(self, key):
        """
        Returns a list of tuples of the value at a given key over the entire history of that key

        Args:
            key (immutable): Any valid dictionary key

        Returns:
            (list): list containing tuples of:

            * **timepoint** (*int*): Integer timepoint for this value
            * **value** (*float*): The value of key at the corresponding timepoint
        """
        return self._data[key]

    def get_value_at_timepoint(self, key, timepoint=-1):
        """
        Returns the value corresponding to a key at the timepoint that is closest to but not greater than timepoint

        Raises a KeyError if key did not exist at timepoint.  Raises an IndexError if no timepoint exists that applies

        Args:
            key (immutable): Any valid dictionary key
            timepoint (int): Integer timepoint to return value for.  If negative, it is interpreted like typical python
                             indexing (-1 means most recent, -2 means second most recent, ...)

        Returns:
            numeric: Value corresponding to key at the timepoint closest to but not over timepoint
        """
        # Handle offsets from most recent
        if timepoint < 0:
            timepoint = self.current_timepoint + timepoint + 1

        # Catch any bad timepoints
        if timepoint > self.current_timepoint or timepoint < 0:
            raise IndexError(f"Invalid timepoint {timepoint}")

        # Get the index of the timepoint that is closest to but not newer than timepoint
        timepoints = np.array([d[0] for d in self._data[key]])
        i_timepoint = np.searchsorted(timepoints, timepoint, side='right') - 1

        if i_timepoint < 0:
            raise KeyError(f'{key} is not a valid key at timepoint {timepoint}')
        return self._data[key][i_timepoint][1]

    def to_dict(self, timepoint=-1):
        """
        Return the state of the data at a given timepoint as a dict

        Args:
            timepoint (int): Integer timepoint to return data as of.  If negative, it is interpreted like typical python
                             indexing (-1 means most recent, -2 means second most recent, ...)

        Returns:
            dict: Data at timepoint
        """
        if timepoint == -1 or timepoint == self.current_timepoint:
            data = {k: self._data[k][timepoint][1] for k in self._data.keys()}
        else:
            data = {}
            for k in self._data.keys():
                try:
                    data[k] = self.get_value_at_timepoint(k, timepoint)
                except KeyError:
                    # Key did not exist in the past timepoint.  Skipping
                    pass
        return data

    def update(self, d):
        """
        Update this instance with a dictionary of data, d (similar to dict.update())

        Keys in d that are present in this object overwrite the previous value.  Keys in d that are missing in this
        object are added.

        All data written from d is given the same timepoint (even if timepoint_mode=implicit) - the addition is treated
        as a single update to the object rather than a series of updates.

        Args:
            d (dict): Dictionary of data to be added here

        Returns:
            None
        """
        # Override the timepoint mode temporarily if necessary.
        timepoint_mode_cache = None
        if self.timepoint_mode == 'implicit':
            timepoint_mode_cache = self.timepoint_mode
            self.timepoint_mode = 'explicit'

        for k, val in d.items():
            self[k] = val

        if timepoint_mode_cache:
            self.timepoint_mode = timepoint_mode_cache
            self.increment_timepoint()

    def increment_timepoint(self):
        """
        Increments the timepoint at which the object is currently writing

        Returns:
            None
        """
        self.current_timepoint += 1
