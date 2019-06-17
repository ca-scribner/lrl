import pandas as pd
import numpy as np
from collections.abc import MutableMapping


class GeneralIterationData:
    """Class to store data about solver iterations

    Data is stored as a list of dictionaries.  This is a placeholder for more advanced storage.  Class gives a minimal
    set of extra bindings for convenience.

    The present object has no checks to ensure consistency between added records (all have same fields, etc.).  If any
    columns are missing from an added record, the underlying Pandas DataFrame will treat these as default missing
    records.
    """

    def __init__(self, columns=None):
        """
        Initialize an instance of the class

        Args:
            columns (list): An optional list of column names for the data (if specified, this sets the order of the
                            columns in any output Pandas DataFrame or csv)

        Returns:
            None
        """
        self._data = []
        self.columns = columns

    def add(self, d):
        """
        Appends a dictionary to the internal list data structure.

        Args:
            d (dict): Dictionary of data to be stored

        Returns:
            None
        """
        self._data.append(d)

    def get(self, i=-1):
        """
        Return the ith entry in the data store (index of storage is in order in which data is committed to this object)

        Args:
            i (int): Index of data to return (can be any valid list index, including -1 and slices)

        Returns:
            ith entry(ies) if the data store
        """
        return self._data[i]

    def to_dataframe(self):
        """
        Returns the data structure as a Pandas DataFrame

        Returns:
            dataframe: Pandas DataFrame of the data
        """
        # Add structure so everything is in same order and not random from dict
        df = pd.DataFrame(self._data, columns=self.columns)
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

    TODO: Explain attributes
    """
    def __init__(self):
        self.rewards = []
        self.walks = []
        self.statistics = []
        self.steps = []
        self.terminals = []

        # Column names/order used for outputting to dataframe
        self.statistics_columns = ['walk_index', 'reward', 'steps', 'terminal',
                                   'reward_mean', 'reward_median', 'reward_std', 'reward_min', 'reward_max',
                                   'steps_mean', 'steps_median', 'steps_std', 'steps_min', 'steps_max',
                                   'terminal_fraction']

    def get_statistic(self, statistic='reward_mean', index=-1):
        """
        Return a lazily computed and memorized statistic about the rewards from walks 0 to index

        If the statistic has not been previous computed, it will be computed here.  See .compute() for definition of
        statistics available

        Side Effects:
            self.statistics[index] will be computed using self.compute() if it has not been already

        Args:
            statistic: See .compute() for available statistics
            index: Walk index at which statistics are computed (statistics are computed for walks 0 through index)

        Returns:
            int or float: Value of the statistic requested
        """
        # if self.statistics[index] is None:
        #     self.compute(index=index)
        # return self.statistics[index][statistic]
        return self.get_statistics(index)[statistic]

    def get_statistics(self, index=-1):
        """
        Return a lazily computed and memorized dictionary of all statistics about the rewards from walks 0 to index

        If the statistic has not been previous computed, it will be computed here.  See .compute() for definition of
        statistics available

        Side Effects:
            self.statistics[index] will be computed using self.compute() if it has not been already

        Args:
            index: Walk index at which statistics are computed (statistics are computed for walks 0 through index)

        Returns:
            int or float: Value of the statistic requested
        """
        if self.statistics[index] is None:
            self.compute(index=index)
        return self.statistics[index]

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
        self.statistics.append(None)

    def compute(self, index=-1, force=False):
        """
        Compute and store statistics about rewards and steps for walks up to and including the indexth walk

        Side Effects:
            self.statistics[index] will be updated

        Args:
            index (int or 'all'): If integer, the index of the walk to compute statistics for (up to and including this
                                  walk).  Eg: If 3, compute statistics for index=3 (eg: reward_mean is the mean of
                                  rewards from walks 0, 1, 2, and 3).
                                  If 'all', compute statistics for all indices, skipping any that have been previously
                                  computed unless force == True
            force (bool): If True, always recompute statistics even if they already exist.  If False, only compute if no
                          previous statistics exist.

        Returns:
            None
        """
        if index == 'all':
            # Compute all indices by calling this recursively
            for i in range(len(self.statistics)):
                self.compute(index=i, force=force)
        else:
            # Compute a single index
            # Interpret index, which could be negative
            if index < 0:
                index = len(self.statistics) + index

            # Get an inclusive slice_end index for grabbing rewards
            slice_end = index + 1

            if (self.statistics[index] is None) or force:
                reward_array = np.array(self.rewards[:slice_end])
                steps_array = np.array(self.steps[:slice_end])
                terminals_array = np.array(self.terminals[:slice_end])
                self.statistics[index] = {
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

        Args:
            include_walks (bool): If True, add column including the entire walk for each iteration

        Returns:
            Pandas DataFrame
        """
        # Ensure everything is computed
        self.compute('all')
        df = pd.DataFrame(self.statistics, columns=self.statistics_columns)
        if include_walks:
            df['walks'] = self.walks
        return df

    def to_csv(self, filename, **kwargs):
        """
        Write statistics to csv via the Pandas DataFrame

        Statistics are output such that each row represents the statistics up to and including that walk.  Each row
        includes fields for:
            walk_index: The walk index of this row
            reward, steps: The reward and steps obtained for this walk
            *_mean, *_median, *_min, *_max, *_std: Statistics for reward or steps results up to and including this walk
            terminal: Whether this walk was terminal (ended by the environment stating the game was finished)
            terminal_fraction: Fraction of runs up until this point that were terminal
        Order of columns is set through self.statistics_columns

        Args:
            filename (str): Filename or full path to output data to
            kwargs (dict): Optional arguments to  be passed to DataFrame.to_csv()

        Returns:
            None
        """
        self.compute('all')
        self.to_dataframe().to_csv(filename, index=False, **kwargs)

    def __str__(self):
        self.compute()

        return 'walk: {}, reward: {}, reward_mean: {}, reward_std: {}, reward_max: {}, reward_min: {}, ' \
               'runs: {}'.format(
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

    Content is stored in dictionary _data as a list of entries of (timepoint, content).  The most recent content is
    always available in _data[-1][1]

    Warnings:
        Deletion of keys is not specifically supported.  Deletion likely works for the most recent timepoint, but the
        history does not handle deleted keys properly
        Only numeric datatypes are supported.  This could be fixed with modifications to the checks for recording
        repeated data in __setitem__
        To avoid logging data just because of rounding error, items are only set if np.isclose(old_val, new_val) is
        False.  Default tolerance for isclose is currently used, but this could be improved easily if needed.
    """
    def __init__(self, timepoint_mode='explicit', tolernace=1e-7):
        """
        Initialize DictWithHistory

        Args:
            timepoint_mode (str):
                explicit: Timepoint incrementing is handled explicitly by the user (the timepoint only changes if the
                          user invokes .update_timepoint()
                implicit: Timepoint incrementing is automatic and occurs on every setting action, including redundant
                          sets (setting a key to a value it already holds).  This is useful for a timehistory of all
                          sets to the object
            tolerance (float): Absolute tolerance to test for when replacing values.  If a value to be set is less than
                               tolerance different from the current value, the current value is not changed.
        """
        self._data = {}

        if timepoint_mode in ['explicit', 'implicit']:
            self.timepoint_mode = timepoint_mode
        else:
            raise ValueError(f'Invalid value for timepoint_mode "{timepoint_mode}"')

        # Current integer timepoint used for writing data
        self.current_timepoint = 0

        self.absolute_tolerance = tolernace

    def __getitem__(self, key):
        # Return most recent _data[item]
        """
        Return the most recent value for key

        Returns:
            Whatever is contained in ._data[key][-1][-1] (return only the most most recent timepoint, not the
            timepoint associated with it)
        """
        return self._data[key][-1][-1]

    def __setitem__(self, key, value):
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
                    matches = np.all(np.isclose(self._data[key][-1][1], value, atol=self.absolute_tolerance, rtol=0.0))
                except (ValueError, TypeError):
                    # ValueError in np.all means the values we're comparing dont broadcast together (might have different
                    # sizes, etc), so they don't match.
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

    def get_value_at_timepoint(self, key, timepoint=-1):
        """
        Returns the value corresponding to a key at the timepoint that is closest to but not over timepoint.

        Raises a KeyError if key did not exist at timepoint.

        Args:
            key (immutable): Any valid dictionary key
            timepoint (int): Integer timepoint to return value for.  If negative, it is interpreted like typical python
                             slicing (-1 means most recent, -2 means second most recent, ...)

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
                             slicing (-1 means most recent, -2 means second most recent, ...)

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
