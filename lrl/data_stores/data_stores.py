import pandas as pd
import numpy as np
from collections import MutableMapping


class GeneralIterationData:
    """Class to store data about solver iterations

    FEATURE: Need to move this elsewhere

    Data is stored as a list of dictionaries.  This is a placeholder for more advanced storage.  Class gives a minimal
    set of extra bindings for convenience.

    """

    def __init__(self, index='iteration', columns=None):
        self._data = []
        self.index = index
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
        Return the ith entry in the data store

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
        # Index here is treated like any other column in the pandas array by default
        df = pd.DataFrame(self._data, columns=self.columns)

        # Old way, which made the index into the Pandas df index (doing it this way means accessing the index by column
        # name wont work, eg: df.loc[:, 'iteration'] does not work.
        # if self.index is not None:
        #     df = df.set_index(self.index)
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

    Statistics are lazily computed and memorized.
    """
    def __init__(self):
        self.rewards = []
        self.walks = []
        self.statistics = []
        self.steps = []

        # Column names/order used for outputting to dataframe
        self.statistics_columns = ['walk_index', 'reward', 'steps',
                                   'reward_mean', 'reward_median', 'reward_std', 'reward_min', 'reward_max',
                                   'steps_mean', 'steps_median', 'steps_std', 'steps_min', 'steps_max']

    def get_statistic(self, statistic='reward_mean', index=-1):
        """
        Return a lazily computed and memorized statistic about the rewards from walks 0 to index

        If the statistic has not been previous computed, it will be computed here

        Side Effects:
            self.statistics[index] will be computed using self.compute() if it has not been already

        Args:
            statistic: Can be (reward or step)
            index: Walk index at which statistics are computed (statistics are computed for walks 0 through index)

        Returns:
            int or float: Value of the statistic requested
        """
        if self.statistics[index] is None:
            self.compute(index=index)
        return self.statistics[index][statistic]

    def add(self, reward, walk):
        self.rewards.append(reward)
        self.steps.append(len(walk))
        self.walks.append(walk)
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
                self.statistics[index] = {}
                self.statistics[index]['reward'] = reward_array[-1]
                self.statistics[index]['reward_mean'] = np.mean(reward_array)
                self.statistics[index]['reward_median'] = np.median(reward_array)
                self.statistics[index]['reward_std'] = np.std(reward_array)
                self.statistics[index]['reward_max'] = np.max(reward_array)
                self.statistics[index]['reward_min'] = np.min(reward_array)
                self.statistics[index]['steps'] = steps_array[-1]
                self.statistics[index]['steps_mean'] = np.mean(steps_array)
                self.statistics[index]['steps_median'] = np.median(steps_array)
                self.statistics[index]['steps_std'] = np.std(steps_array)
                self.statistics[index]['steps_max'] = np.max(steps_array)
                self.statistics[index]['steps_min'] = np.min(steps_array)
                self.statistics[index]['walk_index'] = index

    def to_dataframe(self):
        """
        Return a Pandas DataFrame of the walk statistics

        Returns:
            Pandas DataFrame
        """
        # Ensure everything is computed
        self.compute('all')
        return pd.DataFrame(self.statistics, columns=self.statistics_columns)

    def to_csv(self, filename, **kwargs):
        """
        Write statistics to csv via the Pandas DataFrame

        Statistics are output such that each row represents the statistics up to and including that walk.  Each row
        includes fields for:
            walk_index: The walk index of this row
            reward, steps: The reward and steps obtained for this walk
            *_mean, *_median, *_min, *_max, *_std: Statistics for reward or steps results up to and including this walk

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
    def __init__(self, timepoint_mode='explicit'):
        """
        Initialize DictWithHistory

        Args:
            timepoint_mode (str):
                explicit: Timepoint incrementing is handled explicitly by the user (the timepoint only changes if the
                          user invokes .update_timepoint()
                implicit: Timepoint incrementing is automatic and occurs on every setting action, including redundant
                          sets (setting a key to a value it already holds).  This is useful for a timehistory of all
                          sets to the object
        """
        self._data = {}

        if timepoint_mode in ['explicit', 'implicit']:
            self.timepoint_mode = timepoint_mode
        else:
            raise ValueError(f'Invalid value for timepoint_mode "{timepoint_mode}"')

        # Current integer timepoint used for writing data
        self.current_timepoint = 0

    def __getitem__(self, item):
        # Return most recent _data[item]
        return self._data[item][-1]

    def __setitem__(self, key, value):
        if key not in self._data:
            self._data[key] = [(self.current_timepoint, type(value)(0.0))]
            print(f"Initializing self._data[{key}]={self._data[key][-1]}")

        # If value is close to the most recent entry, do nothing
        if not np.isclose(self._data[key][-1][1], value):
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

    def as_dict(self, timepoint=-1):
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

    def increment_timepoint(self):
        """
        Increments the timepoint at which the object is currently writing

        Returns:
            None
        """
        self.current_timepoint += 1
