import pandas as pd


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
        # Do something to make data
        # df = pd.DataFrame.from_dict(data)
        # df.set_index('iteration')
        df = pd.DataFrame(self._data, columns=self.columns)
        if self.index is not None:
            df = df.set_index(self.index)
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
        self.to_dataframe().to_csv(filename, **kwargs)
