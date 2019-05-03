class GeneralIterationData:
    """Class to store data about solver iterations

    FEATURE: Need to move this elsewhere

    Data is stored as a list of dictionaries.  This is a placeholder for more advanced storage.  Class gives a minimal
    set of extra bindings for convenience.

    """

    def __init__(self):
        self._data = []

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
        pass

    def to_csv(self, filename):
        pass
