from timeit import default_timer as timer
import numpy as np


# Helpers
# FUTURE: Move these to more logical places where people could find and use them


def print_dict_by_row(d, fmt='{key:20s}: {val:d}'):
    """Print a dictionary with a little extra structure, printing a different key/value to each line.

    Args:
        d (dict): Dictionary to be printed
        fmt (str): Format string to be used for printing.  Must contain key and val formatting references

    Returns:
        None

    """
    for k in d:
        print(fmt.format(key=str(k), val=d[k]))


def count_dict_differences(d1, d2, raise_on_missing_key=True, print_differences=False):
    """
    Return the number of differences between two dictionaries.  Useful to compare two policies stored as dictionaries.

    Does not properly handle floats that are approximately equal.

    Optionally raise an error on missing keys (otherwise missing keys are counted as differences)

    Args:
        d1 (dict): Dictionary to compare
        d2 (dict): Dictionary to compare
        raise_on_missing_key (bool): If true, raise KeyError on any keys not shared by both dictionaries
        print_differences (bool): If true, print all differences to screen

    Returns:
        int: Number of differences between the two dictionaries

    """
    keys = d1.keys() | d2.keys()
    differences = 0
    for k in keys:
        try:
            if d1[k] != d2[k]:
                if print_differences:
                    print(f'{k}: {d1.get(k, None)} != {d2.get(k, None)}')
                differences += 1
        except KeyError:
            if raise_on_missing_key:
                raise KeyError("Dictionaries do not have the same keys")
            else:
                differences += 1
    return differences


def dict_differences(d1, d2):
    """
    Return the maximum and mean of the absolute difference between all elements of two dictionaries

    Args:
        d1 (dict): Dictionary to compare
        d2 (dict): Dictionary to compare

    Returns:
        float: Maximum elementwise difference
        float: Sum of elementwise differences
    """
    keys = d1.keys() | d2.keys()
    delta_max = -np.inf
    delta_sum = 0.0
    for k in keys:
        delta = abs(d1[k] - d2[k])
        if delta > delta_max:
            delta_max = delta
        delta_sum += delta
    return delta_max, delta_sum / len(keys)


class Timer:
    """A Simple Timer class
    """
    def __init__(self):
        self.start = timer()

    def elapsed(self):
        return timer() - self.start


def rc_to_xy(row, col, rows):
    """
    Convert from (row, col) coordinates (eg: numpy array) to (x, y) coordinates (bottom left = 0,0)

   (x, y) convention:
        (0,0) in bottom left
        x +ve to the right
        y +ve up
    (row,col) convention:
        (0,0) in top left
        row +ve down
        col +ve to the right

    Args:
        row: This row
        col: This col
        rows: Total rows

    Returns:
        tuple: int x, int y
    """
    x = col
    y = rows - row - 1
    return x, y