from timeit import default_timer as timer
import numpy as np
from collections.abc import Mapping


def print_dict_by_row(d, fmt='{key:20s}: {val:d}'):
    """
    Print a dictionary with a little extra structure, printing a different key/value to each line.

    Args:
        d (dict): Dictionary to be printed
        fmt (str): Format string to be used for printing.  Must contain key and val formatting references

    Returns:
        None
    """
    for k in d:
        print(fmt.format(key=str(k), val=d[k]))


def count_dict_differences(d1, d2, keys=None, raise_on_missing_key=True, print_differences=False):
    """
    Return the number of differences between two dictionaries.  Useful to compare two policies stored as dictionaries.

    Does not properly handle floats that are approximately equal.  Mainly use for int and objects with __eq__

    Optionally raise an error on missing keys (otherwise missing keys are counted as differences)

    Args:
        d1 (dict): Dictionary to compare
        d2 (dict): Dictionary to compare
        keys (list): Optional list of keys to consider for differences.  If None, all keys will be considered
        raise_on_missing_key (bool): If true, raise KeyError on any keys not shared by both dictionaries
        print_differences (bool): If true, print all differences to screen

    Returns:
        int: Number of differences between the two dictionaries

    """
    if keys is None:
        keys = d1.keys() | d2.keys()
    else:
        # Coerce into a set to remove duplicates
        keys = set(keys)
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
    Return the maximum and mean of the absolute difference between all elements of two dictionaries of numbers

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
    """
    A Simple Timer class for timing code
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


def params_to_name(params, n_chars=4, sep='_', first_fields=None, key_remap=None):
    """
    Convert a mappable of parameters into a string for easy test naming

    Warnings:
        Currently includes hard-coded formatting for alpha and epsilon keys

    Args:
        params (dict): Dictionary to convert to a string
        n_chars (int): Number of characters per key to add to string.
                       Eg: if key='abcdefg' and n_chars=4, output will be 'abcd'
        sep (str): Separator character between fields (uses one of these between key and value, and two between
                   different key-value pairs
        first_fields (list): Optional list of keys to write ahead of other keys (otherwise, output order it sorted)
        key_remap (list): List of dictionaries of {key_name: new_key_name} for rewriting keys into more readable strings

    Returns:
        string
    """
    if first_fields is not None:
        keys = [key for key in first_fields if key in params.keys()]
    else:
        keys = []

    keys = keys + [key for key in sorted(params.keys()) if key not in keys]

    if key_remap is None:
        key_remap = {}

    s = ""
    for key in keys:
        try:
            key_printed_name = key_remap[key]
        except KeyError:
            key_printed_name = key

        if len(s) > 0:
            s += sep + sep

        # Add key name
        s += f"{str(key_printed_name)[:n_chars]}{sep}"

        # Handle special case of alpha/epsilon defined by dict
        parsed = False
        if key == 'alpha' or key == 'epsilon' and isinstance(params[key], Mapping):
            try:
                # Add additional remappings to keys to make alpha/epsilon print shorter
                s += f"{str(params[key]['initial_value'])}at{str(params[key]['initial_timestep'])}to" \
                    f"{str(params[key]['final_value'])}at{str(params[key]['final_timestep'])}"

                params_to_name(params[key], n_chars=n_chars, sep=sep, first_fields=first_fields)
                parsed = True
            except TypeError:
                # If this fails, just use the normal method...
                pass

        if not parsed:
            # Add param[key] contents
            if isinstance(params[key], Mapping):
                temp = params_to_name(params[key], n_chars=n_chars, sep=sep, first_fields=first_fields,
                                      key_remap=key_remap)
                s += f"{{{str(temp)}}}"
            else:
                s += f"{str(params[key])}"
    return s
