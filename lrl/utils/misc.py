from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np


# Helpers
# FEATURE: Move these to more logical places where people could find and use them


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


def count_dict_differences(d1, d2, raise_on_missing_key=True):
    """
    Return the number of differences between two dictionaries.  Useful to compare two policies stored as dictionaries.

    Optionally raise an error on missing keys (otherwise missing keys are counted as differences)

    Args:
        d1 (dict): Dictionary to compare
        d2 (dict): Dictionary to compare
        raise_on_missing_key (bool): If true, raise KeyError on any keys not shared by both dictionaries

    Returns:
        int: Number of differences between the two dictionaries

    """
    keys = d1.keys() | d2.keys()
    differences = 0
    for k in keys:
        try:
            if d1[k] != d2[k]:
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


# def plot_value_map(title, v, map_desc, color_map, map_mask=None):
#     """
#
#     :param title:
#     :param v:
#     :param map_desc:
#     :param color_map:
#     :param map_mask: (OPTIONAL) Defines a mask in the same shape of policy that indicates which tiles should be printed.
#                  Only elements that are True will have policy printed on the tile
#     :return:
#     """
#     if map_mask is None:
#         map_mask = np.ones(v.shape, dtype=bool)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # FEATURE: Fix this better
#     font_size = 'xx-small'
#     # font_size = 'x-large'
#     # if v.shape[1] > 16:
#     #     font_size = 'small'
#
#     v_min = np.min(v)
#     v_max = np.max(v)
#     # FEATURE: Disable this in more reasonble way.  Use input arg?
#     # bins = np.linspace(v_min, v_max, 100)
#     # v_red = np.digitize(v, bins)/100.0
#     # # Flip so that numbers are red when low, not high
#     # v_red = np.abs(v_red - 1)
#     for i in range(v.shape[0]):
#         for j in range(v.shape[1]):
#             value = np.round(v[i, j], 1)
#             if len(str(value)) > 3:
#                 font_size = 'xx-small'
#
#     plt.title(title)
#     for i in range(v.shape[0]):
#         for j in range(v.shape[1]):
#             y = v.shape[0] - i - 1
#             x = j
#             p = plt.Rectangle([x, y], 1, 1, edgecolor='k', linewidth=0.1)
#             p.set_facecolor(color_map[map_desc[i, j]])
#             ax.add_patch(p)
#
#             value = np.round(v[i, j], 1)
#
#             # red = v_red[i, j]
#             # if map_desc[i, j] in b'HG':
#             #     continue
#             if map_mask[i, j]:
#
#                 text2 = ax.text(x+0.5, y+0.5, value, size=font_size, weight='bold',
#                                 horizontalalignment='center', verticalalignment='center', color='k')
#                 # text2 = ax.text(x+0.5, y+0.5, value, size=font_size,
#                 #                 horizontalalignment='center', verticalalignment='center', color=(1.0, 1.0-red, 1.0-red))
#                 # text2.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                 #                        path_effects.Normal()])
#
#     plt.axis('off')
#     plt.xlim((0, v.shape[1]))
#     plt.ylim((0, v.shape[0]))
#     plt.tight_layout()
#
#     return watermark(plt)