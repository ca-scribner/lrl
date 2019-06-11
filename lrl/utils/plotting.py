import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(__name__)

# Plotting for BaseSolver objects and data

def plot_solver_convergence(solver, **kwargs):
    """
    Convenience binding to plot convergence statistics for a set of solver objects.

    Also useful as a recipe for custom plotting.

    Args:
        solver (BaseSolver (or child)): Solver object to be plotted
        Other args: See plot_solver_convergence_from_df()

    Returns:
        Axes: Matplotlib axes object
    """
    return plot_solver_convergence_from_df(solver.iteration_data.to_dataframe(), **kwargs)


def plot_solver_convergence_from_df(df, y='delta_max', y_label=None, x='iteration', x_label='Iteration',
                                    data_label=None, ax=None, savefig=None):
    """
    Convenience binding to plot convergence statistics for a set of solver objects.

    Also useful as a recipe for custom plotting.

    Args:
        df (pandas.DataFrame): DataFrame with solver convergence data
        y (str): Convergence statistic to be plotted (eg: delta_max, delta_mean, time, or policy_changes)
        y_label (str): Optional label for y_axis (if omitted, will use y as default name unless axis is already labeled)
        x (str): X axis data (typically 'iteration', but could be any convergence data)
        x_label (str): Optional label for x_axis (if omitted, will use 'Iteration')
        data_label (str): Optional label for the data set (shows up in axes legend)
        ax (Axes): Optional Matplotlib Axes object to add this line to
        savefig (str): Optional filename to save the figure to

    Returns:
        Axes: Matplotlib axes object
    """

    fig, ax = get_ax(ax)
    ax.plot(df.loc[:, x], df.loc[:, y], label=data_label)

    if x_label is not None or (ax.get_xlabel() == ''):
        if x_label is None:
            x_label = x
        ax.set_xlabel(x_label)
    if y_label is not None or (ax.get_ylabel() == ''):
        if y_label is None:
            y_label = y
        ax.set_ylabel(y_label)

    if savefig:
        fig.savefig(savefig)

    return ax


def plot_env(env, ax=None, edgecolor='k', resize_figure=True):
    """
    FUTURE: Add docstring

    Args:
        env:
        ax:
        edgecolor:
        resize_figure: If true, resize the figure to:
         width  = 0.5 * n_cols inches
         height = 0.5 * n_rows inches

    Returns:

    """
    fig, ax = get_ax(ax)

    rows, cols = env.desc.shape
    if resize_figure:
        fig.set_size_inches(cols, rows)

    # Plot each cell of the map as a 1x1 square on the plot.  Note the different numberings:
    # Numbering:
    #   rows: top row is row=0
    #   cols: left col is col=0
    #   x: left col is x=0
    #   y: bot row is y=0

    for row in range(rows):
        for col in range(cols):
            x, y = rc_to_xy(row, col, rows)
            char = env.desc[row, col]
            try:
                facecolor = env.color_map[char]
            except Exception as e:
                facecolor = 'w'
            patch = plt.Rectangle((x, y), width=1, height=1, edgecolor=edgecolor, facecolor=facecolor, linewidth=0.1)
            ax.add_patch(patch)

    ax.axis('off')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    fig.tight_layout()
    return ax


def plot_policies(env, policies, **kwargs):
    """
    TODO: Docstring.  Must handle things like file naming if using savefig (savefig_extra-state?)

    Args:
        env:
        policies:
        **kwargs:

    Returns:

    """
    raise NotImplementedError()


def plot_policy(env, policy, ax=None, color='k', add_env_to_plot=False, size='auto', title=None):
    """
    FUTURE: Add docstring

    Args:
        env:
        policy (np.array): Policy for each grid square in the environment, in the same shape as env.desc
                           For plotting environments where we have multiple states for a given grid square (eg for
                           Racetrack), call plot_policy for each given additional state (eg: for v=(0, 0), v=(1, 0), ..)
        ax:
        color:
        add_env_to_plot:
        size: TODO: AUTO

    Returns:

    """

    fig, ax = get_ax(ax)

    if add_env_to_plot:
        ax = plot_env(env, ax=ax)

    rows, cols = env.desc.shape

    if size == 'auto':
        # Determine an appropriate font size for the policy text based on the longest policy to be printed
        f_char_len = lambda x: len(str(x))
        vectorized = np.vectorize(f_char_len)
        size = choose_text_size(np.max(vectorized(policy)))

    for row in range(rows):
        for col in range(cols):
            x, y = rc_to_xy(row, col, rows)

            # Get center of the current grid square
            x_center = x + 0.5
            y_center = y + 0.5

            # Translate the action into text to be added
            # Remap to special character if env enables this
            action = policy[row, col]
            try:
                action = env.action_as_char[action]
            except (AttributeError, KeyError):
                action = str(action)

            ax.text(x_center, y_center, action, weight='bold', size=size,
                    horizontalalignment='center', verticalalignment='center', color=color)

    if title:
        ax.set_title(title)
    return ax

# FUTURE: Add a "format plot" function to handle all plot default formatting (xy lims, removing labels, subtitle, ...)


# Helpers
def choose_text_size(n_chars, boxsize=1.0):
    """
    Helper to choose an appropriate text size when plotting policies.  Size is chosen based on length of text

    Return is calibrated
    Args:
        n_chars: Text caption to be added to plot
        boxsize (float): Size of box inside which text should print nicely.  Used as a scaling factor.  Default is 1 inch

    Returns:
        Matplotlib-style text size argument
    """
    return min(40.0, 100.0 / n_chars) * boxsize


def policy_dict_to_array(env, policy_dict):
    """
    TODO: Clean this up
    Convert a policy stored as a dictionary into a dictionary of one or more numpy arrays holding the policy

    Policies are

    If policy_dict is keyed by integers denoting state, eg:
        policy_dict = {
            0: policy_0,
            1: policy_1,
            2: policy_2,
            ...
            n: policy_n,
    where all integers from 0..n have an entry, then a single-element list is returned of the form:
        returned = [
            (None, np_array_of_policy),
        ]

    Where:
        np_array_of_policy = [

    returned:
        returned = {

        }

    Args:
        env:
        policy_dict:

    Returns:

    """
    # Convert the policy dict into a list of policy indexed by the integer state number for each state (rather than
    # a tuple of state)
    policy_by_index = np.empty(len(policy_dict), dtype=type(list(policy_dict.keys())[0]))
    try:
        # If state is a tuple, pull out each policy using index_to_state to get a state-index ordered array
        for i in range(len(policy_dict)):
            policy_by_index[i] = policy_dict[env.index_to_state[i]]
    except (KeyError, AttributeError):
        # If AttributeError (env has no index_to_state) or KeyError (index_to_state doesn't haved the keys of
        # policy_dict_by_state) then we likely have a policy_dict that is keyed by state index already
        for i in range(len(policy_dict)):
            policy_by_index[i] = policy_dict[i]

    # Now reshape the array into numpy arrays of the same shape as the env.desc (environment map).  If
    # policy_shaped_array.shape[2] > 1 then we have additional non-xy-location state variables (eg: velocity in
    # Racetrack)
    policy_shaped_array = policy_by_index.reshape(*env.desc.shape, -1)

    # Convert policy to a list of tuples of:
    #   [(non-xy-location_state_vars_0, shaped_policy_for_non-loc_state_vars_0),
    #    (non-xy-location_state_vars_1, shaped_policy_for_non-loc_state_vars_1),
    #    (non-xy-location_state_vars_2, shaped_policy_for_non-loc_state_vars_2),
    #     ...]
    # where the non-location vars are everything after location in state, eg:
    #   for state = (0, 3, 1, -1)
    # we have non-location state of (1, -1) at location (0, 3).  Data is passed back in this way so that we can then
    # output policy maps for each combination of non-location state variables, eg a map of policy with velocity=(0,0),
    # velocity=(1,0), ...
    # If we have only xy-location in state (eg: len(state)==2), then non-xy-location_state_vars = None
    if policy_shaped_array.shape[2] > 1:
        returned = [(env.index_to_state[i][2:], policy_shaped_array[:, :, i])
                    for i in range(policy_shaped_array.shape[2])]
    else:
        returned = [(None, policy_shaped_array[:, :, 0])]

    return returned


def rc_to_xy(row, col, rows):
    """
    Convert from (row, col) coordinates (eg: numpy array) to (x, y) coordinates (bottom left = 0,0)

    Convention:
      rows: top row is row=0
      cols: left col is col=0
      x: left col is x=0
      y: bot row is y=0

    Args:
        row:
        col:
        rows:

    Returns:
        tuple: int x, int y
    """
    x = col
    y = rows - row - 1
    return x, y

def get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax
