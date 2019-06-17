import matplotlib.pyplot as plt
import numpy as np
import csv

from lrl.utils.misc import rc_to_xy

import logging
logger = logging.getLogger(__name__)

# Default plot settings
DEFAULT_PLOT_FORMAT = 'png'
DEFAULT_PLOT_DPI = 150
MAX_PATHS_ON_EPISODE_PLOT = 100

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


def plot_env(env, ax=None, edgecolor='k', resize_figure=True, savefig=None):
    """
    FUTURE: Add docstring

    Args:
        env:
        ax:
        edgecolor:
        resize_figure: If true, resize the figure to:
         width  = 0.5 * n_cols inches
         height = 0.5 * n_rows inches
        savefig:
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

    if savefig:
        ax.get_figure().savefig(f'{savefig}.{DEFAULT_PLOT_FORMAT}', format=DEFAULT_PLOT_FORMAT, dpi=DEFAULT_PLOT_DPI)

    return ax


def plot_solver_results(env, solver=None, policy=None, value=None, savefig=None, **kwargs):
    """
    Convenience function to plot results from a solver over the environment map

    Input can be using a BaseSolver or child object, or by specifying policy and/or value directly via dict or
    DictWithHistory.

    See plot_solver_result() for more info on generation of individual plots and additional arguments for color/precision.

    Args:
        env: Augmented OpenAI Gym-like environment object
        solver (BaseSolver): Solver object used to solve the environment
        policy (dict, DictWithHistory): Policy for the environment, keyed by integer state-index or tuples of state
        value (dict, DictWithHistory): Value function for the environment, keyed by integer state-index or tuples of
                                          state
        savefig (str): If not None, save figures to this name.  For cases with multiple policies per grid square, this
                       will be the suffix on the name (eg: for policy at Vx=1, Vy=2, we get name of savefig_1_2.png)
        **kwargs (dict): Other arguments passed to plot_policy

    Returns:
        list of Matplotlib Axes for the plots
    """
    # Interpret a solver object if specified
    if solver is not None:
        if policy is not None or value is not None:
            raise ValueError("Invalid input - policy or value cannot be specified if solver given as input")
        else:
            policy = getattr(solver, 'policy', None)
            value = getattr(solver, 'value', None)

    # Break policy and value, which are dict-like containers mapping state->X, into a series of numpy arrays shaped the
    # same as env.desc.  See policy_dict_to_array() for more information
    # NOTE: This next section could have more rigorous detection of mismatching numbers of plots, etc., but that seemed
    # like low value.  I think it will raise exceptions if things go wrong as coded now, but not rigorously tested

    # Convert policy and value to their list of tuples, if they are specified, and count the number of plots
    number_of_plots = 1
    axes_titles = None
    policy_list_of_tuples = None
    value_list_of_tuples = None

    if policy is not None:
        # Try to coerce into dict in case it isn't already
        try:
            policy = policy.to_dict()
        except AttributeError:
            pass
        policy_list_of_tuples = policy_dict_to_array(env, policy)
        number_of_plots = max(len(policy_list_of_tuples), number_of_plots)
        axes_titles = [policy_list_of_tuples[i][0] for i in range(len(policy_list_of_tuples))]
    if value is not None:
        # Try to coerce into dict in case it isn't already
        try:
            value = value.to_dict()
        except AttributeError:
            pass
        value_list_of_tuples = policy_dict_to_array(env, value)
        number_of_plots = max(len(value_list_of_tuples), number_of_plots)
        axes_titles = [policy_list_of_tuples[i][0] for i in range(len(value_list_of_tuples))]

    # Apply default values to anything we don't have data for
    if policy is None:
        policy_list_of_tuples = [(None, None)] * number_of_plots
    if value is None:
        value_list_of_tuples = [(None, None)] * number_of_plots

    returned_axes = [None] * number_of_plots
    if axes_titles is None:
        axes_titles = returned_axes[:]

    for i in range(number_of_plots):
        if savefig and axes_titles[i]:
            # Add axes_title to savefig, if required.  Using csv/StringOP feels a bit clunky but it works nicely
            this_sio = csv.StringIO()
            csv_writer = csv.writer(this_sio, delimiter='_')
            csv_writer.writerow(axes_titles[i])

            this_savefig = f'{savefig}_{this_sio.getvalue().strip()}'
        else:
            this_savefig = savefig

        # Actual numpy array of policy/value are the second element of the list_of_tuples.  Title is the first (where
        # both titles should be the same)
        returned_axes[i] = plot_solver_result(env, policy_list_of_tuples[i][1], value_list_of_tuples[i][1],
                                              title=axes_titles[i], savefig=this_savefig)


def plot_policy(env, policy, **kwargs):
    """Convenience binding for plot_policy_or_value().  See plot_policy_or_value for more detail"""
    return plot_solver_result(env, policy=policy, **kwargs)


def plot_value(env, value, **kwargs):
    """Convenience binding for plot_policy_or_value().  See plot_policy_or_value for more detail"""
    return plot_solver_result(env, value=value, **kwargs)


def plot_solver_result(env, policy=None, value=None, ax=None, add_env_to_plot=True, hide_terminal_locations=True,
                       color='k', title=None, savefig=None,
                       size_policy='auto',
                       size_value='auto', value_precision=2):
    """
    FUTURE: Add docstring.  Plot result for a single xy map using a numpy array of shaped policy and/or value


    Args:
        env:
        policy (np.array): Policy for each grid square in the environment, in the same shape as env.desc
                           For plotting environments where we have multiple states for a given grid square (eg for
                           Racetrack), call plot_policy for each given additional state (eg: for v=(0, 0), v=(1, 0), ..)
        value:
        ax:
        add_env_to_plot:
        hide_terminal_locations (bool): If True, all known terminal locations will have no text printed (as policy here
                                        doesn't matter)
        color:
        title:
        savefig:
        size_policy:
        size_value:
        value_precision:

    Returns:
        Matplotlib Axes object
    """
    if not add_env_to_plot and not ((policy is not None) or (value is not None)):
        raise ValueError("Invalid input.  Arguments passed give nothing to plot!")

    fig, ax = get_ax(ax)

    if add_env_to_plot:
        ax = plot_env(env, ax=ax)

    rows, cols = env.desc.shape

    if size_policy == 'auto' and policy is not None:
        # Determine an appropriate font size for the policy text based on the longest policy to be printed
        f_char_len = lambda x: len(str(x))
        vectorized = np.vectorize(f_char_len)
        size_policy = choose_text_size(np.max(vectorized(policy)))

    if size_value == 'auto' and value is not None:
        # Determine an appropriate font size for the value text based on the longest value to be printed
        value = np.around(value, decimals=value_precision)
        f_char_len = lambda x: len(str(x))
        vectorized = np.vectorize(f_char_len)
        size_value = choose_text_size(np.max(vectorized(value)))

    # If policy and value are both being plotted, offset them vertically so they don't overlap
    policy_vertical_shift = 0.0
    value_vertical_shift = 0.0
    if policy is not None and value is not None:
        value_vertical_shift = 0.25
        policy_vertical_shift = -value_vertical_shift

    for row in range(rows):
        for col in range(cols):
            x, y = rc_to_xy(row, col, rows)

            if hide_terminal_locations:
                try:
                    # If we hide terminal locations and this is terminal, skip
                    if env.is_location_terminal[(int(x), int(y))]:
                        continue
                except AttributeError:
                    # environment does not have is_location_terminal.  Ignore request to hide
                    pass

            # Get center of the current grid square
            x_center = x + 0.5
            y_center = y + 0.5

            if policy is not None:
                # Translate the action into text to be added
                # Remap to special character if env enables this
                action = policy[row, col]
                try:
                    text = env.action_as_char[action]
                except (AttributeError, KeyError):
                    text = str(action)
                ax.text(x_center, y_center + policy_vertical_shift, text, weight='bold', size=size_policy,
                        horizontalalignment='center', verticalalignment='center', color=color)

            if value is not None:
                text = str(value[row, col])
                ax.text(x_center, y_center + value_vertical_shift, text, weight='bold', size=size_value,
                        horizontalalignment='center', verticalalignment='center', color=color)

    if title:
        ax.set_title(title)

    if savefig:
        ax.get_figure().savefig(f'{savefig}.{DEFAULT_PLOT_FORMAT}', format=DEFAULT_PLOT_FORMAT, dpi=DEFAULT_PLOT_DPI)
    return ax

# FUTURE: Add a "format plot" function to handle all plot default formatting (xy lims, removing labels, subtitle, ...)


def plot_episodes(episodes, env=None, add_env_to_plot=True, max_episodes=MAX_PATHS_ON_EPISODE_PLOT,
                  alpha=None, color ='k', title=None, ax=None, savefig=None):
    """
    FUTURE: docstring

    Args:
        episodes (list, WalkStatistics): Series of walks to be plotted.  If WalkStatistics instance, .walks will be
                                      extracted
        env:
        add_env_to_plot:
        alpha:
        color:
        title:
        ax:
        max_episodes:
        savefig:

    Returns:

    """
    # Attempt to extract walks from a WalkStatistics instance
    episodes = getattr(episodes, 'walks', episodes)

    fig, ax = get_ax(ax)

    if add_env_to_plot:
        ax = plot_env(env, ax=ax)

    i_episodes = np.arange(len(episodes))
    if len(episodes) > max_episodes:
        i_episodes = np.random.choice(i_episodes, size=max_episodes, replace=False)

    if alpha is None:
        # Below 0.002, the lines don't plot at all!
        alpha = max(1.0 / len(i_episodes), 0.002)

    for i_episode in i_episodes:
        episode = episodes[i_episode]
        ax = plot_episode(episode, env=env, add_env_to_plot=False, alpha=alpha, color=color, title=title, ax=ax)

    if savefig:
        ax.get_figure().savefig(f'{savefig}.{DEFAULT_PLOT_FORMAT}', format=DEFAULT_PLOT_FORMAT, dpi=DEFAULT_PLOT_DPI)

    return ax


def plot_episode(episode, env, add_env_to_plot=False, alpha=None, color='k', title=None, ax=None):
    """
    FUTURE: Docstring

    Args:
        episode:
        env:
        add_env_to_plot:
        alpha:
        color:
        title:
        ax:

    Returns:

    """
    fig, ax = get_ax(ax)

    if add_env_to_plot:
        ax = plot_env(env, ax=ax)

    # Create path
    x = np.zeros(len(episode))
    y = np.zeros(len(episode))

    for i_transition, state in enumerate(episode):
        # Handle index or tuple states in the episode
        try:
            state = env.index_to_state[state]
        except (IndexError, AttributeError, TypeError):
            pass
        x[i_transition] = state[0] + 0.5
        y[i_transition] = state[1] + 0.5

    # FUTURE: Capture direction of the transition here.  Could do a loop and plot each arrow individually, but I
    # believe that makes things really slow.  Tried using ax.quiver, but the arguments/scaling are hard to work with...
    ax.plot(x, y, '-o', color=color, alpha=alpha)
    if title:
        ax.set_title(title)

    return ax


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
    return min(40.0, 80.0 / n_chars) * boxsize


def policy_dict_to_array(env, policy_dict):
    """
    TODO: Clean this up
    Convert a policy stored as a dictionary into a dictionary of one or more policy numpy arrays shaped like env.desc

    Can also be used for a value_dict.

    policy_dict is a dictionary relating state to policy at that state in one of several forms.
    The dictionary can be keyed by state-index or a tuple of state (eg: (x, y, [other_state]), with x=0 in left
    column, y=0 in bottom row)
    If using tuples of state, state may be more than just x,y location as shown above, eg: (x, y, v_x, v_y).  If
    len(state_tuple) > 2, we must plot each additional state separately.

    Translate policy_dict into a policy_list_of_tuples of:
      [(other_state_0, array_of_policy_at_other_state_0),
       (other_state_1, array_of_policy_at_other_state_1),
      ... ]
    where the array_of_policy_at_other_state_* is in the same shape as env.desc (eg: cell [3, 2] of the array is the
    policy for the env.desc[3, 2] location in the env).

    Examples:
        If state is described by tuples of (x, y) (where there is a single unique state for each grid location), eg:
            policy_dict = {
                (0, 0): policy_0_0,
                (0, 1): policy_0_1,
                (0, 2): policy_0_2,
                ...
                (1, 0): policy_2_1,
                (1, 1): policy_2_1,
                ...
                (xmax, ymax): policy_xmax_ymax,
                }
        then a single-element list is returned of the form:
            returned = [
                (None, np_array_of_policy),
            ]
        where np_array_of_policy is of the same shape as env.desc (eg: the map), with each element corresponding to the
        policy at that grid location (for example, cell [3, 2] of the array is the policy for the env.desc[3, 2]
        location in the env).

        If state is described by tuples of (x, y, something_else, [more_something_else...]), for example if
        state = (x, y, Vx, Vy) like below:
            policy_dict = {
                (0, 0, 0, 0): policy_0_0_0_0,
                (0, 0, 1, 0): policy_0_0_1_0,
                (0, 0, 0, 1): policy_0_0_0_1,
                ...
                (1, 0, 0, 0): policy_1_0_0_0,
                (1, 0, 0, 1): policy_1_0_0_1,
                ...
                (xmax, ymax, Vxmax, Vymax): policy_xmax_ymax_Vxmax_Vymax,
        then a list is returned of the form:
            returned = [
            #   (other_state, np_array_of_policies_for_this_other_state)
                ((0, 0), np_array_of_policies_with_Vx-0_Vy-0),
                ((1, 0), np_array_of_policies_with_Vx-0_Vy-0),
                ((0, 1), np_array_of_policies_with_Vx-0_Vy-0),
                ...
                ((Vxmax, Vymax), np_array_of_policies_with_Vxmax_Vymax),
            ]
        where each element corresponds to a different combination of all the non-location state.  This means that each
        element of the list is:
            (Identification of this case, shaped xy-grid of policies for this case)
        and can be easily plotted over the environment's map.

        If policy_dict is keyed by state-index rather than state directly, the same logic as above still applies.

        NOTE: If using an environment (with policy keyed by either index or state) that has more than one unique state
        per grid location (eg: state has more than (x, y)), then environment must also have an index_to_state attribute
        to identify overlapping states.  This constraint exists both for policies keyed by index or state, but the code
        could be refactored to avoid this limitation for state-keyed policies if required.

    Args:
        env: Augmented OpenAI Gym-like environment object
        policy_dict (dict): Dictionary of policy for the environment, keyed by integer state-index or tuples of state

    Returns:
        list of (description, shaped_policy) elements as described above

    """
    # Convert the policy dict into a list of policy indexed by the integer state number for each state (rather than
    # a tuple of state).  The type of the object contained in the list will be determined by sampling the type of an
    # arbitrary element of policy_dict
    policy_by_index = np.empty(len(policy_dict), dtype=type(next(iter(policy_dict.values()))))
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


def get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax
