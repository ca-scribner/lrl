import matplotlib.pyplot as plt


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


# Helpers
def get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax
