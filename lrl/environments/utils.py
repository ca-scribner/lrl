def get_terminal_locations(env):
    """
    Given an environment, return whether each grid location is terminal as a dictionary keyed by (x, y) location

    Definition of terminal here is a location the player cannot legally enter without having the game end (eg: all
    transitions pointing to this location are terminal)

    Args:
        env:

    Returns:

    """
    # Build is_terminal to have an entry for each (x, y) location.
    # If P is keyed by state tuples, take the first two values in the tuple.  Else, try to convert from index to state.
    try:
        # This will work if state is a tuple.  Take just the x,y location from state
        is_terminal = {k[:2]: True for k in env.P}
    except TypeError:
        # If state is not subscriptable, try to convert to a tuple
        is_terminal = {env.index_to_state[k]: True for k in env.P}

    print(f'is_terminal = {is_terminal} (before updates)')

    # For all transitions, if non-terminal then set the corresponding is_terminal entry to False
    for state in env.P:
        for action in env.P[state]:
            for transition in env.P[state][action]:
                if not transition[3]:
                    # Non-terminal
                    try:
                        # This will work if state is a tuple.  Take just the x,y location from state
                        is_terminal[transition[1][:2]] = False
                    except TypeError:
                        # If transition[1] is not subscriptable, try to convert index to state
                        is_terminal[env.index_to_state[transition[1]]] = False
    print(f'is_terminal = {is_terminal} (after updates)')

    return is_terminal
