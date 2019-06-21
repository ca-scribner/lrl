import numpy as np

from gym.envs.toy_text import discrete


RACETRACK_DEFAULT_REWARDS = {
    'crash': -100,
    'time': -1,
    'goal': 100,
}

# Character map for the map characters (say that 10 times fast!)
CHAR_MAP = {
    "G": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['crash'], 'terminal': True, 'color': 'forestgreen'},
    "S": {'start_prob': 1., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'lightgreen'},
    " ": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'darkgray'},
    "F": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['goal'], 'terminal': True, 'color': 'gold'},
    "O": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'gray', 'chance_to_slip': 0.25}
}

TRACKS = {
    # Naming of tracks is "number_of_units_in_x by number_of_units_in_y"
    # These are default tracks, but you can make your own!
    '3x4_basic': [
        "GGG",
        "GFG",
        "GSG",
        "GGG",
    ],
    '5x4_basic': [
        "GGGGG",
        "G   G",
        "GSGFG",
        "GGGGG",
    ],
    '10x10': [
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GGG     FF",
        "GGG     FF",
        "GG   GGGGG",
        "GG   GGGGG",
        "GG    GGGG",
        "GG   SGGGG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
    ],
    '10x10_basic': [
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GG     FGG",
        "GG      GG",
        "GG      GG",
        "GG      GG",
        "GG      GG",
        "GGS     GG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
    ],
    '10x10_all_oil': [
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GGOOOOOFGG",
        "GGOOOOOOGG",
        "GGOOOOOOGG",
        "GGOOOOOOGG",
        "GGOOOOOOGG",
        "GGSOOOOOGG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
    ],
    '15x15_basic': [
        "GGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGG",
        "GG         FGG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GG          GG",
        "GGS         GG",
        "GGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGG",
    ],
    '20x20_basic': [
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GG               FGG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GG                GG",
        "GGS               GG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
    ],
    '20x20_all_oil': [
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGOOOOOOOOOOOOOOOFGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGOOOOOOOOOOOOOOOOGG",
        "GGSOOOOOOOOOOOOOOOGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
    ],
    '30x30_basic': [
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
        "GG                         FGG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GG                          GG",
        "GGS                         GG",
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    ],
    '20x10_U_all_oil': [
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGOOOOOOOOOOOOOOOGG",
        "GGGOOOOOGGGGOOOOOOGG",
        "GGOOOOGGGGGGGGOOOOGG",
        "GGOOOGGGGGGGGGGOOOGG",
        "GGOOOOGGGGGGGGOOOOGG",
        "GGOOOOOOSGGGGGOOOOGG",
        "GGGGGGGGGGGGGGFFFFGG",
        "GGGGGGGGGGGGGGFFFFGG",
    ],
    '20x10_U': [
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGG     OOOO      GG",
        "GGG     GGGG      GG",
        "GGOOOOGGGGGGGGOOOOGG",
        "GGOOOGGGGGGGGGGOOOGG",
        "GG    GGGGGGGG    GG",
        "GG      SGGGGG    GG",
        "GGGGGGGGGGGGGGFFFFGG",
        "GGGGGGGGGGGGGGFFFFGG",
    ],
    '10x10_oil': [
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GGG   OOFF",
        "GGG     FF",
        "GG   GGGGG",
        "GG   GGGGG",
        "GG    GGGG",
        "GG   SGGGG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
    ],
    '20x15_risky': [
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GG                GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGGGGGGGGGG GG",
        "GG GGGGGOOOOGGGGG GG",
        "GGG GGGOGGGGOGGG GGG",
        "GGGG  SGGGGGGF  GGGG",
        "GGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGG",
    ]
}


class Racetrack(discrete.DiscreteEnv):
    """
    A car-race-like environment that uses location and velocity for state and acceleration for actions, in 2D

    Loosely inspired by the Racetrack example of Sutton and Barto's Reinforcement Learning (Exercise 5.8,
    http://www.incompleteideas.net/book/the-book.html)

    The objective of this environment is to traverse a racetrack from a start location to any goal location.
    Reaching a goal location returns a large reward and terminates the episode, whereas landing on a grass location
    returns a large negative reward and terminates the episode.  All non-terminal transitions return a small negative
    reward.  Oily road surfaces are non-terminal but also react to an agent's action stochastically, sometimes causing
    an Agent to "slip" whereby their requested action is ignored (interpreted as if a=(0,0)).

    The tiles in the environment are:

    * (blank): Clean open (deterministic) road
    * O: Oily (stochastic) road
    * G: (terminal) grass
    * S: Starting location (agent starts at a random starting location).  After starting, S tiles behave like open road
    * F: Finish location(s) (agent must reach any of these tiles to receive positive reward

    The state space of the environment is described by xy location and xy velocity (with maximum velocity being a
    user-specified parameter).  For example, s=(3, 5, 1, -1) means the Agent is currently in the x=3, y=5 location
    with Vx=1, Vy=-1.

    The action space of the environment is xy acceleration (with maximum acceleration being a user-specified parameter).
    For example, a=(-2, 1) means ax=-2, ay=-1.  Transitions are determined by the current velocity as well as the
    requested acceleration (with a cap set by Vmax of the environment), for example:

    * s=(3, 5, 1, -1), a=(-3, 1) --> s_prime=(1, 5, -2, 0)

    But if vx_max == +-1 then:

    * s=(3, 5, 1, -1), a=(-3, 1) --> s_prime=(2, 5, -1, 0)

    Note that sign conventions for location are:

    * x: 0 at leftmost column, positive to the right
    * y: 0 at bottommost row, positive up

    Args:
        track (list): List of strings describing the track (see racetrack_tracks.py for examples)
        x_vel_limits (tuple): (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
        y_vel_limits (tuple): (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
        x_accel_limits (tuple): (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
        y_accel_limits (tuple): (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
        max_total_accel (int): (OPTIONAL) Integer maximum total acceleration in one action.  Total acceleration is computed
                            by abs(x_a)+abs(y_a), representing the sum of change in acceleration in both directions
                            Default is infinite (eg: any accel described by x and y limits)

    Notes:
        See also discrete.DiscreteEnv for additional attributes, members, and arguments (missing due here to Sphinx bug
        with inheritance in docs)

    DOCTODO: Add examples
    """

    def __init__(self, track=None, x_vel_limits=None, y_vel_limits=None,
                 x_accel_limits=None, y_accel_limits=None, max_total_accel=2):

        if track is None:
            track = TRACKS['10x10']
        elif isinstance(track, str):
            track = TRACKS[track]
        #: list: List of strings describing track
        self.track = track

        #: np.array: Numpy character array of the track (better for printing on screen/accessing track at xy locations)
        self.desc = np.asarray(self.track, dtype='c')

        if x_vel_limits is None:
            x_vel_limits = (-2, 2)
        self._x_vel_limits = x_vel_limits

        if y_vel_limits is None:
            y_vel_limits = (-2, 2)
        self._y_vel_limits = y_vel_limits

        if x_accel_limits is None:
            x_accel_limits = (-2, 2)
        self.x_accel_limits = x_accel_limits

        if y_accel_limits is None:
            y_accel_limits = (-2, 2)
        self._y_accel_limits = y_accel_limits

        self._max_total_accel = max_total_accel

        self._char_map = CHAR_MAP
        #: dict: Map from grid tile type to display color
        self.color_map = {k.encode(): self._char_map[k]['color'] for k in self._char_map}

        # Generate an OpenAI Gym DiscreteEnv style P matrix
        n_states, n_actions, p, starting_probability, action_map, terminal_locations = \
            track_to_p_matrix(track=self.track, char_map=self._char_map,
                              x_vel_limits=self._x_vel_limits, y_vel_limits=self._y_vel_limits,
                              x_accel_limits=self.x_accel_limits, y_accel_limits=self._y_accel_limits,
                              max_total_accel=self._max_total_accel)

        # Instantiate the DiscreteEnv parent
        super().__init__(n_states, n_actions, p, starting_probability)

        #: list: Attribute to map from state index to full tuple describing state
        #:
        #: Ex: index_to_state[state_index] -> state_tuple
        self.index_to_state = [x[0] for x in list(self.P.items())]

        #: dict: Attribute to map from state tuple to state index
        #:
        #: Ex: state_to_index[state_tuple] -> state_index
        self.state_to_index = {k: i for i, k in enumerate(self.index_to_state)}
        self.index_to_action = action_map

        #: dict: Attribute to map whether a state is terminal (eg: no rewards/transitions leading out of state).
        #:
        #: Keyed by state tuple
        self.is_location_terminal = terminal_locations

        # super().__init__ sets self.s as an index, whereas Racetrack uses a state tuple.  Convert s to it's state tuple
        #: int, tuple: Current state (inherited from parent)
        self.s = list(self.P.items())[self.s][0]

    def reset(self):
        """
        Reset the environment to a random starting location

        :return: None
        """
        super().reset()
        self.s = list(self.P.items())[self.s][0]
        self.lastaction = None
        return self.s

    def render(self, mode='human', current_location='*'):
        """
        Render the environment.

        Warnings:
            This method does not follow the prototype of it's parent.  It is presently a very simple version for
            printing the environment's current state to the screen

        Args:
            mode: (NOT USED)
            current_location: Character to denote the current location

        Returns:
            None
        """
        print(f"Vx={self.s[2]}, Vy={self.s[3]}")
        print_track(self.track, xy=[self.s[0:2]], print_as=current_location)

    def step(self, a):
        """
        Take a step in the environment.

        This wraps the parent object's step(), interpreting integer actions as mapped to human-readable actions

        Args:
            a (tuple, int): Action to take, either as an integer (0..nA-1) or true action (tuple of (x_accel,y_accel))

        Returns:
            Next state, either as a tuple or int depending on type of state used
        """
        try:
            return super().step(a)
        except KeyError:
            a = self.index_to_action[a]
            return super().step(a)


# Helpers
def print_track(track, rc=None, xy=None, print_as='*'):
    """
    Print a track, optionally with locations highlighted.

    Locations can be specified by a list of row-col or x-y coordinates, but not a mixture

    Args:
        track (list): List of strings describing the track
        rc (list): Highlighted locations (printed as print_as)
        xy (list): Highlighted locations (printed as print_as)
        print_as (str): Character to use to print any highlighted locations

    Side Effects:
        Track is printed to screen

    Returns:
        None
    """
    if rc is None:
        if xy is None:
            rc = []
        else:
            rc = [xy_to_rc(track, *this_xy) for this_xy in xy]

    # If print_as is a single string, print all locations as that same code
    if isinstance(print_as, str):
        print_as = [print_as for _ in range(len(rc))]

    this_location = track.copy()
    for i, this_rc in enumerate(rc):
        this_location[this_rc[0]] = \
            this_location[this_rc[0]][:this_rc[1]] + print_as[i] + this_location[this_rc[0]][this_rc[1] + 1:]
    for line in this_location:
        print(line)


def track_to_p_matrix(track, char_map=CHAR_MAP, x_vel_limits=None, y_vel_limits=None,
                      x_accel_limits=None, y_accel_limits=None, max_total_accel=np.inf):
    """
    Converts a map described by a list of strings to P-matrix format for an OpenAI Gym Discrete environment.

    Maps are specified using the following characters:

    * G: (Grass) Terminal location with reward
    * S: Starting location (can be one or more) with small negative reward.  Note that starting state will always have
      0 velocity.  FUTURE: Add random velocity to start
    * " " (single space): Open track with small negative reward
    * F: Finish location with reward
    * O: Oil (slippery tile where an action randomly may not work as expected)

    Rewards and terminal status are assessed based on someone ENTERING that state (eg: if you travel from a starting
    location to a wall, you get the wall's reward and terminal status)

    Sign conventions:
    * x: 0 at leftmost column, positive to the right
    * y: 0 at bottommost row, positive up

    Args:
        track: List of strings describing the track
        char_map: Character map that maps track characters to P matrix
        x_vel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
        y_vel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
        x_accel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
        y_accel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
        max_total_accel: (OPTIONAL) Integer maximum total acceleration in one action.  Total acceleration is computed
            by abs(x_a)+abs(y_a), representing the sum of change in acceleration in both directions
            Default is infinite (eg: any accel described by x and y limits)

    Returns:
        (tuple): tuple containing:

        * **n_states** (*int*): Number of states
        * **n_actions** (*int*): Number of actions
        * **p** (*dict*): P matrix in DiscreteEnv format
        * **starting_probability** (*list*): Probability for all starting locations
        * **action_map** (*dict*): Map between tuple and integer actions
        * **terminal_locations** (*dict*): Map of locations which are terminal
    """
    # Defaults:
    if x_vel_limits is None:
        x_vel_limits = (-2, 2)
    if y_vel_limits is None:
        y_vel_limits = (-2, 2)
    if x_accel_limits is None:
        x_accel_limits = (-2, 2)
    if y_accel_limits is None:
        y_accel_limits = (-2, 2)

    x_vels = range(x_vel_limits[0], x_vel_limits[1] + 1)
    y_vels = range(y_vel_limits[0], y_vel_limits[1] + 1)
    x_accels = range(x_accel_limits[0], x_accel_limits[1] + 1)
    y_accels = range(x_accel_limits[0], y_accel_limits[1] + 1)

    p = {}
    starting_probability = []
    terminal_locations = {}

    # Build list of all available actions.  Filter based on maximum acceleration allowed in a single step, if specified
    actions_as_list = []
    # actions_as_list = [(x_accel, y_accel) for x_accel in x_accels for y_accel in y_accels]
    for x_accel in x_accels:
        for y_accel in y_accels:
            if abs(x_accel) + abs(y_accel) <= max_total_accel:
                actions_as_list.append((x_accel, y_accel))
    for i_row in range(len(track)):
        for i_col in range(len(track[i_row])):
            x, y = rc_to_xy(track, i_row, i_col)
            # Record whether this location is terminal
            terminal_locations[(x, y)] = char_map[track[i_row][i_col]]['terminal']
            for x_vel in x_vels:
                for y_vel in y_vels:
                    actions = {}
                    if x_vel == 0 and y_vel == 0:
                        starting_probability.append(char_map[track[i_row][i_col]]['start_prob'])
                    else:
                        starting_probability.append(0.0)
                    for x_accel, y_accel in actions_as_list:
                        # If this state is terminal, Make all actions point back to itself for 0 reward and also
                        # be terminal (this is how the value iteration wants it)
                        if char_map[track[i_row][i_col]]['terminal']:
                            next_state = (x, y, x_vel, y_vel)
                            reward = 0
                            terminal = True
                            actions[(x_accel, y_accel)] = [(1.0, next_state, reward, terminal)]
                        else:
                            next_x_vel = accel_within_limits(x_vel, x_accel, x_vel_limits)
                            next_y_vel = accel_within_limits(y_vel, y_accel, y_vel_limits)

                            next_x = x + next_x_vel
                            next_y = y + next_y_vel

                            # Next location in i_row, i_col
                            next_r, next_c = xy_to_rc(track, next_x, next_y)

                            next_state = (next_x, next_y, next_x_vel, next_y_vel)

                            # Get reward associated with entering next state, and decide if this action is terminal
                            try:
                                reward = char_map[track[next_r][next_c]]['reward']
                                terminal = char_map[track[next_r][next_c]]['terminal']
                            except IndexError:
                                raise IndexError("Caught IndexError while building Racetrack.  Likely cause is a "
                                                 "max velocity that is creater than the wall padding around the track "
                                                 "(leading to a car that can exit the track entirely)")

                            # Define probabilistic results of this action
                            if track[i_row][i_col] == 'O':
                                # Oil is slippery!  Add a chance of slipping and having no acceleration on this move
                                next_x_if_slipped = x + x_vel
                                next_y_if_slipped = y + y_vel
                                next_state_if_slipped = (next_x_if_slipped, next_y_if_slipped, x_vel, y_vel)

                                # Next location in i_row, i_col
                                next_r_if_slipped, next_c_if_slipped = \
                                    xy_to_rc(track, next_x_if_slipped, next_y_if_slipped)

                                # Get reward associated with entering next state, and decide if this action is terminal
                                reward_if_slipped = char_map[track[next_r_if_slipped][next_c_if_slipped]]['reward']
                                terminal_if_slipped = char_map[track[next_r_if_slipped][next_c_if_slipped]]['terminal']

                                actions[(x_accel, y_accel)] = [
                                    # (prob, next_state, reward, boolean_for_terminal_state)
                                    (1.0 - char_map['O']['chance_to_slip'], next_state, reward, terminal),
                                    (char_map['O']['chance_to_slip'], next_state_if_slipped, reward_if_slipped,
                                     terminal_if_slipped),
                                ]
                            else:
                                # Clean track - everything works well
                                actions[(x_accel, y_accel)] = [
                                    # (prob, next_state, reward, boolean_for_terminal_state)
                                    (1.0, next_state, reward, terminal),
                                ]
                    p[(x, y, x_vel, y_vel)] = actions

    n_states = len(p)
    n_actions = len(actions_as_list)
    return n_states, n_actions, p, starting_probability, actions_as_list, terminal_locations


def accel_within_limits(v, a, v_range):
    """
    Accelerate the car while clipping to a velocity range

    Args:
        v (int): starting velocity
        a (int): acceleration
        v_range (tuple): min and max velocity

    Returns:
        (int): velocity, clipped to min/max v_range
    """
    v = v + a
    v = max(v, v_range[0])
    v = min(v, v_range[1])
    return v

def xy_to_rc(track, x, y):
    """
    Convert a track (x, y) location to (row, col)

    (x, y) convention

    * (0,0) in bottom left
    * x +ve to the right
    * y +ve up

    (row,col) convention:

    * (0,0) in top left
    * row +ve down
    * col +ve to the right

    Args:
        track (list): List of strings describing the track
        x (int): x coordinate to be converted
        y (int): y coordinate to be converted

    Returns:
        tuple: (row, col)
    """
    r = (len(track) - 1) - y
    c = x
    return r, c


def rc_to_xy(track, r, c):
    """
    Convert a track (row, col) location to (x, y)

    (x, y) convention

    * (0,0) in bottom left
    * x +ve to the right
    * y +ve up

    (row,col) convention:

    * (0,0) in top left
    * row +ve down
    * col +ve to the right

    Args:
        track (list): List of strings describing the track
        r (int): row coordinate to be converted
        c (int): col coordinate to be converted

    Returns:
        tuple: (x, y)
    """
    x = c
    y = (len(track) - 1) - r
    return x, y
