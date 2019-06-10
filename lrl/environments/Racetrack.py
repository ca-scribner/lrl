import numpy as np

from gym.envs.toy_text import discrete


RACETRACK_DEFAULT_REWARDS = {
    'crash': -100,
    'time': -1,
    'goal': 100,
}


# Character map for the map characters (say that 10 times fast!)
CHAR_MAP = {
    "W": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['crash'], 'terminal': True, 'color': 'forestgreen'},
    "S": {'start_prob': 1., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'dodgerblue'},
    " ": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'darkgray'},
    "G": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['goal'], 'terminal': True, 'color': 'gold'},
    "O": {'start_prob': 0., 'reward': RACETRACK_DEFAULT_REWARDS['time'], 'terminal': False, 'color': 'gray', 'chance_to_slip': 0.25}
}

TRACKS = {
    # Naming as units_in_x x units_in_y
    '3x4_basic': [
        "WWW",
        "WGW",
        "WSW",
        "WWW",
    ],
    '5x4_basic': [
        "WWWWW",
        "W   W",
        "WSWGW",
        "WWWWW",
    ],
    '10x10': [
        "WWWWWWWWWW",
        "WWWWWWWWWW",
        "WWW     GG",
        "WWW     GG",
        "WW   WWWWW",
        "WW   WWWWW",
        "WW    WWWW",
        "WW   SWWWW",
        "WWWWWWWWWW",
        "WWWWWWWWWW",
    ],
    '10x10_basic': [
        "WWWWWWWWWW",
        "WWWWWWWWWW",
        "WW     GWW",
        "WW      WW",
        "WW      WW",
        "WW      WW",
        "WW      WW",
        "WWS     WW",
        "WWWWWWWWWW",
        "WWWWWWWWWW",
    ],
    '15x15_basic': [
        "WWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWW",
        "WW         GWW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WW          WW",
        "WWS         WW",
        "WWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWW",
    ],
    '20x20_basic': [
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WW               GWW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WW                WW",
        "WWS               WW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
    ],
    '20x20_all_oil': [
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWOOOOOOOOOOOOOOOGWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWOOOOOOOOOOOOOOOOWW",
        "WWSOOOOOOOOOOOOOOOWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
    ],
    '30x30_basic': [
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
        "WW                         GWW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WW                          WW",
        "WWS                         WW",
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    ],
    '20x10_U_all_oil': [
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWOOOOOOOOOOOOOOOWW",
        "WWWOOOOOWWWWOOOOOOWW",
        "WWOOOOWWWWWWWWOOOOWW",
        "WWOOOWWWWWWWWWWOOOWW",
        "WWOOOOWWWWWWWWOOOOWW",
        "WWOOOOOOSWWWWWOOOOWW",
        "WWWWWWWWWWWWWWGGGGWW",
        "WWWWWWWWWWWWWWGGGGWW",
    ],
    '20x10_U': [
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWW     OOOO      WW",
        "WWW     WWWW      WW",
        "WWOOOOWWWWWWWWOOOOWW",
        "WWOOOWWWWWWWWWWOOOWW",
        "WW    WWWWWWWW    WW",
        "WW      SWWWWW    WW",
        "WWWWWWWWWWWWWWGGGGWW",
        "WWWWWWWWWWWWWWGGGGWW",
    ],
    '10x10_oil': [
        "WWWWWWWWWW",
        "WWWWWWWWWW",
        "WWW   OOGG",
        "WWW     GG",
        "WW   WWWWW",
        "WW   WWWWW",
        "WW    WWWW",
        "WW   SWWWW",
        "WWWWWWWWWW",
        "WWWWWWWWWW",
    ],
    '20x15_risky': [
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WW                WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWWWWWWWWWW WW",
        "WW WWWWWOOOOWWWWW WW",
        "WWW WWWOWWWWOWWW WWW",
        "WWWW  SWWWWWWG  WWWW",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWWW",
    ]
}


class Racetrack(discrete.DiscreteEnv):

    def __init__(self, track=None, char_map=CHAR_MAP, x_vel_limits=None, y_vel_limits=None,
                 x_accel_limits=None, y_accel_limits=None, max_total_accel=2, seed=None, verbose=False):
        # NOTE: If adding arguments, make sure to update .new_instance().  Is there a better way to do this?
        # Defaults:
        if track is None:
            track = TRACKS['10x10']
        elif isinstance(track, str):
            track = TRACKS[track]
        self.track = track
        self.desc = np.asarray(self.track, dtype='c')
        self.char_map = char_map

        if x_vel_limits is None:
            x_vel_limits = (-2, 2)
        self.x_vel_limits = x_vel_limits
        if y_vel_limits is None:
            y_vel_limits = (-2, 2)
        self.y_vel_limits = y_vel_limits
        if x_accel_limits is None:
            x_accel_limits = (-2, 2)
        self.x_accel_limits = x_accel_limits
        if y_accel_limits is None:
            y_accel_limits = (-2, 2)
        self.y_accel_limits = y_accel_limits

        self.max_total_accel = max_total_accel

        self.verbose = verbose

        self._colors = {k.encode(): char_map[k]['color'] for k in char_map}

        n_states, n_actions, p, starting_probability, action_map, terminal_locations = \
            track_to_p_matrix(track=self.track, char_map=self.char_map,
                              x_vel_limits=self.x_vel_limits, y_vel_limits=self.y_vel_limits,
                              x_accel_limits=self.x_accel_limits, y_accel_limits=self.y_accel_limits,
                              max_total_accel=self.max_total_accel, verbose=False)

        super().__init__(n_states, n_actions, p, starting_probability)


        # Maps to convert indexed states/actions to actual (tuple) states and actions
        self.index_to_state = [x[0] for x in list(self.P.items())]
        self.state_to_index = {k: i for i, k in enumerate(self.index_to_state)}
        self.index_to_action = action_map

        # xy location to isTerminal map (helps with plotting later)
        self.is_location_terminal = terminal_locations

        # Parent __init__ gets self.s as an index.  Convert to the state tuple
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

        :param mode:
        :param current_location: Character to denote the current location
        :return: None
        """

        print(f"Vx={self.s[2]}, Vy={self.s[3]}")
        print_track(self.track, xy=[self.s[0:2]], print_as=current_location)

    def step(self, a):
        """
        Wrap parent step, interpreting integer actions as mapped to human-readable actions

        :param a: Action to take, either as an integer (0..nA-1) or true action (tuple of (x_accel,y_accel))
        :return: Same as parent step
        """

        try:
            return super().step(a)
        except KeyError:
            a = self.index_to_action[a]
            return super().step(a)

    def new_instance(self):
        return Racetrack(track=self.track, char_map=self.char_map,
                         x_vel_limits=self.x_vel_limits, y_vel_limits=self.y_vel_limits,
                         x_accel_limits=self.x_accel_limits, y_accel_limits=self.y_accel_limits,
                         max_total_accel=self.max_total_accel, seed=None, verbose=self.verbose)

    def colors(self):
        return self._colors

    def directions(self):
        return self.index_to_action


def print_track(track, rc=None, xy=None, print_as='*'):
    """
    Print a track, optionally with locations highlighted.

    Locations can be specified by a list of row-col or x-y coordinates, but not a mixture

    :param track: List of characters describing the track
    :param rc:
    :param xy:
    :param print_as:
    :return:
    """
    if rc is None:
        if xy is None:
            rc = []
        else:
            rc = [xy_to_rc(track, *this_xy) for this_xy in xy]

    # If print_as is a single string, print all locations as that same code
    if isinstance(print_as, str):
        print_as = [print_as for i in range(len(rc))]

    this_location = track.copy()
    for i, this_rc in enumerate(rc):
        this_location[this_rc[0]] = \
            this_location[this_rc[0]][:this_rc[1]] + print_as[i] + this_location[this_rc[0]][this_rc[1] + 1:]
    for line in this_location:
        print(line)


def xy_to_rc(track, x, y):
    """
    Convert a track (x, y) location to (row, col)

    (x, y) convention:
        (0,0) in bottom left
        x +ve to the right
        y +ve up
    (row,col) convention:
        (0,0) in top left
        row +ve down
        col +ve to the right

    :param track: Track owning the positions
    :param x: x coordinate to be converted
    :param y: y coordinate to be converted
    :return: Tuple of (row, col)
    """
    r = (len(track) - 1) - y
    c = x
    return r, c


def rc_to_xy(track, r, c):
    """
    Convert a track (row, col) location to (x, y)

    (x, y) convention:
        (0,0) in bottom left
        x +ve to the right
        y +ve up
    (row,col) convention:
        (0,0) in top left
        row +ve down
        col +ve to the right

    :param track: Track owning the positions
    :param row: row coordinate to be converted
    :param col: col coordinate to be converted
    :return: Tuple of (x, y)
    """
    x = c
    y = (len(track) - 1) - r
    return x, y


def track_to_p_matrix(track, char_map=CHAR_MAP, x_vel_limits=None, y_vel_limits=None,
                      x_accel_limits=None, y_accel_limits=None, max_total_accel=np.inf, verbose=True):
    """
    Converts a map described by a list of strings to P-matrix format for an OpenAI Gym Discrete environment.

    Maps are specified using the following characters:
        W: (Wall) Terminal location with reward
        S: Starting location (can be one or more) with small negative reward.  Note that starting state will always have
           0 velocity
           FUTURE: Add velocity to start
        " " (single space): Open track with small negative reward
        G: Goal location with reward
    Rewards and terminal status are assessed based on someone ENTERING that state (eg: if you travel from a starting
    location to a wall, you get the wall's reward and terminal status)

    Sign conventions:
        x: 0 at leftmost column, positive to the right
        y: 0 at bottommost row, positive up

    :param track: List of strings describing the track
    :param char_map: Character map that maps track characters to P matrix
    :param x_vel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
    :param y_vel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
    :param x_accel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in x.  Default is (-2, 2).
    :param y_accel_limits: (OPTIONAL) Tuple of (min, max) valid acceleration in y.  Default is (-2, 2).
    :param max_total_accel: (OPTIONAL) Integer maximum total acceleration in one action.  Total acceleration is computed
                            by abs(x_a)+abs(y_a), representing the sum of change in acceleration in both directions
                            Default is infinite (eg: any accel described by x and y limits)
    :param verbose: If True, print debug statements
    :return: TBD
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
            if verbose:
                print_track(track, rc=[(i_row, i_col)])
            for x_vel in x_vels:
                for y_vel in y_vels:
                    actions = {}
                    if x_vel == 0 and y_vel == 0:
                        starting_probability.append(char_map[track[i_row][i_col]]['start_prob'])
                    else:
                        starting_probability.append(0.0)
                    if verbose:
                        print(f's = {(x, y, x_vel, y_vel)} ({track[i_row][i_col]} w/start prob = {starting_probability[-1]})')
                    for x_accel, y_accel in actions_as_list:
                        # If this state is terminal, Make all actions point back to itself for 0 reward and also
                        # be terminal (this is how the value iteration wants it)
                        if char_map[track[i_row][i_col]]['terminal']:
                            next_state = (x, y, x_vel, y_vel)
                            reward = 0
                            terminal = True
                            actions[(x_accel, y_accel)] = [(1.0, next_state, reward, terminal)]
                            if verbose:
                                print("\tTerminal state - adding dummy action")
                                print(f'\ta= {(x_accel, y_accel)} -> v={(x_vel, y_vel)}, s` = {next_state}(xy) r={reward}, done={terminal}')
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
                            if verbose:
                                print(f'\ta= {(x_accel, y_accel)} -> v={(next_x_vel, next_y_vel)}, s` = {next_state}(xy) ({next_r, next_c} (rc)), r={reward}, done={terminal}')
                    p[(x, y, x_vel, y_vel)] = actions

    n_states = len(p)
    n_actions = len(actions_as_list)
    return n_states, n_actions, p, starting_probability, actions_as_list, terminal_locations


def accel_within_limits(v, a, v_range):
    v = v + a
    v = max(v, v_range[0])
    v = min(v, v_range[1])
    return v
