# frozen_lake is a slightly modified version from https://github.com/cmaron/CS-7641-assignments/tree/master/assignment4,
# which was a moderately modified version of https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py,
# adding rewards passed out every step in addition to finding the goal (to make the problem easier for the learner)

import numpy as np

import sys
from contextlib import closing

from six import StringIO

from gym import utils
from gym.envs.toy_text import discrete

from lrl.environments.utils import get_terminal_locations
from lrl.utils import rc_to_xy


DEFAULT_STEP_REWARD = -0.01
DEFAULT_HOLE_REWARD = -1.0
DEFAULT_GOAL_REWARD = 1.0
DEFAULT_STEP_PROB = 0.75

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFHFF",
        "FHFFHHFF",
        "FFFHFFFG"
    ],
    "12x12": [
        "SFFFFHFFFFFF",
        "FFFFFFFFFFFF",
        "FHFFFFFFFFFF",
        "FFFFFFFFHFFF",
        "FFFHFFFFHFFF",
        "FFFFFFHHHFFF",
        "HHFHFFFFHFFF",
        "FFFHFFFFHFFF",
        "FFFFFFFFFFFF",
        "FFFHFFFFFFFF",
        "FFHFHFFFFFFF",
        "FHFFFFFHHFFG"
    ],
    "15x15": [
        "SFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFHFFFFFFHFFFF",
        "FFFFFFFFFFHFFFF",
        "FFFFFFFFFHHHFFF",
        "HHHFFFFFFFHFFFF",
        "FFFFHFFFFFHFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FHFHFHFFFFFFFFF",
        "FFHFFFFFFHHFFFG"
    ],
    "20x20": [
        "SFFFFFFHHHFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFHFFFFFFFHHFFFFFFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFHFFFFFFFHHFF",
        "FFFFFHFFFFHHFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHHHHHHHFF",
        "HHHHFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFHHHFFFHHFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFHFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHFFFFFFFF",
        "FHHFFFHFFFFHFFFFFHFF",
        "FHHFHFHFFFFFFFFFFFFF",
        "FFFHFFFFFHFFFFHHFHFG"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1 - p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class RewardingFrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF

        FHFH

        FFFH

        HFFG

    * S: starting point, safe
    * F: frozen surface, safe
    * H: hole, fall to your doom
    * G: goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    There are configurable rewards for reaching the goal, falling in a hole, and simply taking a step.
    The hole and step rewards are configurable when creating an instance of the problem.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", step_reward=None, hole_reward=None, goal_reward=None,
                 is_slippery=True, step_prob=None):

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        if step_reward is None:
            step_reward = DEFAULT_STEP_REWARD
        self.step_reward = step_reward
        if hole_reward is None:
            hole_reward = DEFAULT_HOLE_REWARD
        self.hole_reward = hole_reward
        if goal_reward is None:
            goal_reward = DEFAULT_GOAL_REWARD
        self.goal_reward = goal_reward

        self.is_slippery = is_slippery
        if step_prob is None:
            step_prob = DEFAULT_STEP_PROB
        self.step_prob = step_prob
        self.slip_prob = (1.0 - self.step_prob) / 2

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'G':
                        li.append((1.0, s, self.goal_reward, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                if newletter in b'FS':
                                    rew = self.step_reward
                                elif newletter == b'H':
                                    rew = self.hole_reward
                                if b == a:
                                    li.append((self.step_prob, newstate, rew, done))
                                else:
                                    li.append((self.slip_prob, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            if newletter in b'FS':
                                rew = self.step_reward
                            elif newletter == b'H':
                                rew = self.hole_reward
                            li.append((1.0, newstate, rew, done))

        super().__init__(nS, nA, P, isd)

        # Additional fields that make actions in the solvers easier:
        # Add some maps for easier plotting:
        # Color_map to relate terrain type to background color (note keyed by byte-string rather than string
        # as per convention used by FrozenLakeEnv.desc)
        self.color_map = {
            b'G': 'gold',
            b'S': 'lightgreen',
            b'F': 'dodgerblue',
            b'H': 'b',
        }

        # Map of action to character for plotting policy
        self.action_as_char = {
            0: '⬅',
            1: '⬇',
            2: '➡',
            3: '⬆',
            4: '',
        }

        # Add conversion for index-->state_tuple and state_tuple-->index
        # state_tuple is (x, y) where x=0 is the leftmost column and y=0 is the bottom row

        self.index_to_state = [rc_to_xy(r, c, self.desc.shape[0]) for r, row in enumerate(self.desc)
                          for c in range(row.shape[0])]
        self.state_to_index = {k: i for i, k in enumerate(self.index_to_state)}

        # Build a dictionary that denotes whether each location is terminal.  Dictionary is keyed by (x,y) grid location
        # tuples
        self.is_location_terminal = get_terminal_locations(self)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
