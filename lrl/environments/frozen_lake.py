from gym.envs.toy_text import frozen_lake

from lrl.environments.utils import get_terminal_locations
from lrl.utils import rc_to_xy


class FrozenLakeEnv(frozen_lake.FrozenLakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
