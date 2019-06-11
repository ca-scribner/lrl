from gym.envs.toy_text import frozen_lake


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
