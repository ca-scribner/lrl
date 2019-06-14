import pytest

from lrl import environments
from lrl.environments import utils


# Constants
rt_5x4_is_terminal = {
    (0, 3): True,
    (1, 3): True,
    (2, 3): True,
    (3, 3): True,
    (4, 3): True,
    (0, 2): True,
    (1, 2): False,
    (2, 2): False,
    (3, 2): False,
    (4, 2): True,
    (0, 1): True,
    (1, 1): False,
    (2, 1): True,
    (3, 1): True,
    (4, 1): True,
    (0, 0): True,
    (1, 0): True,
    (2, 0): True,
    (3, 0): True,
    (4, 0): True
    }

lake_4x4_is_terminal = {
    (0, 3): False,
    (1, 3): False,
    (2, 3): False,
    (3, 3): False,
    (0, 2): False,
    (1, 2): True,
    (2, 2): False,
    (3, 2): True,
    (0, 1): False,
    (1, 1): False,
    (2, 1): False,
    (3, 1): True,
    (0, 0): True,
    (1, 0): False,
    (2, 0): False,
    (3, 0): True,
    }


@pytest.fixture
def supply_racetrack_5x4():
    """
    Use factory approach instead of returning the env directly so we can create multiple copies of env in a test

    Returns:
        A factory that generates a Racetrack instance for testing
    """
    class Factory:
        @staticmethod
        def get():
            return environments.get_racetrack(
                track='5x4_basic',
                x_vel_limits=(-1, 1),
                y_vel_limits=(-1, 1),
                x_accel_limits=(-1, 1),
                y_accel_limits=(-1, 1),
                max_total_accel=2,
                )
    return Factory()


def test_environment_get_terminal_locations(supply_racetrack_5x4):
    """
    Test function get_terminal_locations

    Args:
        supply_racetrack_5x4:

    Returns:

    """
    rt = supply_racetrack_5x4.get()

    # Compare also to a known solution
    assert rt.is_location_terminal == rt_5x4_is_terminal
    # Racetrack knows its own terminal locations through a different method.  Compare to get_terminal_locations
    assert rt.is_location_terminal == utils.get_terminal_locations(rt)

    lake = environments.frozen_lake.RewardingFrozenLakeEnv()
    # Compare to a known solution
    assert lake.is_location_terminal == lake_4x4_is_terminal
    # This test is redundant because lake uses get_terminal_locations in its definition
    # assert lake.is_location_terminal == utils.get_terminal_locations(lake)
