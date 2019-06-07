import pytest
import itertools

from lrl.solvers.planners import PolicyIteration, ValueIteration
from lrl import environments


@pytest.fixture
def supply_racetrack_5x4():
    """
    Use factory approach instead of returning the env directly so we can create multiple copies of env in a test

    Returns:
        A factory that generates a Racetrack instance for testing
    """
    class Factory:
        def get(self):
            return environments.get_racetrack(
                track='5x4_basic',
                x_vel_limits=(-1, 1),
                y_vel_limits=(-1, 1),
                x_accel_limits=(-1, 1),
                y_accel_limits=(-1, 1),
                max_total_accel=2,
                )
    return Factory()


def test_policy_iteration_racetrack(supply_racetrack_5x4):
    """
    Test PolicyIteration's functionality

    Not sure how to nicely test everything - anything but simple tests are hard to hand calculate.  Spot testing the
    policy output from a simple test, as well as its path's rewards, for now

    Args:
        supply_racetrack_5x4: Factory to generate a simple Racetrack env

    Returns:
        None
    """
    # Build environment and policy iteration object
    env = supply_racetrack_5x4.get()
    pi = PolicyIteration(env)

    # Solve environment
    pi.iterate_to_convergence()

    # Spot test policy results
    assert pi.policy[(1, 1, 0, 0)] == (0, 1)
    assert pi.policy[(1, 1, -1, -1)] == (1, 1)
    assert pi.policy[(1, 2, 0, 1)] == (1, -1)
    assert pi.policy[(1, 2, 1, 1)] == (0, -1)
    assert pi.policy[(3, 2, 1, 1)] == (-1, -1)
    assert pi.policy[(3, 2, 0, 0)] == (0, -1)

    # Test outcome of running best policy
    states, rewards, terminal = pi.run_policy()
    assert states == [(1, 1, 0, 0), (1, 2, 0, 1), (2, 2, 1, 0), (3, 1, 1, -1)]
    assert rewards == pytest.approx([0.0, -1., -1., 100])
    assert terminal is True

    # Spot test a few values from the value function
    assert pi.value[(1, 1, 0, 0)] == pytest.approx(79.1)
    assert pi.value[(1, 2, 0, 1)] == pytest.approx(89.)
    assert pi.value[(2, 2, 1, 0)] == pytest.approx(100.)

    # Test run_policy() by scoring this policy iteratively, from standard initial and another state initial state
    assert pi.score_policy(iters=3).get_statistic(statistic='reward_mean') == pytest.approx(98)
    assert pi.score_policy(iters=3, initial_state=(1, 2, 0, 0)).get_statistic(statistic='reward_mean') == \
           pytest.approx(99)

# Below test is unnecessary as the pi == vi test below covers it, but left it in in case there's a reason to add later
# def test_value_iteration_racetrack(supply_racetrack_5x4):
#     """
#     Test ValueIteration's functionality
#
#     Not sure how to nicely test everything - anything but simple tests are hard to hand calculate.  Spot testing the
#     policy output from a simple test, as well as its path's rewards, for now
#
#     Args:
#         supply_racetrack_5x4: Factory to generate a simple Racetrack env
#
#     Returns:
#         None
#     """
#     # Build environment and policy iteration object
#     env = supply_racetrack_5x4.get()
#     vi = ValueIteration(env)
#
#     # Solve environment
#     vi.iterate_to_convergence()
#
#     # Spot test policy results
#     assert vi.policy[(1, 1, 0, 0)] == (0, 1)
#     assert vi.policy[(1, 1, -1, -1)] == (1, 1)
#     assert vi.policy[(1, 2, 0, 1)] == (1, -1)
#     assert vi.policy[(1, 2, 1, 1)] == (0, -1)
#     assert vi.policy[(3, 2, 1, 1)] == (-1, -1)
#     assert vi.policy[(3, 2, 0, 0)] == (0, -1)
#
#     # Test outcome of running best policy
#     states, rewards, terminal = vi.run_policy()
#     assert states == [(1, 1, 0, 0), (1, 2, 0, 1), (2, 2, 1, 0), (3, 1, 1, -1)]
#     assert rewards == pytest.approx([0.0, -1., -1., 100])
#     assert terminal is True
#
#     # Spot test a few values from the value function
#     assert vi.value[(1, 1, 0, 0)] == pytest.approx(79.1)
#     assert vi.value[(1, 2, 0, 1)] == pytest.approx(89.)
#     assert vi.value[(2, 2, 1, 0)] == pytest.approx(100.)


def test_compare_planners_racetrack(supply_racetrack_5x4):
    """
    Test PolicyIteration and ValueIteration versus each other (should arrive at similar results)

    Args:
        supply_racetrack_5x4: Factory to generate a simple Racetrack env

    Returns:
        None
    """
    # Build environments and planners
    planners = {
        'pi': PolicyIteration(supply_racetrack_5x4.get()),
        'vi': ValueIteration(supply_racetrack_5x4.get())
    }

    # Solve environment
    for planner in planners:
        planners[planner].iterate_to_convergence()

    # Compare policy and value functions for all combinations
    combinations = list(itertools.combinations(planners.keys(), 2))

    for p1, p2 in combinations:
        assert planners[p1].policy == planners[p2].policy
        assert planners[p1].value == pytest.approx(planners[p2].value)
