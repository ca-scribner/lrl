import pytest
import itertools

from gym.envs.toy_text import frozen_lake

from lrl.solvers.planners import PolicyIteration, ValueIteration
from lrl.solvers.learners import QLearning
from lrl import environments


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

@pytest.mark.parametrize(
    "solver_settings",
    [
        ({'alpha': None, 'epsilon': None, 'max_iters': 1000, 'num_episodes_for_convergence': 10}),
        ({'alpha': 0.1, 'epsilon': 0.1, 'max_iters': 1000, 'num_episodes_for_convergence': 10}),
        ({'alpha': 0.1, 'epsilon': 0.2, 'max_iters': 4000, 'num_episodes_for_convergence': 10}),
        ({'alpha': 0.05, 'epsilon': 0.1, 'max_iters': 1000, 'num_episodes_for_convergence': 10}),
    ]
)
def test_qlearning_iteration_racetrack(supply_racetrack_5x4, solver_settings):
    """
    Spot test QLearning's functionality

    Hard to rigorously test this.  Can't get env and solvers to all seed correctly for direct reproducibility.
    For now I'm just testing against the deterministic score a good walk should get... This is more a verification than
    validation.

    Args:
        supply_racetrack_5x4: Factory to generate a simple Racetrack env

    Returns:
        None

    """
    # Build environment and policy iteration object
    env = supply_racetrack_5x4.get()
    ql = QLearning(env, **solver_settings)

    # Solve environment (should converge)
    ql.iterate_to_convergence()

    # Test outcome of running best policy
    states, rewards, terminal = ql.run_policy()
    assert states == [(1, 1, 0, 0), (2, 2, 1, 1), (3, 2, 1, 0), (3, 1, 0, -1)]
    assert rewards == pytest.approx([0.0, -1., -1., 100])
    assert terminal is True

    # Test run_policy() by scoring this policy iteratively, from standard initial and another initial state
    assert ql.score_policy(iters=3).get_statistic(statistic='reward_mean') == pytest.approx(98)


# Tried to parameterize the testing of multiple tracks, but have stochastic tiles which means value function doesn't
# always match, etc... got pretty messy.
@pytest.fixture(params=[
    {'test_policy': True,
     'test_value': True,
     'track_params': dict(track='5x4_basic', x_vel_limits=(-1, 1), y_vel_limits=(-1, 1),
                          x_accel_limits=(-1, 1), y_accel_limits=(-1, 1), max_total_accel=2)},
    {'test_policy': True,
     'test_value': True,
     'track_params': dict(track='10x10_basic', x_vel_limits=(-1, 2), y_vel_limits=(-2, 1),
                          x_accel_limits=(-2, 2), y_accel_limits=(-2, 2), max_total_accel=3)},
    {'test_policy': True,
     'test_value': False,
     'track_params': dict(track='10x10_all_oil', x_vel_limits=(-1, 1), y_vel_limits=(-1, 1),
                          x_accel_limits=(-1, 1), y_accel_limits=(-1, 1), max_total_accel=2)},
])
def supply_racetracks(request):
    print('supply_racetracks')
    print(f'type(request) = {type(request)}')
    print(f'request.param = {request.param}')

    class Factory:
        test_policy = request.param['test_policy']
        test_value = request.param['test_value']
        @staticmethod
        def get():
            return environments.get_racetrack(**request.param['track_params'])
    return Factory()


def test_compare_solvers_racetrack_param(supply_racetracks):
    """
    Test PolicyIteration and ValueIteration versus each other (should arrive at the same policy and similar value func)

    Args:
        supply_racetrack: Factory to generate a Racetrack env

    Returns:
        None
    """
    # Build environments and planners
    planners = {
        'pi': PolicyIteration(supply_racetracks.get()),
        'vi': ValueIteration(supply_racetracks.get()),
    }

    # Solve environment
    for planner in planners:
        planners[planner].iterate_to_convergence()

    # Compare policy and value functions for all combinations
    combinations = list(itertools.combinations(planners.keys(), 2))

    for p1, p2 in combinations:
        if supply_racetracks.test_policy:
            assert planners[p1].policy == planners[p2].policy
        if supply_racetracks.test_value:
            assert planners[p1].value == pytest.approx(planners[p2].value)


# def test_policy_iteration_lake():
#     lake = frozen_lake.FrozenLakeEnv()
#
#     pi = PolicyIteration(lake)
#     pi.iterate_to_convergence()
#
#     # Spot test policy results
#     assert pi.policy[(1, 1, 0, 0)] == (0, 1)
#     assert pi.policy[(1, 1, -1, -1)] == (1, 1)
#     assert pi.policy[(1, 2, 0, 1)] == (1, -1)
#     assert pi.policy[(1, 2, 1, 1)] == (0, -1)
#     assert pi.policy[(3, 2, 1, 1)] == (-1, -1)
#     assert pi.policy[(3, 2, 0, 0)] == (0, -1)
#
#     # Test outcome of running best policy
#     states, rewards, terminal = pi.run_policy()
#     assert states == [(1, 1, 0, 0), (1, 2, 0, 1), (2, 2, 1, 0), (3, 1, 1, -1)]
#     assert rewards == pytest.approx([0.0, -1., -1., 100])
#     assert terminal is True
#
#     # Spot test a few values from the value function
#     assert pi.value[(1, 1, 0, 0)] == pytest.approx(79.1)
#     assert pi.value[(1, 2, 0, 1)] == pytest.approx(89.)
#     assert pi.value[(2, 2, 1, 0)] == pytest.approx(100.)
#
#     # Test run_policy() by scoring this policy iteratively, from standard initial and another state initial state
#     assert pi.score_policy(iters=3).get_statistic(statistic='reward_mean') == pytest.approx(98)
#     assert pi.score_policy(iters=3, initial_state=(1, 2, 0, 0)).get_statistic(statistic='reward_mean') == \
#            pytest.approx(99)
