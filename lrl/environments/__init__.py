import gym
from gym.envs.registration import register

# from .cliff_walking import *
# from .frozen_lake import *
from .Racetrack import *

# FUTURE: Should I register environments here?  Or in env module?  Or?
# FUTURE: Do I even need a construction function here at all?  What is benefit/downside?

# LAKE_STEP_PROB = 0.5
# LAKE_STEP_REW = -0.1
# LAKE_HOLE_REW = -10
# LAKE_GOAL_REW = 10
# CLIFF_WIND_PROB = 0.1
# CLIFF_STEP_REW = -1
# CLIFF_FALL_REW = -100
# CLIFF_GOAL_REW = 100

RACETRACK_TRACK_DEFAULT = '10x10'


# __all__ = ['RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv', 'Racetrack']
__all__ = ['Racetrack']

# register(
#     id='RewardingFrozenLake4x4-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLake8x8-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLake12x12-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLake15x15-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLake20x20-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLakeNoRewards4x4-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLakeNoRewards8x8-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLakeNoRewards12x12-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLakeNoRewards15x15-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='RewardingFrozenLakeNoRewards20x20-v0',
#     entry_point='environments:RewardingFrozenLakeEnv',
# )

# register(
#     id='CliffWalking4x12-v0',
#     entry_point='environments:WindyCliffWalkingEnv',
# )

# register(
#     id='WindyCliffWalking4x12-v0',
#     entry_point='environments:WindyCliffWalkingEnv',
# )

# register(
#     id='Racetrack10x10-v0',
#     entry_point='environments:Racetrack',
#     )

register(
    id='Racetrack-v0',
    entry_point='lrl.environments:Racetrack',
    )


# def get_rewarding_frozen_lake_4x4_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                               hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '4x4', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLake4x4-v0', **kwargs)


# def get_rewarding_frozen_lake_8x8_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                               hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '8x8', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLake8x8-v0', **kwargs)


# def get_large_rewarding_frozen_lake_12x12_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                       hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '12x12', 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW}
#     return gym.make('RewardingFrozenLake12x12-v0', **kwargs)


# def get_large_rewarding_frozen_lake_15x15_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                       hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '15x15', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLake15x15-v0', **kwargs)


# def get_large_rewarding_frozen_lake_20x20_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                       hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '20x20', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLake20x20-v0', **kwargs)


# def get_frozen_lake_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, hole_rew=LAKE_HOLE_REW, \
#                                 goal_rew=LAKE_GOAL_REW):
#     return gym.make('FrozenLake-v0')


# def get_rewarding_no_reward_frozen_lake_4x4_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                         hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '4x4', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLakeNoRewards4x4-v0', **kwargs)


# def get_rewarding_no_reward_frozen_lake_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                     hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '8x8', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLakeNoRewards8x8-v0', **kwargs)


# def get_large_rewarding_no_reward_frozen_lake_12x12_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                                 hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '12x12', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLakeNoRewards12x12-v0', **kwargs)


# def get_large_rewarding_no_reward_frozen_lake_15x15_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                                 hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '15x15', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLakeNoRewards15x15-v0', **kwargs)


# def get_large_rewarding_no_reward_frozen_lake_20x20_environment(step_prob=LAKE_STEP_PROB, step_rew=LAKE_STEP_REW, \
#                                                                 hole_rew=LAKE_HOLE_REW, goal_rew=LAKE_GOAL_REW):
#     kwargs={'map_name': '20x20', 'step_rew': step_rew, 'hole_rew': hole_rew, 'goal_rew': goal_rew}
#     return gym.make('RewardingFrozenLakeNoRewards20x20-v0', **kwargs)


# def get_cliff_walking_4x12_environment(wind_prob=CLIFF_WIND_PROB, step_rew=CLIFF_STEP_REW, \
#                                        fall_rew=CLIFF_FALL_REW, goal_rew=CLIFF_GOAL_REW):
#     kwargs={'wind_prob': 0, 'step_rew': step_rew, 'fall_rew': fall_rew, 'goal_rew': goal_rew},
#     return gym.make('CliffWalking4x12-v0', **kwargs)


# def get_windy_cliff_walking_4x12_environment(wind_prob=CLIFF_WIND_PROB, step_rew=CLIFF_STEP_REW, \
#                                              fall_rew=CLIFF_FALL_REW, goal_rew=CLIFF_GOAL_REW):
#     return gym.make('WindyCliffWalking4x12-v0')


def get_racetrack(track=None,
                  x_vel_limits=None, y_vel_limits=None,
                  x_accel_limits=None, y_accel_limits=None,
                  max_total_accel=None,):
    """
    Returns a Racetrack environment initialized as an OpenAI Gym environment

    Args:
        See Racetrack() docs

    Returns:
        Instance of an OpenAI Gym environment
    """
    kwargs={'track': track, 
            'x_vel_limits': x_vel_limits, 'y_vel_limits': y_vel_limits, 
            'x_accel_limits': x_accel_limits, 'y_accel_limits': y_accel_limits, 
            'max_total_accel': max_total_accel, }
    return gym.make("Racetrack-v0", **kwargs)