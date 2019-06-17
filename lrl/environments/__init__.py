import gym
from gym.envs.registration import register

from .Racetrack import *


__all__ = ['Racetrack', 'frozen_lake']


# Is this needed?  Can I just make the envs directly?
register(
    id='Racetrack-v0',
    entry_point='lrl.environments:Racetrack',
    )


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
    kwargs = {'track': track,
              'x_vel_limits': x_vel_limits, 'y_vel_limits': y_vel_limits,
              'x_accel_limits': x_accel_limits, 'y_accel_limits': y_accel_limits,
              'max_total_accel': max_total_accel, }
    return gym.make("Racetrack-v0", **kwargs)
