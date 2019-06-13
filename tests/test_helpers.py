import pytest

from lrl.solvers.learners import decay_functions


@pytest.mark.parametrize(
    "settings,inputs,expected",
    [
        ({'type': 'constant', 'initial_value': 10.0},
         [0, 1.0, 10, 1000001.5],
         [10.0, 10.0, 10.0, 10.0]),

        ({'type': 'constant', 'initial_value': 65.5},
         [0, 1.0, 10, 1000001.5],
         [65.5, 65.5, 65.5, 65.5]),

        ({'type': 'linear', 'initial_value': 10.0, 'initial_timestep': 0, 'final_value': 20.0, 'final_timestep': 20},
         [0,    1,    5,    10,   19,   20,   30],
         [10.0, 10.5, 12.5, 15.0, 19.5, 20.0, 20.0]),

        ({'type': 'linear', 'initial_value': 10.0, 'initial_timestep': 10, 'final_value': 20.0, 'final_timestep': 20},
         [0,    1,    5,    10,   19,   20,   30],
         [10.0, 10.0, 10.0, 10.0, 19.0, 20.0, 20.0]),

        ({'type': 'linear', 'initial_value': -10.0, 'initial_timestep': 100, 'final_value': 0.0, 'final_timestep': 600},
         [0,      50,    100,    200,   350,   600,   1000],
         [-10.0, -10.0, -10.0,  -8.0,  -5.0,   0.0,   0.0]),

    ]
)
def test_decay_functions(settings, inputs, expected):
    """
    Test decay_functions helper

    Returns:
        None
    """

    decay_function = decay_functions(settings)
    for x_in, x_expected in zip(inputs, expected):
        assert decay_function(x_in) == pytest.approx(x_expected)
