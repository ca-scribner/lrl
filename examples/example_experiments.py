import lrl
from lrl import environments
from lrl.utils.experiment_runners import run_experiments

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
                    level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


########################################################################################################################
# Run settings (modify this)
########################################################################################################################

# Build a set of cases to run.
# Environments are defined as instanced environments (complete settings already set) in a dictionary environments
# Solvers are defined by sets of parameters.  The solver_param_grid object is passed to
# sklearn.model_selection.ParameterGrid (eg: the same routine that parses the param_grid argument for
# sklearn.model_selection.GridSearchCV()) to build a list of cases.
#
# Each settings case (combination of solver + single set of settings for that solver) will be run on each environment.
# Output from this case is described in the run_experiments() routine

# Top level directory for result output
output_path = './output/'

# Define the environments
# Dictionary of instanced environments to use in runs (keys will be used as the environment names for output)
environments = {
    'rt_5x4_basic': environments.get_racetrack(track='5x4_basic',
                                               x_vel_limits=(-1, 1),
                                               y_vel_limits=(-1, 1),
                                               x_accel_limits=(-1, 1),
                                               y_accel_limits=(-1, 1),
                                               max_total_accel=2,
                                               ),
    'rt_20x10_U': environments.get_racetrack(track='20x10_U',
                                             x_vel_limits=(-2, 2),
                                             y_vel_limits=(-2, 2),
                                             x_accel_limits=(-2, 2),
                                             y_accel_limits=(-2, 2),
                                             max_total_accel=2,
                                             ),
    # 'rt_10x10_all_oil': environments.get_racetrack(track='10x10_all_oil',
    #                                                x_vel_limits=(-2, 2),
    #                                                y_vel_limits=(-2, 2),
    #                                                x_accel_limits=(-2, 2),
    #                                                y_accel_limits=(-2, 2),
    #                                                max_total_accel=2,
    #                                                ),
    # 'lake_4x4': environments.frozen_lake.RewardingFrozenLakeEnv(map_name='4x4', is_slippery=True),
    'lake_8x8': environments.frozen_lake.RewardingFrozenLakeEnv(map_name='8x8', is_slippery=True),
}

# Define solver settings
# The only important part here is the solver_param_grid passed to run_experiments().  All other settings are just
# convenient ways to build solver_param_grid

# Settings for PI, VI, and QL
gammas = [0.5, 0.95]

# Settings for PI and VI
max_iters_planners = [500]

# Settings for QL only
alphas = [
    # 0.1,  # Constant alpha
    {'type': 'linear',  # Linear decay of alpha
     'initial_value': 0.2, 'initial_timestep': 0,
     'final_value': 0.05, 'final_timestep': 1500, },
]
epsilons = [
    # 0.1,  # Constant epsilon
    {'type': 'linear',  # Linear decay of epsilon
     'initial_value': 0.2, 'initial_timestep': 0,
     'final_value': 0.05, 'final_timestep': 1500, },
]
max_iters_ql = [5000]

# Combine all settings into a param_grid.  run_experiments() uses sklearn.model_selection.ParameterGrid to convert
# solver_param_grid to individual cases
solver_param_grid = [
    {'solver': ['PI', 'VI'],
     'gamma': gammas,
     'max_iters': max_iters_planners,
     'score_while_training': [{'n_trains_per_eval': 2, 'n_evals': 500}],
     },
    {'solver': ['QL'],
     'gamma': gammas,
     'alpha': alphas,
     'epsilon': epsilons,
     'max_iters': max_iters_ql,
     'score_while_training': [True],
     }
]

# Call for experiments.  Do not modify unless you know what you're doing
run_experiments(environments=environments, solver_param_grid=solver_param_grid, output_path=output_path)
