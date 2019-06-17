import shutil
import os
import pandas as pd

from lrl import solvers
from lrl.utils import misc, plotting
from sklearn.model_selection import ParameterGrid

import logging
logger = logging.getLogger(__name__)

SOLVER_DICT = {
    'PI': solvers.PolicyIteration,
    'VI': solvers.ValueIteration,
    'QL': solvers.QLearning,
}


def run_experiment(env, params, output_path):
    """
    Run a single experiment (env/solver combination), outputing results to a given location

    FUTURE: Improve easy reproducibility by outputting a settings file or similar?  Could use gin-config or just output
     params.  Outputting params doesn't cover env though...

    Args:
        env:
        params:
        output_path:

    Returns:
        dict of:
            solver: Fully populated solver object (after solving env)
            scored_results: WalkStatistics object of results from scoring the final policy
            solve_time: Time in seconds used to solve the env (eg: run solver.iterate_to_convergence())
    """
    # Create the output path, cleaning out old results if needed
    if os.path.exists(output_path):
        logging.warning(f"Deleting old results from {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Set up the solver
    solver_name = params['solver']

    # Extract solver settings from params
    solver_settings = {k: v for k, v in params.items() if k != 'solver'}

    # Setup case
    solver = SOLVER_DICT[solver_name](env, **solver_settings)

    # Solve
    timer = misc.Timer()
    solver.iterate_to_convergence()
    solve_time = timer.elapsed()

    # Score result
    scored_results = solver.score_policy()

    # Capture results
    solver.scoring_summary.to_csv(f'{output_path}/intermediate_scoring_results.csv')
    solver.iteration_data.to_csv(f'{output_path}/iteration_data.csv')
    try:
        solver.walk_statistics.to_dataframe(include_walks=True).to_csv(f'{output_path}/training_episodes.csv')
    except AttributeError:
        # walk_statistics only exists for some solvers
        pass

    # Plot relevant plots
    plotting.plot_solver_results(env, solver=solver, savefig=f'{output_path}/solver_results')
    plotting.plot_episodes(scored_results.walks, env, max_episodes=1000, savefig=f'{output_path}/scored_episodes.png')
    try:
        plotting.plot_episodes(solver.walk_statistics.walks, env, max_episodes=1000,
                               savefig=f'{output_path}/training_episodes.png')
    except AttributeError:
        # walk_statistics only exists for some solvers
        pass

    return {'solver': solver, 'scored_results': scored_results, 'solve_time': solve_time}


def run_experiments(environments, solver_param_grid, output_path='./output/'):
    """
    Runs a set of experiments defined by param_grid, writing results to output_path

    Args:
        environments:
        solver_param_grid:
        output_path:

    Outputs:
        grid_search_summary.csv: high-level summary of results
        env_name/case_name: Directory with detailed results for each env/case combination

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Use ParameterGrid to build a list of cases to run
    # Loop through all cases, but group the order by solver
    cases = sorted(list(ParameterGrid(solver_param_grid)), key=lambda k: k['solver'])

    # Initialize column name lists for saving summary data later
    # Get all parameters noted in the cases dicts and sort them for some deterministic ordering
    param_columns = sorted(list(set(k for param_dict in cases for k in param_dict)))
    other_columns = ['solve_time']
    score_columns = ['reward_mean', 'reward_median', 'reward_std', 'reward_min', 'reward_max', 'steps_mean',
                     'steps_median', 'steps_std', 'steps_min', 'steps_max']

    for i_env, (env_name, env) in enumerate(environments.items()):
        logging.info(f"Running {env_name} ({i_env}/{len(environments)})")
        env_path = f"{output_path}/{env_name}/"

        if not os.path.exists(env_path):
            os.makedirs(env_path)

        # Initialize summary results dataframe using previously set up column names
        summary_df = pd.DataFrame(columns=param_columns + other_columns + score_columns)

        for i_params, params in enumerate(cases):

            case_name = misc.params_to_name(params, first_fields=['solver'])
            case_path = f'{env_path}/{case_name}'

            logger.info(f'Running {params["solver"]} (settings case {i_params}/{len(cases)})')
            logger.debug(f'Full case name: {case_name}')

            results = run_experiment(env, params, case_path)

            # Capture this run's results in a central df
            this_data = params.copy()
            this_data['solve_time'] = results['solve_time']

            # Update this_data with a dict produced by:
            # - Get scored_results as a dataframe (only score_columns)
            # - pull the last row (statistics for the entire scoring run) as a pd.Series
            # - convert pd.Series to a dictionary
            this_data.update(results['scored_results'].to_dataframe().loc[:, score_columns].iloc[-1].to_dict())
            summary_df = summary_df.append(this_data, ignore_index=True)

        summary_df.to_csv(f'{env_path}/grid_search_summary.csv')
