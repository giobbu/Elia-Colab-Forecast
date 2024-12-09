from pathlib import Path
import pickle
import os
from loguru import logger

def load_or_initialize_results(ens_params, buyer_resource_name):
    " Load or initialize results dictionary"
    file_info = ens_params['save_info'] + buyer_resource_name + '_' + ens_params['save_file']
    file_path = Path(file_info)
    if file_path.is_file():
        with open(file_info, 'rb') as handle:
            results_challenge_dict = pickle.load(handle)
        iteration = results_challenge_dict['iteration'] + 1
        best_results = results_challenge_dict['wind_power']['best_results']
        best_results_var = results_challenge_dict['wind_power_variability']['best_results']
    else:
        iteration = 0
        best_results = {}
        best_results_var = {}
    return file_info, iteration, best_results, best_results_var


# create function to remove previous day pickle file
def delete_previous_day_pickle():
    "Delete previous day pickle file"
    filename = './info_model/b1r1_previous_day.pickle'
    try:
        os.remove(filename)
        logger.opt(colors = True).warning('previous day pickle file removed')
    except OSError:
        pass