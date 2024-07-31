from loguru import logger
import pickle
import numpy as np
from collections import defaultdict
from source.ensemble.stack_generalization.test_importance.first_stage_importance import wind_power_importance
from source.ensemble.stack_generalization.test_importance.second_stage_importance import wind_power_ramp_importance

def load_model_info(file_path):
    " Load model info"
    assert file_path.endswith('.pickle'), 'The file must be a pickle file'
    try:
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        logger.error(f"Failed to load model info from {file_path}: {e}")
        return None

def calculate_contributions(results_challenge_dict, ens_params, y_test, forecast_range):
    " Calculate the contributions of the models in the ensemble"
    assert isinstance(results_challenge_dict, dict), 'The results_challenge_dict must be a dictionary'
    assert isinstance(ens_params, dict), 'The ens_params must be a dictionary'
    assert isinstance(y_test, np.ndarray), 'The y_test must be a numpy array'
    results_contributions = defaultdict(dict)
    try:
        # wind power importance
        results_contributions = wind_power_importance(results_challenge_dict, ens_params, y_test, results_contributions)
        # wind power ramp importance
        results_contributions = wind_power_ramp_importance(results_challenge_dict, ens_params, y_test, previous_day_forecast_range, results_contributions)
    except Exception as e:
        logger.error(f"Failed to calculate contributions: {e}")
    return results_contributions