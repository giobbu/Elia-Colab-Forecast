from loguru import logger
import pickle
import numpy as np
from collections import defaultdict
from source.ensemble.stack_generalization.test_importance.first_stage_importance_shap import wind_power_importance
from source.ensemble.stack_generalization.test_importance.second_stage_importance_shap import wind_power_variability_importance
import time

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
    assert y_test.shape[0]>=92 and y_test.shape[0]<=100, 'The y_test must be between 92 and 100'
    results_contributions = defaultdict(dict)
    # wind power importance
    start_1_stage = time.time()
    y_test_1st_stage = y_test[-96:]
    results_contributions = wind_power_importance(results_challenge_dict, ens_params, y_test_1st_stage, results_contributions)
    end_1_stage = time.time() - start_1_stage
    logger.debug(f'Computational Time for 1-Stage {end_1_stage}')
    # wind power ramp importance
    start_2_stage = time.time()
    y_test_2nd_stage = y_test[-192:]
    results_contributions = wind_power_variability_importance(results_challenge_dict, ens_params, y_test_2nd_stage, forecast_range, results_contributions)
    end_2_stage = time.time() - start_2_stage
    logger.debug(f'Computational Time for 2-Stage {end_2_stage}')
    return results_contributions
