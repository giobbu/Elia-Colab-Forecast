from loguru import logger
import pandas as pd
from source.ensemble.stack_generalization.test_importance.forecasters_contributions import load_model_info, calculate_contributions

def compute_forecasters_contributions(buyer_resource_name, ens_params, y_test, forecast_range):
    """Compute the contributions of the forecasters for the buyer resource.
    Args:
        buyer_resource_name: Name of the buyer resource
        ens_params: Dictionary with ensemble parameters
        y_test: Series with the true values
        forecast_range: DatetimeIndex with the forecast range
    Returns:
        results_contributions: Dictionary with the contributions of the forecasters
    """
    assert isinstance(buyer_resource_name, str), 'The buyer_resource_name must be a string'
    assert isinstance(ens_params, dict), 'The ens_params must be a dictionary'
    assert isinstance(y_test, pd.Series), 'The y_test must be a pandas Series'
    assert isinstance(forecast_range, pd.DatetimeIndex), 'The forecast_range must be a pandas DatetimeIndex'
    try:
        # Retrieve path of the previous day result
        path_previous_day_result = f"{ens_params['save_info']}{buyer_resource_name}_{ens_params['save_file']}"
        # Load model info from file
        logger.info(f"Load model info from file: {path_previous_day_result}")
        results_challenge_dict = load_model_info(path_previous_day_result)
        # Calculate the contribution for each forecaster
        logger.info(f"Get the contributions for the buyer resource: {buyer_resource_name}")
        results_contributions = calculate_contributions(results_challenge_dict, ens_params, y_test, forecast_range)
        return results_contributions
    except Exception as e:
        logger.error(f"Error in compute_forecasters_contributions: {e}")
        return None