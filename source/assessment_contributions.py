from loguru import logger
from source.ensemble.stack_generalization.test_importance.forecasters_contributions import load_model_info, calculate_contributions

def compute_forecasters_contributions(buyer_resource_name, ens_params, y_test, forecast_range):
    " Compute the contributions of the forecasters for the buyer resource"
    assert isinstance(buyer_resource_name, str), 'The buyer_resource_name must be a string'
    previous_day_result = f"{ens_params['save_info']}{buyer_resource_name}_{ens_params['save_file']}"
    logger.info(f"Load model info from file: {previous_day_result}")
    results_challenge_dict = load_model_info(previous_day_result)
    logger.info(f"Get the contributions for the buyer resource: {buyer_resource_name}")
    results_contributions = calculate_contributions(results_challenge_dict, ens_params, y_test, forecast_range)
    return results_contributions