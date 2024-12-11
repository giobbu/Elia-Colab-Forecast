import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

from source.utils.file_read import read_csv_file, filter_data, replace_nan_values
from source.utils.generate_timestamp import generate_timestamps
from source.simulation.submission_module import submission_forecasters
from source.simulation.buyer_module import prepare_buyer_data
from source.ml_engine import create_ensemble_forecasts
from source.ensemble.stack_generalization.wind_ramp.detector import wind_ramp_detector
from source.assessment_contributions import compute_forecasters_contributions
from source.utils.session_ml_info import delete_previous_day_pickle

# Configuration settings
from config.simulation_setting_with_ramps import Simulation, Stack

def main(sim_params, ens_params):
    # Set random seed
    np.random.seed(sim_params['random_seed'])

    # Read CSV file
    df_processed = read_csv_file(sim_params['csv_filename'], sim_params['list_columns'], sim_params['starting_period'], sim_params['ending_period'])

    # Replace NaN values if specified
    if sim_params['replace_nan']:
        df_processed = replace_nan_values(sim_params, df_processed)

    # Remove previous day pickle file
    logger.info(' ')
    delete_previous_day_pickle()


    # Loop over test days
    for i in tqdm(range(sim_params['num_test_days']), desc='Testing Days'):

        # Generate timestamps for training and prediction
        start_training_timestamp, end_training_timestamp, start_prediction_timestamp, end_prediction_timestamp = generate_timestamps(
            sim_params['start_training'], i, sim_params['window_size'])

        # Trim data for training and testing
        df_train = filter_data(df_processed, start_training_timestamp, end_training_timestamp, string='training')
        df_test = filter_data(df_processed, start_prediction_timestamp, end_prediction_timestamp, string='testing')

        # ----------------------------> FORECASTERS SUBMISSION <----------------------------

        logger.debug("Forecasters submission ...")
        df_market, df_train, df_test = submission_forecasters(sim_params, df_train, df_test)

        # ----------------------------> MARKET OPERATOR DATA <----------------------------

        logger.debug("Market operator data ...")
        df_buyer, forecast_range = prepare_buyer_data(df_train, df_test, start_prediction_timestamp, end_prediction_timestamp)

        # ----------------------------> PREDICO PLATFORM ML ENGINE <----------------------------

        # ----------------------------> ENSEMBLE FORECASTS <----------------------------

        logger.debug("Wind ensemble forecasts ...")
        results_ensemble_forecasts = create_ensemble_forecasts(
            ens_params=ens_params,
            df_buyer=df_buyer,
            df_market=df_market,
            end_training_timestamp=end_training_timestamp,
            forecast_range=forecast_range,
            challenge_usecase='simulation',
            simulation=True
        )

        # ----------------------------> WIND RAMP DETECTION <----------------------------

        logger.debug("Wind ramp detection ...")
        pred_variability_insample = results_ensemble_forecasts['wind_power_ramp']['predictions_insample']
        pred_variability_outsample = results_ensemble_forecasts['wind_power_ramp']['predictions_outsample']
        
        # Wind ramp detection logic
        alarm_status, df_ramp_clusters = wind_ramp_detector(
            ens_params=Stack.params,
            df_pred_variability_insample=pred_variability_insample,
            df_pred_variability_outsample=pred_variability_outsample
        )

        logger.info(f"Alarm status: {alarm_status}")
        if df_ramp_clusters is not None:
            logger.info(f"Ramp clusters: {df_ramp_clusters.cluster_id.unique()}")
            logger.info(f"Ramp clusters datetime: {df_ramp_clusters.index}")

        # ----------------------------> ASSESSMENT CONTRIBUTIONS <----------------------------

        logger.debug("Forecasters contributions ...")
        y_test = df_test['measured'].values
        forecasters_contributions = compute_forecasters_contributions(
            sim_params['buyer_resource_name'], ens_params, y_test, forecast_range
        )
        logger.info(f"Forecasters contributions: {forecasters_contributions}")



# Example of how the main function might be called:
if __name__ == "__main__":

    sim_params = Simulation.testing_period  # Simulation parameters
    ens_params = Stack.params  # QRA Ensemble parameters

    # Call the main function
    main(sim_params, ens_params)
