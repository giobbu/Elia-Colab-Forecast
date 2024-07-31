from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_augmented_dataframe_2stage

def second_stage_permuted_score(predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, df_train_ensemble, df_test_ensemble, y_train, predictions_insample, order_diff, max_lags_var, augment_var, start_prediction_timestamp, end_prediction_timestamp, var_fitted_model):
    "Compute the permuted score for a single predictor in the second stage model."
    X_test_permuted = X_test_augmented.copy()
    X_test_permuted[:, predictor_index] = np.random.permutation(X_test_augmented[:, predictor_index])
    permuted_predictions_outsample = fitted_model.predict(X_test_permuted)
    df_2stage_permuted = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, permuted_predictions_outsample)
    df_2stage_processed_permuted = create_augmented_dataframe_2stage(df_2stage_permuted, order_diff, max_lags=max_lags_var, augment=augment_var)
    df_2stage_test_permuted = df_2stage_processed_permuted[(df_2stage_processed_permuted.index >= start_prediction_timestamp) & (df_2stage_processed_permuted.index <= end_prediction_timestamp)]
    X_test_2stage_permuted, y_test_2stage_permuted = df_2stage_test_permuted.drop(columns=['targets']).values, df_2stage_test_permuted['targets'].values
    permutation_score = score_functions[quantile](var_fitted_model, X_test_2stage_permuted, y_test_2stage_permuted)['mean_loss']
    return  permutation_score

def second_stage_permutation_importance(y_test, parameters_model, quantile, info_previous_day_second_stage, start_prediction_timestamp, end_prediction_timestamp):
    "Compute permutation importances for the second stage model."
    # Define the score functions for different quantiles
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    num_permutations = parameters_model['nr_permutations']
    assert num_permutations > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    order_diff = parameters_model['order_diff']
    max_lags_var = parameters_model['max_lags_var']
    augment_var = parameters_model['augment_var']
    #get the model
    fitted_model = info_previous_day_second_stage[quantile]['fitted_model']
    var_fitted_model = info_previous_day_second_stage[quantile]['var_fitted_model']
    # get the data
    X_test_augmented = info_previous_day_second_stage[quantile]['X_test_augmented']
    df_train_ensemble_augmented = info_previous_day_second_stage[quantile]['df_train_ensemble_augmented']
    X_train_augmented = info_previous_day_second_stage[quantile]['X_train_augmented']
    df_train_ensemble = info_previous_day_second_stage[quantile]['df_train_ensemble']
    df_test_ensemble = info_previous_day_second_stage[quantile]['df_test_ensemble']
    y_train = info_previous_day_second_stage[quantile]['y_train']

    # Generate predictions from the first-stage model
    predictions_insample = fitted_model.predict(X_train_augmented)
    predictions_outsample = fitted_model.predict(X_test_augmented)
    # Create and preprocess the two-stage dataframe
    df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample)
    df_2stage_processed = create_augmented_dataframe_2stage(df_2stage, order_diff, max_lags=max_lags_var, augment=augment_var)
    # Split the processed dataframe into test sets
    df_2stage_test = df_2stage_processed[(df_2stage_processed.index >= start_prediction_timestamp) & (df_2stage_processed.index <= end_prediction_timestamp)]
    X_test_2stage, y_test_2stage = df_2stage_test.drop(columns=['targets']).values, df_2stage_test['targets'].values
    # Compute the original score
    base_score = score_functions[quantile](var_fitted_model, X_test_2stage, y_test_2stage)['mean_loss']
    importance_scores = []
    # Loop through each predictor
    for predictor_index in range(X_test_augmented.shape[1]):
        predictor_name = df_train_ensemble_augmented.drop(columns=['norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        permuted_scores = Parallel(n_jobs=-1)(delayed(second_stage_permuted_score)(
            predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile,
            df_train_ensemble, df_test_ensemble, y_train, predictions_insample, order_diff, max_lags_var,
            augment_var, start_prediction_timestamp, end_prediction_timestamp, var_fitted_model
        ) for _ in range(num_permutations))
        # Calculate mean contribution for the predictor
        mean_contribution = max(0, np.mean(permuted_scores) - base_score)
        importance_scores.append({'predictor': predictor_name, 'contribution': mean_contribution})
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores).sort_values(by='contribution', ascending=False)
    # Drop the forecasters standard deviation and variance rows
    results_df = results_df[~results_df.predictor.isin(['forecasters_var', 'forecasters_std'])]
    # Normalize contributions
    results_df['contribution'] = results_df['contribution']/results_df['contribution'].sum()
    return results_df

def wind_power_ramp_importance(results_challenge_dict, ens_params, y_test, forecast_range, results_contributions):
    " Get the importance of the wind power ramp"
    assert 'wind_power_ramp' in results_challenge_dict.keys(), 'The key wind_power_variability is not present in the results_challenge_dict'
    assert 'info_contributions' in results_challenge_dict['wind_power_ramp'].keys(), 'The key info_contributions is not present in the results_challenge_dict'
    assert 'quantiles' in ens_params.keys(), 'The key quantiles is not present in the ens_params'
    assert 'nr_permutations' in ens_params.keys(), 'The key nr_permutations is not present in the ens_params'
    logger.opt(colors=True).info(f'<blue>--</blue>' * 79)
    logger.opt(colors=True).info(f'<blue>Wind Power Ramp</blue>')
    # Get the info from the previous day
    info_previous_day_second_stage = results_challenge_dict['wind_power_ramp']['info_contributions']
    num_permutations = ens_params['nr_permutations']
    logger.info(f'Number of permutations: {num_permutations}')
    for quantile in ens_params['quantiles']:
        logger.opt(colors=True).info(f'<blue>Quantile: {quantile}</blue>')
        # Get the contributions
        df_contributions = second_stage_permutation_importance(
            y_test_prev=y_test, 
            parameters_model=ens_params, 
            quantile=quantile, 
            info_previous_day_second_stage=info_previous_day_second_stage, 
            forecast_range = forecast_range
        )
        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power_ramp'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions