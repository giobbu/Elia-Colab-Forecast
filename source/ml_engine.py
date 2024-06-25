from loguru import logger
import pandas as pd
import gc
import pickle
from tqdm import tqdm
from pathlib import Path
from source.utils.data_preprocess import normalize_dataframe
from source.ensemble.stack_generalization.feature_engineering.data_augmentation import create_augmented_dataframe
from source.ensemble.stack_generalization.data_preparation.data_train_test import prepare_train_test_data, get_numpy_Xy_train_test
from source.ensemble.stack_generalization.ensemble_model import predico_ensemble_predictions_per_quantile, predico_ensemble_variability_predictions
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_augmented_dataframe_2stage, create_var_ensemble_dataframe
from source.ensemble.stack_generalization.utils.results import collect_quantile_ensemble_predictions, create_ensemble_dataframe

def create_ensemble_forecasts(ens_params,
                                df_buyer,
                                df_market,
                                forecast_range,
                                challenge_usecase = None,
                                simulation = False,):
    " Create ensemble forecasts for wind power and wind power variability using forecasters predictions"

    start_prediction_timestamp = forecast_range[0] 
    end_prediction_timestamp = forecast_range[-1]

    df_ensemble_quantile50 = df_market[[ name for name in df_market.columns if 'q50' in name]]  # get the quantile 50 predictions
    df_ensemble_quantile10 = df_market[[ name for name in df_market.columns if 'q10' in name]]  # get the quantile 10 predictions
    df_ensemble_quantile90 = df_market[[ name for name in df_market.columns if 'q90' in name]]  # get the quantile 90 predictions

    buyer_resource_name = df_buyer.columns[0]  # get the name of the buyer resource
    maximum_capacity = df_buyer[buyer_resource_name].max()  # get the maximum capacity
    
    # set solver for quantile regression
    from sklearn.utils.fixes import parse_version, sp_version
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    if ens_params['model_type'] == 'LR':
        assert ens_params['normalize'] == True, "Normalization must be True for model_type 'LR'"
 

    # ML ENGINE PREDICO PLATFORM
    logger.info('  ')
    logger.opt(colors=True).info(f'<fg 250,128,114> PREDICO Machine Learning Engine </fg 250,128,114> ')
    logger.info('  ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Predictions from {str(start_prediction_timestamp)} to {str(end_prediction_timestamp)} </fg 250,128,114> ')
    logger.info('  ')

    logger.opt(colors=True).info(f'<fg 250,128,114> Buyer Resource Name: {buyer_resource_name} </fg 250,128,114>')
    logger.opt(colors=True).info(f'<fg 250,128,114> Maximum Capacity: {maximum_capacity} </fg 250,128,114>')
    logger.info('  ')

    # Logging
    logger.opt(colors=True).info(f'<fg 250,128,114> Collecting forecasters prediction for ensemble learning - model: {ens_params["model_type"]} </fg 250,128,114>')
    logger.info('  ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Forecasters Ensemble DataFrame </fg 250,128,114>')

    # Normalize dataframes
    if ens_params['normalize']:
        logger.info('   ')
        logger.opt(colors=True).info(f'<fg 250,128,114> Normalize DataFrame </fg 250,128,114>')
        df_ensemble_normalized = normalize_dataframe(df_ensemble_quantile50, maximum_capacity)

        # Normalize dataframes quantile predictions
        if ens_params['add_quantile_predictions']:
            logger.opt(colors=True).info(f'<fg 250,128,114> -- Add quantile predictions </fg 250,128,114>')
            df_ensemble_normalized_quantile10 = normalize_dataframe(df_ensemble_quantile10, maximum_capacity)
            df_ensemble_normalized_quantile90 = normalize_dataframe(df_ensemble_quantile90, maximum_capacity)
        else:
            df_ensemble_normalized_quantile10 = df_ensemble_normalized_quantile90 = None
    else:
        df_ensemble_normalized = df_ensemble_quantile50.copy()
        if ens_params['add_quantile_predictions']:
            df_ensemble_normalized_quantile10 = df_ensemble_quantile10.copy()
            df_ensemble_normalized_quantile90 = df_ensemble_quantile90.copy()
        else:
            df_ensemble_normalized_quantile10 = df_ensemble_normalized_quantile90 = None
    
    # Augment dataframes
    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Augment DataFrame </fg 250,128,114>')
    df_ensemble_normalized_lag = create_augmented_dataframe(df_ensemble_normalized, 
                                                            max_lags=ens_params['max_lags'], 
                                                            forecasters_diversity=ens_params['forecasters_diversity'], 
                                                            lagged=ens_params['lagged'], 
                                                            augmented=ens_params['augment'],
                                                            differenciate=ens_params['differenciate'])
    # Augment dataframes quantile predictions
    if ens_params['add_quantile_predictions']:
        logger.opt(colors=True).info(f'<fg 250,128,114> -- Augment quantile predictions </fg 250,128,114>')

        # Augment with predictions quantile 10
        df_ensemble_normalized_lag_quantile10 = create_augmented_dataframe(df_ensemble_normalized_quantile10,
                                                                            max_lags=ens_params['max_lags'], 
                                                                            forecasters_diversity=ens_params['forecasters_diversity'], 
                                                                            lagged=ens_params['lagged'], 
                                                                            augmented=ens_params['augment'], 
                                                                            differenciate=ens_params['differenciate'])
        # Augment with predictions quantile 90
        df_ensemble_normalized_lag_quantile90 = create_augmented_dataframe(df_ensemble_normalized_quantile90, 
                                                                            max_lags=ens_params['max_lags'], 
                                                                            forecasters_diversity=ens_params['forecasters_diversity'], 
                                                                            lagged=ens_params['lagged'], 
                                                                            augmented=ens_params['augment'], 
                                                                            differenciate=ens_params['differenciate'])
    else:
        df_ensemble_normalized_lag_quantile10 = df_ensemble_normalized_lag_quantile90 = None
    
    # Normalize dataframe
    if ens_params['normalize']:
        df_buyer_norm = normalize_dataframe(df_buyer, maximum_capacity)
    else:
        df_buyer_norm = df_buyer.copy()
        df_buyer_norm = df_buyer_norm.add_prefix('norm_')
    
    # Differentiate dataframe
    df_buyer_norm_diff = df_buyer_norm.copy()
    lst_cols_diff = ['diff_' + name for name in list(df_buyer_norm.columns)]
    df_buyer_norm_diff.columns = lst_cols_diff
    
    # Split train and test dataframes
    df_train_norm_diff = df_buyer_norm_diff[df_buyer_norm_diff.index < start_prediction_timestamp]
    df_test_norm_diff = df_buyer_norm_diff[df_buyer_norm_diff.index >= start_prediction_timestamp]

    df_train_ensemble, df_test_ensemble = prepare_train_test_data(buyer_resource_name, df_ensemble_normalized_lag, df_train_norm_diff, df_test_norm_diff, start_prediction_timestamp, ens_params['max_lags'])
    
    # Split train and test dataframes quantile predictions
    if ens_params['add_quantile_predictions']:
        df_train_ensemble_quantile10 = df_ensemble_normalized_lag_quantile10[df_ensemble_normalized_lag_quantile10.index < start_prediction_timestamp]
        df_test_ensemble_quantile10 = df_ensemble_normalized_lag_quantile10[df_ensemble_normalized_lag_quantile10.index >= start_prediction_timestamp]
        df_train_ensemble_quantile90 = df_ensemble_normalized_lag_quantile90[df_ensemble_normalized_lag_quantile90.index < start_prediction_timestamp]
        df_test_ensemble_quantile90 = df_ensemble_normalized_lag_quantile90[df_ensemble_normalized_lag_quantile90.index >= start_prediction_timestamp]
    else:
        df_train_ensemble_quantile10 = df_test_ensemble_quantile10 = None
        df_train_ensemble_quantile90 = df_test_ensemble_quantile90 = None
    
    # Assert df_test matches df_ensemble_test
    assert (df_test_norm_diff.index == df_test_ensemble.index).all(),'Datetime index are not equal'

    # Make X-y train and test sets
    X_train, y_train, X_test, y_test = get_numpy_Xy_train_test(df_train_ensemble, df_test_ensemble)
    
    # Make X-y train and test sets quantile predictions
    if ens_params['add_quantile_predictions']:
        X_train_quantile10 = df_train_ensemble_quantile10.values
        X_test_quantile10 = df_test_ensemble_quantile10.values
        X_train_quantile90 = df_train_ensemble_quantile90.values
        X_test_quantile90 = df_test_ensemble_quantile90.values
    else:
        X_train_quantile10 = X_test_quantile10 = None
        X_train_quantile90 = X_test_quantile90 = None
    
    # Assert no NaNs in train ensemble
    assert df_train_ensemble.isna().sum().sum() == 0
    
    # chech if file with model info exists
    file_info = ens_params['save_info'] + buyer_resource_name + '_' + ens_params['save_file']
    if Path(file_info).is_file():
        with open(file_info, 'rb') as handle:
            results_challenge_dict = pickle.load(handle)
        iteration = results_challenge_dict['iteration'] + 1
        best_results = results_challenge_dict['wind_power']['best_results']
        best_results_var = results_challenge_dict['wind_power_variability']['best_results']
    else:
        iteration = 0

    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Iteration {iteration} </fg 250,128,114>')

    # Run ensemble learning
    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Compute Ensemble Predictions </fg 250,128,114>')


    # dictioanry to store predictions
    predictions = {}
    if iteration == 0:
        best_results = {}
    previous_day_results_first_stage = {}

    # Loop over quantiles
    for quantile in tqdm(ens_params['quantiles'], desc='Quantile Regression'):

        # Run ensemble learning
        predictions, best_results, fitted_model, X_train_augmented, X_test_augmented, df_train_ensemble_augmented = predico_ensemble_predictions_per_quantile(abs_differenciate=ens_params['compute_abs_difference'], 
                                                                                                                                        X_train=X_train, X_test=X_test, y_train=y_train, df_train_ensemble=df_train_ensemble, 
                                                                                                                                        predictions=predictions, quantile=quantile, add_quantiles=ens_params['add_quantile_predictions'], 
                                                                                                                                        augment_q50=ens_params['augment_q50'], nr_cv_splits=ens_params['nr_cv_splits'], model_type=ens_params['model_type'], solver=solver, 
                                                                                                                                        gbr_update_every_days=ens_params['gbr_update_every_days'], gbr_config_params=ens_params['gbr_config_params'], 
                                                                                                                                        lr_config_params=ens_params['lr_config_params'], plot_importance_gbr=ens_params['plot_importance_gbr'], 
                                                                                                                                        best_results=best_results, iteration=iteration, 
                                                                                                                                        X_train_quantile10=X_train_quantile10, X_test_quantile10=X_test_quantile10, df_train_ensemble_quantile10=df_train_ensemble_quantile10, 
                                                                                                                                        X_train_quantile90=X_train_quantile90, X_test_quantile90=X_test_quantile90, df_train_ensemble_quantile90=df_train_ensemble_quantile90)
        # Store results
        previous_day_results_first_stage[quantile] = {"fitted_model" : fitted_model, 
                                                        "X_train_augmented" : X_train_augmented, 
                                                        "X_test_augmented" : X_test_augmented, 
                                                        "df_train_ensemble_augmented" : df_train_ensemble_augmented}
        
        # compute variability predictions with as input the predictions of the first stage
        if ens_params['compute_second_stage'] and quantile == 0.5:
            logger.info('   ')
            logger.opt(colors=True).info(f'<fg 72,201,176> Compute Variability Predictions </fg 72,201,176>')
            
            predictions_insample = fitted_model.predict(X_train_augmented)
            predictions_outsample = fitted_model.predict(X_test_augmented)
            
            # Create 2-stage dataframe
            df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample)

            # Augment 2-stage dataframe
            df_2stage_buyer = create_augmented_dataframe_2stage(df_2stage, ens_params['order_diff'], max_lags=ens_params['max_lags_var'], augment=ens_params['augment_var'])
            
            # Split 2-stage dataframe
            df_2stage_train = df_2stage_buyer[df_2stage_buyer.index < start_prediction_timestamp]
            df_2stage_test = df_2stage_buyer[df_2stage_buyer.index >= start_prediction_timestamp]
            
            # Normalize 2-stage dataframe
            X_train_2stage = df_2stage_train.drop(columns=['targets']).values
            y_train_2stage = df_2stage_train['targets'].values
            X_test_2stage = df_2stage_test.drop(columns=['targets']).values

            # dictioanry to store variability predictions
            variability_predictions = {}
            if iteration == 0:
                best_results_var = {}
            previous_day_results_second_stage = {}

            # Loop over quantiles
            for quantile in tqdm(ens_params['quantiles'], desc='Quantile Regression'):

                # Run ensemble learning
                variability_predictions, best_results_var, var_fitted_model = predico_ensemble_variability_predictions(X_train_2stage=X_train_2stage, y_train_2stage=y_train_2stage, X_test_2stage=X_test_2stage,
                                                                                                                variability_predictions=variability_predictions, quantile=quantile, nr_cv_splits=ens_params['nr_cv_splits'], 
                                                                                                                var_model_type=ens_params['var_model_type'], solver=solver, 
                                                                                                                var_gbr_config_params=ens_params['var_gbr_config_params'], var_lr_config_params=ens_params['var_lr_config_params'], 
                                                                                                                gbr_update_every_days=ens_params['gbr_update_every_days'], iteration=iteration, best_results_var=best_results_var)
                
                # Store results
                previous_day_results_second_stage[quantile] = {"fitted_model": fitted_model, 
                                                                "var_fitted_model": var_fitted_model, 
                                                                "X_train_augmented": X_train_augmented, 
                                                                "X_test_augmented": X_test_augmented, 
                                                                "df_train_ensemble_augmented": df_train_ensemble_augmented, 
                                                                "df_train_ensemble": df_train_ensemble, 
                                                                "df_test_ensemble": df_test_ensemble,
                                                                "y_train": y_train}
            # Collect quantile variability predictions
            var_predictions_dict = collect_quantile_ensemble_predictions(ens_params['quantiles'], df_2stage_test, variability_predictions)

            # collect results as dataframe
            df_var_ensemble = create_var_ensemble_dataframe(buyer_resource_name, 
                                                            ens_params['quantiles'], 
                                                            var_predictions_dict, 
                                                            df_2stage_test)
            
            # melt dataframe
            df_var_ensemble_melt = pd.melt(df_var_ensemble.reset_index(), id_vars='datetime', value_vars=df_var_ensemble.columns)

            del df_2stage, df_2stage_buyer, df_2stage_train
            gc.collect()
        
        del X_train_augmented, X_test_augmented, df_train_ensemble_augmented
        gc.collect()
    
    # Collect quantile predictions
    quantile_predictions_dict = collect_quantile_ensemble_predictions(ens_params['quantiles'], df_test_norm_diff, predictions)

    # collect results as dataframe
    df_pred_ensemble = create_ensemble_dataframe(buyer_resource_name,
                                                ens_params['quantiles'],
                                                quantile_predictions_dict, 
                                                df_test_norm_diff)
    
    if simulation:

        assert  challenge_usecase == 'simulation', 'challenge_usecase must be "simulation"'

        # collect results as dataframe
        df_results_wind_power = pd.concat([df_pred_ensemble, df_test_norm_diff['diff_norm_' + buyer_resource_name]], axis=1) 
        df_results_wind_power_variability = pd.concat([df_var_ensemble, df_2stage_test['targets']], axis=1)

        # collect results as dictionary of dictionaries
        results_challenge_dict_simu = {'previous_lt': start_prediction_timestamp,
                                    'iteration': iteration,
                                    'wind_power': 
                                        {'predictions': df_results_wind_power, 
                                            'info_contributions': previous_day_results_first_stage,
                                            'best_results': best_results,
                                            'df_test': df_test_ensemble},
                                    'wind_power_variability': 
                                        {'predictions': df_results_wind_power_variability, 
                                            'info_contributions': previous_day_results_second_stage,
                                            'best_results': best_results_var,
                                            'df_test': df_2stage_test}
                                        }
        # save results
        with open(file_info, 'wb') as handle:
            pickle.dump(results_challenge_dict_simu, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return results_challenge_dict_simu
    else:
        # melt dataframe
        df_pred_ensemble_melt = pd.melt(df_pred_ensemble.reset_index(), id_vars='datetime', value_vars=df_pred_ensemble.columns)

        # collect results as dictionary of dictionaries
        results_challenge_dict = {'previous_lt': start_prediction_timestamp,
                                    'iteration': iteration,
                                    'wind_power': 
                                        {'predictions': df_pred_ensemble_melt, 
                                            'info_contributions': previous_day_results_first_stage,
                                            'best_results': best_results},
                                    'wind_power_variability': 
                                        {'predictions': df_var_ensemble_melt, 
                                            'info_contributions': previous_day_results_second_stage,
                                            'best_results': best_results_var}
                                        }
        # save results
        with open(file_info, 'wb') as handle:
            pickle.dump(results_challenge_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        assert  challenge_usecase == 'wind_power' or challenge_usecase == 'wind_power_variability', 'challenge_usecase must be either "wind_power" or "wind_power_variability"'
        return results_challenge_dict[challenge_usecase]['predictions']
    