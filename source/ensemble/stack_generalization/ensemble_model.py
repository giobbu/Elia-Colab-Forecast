from source.ensemble.stack_generalization.feature_engineering.data_augmentation import augment_with_quantiles
from source.ensemble.stack_generalization.hyperparam_optimization.optimization import optimize_model, initialize_train_and_predict, permutation_quantile_regression
from loguru import logger
import pandas as pd
import numpy as np

def predico_ensemble_predictions_per_quantile(ens_params, 
                                                X_train, X_test, y_train, df_train_ensemble,  
                                                predictions, quantile,
                                                best_results, iteration, 
                                                X_train_quantile10=np.array([]), X_test_quantile10=np.array([]), df_train_ensemble_quantile10=pd.DataFrame(), 
                                                X_train_quantile90=np.array([]), X_test_quantile90=np.array([]), df_train_ensemble_quantile90=pd.DataFrame()):
    """ Run ensemble predictions for a specific quantile.
    args:
        ens_params: dict, ensemble parameters
        X_train: np.array, training data
        X_test: np.array, testing data
        y_train: np.array, target data
        df_train_ensemble: pd.DataFrame, training data
        predictions: dict, predictions
        quantile: float, quantile
        best_results: dict, best results
        iteration: int, iteration number
        X_train_quantile10: np.array, training data for quantile 10
        X_test_quantile10: np.array, testing data for quantile 10
        df_train_ensemble_quantile10: pd.DataFrame, training data for quantile 10
        X_train_quantile90: np.array, training data for quantile 90
        X_test_quantile90: np.array, testing data for quantile 90
        df_train_ensemble_quantile90: pd.DataFrame, training data for quantile 90
    returns:
            results: dict, results
    """
    
    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Run ensemble predictions for quantile {quantile} </fg 250,128,114>')

    # Initialize variables
    add_quantiles = ens_params['add_quantile_predictions'] 
    augment_q50 = ens_params['augment_q50']
    nr_cv_splits = ens_params['nr_cv_splits'] 
    model_type = ens_params['model_type']
    solver = ens_params['solver']
    gbr_update_every_days = ens_params['gbr_update_every_days'] 
    gbr_config_params = ens_params['gbr_config_params']
    lr_config_params = ens_params['lr_config_params']

    assert model_type in ['GBR', 'LR'], 'Invalid model type'

    # Initialize variables
    X_train_augmented, X_test_augmented, df_train_ensemble_augmented = X_train, X_test, df_train_ensemble
    # Augment the training and testing data with the quantiles predictions
    if add_quantiles:
        logger.opt(colors=True).info(f'<fg 250,128,114> Augmenting training and testing data with quantiles </fg 250,128,114>')
        # Augment the training and testing data with the quantiles predictions
        X_train_augmented, X_test_augmented, df_train_ensemble_augmented = augment_with_quantiles(
                                                                                        X_train, X_test, df_train_ensemble,
                                                                                        X_train_quantile10, X_test_quantile10, df_train_ensemble_quantile10,
                                                                                        X_train_quantile90, X_test_quantile90, df_train_ensemble_quantile90,
                                                                                        quantile, augment_q50=augment_q50)
    
    # if ens_params['conformalized_qr']:
    #     # retain first two days for calibration
    #     X_train_augmented = X_train_augmented[ens_params['day_calibration']*96:]
    #     y_train = y_train[ens_params['day_calibration']*96:]
    #     X_calibrate_augmented = X_train_augmented[:ens_params['day_calibration']*96]
    #     y_calibrate = y_train[:ens_params['day_calibration']*96]
    #     df_train_ensemble_augmented = df_train_ensemble_augmented.iloc[ens_params['day_calibration']*96:]

    # Optimize model hyperparameters
    if iteration % gbr_update_every_days == 0:  # Optimize hyperparameters every gbr_update_every_days
        logger.opt(colors=True).info(f'<fg 250,128,114> Optimizing model hyperparameters - updating every {gbr_update_every_days} days</fg 250,128,114>')
        best_score, best_params = optimize_model(X_train_augmented, y_train, quantile,
                                                    nr_cv_splits, model_type, solver, gbr_config_params,
                                                    lr_config_params)
        best_results[quantile] = [('best_score', best_score), ('params', best_params)]
    else:
        logger.opt(colors=True).info(f'<fg 250,128,114> Using best hyperparameters from first iteration </fg 250,128,114>')
        best_params = best_results[quantile][1][1]

    # Initialize, fit and predict
    fitted_model, predictions = initialize_train_and_predict(predictions, model_type, quantile, best_params, solver, X_train_augmented, y_train, X_test_augmented) 

    # Store results
    results = {'predictions': predictions, 'best_results': best_results, 'fitted_model': fitted_model, 
                'X_train_augmented': X_train_augmented, 'X_test_augmented': X_test_augmented,
                'df_train_ensemble_augmented': df_train_ensemble_augmented}
    
    # Compute p-values for the coefficients
    if ens_params['model_type'] == 'LR':
        # Compute p-values for the coefficients
        coefs, p_values_permutation = permutation_quantile_regression(best_params, solver, X_train_augmented, y_train, quantile, n_permutations=ens_params['nr_pvalues_permutations'])
        # Bonferroni correction
        is_significant = p_values_permutation < ens_params['alpha']/len(coefs)
        model_summary = pd.DataFrame({
                                        "Predictor": df_train_ensemble_augmented.drop(['norm_targ'], axis=1).columns,
                                        "Coefs": coefs,
                                        "p-values": p_values_permutation,
                                        "significant": is_significant
                                    })
        model_summary = model_summary.sort_values(by="Coefs", ascending=False)
        # Store results
        results['coefs'] = coefs
        results['p_values'] = np.array([round(p_values_permutation[i], 4) for i in range(len(p_values_permutation))])
        results['model-summary'] = model_summary
        
        # logger.info('Model summary')
        # logger.info(model_summary[model_summary['significant'] == True])
    
    # # Store calibration data
    # if ens_params['conformalized_qr']:
    #     results['X_calibrate_augmented'] = X_calibrate_augmented
    #     results['y_calibrate'] = y_calibrate

    return results 


def predico_ensemble_variability_predictions(ens_params, 
                                            X_train_2stage,
                                            y_train_2stage, 
                                            X_test_2stage, 
                                            variability_predictions, 
                                            quantile, 
                                            iteration, 
                                            best_results_var, 
                                            variability_predictions_insample,
                                            variability_predictions_outsample):
    """ Run ensemble variability predictions 
    args:
        ens_params: dict, ensemble parameters
        X_train_2stage: np.array, training data for the 2nd stage
        y_train_2stage: np.array, target data for the 2nd stage
        X_test_2stage: np.array, testing data for the 2nd stage
        variability_predictions: dict, predictions
        quantile: float, quantile
        iteration: int, iteration number
        best_results_var: dict, best results
        variability_predictions_insample: dict, insample predictions
        variability_predictions_outsample: dict, outsample predictions
    returns:
        results: dict, results
    """
    logger.opt(colors=True).info(f'<fg 72,201,176> Run ensemble variability predictions for quantile {quantile} </fg 72,201,176>')
    # Initialize variables
    nr_cv_splits = ens_params['nr_cv_splits'] 
    var_model_type = ens_params['var_model_type'] 
    solver = ens_params['solver'] 
    var_gbr_config_params = ens_params['var_gbr_config_params'] 
    var_lr_config_params = ens_params['var_lr_config_params'] 
    gbr_update_every_days = ens_params['gbr_update_every_days'] 

    assert var_model_type in ['GBR', 'LR'], 'Invalid model type'

    # Optimize model hyperparameters
    if iteration % gbr_update_every_days == 0:  # Optimize hyperparameters every gbr_update_every_days
        logger.opt(colors=True).info(f'<fg 72,201,176> Optimizing model hyperparameters - updating every {gbr_update_every_days} days</fg 72,201,176>')
        best_score, best_params_var = optimize_model(X_train_2stage, y_train_2stage, quantile, nr_cv_splits, var_model_type, solver, var_gbr_config_params, var_lr_config_params)
        best_results_var[quantile] = [('best_score', best_score), ('params', best_params_var)]
    else:
        logger.opt(colors=True).info(f'<fg 72,201,176> Using best hyperparameters from first iteration </fg 72,201,176>')
        best_params_var = best_results_var[quantile][1][1]

    # Initialize, fit and predict
    var_fitted_model, variability_predictions, variability_predictions_insample, variability_predictions_outsample = initialize_train_and_predict(variability_predictions, var_model_type, quantile, best_params_var, solver, X_train_2stage, y_train_2stage, X_test_2stage, insample=True, 
                                                                                                                                                    predictions_insample = variability_predictions_insample,
                                                                                                                                                    predictions_outsample = variability_predictions_outsample)  

    # Store results
    results = {'variability_predictions': variability_predictions, 
                'best_results_var': best_results_var, 
                'var_fitted_model': var_fitted_model,
                'variability_predictions_insample': variability_predictions_insample,
                'variability_predictions_outsample': variability_predictions_outsample}

    return results