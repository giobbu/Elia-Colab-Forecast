from source.ensemble.stack_generalization.feature_engineering.data_augmentation import augment_with_quantiles
from source.ensemble.stack_generalization.hyperparam_optimization.optimization import optimize_model, initialize_model
from source.ensemble.stack_generalization.train_importance.plot_importance import plot_feature_importance
from loguru import logger
import pandas as pd
import numpy as np

def run_ensemble_predictions_per_quantile(abs_differenciate, X_train, X_test, y_train, df_train_ensemble,  
                                    predictions, quantile, iteration, add_quantiles, augment_q50,
                                    nr_cv_splits, model_type, solver, 
                                    gbr_update_every_days, gbr_config_params, lr_config_params,
                                    plot_importance_gbr, best_results,
                                    X_train_quantile10=None, X_test_quantile10=None, df_train_ensemble_quantile10=None, 
                                    X_train_quantile90=None, X_test_quantile90=None, df_train_ensemble_quantile90=None):
    " Run ensemble predictions for a specific quantile."
    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Run ensemble predictions for quantile {quantile} </fg 250,128,114>')

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
    # Initialize model
    model = initialize_model(model_type, quantile, best_params, solver)
    # Fit model
    fitted_model = model.fit(X_train_augmented, y_train)
    if plot_importance_gbr and model_type == 'GBR':
        logger.opt(colors=True).info(f'<fg 250,128,114> GBR feature importance </fg 250,128,114>')
        plot_feature_importance(fitted_model.feature_importances_,
                                df_train_ensemble_augmented.drop(columns=['diff_norm_targ']))
    # Make predictions
    raw_predictions = fitted_model.predict(X_test_augmented)
    if not abs_differenciate:
        raw_predictions[raw_predictions < 0] = 0  # Set negative predictions to 0
    predictions[quantile] = raw_predictions  # Store predictions
    return predictions, best_results, fitted_model, X_train_augmented, X_test_augmented, df_train_ensemble_augmented


def run_ensemble_variability_predictions(X_train_2stage, y_train_2stage, X_test_2stage, variability_predictions, quantile, nr_cv_splits, var_model_type, solver, var_gbr_config_params, var_lr_config_params, gbr_update_every_days, iteration, best_results_var):
    " Run ensemble variability predictions"
    assert var_model_type in ['GBR', 'LR'], 'Invalid model type'
    logger.opt(colors=True).info(f'<fg 72,201,176> Run ensemble variability predictions for quantile {quantile} </fg 72,201,176>')
    # Optimize model hyperparameters
    if iteration % gbr_update_every_days == 0:  # Optimize hyperparameters every gbr_update_every_days
        logger.opt(colors=True).info(f'<fg 72,201,176> Optimizing model hyperparameters - updating every {gbr_update_every_days} days</fg 72,201,176>')
        best_score, best_params_var = optimize_model(X_train_2stage, y_train_2stage, quantile, nr_cv_splits, var_model_type, solver, var_gbr_config_params, var_lr_config_params)
        best_results_var[quantile] = [('best_score', best_score), ('params', best_params_var)]
    else:
        logger.opt(colors=True).info(f'<fg 72,201,176> Using best hyperparameters from first iteration </fg 72,201,176>')
        best_params_var = best_results_var[quantile][1][1]
    model = initialize_model(var_model_type, quantile, best_params_var, solver)  # Initialize model
    var_fitted_model = model.fit(X_train_2stage, y_train_2stage)  # Fit model
    raw_variability_predictions = var_fitted_model.predict(X_test_2stage)  # Make predictions
    variability_predictions[quantile] = raw_variability_predictions  # Store predictions
    return variability_predictions, best_results_var, var_fitted_model



def predico_ensemble_predictions_per_quantile(abs_differenciate, X_train, X_test, y_train, df_train_ensemble,  
                                    predictions, quantile, iteration, add_quantiles, augment_q50,
                                    nr_cv_splits, model_type, solver, 
                                    gbr_update_every_days, gbr_config_params, lr_config_params,
                                    plot_importance_gbr, best_results,
                                    X_train_quantile10=np.array([]), X_test_quantile10=np.array([]), df_train_ensemble_quantile10=pd.DataFrame(), 
                                    X_train_quantile90=np.array([]), X_test_quantile90=np.array([]), df_train_ensemble_quantile90=pd.DataFrame()):
    " Run ensemble predictions for a specific quantile."
    logger.info('   ')
    logger.opt(colors=True).info(f'<fg 250,128,114> Run ensemble predictions for quantile {quantile} </fg 250,128,114>')

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
    # Initialize model
    model = initialize_model(model_type, quantile, best_params, solver)
    # Fit model
    fitted_model = model.fit(X_train_augmented, y_train)
    if plot_importance_gbr and model_type == 'GBR':
        logger.opt(colors=True).info(f'<fg 250,128,114> GBR feature importance </fg 250,128,114>')
        plot_feature_importance(fitted_model.feature_importances_,
                                df_train_ensemble_augmented.drop(columns=['diff_norm_targ']))
    # Make predictions
    raw_predictions = fitted_model.predict(X_test_augmented)
    if not abs_differenciate:
        raw_predictions[raw_predictions < 0] = 0  # Set negative predictions to 0
    predictions[quantile] = raw_predictions  # Store predictions
    return predictions, best_results, fitted_model, X_train_augmented, X_test_augmented, df_train_ensemble_augmented


def predico_ensemble_variability_predictions(X_train_2stage, y_train_2stage, X_test_2stage, variability_predictions, quantile, nr_cv_splits, var_model_type, solver, var_gbr_config_params, var_lr_config_params, gbr_update_every_days, iteration, best_results_var):
    " Run ensemble variability predictions"
    assert var_model_type in ['GBR', 'LR'], 'Invalid model type'
    logger.opt(colors=True).info(f'<fg 72,201,176> Run ensemble variability predictions for quantile {quantile} </fg 72,201,176>')
    # Optimize model hyperparameters
    if iteration % gbr_update_every_days == 0:  # Optimize hyperparameters every gbr_update_every_days
        logger.opt(colors=True).info(f'<fg 72,201,176> Optimizing model hyperparameters - updating every {gbr_update_every_days} days</fg 72,201,176>')
        best_score, best_params_var = optimize_model(X_train_2stage, y_train_2stage, quantile, nr_cv_splits, var_model_type, solver, var_gbr_config_params, var_lr_config_params)
        best_results_var[quantile] = [('best_score', best_score), ('params', best_params_var)]
    else:
        logger.opt(colors=True).info(f'<fg 72,201,176> Using best hyperparameters from first iteration </fg 72,201,176>')
        best_params_var = best_results_var[quantile][1][1]
    model = initialize_model(var_model_type, quantile, best_params_var, solver)  # Initialize model
    var_fitted_model = model.fit(X_train_2stage, y_train_2stage)  # Fit model
    raw_variability_predictions = var_fitted_model.predict(X_test_2stage)  # Make predictions
    variability_predictions[quantile] = raw_variability_predictions  # Store predictions
    return variability_predictions, best_results_var, var_fitted_model