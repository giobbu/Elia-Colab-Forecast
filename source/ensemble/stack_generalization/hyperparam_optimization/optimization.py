import numpy as np
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Lasso

from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_gbr import optimize_gbr
from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_lr import optimize_lr


def optimize_model(X_train, y_train, quantile, nr_cv_splits, model_type, solver, gbr_config_params, lr_config_params):
    " Optimize selected model hyperparameters."
    if model_type == 'GBR':
        best_score, best_params = optimize_gbr(X_train, y_train, quantile, nr_cv_splits, gbr_config_params)
    elif model_type == 'LR':
        best_score, best_params = optimize_lr(X_train, y_train, quantile, nr_cv_splits, solver, lr_config_params)
    else:
        raise ValueError('"model_type" is not valid')
    logger.info(f'best_score {round(best_score, 3)}')
    logger.info(f'best_params {best_params}')
    return best_score, best_params

def initialize_model(model_type, quantile, best_params, solver):
    " Initialize selected model."
    assert best_params is not None, "Best parameters must be provided"
    if model_type == 'GBR':
        if quantile == 0.5:
            model = HistGradientBoostingRegressor(**best_params)
        else:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **best_params)
    elif model_type == 'LR':
        if quantile == 0.5:
            model = Lasso(**best_params)  #LinearRegression(**best_params)
        else:
            model = QuantileRegressor(quantile=quantile, solver=solver, **best_params)
    else:
        raise ValueError('"model_type" is not valid')
    return model

def initialize_train_and_predict(predictions, 
                                model_type, 
                                quantile, 
                                best_params, 
                                solver, 
                                X_train, 
                                y_train, 
                                X_test, 
                                insample=False, 
                                predictions_insample=None, 
                                predictions_outsample=None):
    """
    Initializes, fits a model, and makes predictions.
    """
    # Initialize model with best params
    model = initialize_model(model_type, quantile, best_params, solver)
    # Fit model
    fitted_model = model.fit(X_train, y_train)
    # Make predictions
    predictions[quantile] = fitted_model.predict(X_test)
    # Return fitted model and predictions
    if not insample:
        return fitted_model, predictions
    elif insample:
        predictions_insample[quantile] = fitted_model.predict(X_train)
        predictions_outsample[quantile] = fitted_model.predict(X_test) 
        return fitted_model, predictions, predictions_insample, predictions_outsample

def permutation_quantile_regression(best_params, solver, X, y, quantile, n_permutations=100):
    " Perform permutation-based quantile regression to calculate p-values."
    coefs = []
    # Fit the model on the original dataset to get the observed coefficients
    model_original = QuantileRegressor(quantile=quantile, solver=solver, **best_params).fit(X, y)
    coefs_original = model_original.coef_
    for _ in range(n_permutations):
        # Permute y (random shuffle)
        y_permuted = np.random.permutation(y)
        if quantile == 0.5:
            # Fit the model on the permuted dataset
            model = Lasso(**best_params).fit(X, y_permuted)
        else:
            # Fit the model on the permuted dataset
            model = QuantileRegressor(quantile=quantile, solver=solver, **best_params).fit(X, y_permuted)
        coefs.append(model.coef_)
    # Convert the list of coefficients to a NumPy array
    coefs = np.array(coefs)
    # Calculate p-values by comparing original coefficients to the permuted coefficients
    p_values = np.mean(np.abs(coefs) >= np.abs(coefs_original), axis=0)
    return coefs_original, p_values