import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import QuantileRegressor, Lasso
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import evaluate

def optimize_lr(X_train, y_train, quantile, nr_cv_splits, solver, params):
    """ Hyperparameter optimization for Quantile Linear Regression. 
    args:
        X_train: np.array, training data
        y_train: np.array, target data
        quantile: float, quantile
        nr_cv_splits: int, number of cross-validation splits
        solver: str, solver
        params: dict, parameters
    returns:
        best_score: float, best score
        best_lr_params: dict, best parameters"""
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert quantile in [0.1, 0.5, 0.9], "Invalid quantile value. Must be 0.1, 0.5, or 0.9."
    assert isinstance(nr_cv_splits, int), "nr_cv_splits should be an integer"
    assert isinstance(params, dict), "params should be a dictionary"
    assert 'alpha' in params, "alpha must be provided"
    assert 'fit_intercept' in params, "fit_intercept must be provided"
    assert params['alpha'] is not None, 'alpha must be provided'
    assert params['fit_intercept'] is not None, 'fit_intercept must be provided'
    best_lr_params = None
    best_score=np.exp(1000000)
    ts_cv = TimeSeriesSplit(n_splits=nr_cv_splits)
    for alpha in params['alpha']:
        for fit_intercept in params['fit_intercept']:
            lr_params = dict(
                alpha=alpha,
                fit_intercept=fit_intercept)
            if quantile == 0.5:
                lr = Lasso(**lr_params)  
            else:
                lr = QuantileRegressor(quantile=quantile, solver=solver, **lr_params)
            mean_cv_score = evaluate(lr, X_train, y_train, cv=ts_cv, quantile=quantile)  
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_lr_params = lr_params
    return best_score, best_lr_params