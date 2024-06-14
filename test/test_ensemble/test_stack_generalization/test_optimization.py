import pytest
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from source.ensemble.stack_generalization.hyperparam_optimization.optimization import optimize_model, initialize_model


def test_optimize_model_invalid_model_type(mock_data_optimization):
    "Test optimize_model function with invalid model_type."
    X_train, y_train, quantile, nr_cv_splits, solver, gbr_config_params, lr_config_params = mock_data_optimization
    best_score, best_params = optimize_model(X_train, y_train, quantile, nr_cv_splits, 'GBR', solver, gbr_config_params, lr_config_params)
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    best_score, best_params = optimize_model(X_train, y_train, quantile, nr_cv_splits, 'LR', solver, gbr_config_params, lr_config_params)
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    with pytest.raises(ValueError, match='"model_type" is not valid'):
        optimize_model(X_train, y_train, quantile, nr_cv_splits, 'INVALID_MODEL_TYPE', solver, gbr_config_params, lr_config_params)

def test_initialize_model_invalid_type(best_params_gbr):
    "Test initialize_model function with invalid model_type."
    with pytest.raises(ValueError, match='"model_type" is not valid'):
        initialize_model('INVALID', 0.5, best_params_gbr, None)

def test_initialize_model_missing_params():
    "Test initialize_model function with missing best_params."
    with pytest.raises(AssertionError, match="Best parameters must be provided"):
        initialize_model('GBR', 0.5, None, None)

def test_initialize_model_lr_50(best_params_lr):
    "Test initialize_model function with default loss."
    del best_params_lr['alpha']
    model = initialize_model('LR', 0.5, best_params_lr, None)
    assert isinstance(model, LinearRegression)
    assert model.fit_intercept is True

def test_initialize_model_lr_quantile(best_params_lr):
    "Test initialize_model function with quantile loss."
    model = initialize_model('LR', 0.1, best_params_lr, "highs")
    assert isinstance(model, QuantileRegressor)
    assert model.quantile == 0.1
    assert model.solver == "highs"
    assert model.alpha == 0.1

def test_initialize_model_gbr_50(best_params_gbr):
    "Test initialize_model function with default loss."
    model = initialize_model('GBR', 0.5, best_params_gbr, None)
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.learning_rate == 0.1
    assert model.max_features == 1.0
    assert model.max_depth == 5
    assert model.max_iter == 100

def test_initialize_model_gbr_quantile(best_params_gbr):
    "Test initialize_model function with quantile loss."
    model = initialize_model('GBR', 0.1, best_params_gbr, None)
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.loss == 'quantile'
    assert model.quantile == 0.1