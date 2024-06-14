import pytest
from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_gbr import optimize_gbr
from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_lr import optimize_lr


def test_optimize_gbr(mock_data_optimize_gbr):
    "Test optimize_gbr function."
    X_train, y_train, params = mock_data_optimize_gbr
    # Test for valid inputs
    best_score, best_gbr_params = optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits=3, params=params)
    assert isinstance(best_score, float)
    assert isinstance(best_gbr_params, dict)
    assert all(key in best_gbr_params for key in ['learning_rate', 'max_features', 'max_iter', 'max_depth'])
    # Test for invalid quantile
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.3, nr_cv_splits=3, params=params)
    # Test for non-numpy array X_train
    with pytest.raises(AssertionError):
        optimize_gbr(list(X_train), y_train, quantile=0.1, nr_cv_splits=3, params=params)
    # Test for non-numpy array y_train
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, list(y_train), quantile=0.1, nr_cv_splits=3, params=params)
    # Test for non-integer nr_cv_splits
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits='3', params=params)
    # Test for missing learning_rate
    invalid_params = params.copy()
    invalid_params['learning_rate'] = None
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits=3, params=invalid_params)
    # Test for missing max_features
    invalid_params = params.copy()
    invalid_params['max_features'] = None
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits=3, params=invalid_params)
    # Test for missing max_depth
    invalid_params = params.copy()
    invalid_params['max_depth'] = None
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits=3, params=invalid_params)
    # Test for missing max_iter
    invalid_params = params.copy()
    invalid_params['max_iter'] = None
    with pytest.raises(AssertionError):
        optimize_gbr(X_train, y_train, quantile=0.1, nr_cv_splits=3, params=invalid_params)


def test_optimize_lr(mock_data_optimize_lr):
    "Test optimize_lr function."
    X_train, y_train, params = mock_data_optimize_lr
    # Test for valid inputs
    best_score, best_lr_params = optimize_lr(X_train, y_train, quantile=0.1, nr_cv_splits=3, solver='highs', params=params)
    assert isinstance(best_score, float)
    assert isinstance(best_lr_params, dict)
    assert 'fit_intercept' in best_lr_params
    assert 'alpha' in best_lr_params
    # Test for valid inputs with quantile 0.5
    best_score, best_lr_params = optimize_lr(X_train, y_train, quantile=0.5, nr_cv_splits=3, solver='highs', params=params)
    assert isinstance(best_score, float)
    assert isinstance(best_lr_params, dict)
    assert 'fit_intercept' in best_lr_params
    assert 'alpha' not in best_lr_params  # alpha is not used for LinearRegression
    # Test for invalid quantile
    with pytest.raises(AssertionError):
        optimize_lr(X_train, y_train, quantile=0.3, nr_cv_splits=3, solver='highs', params=params)
    # Test for non-numpy array X_train
    with pytest.raises(AssertionError):
        optimize_lr(list(X_train), y_train, quantile=0.1, nr_cv_splits=3, solver='highs', params=params)
    # Test for non-numpy array y_train
    with pytest.raises(AssertionError):
        optimize_lr(X_train, list(y_train), quantile=0.1, nr_cv_splits=3, solver='highs', params=params)
    # Test for non-integer nr_cv_splits
    with pytest.raises(AssertionError):
        optimize_lr(X_train, y_train, quantile=0.1, nr_cv_splits='3', solver='highs', params=params)
    # Test for missing alpha
    invalid_params = params.copy()
    invalid_params['alpha'] = None
    with pytest.raises(AssertionError):
        optimize_lr(X_train, y_train, quantile=0.1, nr_cv_splits=3, solver='highs', params=invalid_params)
    # Test for missing fit_intercept
    invalid_params = params.copy()
    invalid_params['fit_intercept'] = None
    with pytest.raises(AssertionError):
        optimize_lr(X_train, y_train, quantile=0.1, nr_cv_splits=3, solver='highs', params=invalid_params)