import pytest
from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_gbr import optimize_gbr
from source.ensemble.stack_generalization.hyperparam_optimization.models.quantile_lr import optimize_lr

def test_optimize_gbr_valid_inputs(mock_data_optimize_gbr):
    " Test optimize_gbr with valid inputs "
    # Unpack the mock data
    X_train, y_train, params, nr_cv_splits, quantile = mock_data_optimize_gbr
    # optimize_gbr should return two elements
    best_score, best_gbr_params = optimize_gbr(X_train, y_train, quantile, nr_cv_splits, params)
    # Check the return types
    assert isinstance(best_score, float), "best_score should be a float"
    assert isinstance(best_gbr_params, dict), "best_gbr_params should be a dictionary"
    # Check that best_gbr_params contains the required keys
    required_keys = ['learning_rate', 'max_features', 'max_iter', 'max_depth', 'random_state']
    for key in required_keys:
        assert key in best_gbr_params, f"Missing key in best_gbr_params: {key}"

def test_optimize_gbr_invalid_X_train(mock_data_optimize_gbr):
    " Test optimize_gbr with invalid X_train "
    # Unpack the mock data
    _, y_train, params, nr_cv_splits, quantile = mock_data_optimize_gbr
    with pytest.raises(AssertionError, match="X_train should be a numpy array"):
        optimize_gbr("invalid_X_train", y_train, quantile, nr_cv_splits, params)

def test_optimize_gbr_invalid_y_train(mock_data_optimize_gbr):
    " Test optimize_gbr with invalid y_train "
    # Unpack the mock data
    X_train, _, params, nr_cv_splits, quantile = mock_data_optimize_gbr
    with pytest.raises(AssertionError, match="y_train should be a numpy array"):
        optimize_gbr(X_train, "invalid_y_train", quantile, nr_cv_splits, params)

def test_optimize_gbr_invalid_quantile(mock_data_optimize_gbr):
    " Test optimize_gbr with invalid quantile "
    # Unpack the mock data
    X_train, y_train, params, nr_cv_splits, _ = mock_data_optimize_gbr
    with pytest.raises(AssertionError, match="Invalid quantile value"):
        optimize_gbr(X_train, y_train, 0.25, nr_cv_splits, params)

def test_optimize_gbr_invalid_nr_cv_splits(mock_data_optimize_gbr):
    " Test optimize_gbr with invalid nr_cv_splits "
    # Unpack the mock data
    X_train, y_train, params, _, quantile = mock_data_optimize_gbr
    with pytest.raises(AssertionError, match="nr_cv_splits should be an integer"):
        optimize_gbr(X_train, y_train, quantile, "invalid_nr_cv_splits", params)

def test_optimize_gbr_invalid_params(mock_data_optimize_gbr):
    " Test optimize_gbr with invalid params "
    # Unpack the mock data
    X_train, y_train, _, nr_cv_splits, quantile = mock_data_optimize_gbr
    with pytest.raises(AssertionError, match="params should be a dictionary"):
        optimize_gbr(X_train, y_train, quantile, nr_cv_splits, "invalid_params")