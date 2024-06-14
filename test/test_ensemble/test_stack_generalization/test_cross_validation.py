import pytest
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90, evaluate

def test_score_func_10_valid_input(mock_data_score_func):
    "Test score_func_10 with valid input"
    estimator, X, y = mock_data_score_func
    # Test for valid inputs
    result = score_func_10(estimator, X, y)
    assert isinstance(result, dict)
    assert "mean_pinball_loss" in result
    assert isinstance(result["mean_pinball_loss"], float)

def test_score_func_50_valid_input(mock_data_score_func):
    "Test score_func_50 with valid input"
    estimator, X, y = mock_data_score_func
    # Test for valid inputs
    result = score_func_50(estimator, X, y)
    assert isinstance(result, dict)
    assert "mean_pinball_loss" in result
    assert isinstance(result["mean_pinball_loss"], float)

def test_score_func_90_valid_input(mock_data_score_func):
    "Test score_func_90 with valid input"
    estimator, X, y = mock_data_score_func
    # Test for valid inputs
    result = score_func_90(estimator, X, y)
    assert isinstance(result, dict)
    assert "mean_pinball_loss" in result
    assert isinstance(result["mean_pinball_loss"], float)

def test_evaluate(mock_data_evaluate):
    "Test evaluate with valid input"
    estimator, X, y, cv, quantile = mock_data_evaluate
    # Test for valid inputs
    score_mean = evaluate(estimator, X, y, cv=cv, quantile=quantile)
    assert isinstance(score_mean, float)
    with pytest.raises(AssertionError):
        evaluate(estimator, list(X), y, cv=cv, quantile=quantile)
    with pytest.raises(AssertionError):
        evaluate(estimator, X, list(y), cv=cv, quantile=quantile)
    with pytest.raises(AssertionError):
        evaluate(estimator, X, y, cv=cv, quantile=0.3)