import pytest
import pandas as pd
import numpy as np
from source.ensemble.stack_generalization.data_preparation.data_train_test import prepare_train_test_data, get_numpy_Xy_train_test

def test_prepare_train_test_data_parameter_types(mock_train_test_split_data):
    "Test if the function raises AssertionError for invalid parameter types"
    df_ensemble, df_val, df_test, start_predictions, max_lag = mock_train_test_split_data
    # Modify types to provoke assertion errors
    with pytest.raises(AssertionError, match="df_ensemble should be a DataFrame"):
        prepare_train_test_data(None, df_val, df_test, start_predictions, max_lag)
    with pytest.raises(AssertionError, match="df_val should be a DataFrame"):
        prepare_train_test_data(df_ensemble, None, df_test, start_predictions, max_lag)
    with pytest.raises(AssertionError, match="df_test should be a DataFrame"):
        prepare_train_test_data(df_ensemble, df_val, None, start_predictions, max_lag)
    with pytest.raises(AssertionError, match="start_predictions should be a Timestamp"):
        prepare_train_test_data(df_ensemble, df_val, df_test, "2024-01-06", max_lag)
    with pytest.raises(AssertionError, match="max_lag should be an integer"):
        prepare_train_test_data(df_ensemble, df_val, df_test, start_predictions, "2")
    with pytest.raises(AssertionError, match="max_lag should be greater than 0"):
        prepare_train_test_data(df_ensemble, df_val, df_test, start_predictions, 0)

def test_prepare_train_test_data_missing_column(mock_train_test_split_data):
    "Test if the function raises AssertionError if 'diff_norm_measured' column is missing"
    df_ensemble, df_val, df_test, start_predictions, max_lag = mock_train_test_split_data
    # Remove 'diff_norm_measured' from df_val
    df_val_without_column = df_val.drop(columns=['diff_norm_measured'])
    # Check if assertion error is raised
    with pytest.raises(AssertionError, match="diff_norm_measured should be in df_val columns"):
        prepare_train_test_data(df_ensemble, df_val_without_column, df_test, start_predictions, max_lag)
    # Remove 'diff_norm_measured' from df_test
    df_test_without_column = df_test.drop(columns=['diff_norm_measured'])
    # Check if assertion error is raised
    with pytest.raises(AssertionError, match="diff_norm_measured should be in df_test columns"):
        prepare_train_test_data(df_ensemble, df_val, df_test_without_column, start_predictions, max_lag)

def test_prepare_train_test_data(mock_train_test_split_data):
    "Test prepare_train_test_data with valid data"
    df_ensemble, df_val, df_test, start_predictions, max_lag = mock_train_test_split_data
    df_train_ensemble, df_test_ensemble = prepare_train_test_data(df_ensemble, df_val, df_test, start_predictions, max_lag)
    # Check the type of the result
    assert isinstance(df_train_ensemble, pd.DataFrame)
    assert isinstance(df_test_ensemble, pd.DataFrame)
    # Check the content of the result
    assert 'diff_norm_targ' in df_train_ensemble.columns
    assert 'diff_norm_targ' in df_test_ensemble.columns
    # Check the values
    assert all(df_train_ensemble['diff_norm_targ'].values == df_val['diff_norm_measured'].values[max_lag:])
    assert all(df_test_ensemble['diff_norm_targ'].values == df_test['diff_norm_measured'].values)

def test_get_numpy_Xy_train_test(mock_data_for_get_numpy_Xy_train_test):
    "Test get_numpy_Xy_train_test with valid data"
    df_train_ensemble, df_test_ensemble = mock_data_for_get_numpy_Xy_train_test
    X_train, y_train, X_test, y_test = get_numpy_Xy_train_test(df_train_ensemble, df_test_ensemble)
    # Check the type of the result
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    # Check the shape of the arrays
    assert X_train.shape == (len(df_train_ensemble), len(df_train_ensemble.columns) - 1)
    assert y_train.shape == (len(df_train_ensemble),)
    assert X_test.shape == (len(df_test_ensemble), len(df_test_ensemble.columns) - 1)
    assert y_test.shape == (len(df_test_ensemble),)
    # Check the content of X_train and X_test
    assert np.array_equal(X_train[:, 0], df_train_ensemble['feature1'].values)
    assert np.array_equal(X_train[:, 1], df_train_ensemble['feature2'].values)
    assert np.array_equal(X_test[:, 0], df_test_ensemble['feature1'].values)
    assert np.array_equal(X_test[:, 1], df_test_ensemble['feature2'].values)
    # Check the content of y_train and y_test
    assert np.array_equal(y_train, df_train_ensemble['target'].values)
    assert np.array_equal(y_test, df_test_ensemble['target'].values)
    # Check for invalid inputs
    with pytest.raises(AssertionError):
        get_numpy_Xy_train_test(None, df_test_ensemble)
    with pytest.raises(AssertionError):
        get_numpy_Xy_train_test(df_train_ensemble, None)


