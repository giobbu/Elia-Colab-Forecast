import pytest
import pandas as pd
import numpy as np
from source.ensemble.stack_generalization.data_preparation.data_train_test import get_numpy_Xy_train_test


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


