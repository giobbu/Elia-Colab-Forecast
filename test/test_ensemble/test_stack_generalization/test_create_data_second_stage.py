import pytest
import numpy as np
import pandas as pd
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_var_ensemble_dataframe

def test_create_2stage_dataframe(mock_data_2stage_dataframe):
    "Test create_2stage_dataframe function."
    df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample = mock_data_2stage_dataframe
    df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample)
    assert isinstance(df_2stage, pd.DataFrame), "Output should be a DataFrame"
    assert 'predictions' in df_2stage.columns, "Output DataFrame should contain 'predictions' column"
    assert 'targets' in df_2stage.columns, "Output DataFrame should contain 'targets' column"
    expected_index = df_train_ensemble.index.tolist() + df_test_ensemble.index.tolist()
    assert df_2stage.index.tolist() == expected_index, "Output DataFrame index should match concatenated indices of train and test DataFrames"
    with pytest.raises(AssertionError, match="Length mismatch between train data and in-sample predictions"):
        create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample[:-1], predictions_outsample)
    with pytest.raises(AssertionError, match="Length mismatch between test data and out-sample predictions"):
        create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample[:-1])
    with pytest.raises(AssertionError, match="Length mismatch between targets and in-sample predictions"):
        create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train[:-1], y_test, predictions_insample, predictions_outsample)
    with pytest.raises(AssertionError, match="Length mismatch between targets and out-sample predictions"):
        create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test[:-1], predictions_insample, predictions_outsample)

def test_create_var_ensemble_dataframe(mock_data_var_ensemble_dataframe):
    "Test create_var_ensemble_dataframe function."
    quantiles, quantile_predictions_dict, df_test = mock_data_var_ensemble_dataframe
    df_pred_ensemble = create_var_ensemble_dataframe(quantiles, quantile_predictions_dict, df_test)
    # Check output DataFrame
    assert isinstance(df_pred_ensemble, pd.DataFrame), "Output should be a DataFrame"
    assert 'targets' in df_pred_ensemble.columns, "Output DataFrame should contain 'targets' column"
    # Check quantile columns
    for quantile in quantiles:
        quantile_col = str(int(quantile*100)) + '_var_predictions'
        assert quantile_col in df_pred_ensemble.columns, f"Output DataFrame should contain '{quantile_col}' column"
    # Check index and targets
    assert df_pred_ensemble.index.tolist() == df_test.index.tolist(), "Output DataFrame index should match test DataFrame index"
    # Check targets
    assert np.array_equal(df_pred_ensemble['targets'].values, df_test['targets'].values), "Targets should match test DataFrame targets"
    # Check quantile predictions
    with pytest.raises(AssertionError, match="Length mismatch between quantiles and quantile predictions"):
        invalid_quantiles = [0.1, 0.5]  # fewer quantiles than predictions
        create_var_ensemble_dataframe(invalid_quantiles, quantile_predictions_dict, df_test)
    # Check quantile predictions
    with pytest.raises(AssertionError, match="Length mismatch between test data and predictions"):
        invalid_predictions_dict = {
            0.1: [(date, np.random.rand()) for date in df_test.index[:-1]],
            0.5: [(date, np.random.rand()) for date in df_test.index[:-1]],
            0.9: [(date, np.random.rand()) for date in df_test.index[:-1]],
        }
        create_var_ensemble_dataframe(quantiles, invalid_predictions_dict, df_test)
