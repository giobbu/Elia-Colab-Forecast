import pytest
import pandas as pd
import numpy as np
from source.ensemble.combination_scheme.avg_weights import calculate_weights, normalize_weights, calculate_combination_forecast, create_weighted_avg_df, calculate_weighted_avg

def test_calculate_weights_empty_dataframe():
    "Test calculate_weights with empty dataframe"
    df = pd.DataFrame()
    with pytest.raises(AssertionError, match='Dataframe is empty'):
        calculate_weights(df)

def test_calculate_weights_valid_dataframe(data):
    "Test calculate_weights with valid dataframe"
    df = pd.DataFrame(data)
    lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight = calculate_weights(df)
    assert 'forecast' in lst_cols_forecasts
    assert 'confidence10' in lst_cols_forecasts
    assert 'confidence90' in lst_cols_forecasts
    assert lst_q10_weight[0]['confidence10'] > 0
    assert lst_q50_weight[0]['forecast'] > 0
    assert lst_q90_weight[0]['confidence90'] > 0

def test_calculate_weights_invalid_column(invalid_data):
    "Test calculate_weights with invalid column"
    df = invalid_data
    with pytest.raises(ValueError, match='Not a valid column'):
        calculate_weights(df)

def test_normalize_weights_single_item(single_weight):
    "Test normalize_weights with single item"
    lst_weight = single_weight
    normalized = normalize_weights(lst_weight)
    assert normalized == [{'forecast': 1}], f"Expected [{'forecast': 1}], but got {normalized}"

def test_normalize_weights_multiple_items(multiple_weights):
    "Test normalize_weights with multiple items"
    lst_weight = multiple_weights
    normalized = normalize_weights(lst_weight)
    expected_sum = 1.0
    actual_sum = sum([list(d.values())[0] for d in normalized])
    assert actual_sum == pytest.approx(expected_sum), f"Expected sum of normalized weights to be 1, but got {actual_sum}"

def test_calculate_combination_forecast_forecasts_confidence10_confidence90(mock_data_forecasts_confidence10_confidence90, mock_weights_forecasts_confidence10_confidence90):
    "Test calculate_combination_forecast with valid data"
    df = mock_data_forecasts_confidence10_confidence90
    lst_cols_forecasts = ['forecast1', 'forecast2', 'forecast3', 'confidence101', 'confidence102', 'confidence901', 'confidence902']
    norm_lst_q50_pb_loss, norm_lst_q10_pb_loss, norm_lst_q90_pb_loss = mock_weights_forecasts_confidence10_confidence90

    combination_forecast, combination_quantile10, combination_quantile90 = calculate_combination_forecast(
        df, lst_cols_forecasts, norm_lst_q50_pb_loss, norm_lst_q10_pb_loss, norm_lst_q90_pb_loss
    )
    expected_combination_forecast = (
        df['forecast1'] * 0.3 +
        df['forecast2'] * 0.4 +
        df['forecast3'] * 0.3
    )
    expected_combination_quantile10 = (
        df['confidence101'] * 0.6 +
        df['confidence102'] * 0.4
    )
    expected_combination_quantile90 = (
        df['confidence901'] * 0.7 +
        df['confidence902'] * 0.3
    )
    np.testing.assert_array_almost_equal(combination_forecast, expected_combination_forecast)
    np.testing.assert_array_almost_equal(combination_quantile10, expected_combination_quantile10)
    np.testing.assert_array_almost_equal(combination_quantile90, expected_combination_quantile90)


def test_create_weighted_avg_df(mock_data_for_weighted_avg, mock_combination_forecast, mock_combination_quantile10, mock_combination_quantile90):
    "Test create_weighted_avg_df with valid data"
    df = mock_data_for_weighted_avg
    combination_forecast = mock_combination_forecast
    combination_quantile10 = mock_combination_quantile10
    combination_quantile90 = mock_combination_quantile90
    result_df = create_weighted_avg_df(df, combination_forecast, combination_quantile10, combination_quantile90)
    expected_df = pd.DataFrame({
        'Q10': combination_quantile10,
        'mean_prediction': combination_forecast,
        'Q90': combination_quantile90,
        'diff_norm_measured': df['diff_norm_measured']
    }, index=df.index)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_calculate_weighted_avg(mock_calculate_weighted_avg_data):

    df_train_norm_diff, df_test_norm_diff, start_predictions = mock_calculate_weighted_avg_data

    print(df_test_norm_diff)

    # Test without var
    df_weighted_avg, dict_weights = calculate_weighted_avg(df_train_norm_diff, df_test_norm_diff, start_predictions, window_size_valid=1, var=False)
    
    assert 'mean_prediction' in df_weighted_avg.columns
    assert 'Q10' in df_weighted_avg.columns
    assert 'Q90' in df_weighted_avg.columns
    assert 'diff_norm_measured' in df_weighted_avg.columns
    assert 0.5 in dict_weights
    assert 0.1 in dict_weights
    assert 0.9 in dict_weights
    
    # Test with var
    df_weighted_avg_var, dict_weights_var = calculate_weighted_avg(df_train_norm_diff, df_test_norm_diff, start_predictions, window_size_valid=1, var=True)
    
    assert 'mean_prediction' in df_weighted_avg_var.columns
    assert 'diff_norm_measured' in df_weighted_avg_var.columns
    assert 0.5 in dict_weights_var

