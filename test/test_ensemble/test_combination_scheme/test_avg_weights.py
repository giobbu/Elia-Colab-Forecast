import pytest
import pandas as pd
import numpy as np
from source.ensemble.combination_scheme.avg_weights import normalize_weights, calculate_combination_forecast


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
