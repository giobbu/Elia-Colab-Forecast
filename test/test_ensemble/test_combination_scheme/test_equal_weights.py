import pandas as pd
import pytest
from source.ensemble.combination_scheme.equal_weights import calculate_equal_weights

def test_calculate_equal_weights(mock_calculate_equal_weights_data):
    " Test calculate_equal_weights with valid data "
    df_test_norm_diff = mock_calculate_equal_weights_data
    df_equal_weights = calculate_equal_weights(df_test_norm_diff, start_prediction_timestamp='2022-01-01 00:00:00')
    expected_data = {
        'q10_equal_weights': [float(i - 1) for i in range(96)],
        'q50_equal_weights': [float(i) for i in range(96)],
        'q90_equal_weights': [float(i + 1) for i in range(96)],
        'targets': [i + 3 for i in range(96)],
    }
    expected_df = pd.DataFrame(expected_data, index=pd.date_range(start='2022-01-01 00:00:00', periods=96, freq='D'))
    pd.testing.assert_frame_equal(df_equal_weights, expected_df)

def test_calculate_equal_weights_missing_column(mock_calculate_equal_weights_missing_data):
    " Test calculate_equal_weights with missing norm_measured column "
    df_test_norm_diff = mock_calculate_equal_weights_missing_data
    with pytest.raises(AssertionError, match="norm_measured column is missing"):
        calculate_equal_weights(df_test_norm_diff, start_prediction_timestamp='2022-01-01 00:00:00')