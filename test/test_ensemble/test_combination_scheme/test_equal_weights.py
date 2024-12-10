import pytest
import pandas as pd
from source.ensemble.combination_scheme.equal_weights import calculate_equal_weights


# def test_calculate_equal_weights(mock_calculate_equal_weights_data):
#     " Test calculate_equal_weights with valid data"
#     df_test_norm_diff = mock_calculate_equal_weights_data
#     df_equal_weights = calculate_equal_weights(df_test_norm_diff)
#     expected_data = {
#         'Q10': [0.1, 0.2, 0.3],
#         'mean_prediction': [1.0, 1.1, 1.2],
#         'Q90': [1.9, 2.0, 2.1],
#         'diff_norm_measured': [1.5, 1.6, 1.7]
#     }
#     expected_df = pd.DataFrame(expected_data)
#     pd.testing.assert_frame_equal(df_equal_weights, expected_df)

# def test_calculate_equal_weights_missing_column(mock_calculate_equal_weights_missing_data):
#     df_test_norm_diff = mock_calculate_equal_weights_missing_data
#     with pytest.raises(AssertionError, match="diff_norm_measured column is missing"):
#         calculate_equal_weights(df_test_norm_diff)