import pandas as pd
from source.utils.ensemble_predictions import dict2df_predictions, dict2df_quantiles10, dict2df_quantiles90


def test_dict2df_predictions(sample_prediction_dict):
    "Test that the dictionary is converted to a DataFrame correctly"
    col_name = 'blabla'
    result = dict2df_predictions(sample_prediction_dict, col_name)
    # Check if DataFrame is created with correct columns and index
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert result.columns == [col_name + '_pred']
    # Check if DataFrame values match the input dictionary
    expected_values = [10, 20, 30]
    assert result[col_name + '_pred'].tolist() == expected_values

def test_dict2df_quantiles10(sample_quantile_prediction_dict):
    "Test that the dictionary is converted to a DataFrame correctly"
    col_name = 'blabla'
    result = dict2df_quantiles10(sample_quantile_prediction_dict, col_name)
    # Check if DataFrame is created with correct columns and index
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert result.columns == [col_name + '_quantile10']
    # Check if DataFrame values match the input dictionary
    expected_values = [10, 20, 30]
    assert result[col_name + '_quantile10'].tolist() == expected_values

def test_dict2df_quantiles90(sample_quantile_prediction_dict):
    "Test that the dictionary is converted to a DataFrame correctly"
    col_name = 'blabla'
    result = dict2df_quantiles90(sample_quantile_prediction_dict, col_name)
    # Check if DataFrame is created with correct columns and index
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert result.columns == [col_name + '_quantile90']
    # Check if DataFrame values match the input dictionary
    expected_values = [10, 20, 30]
    assert result[col_name + '_quantile90'].tolist() == expected_values