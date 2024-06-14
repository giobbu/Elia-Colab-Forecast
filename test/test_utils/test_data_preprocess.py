import pytest
import pandas as pd
from source.utils.data_preprocess import scale, detect_ramp_event, differentiate_dataframe

def test_scale_max_cap(sample_data_preprocess):
    "Test that the maximum capacity is greater than 0"
    df = sample_data_preprocess
    col_name = 'values'
    with pytest.raises(AssertionError, match="Maximum capacity must be greater than 0"):
        scale(df, col_name, -10)
    with pytest.raises(AssertionError, match="Maximum capacity must be greater than 0"):
        scale(df, col_name, 0)

def test_scale_valid(sample_data_preprocess):
    "Test that the values are correctly scaled"
    df = sample_data_preprocess
    col_name = 'values'
    max_cap = 50
    result = scale(df, col_name, max_cap)
    # Check if the values are correctly scaled
    expected_values = [0.2, 0.4, 0.6, 0.8]
    assert result.tolist() == expected_values
    # Check if all values are between 0 and 1
    assert all(0 <= x <= 1 for x in result)

def test_detect_ramp_event_valid(sample_df_ramp_event):
    "Test that ramp event is correctly detected"
    df = sample_df_ramp_event
    ramp_threshold = 0.3
    result = detect_ramp_event(df, ramp_threshold)
    # Check if the ramp_event columns are added
    assert 'ramp_event' in result.columns
    assert 'ramp_event_up' in result.columns
    assert 'ramp_event_down' in result.columns
    # Check the values in the ramp_event columns
    expected_ramp_event = [0, 1, 1, 0, 1]
    expected_ramp_event_up = [0, 1, 0, 0, 0]
    expected_ramp_event_down = [0, 0, 1, 0, 1]
    assert result['ramp_event'].tolist() == expected_ramp_event
    assert result['ramp_event_up'].tolist() == expected_ramp_event_up
    assert result['ramp_event_down'].tolist() == expected_ramp_event_down

def test_detect_ramp_event_threshold_zero(sample_df_ramp_event):
    "Test that the ramp threshold is greater than 0"
    with pytest.raises(AssertionError, match="Ramp threshold must be greater than 0"):
        detect_ramp_event(sample_df_ramp_event, 0)

def test_detect_ramp_event_threshold_negative(sample_df_ramp_event):
    "Test that the ramp threshold is greater than 0"
    with pytest.raises(AssertionError, match="Ramp threshold must be greater than 0"):
        detect_ramp_event(sample_df_ramp_event, -0.1)

def test_differentiate_dataframe_single_row():
    "Test that the function raises an AssertionError if the DataFrame has only one row"
    df = pd.DataFrame({'value1': [10], 'value2': [5]})
    with pytest.raises(AssertionError, match="Input DataFrame must have more than one row"):
        differentiate_dataframe(df)

def test_differentiate_dataframe_empty_df():
    "Test that the function raises an AssertionError if the DataFrame is empty"
    df = pd.DataFrame(columns=['value1', 'value2'])
    with pytest.raises(AssertionError, match="Input DataFrame must have more than one row"):
        differentiate_dataframe(df)

def test_differentiate_dataframe_non_numeric():
    "Test that the function raises a TypeError if the DataFrame contains non-numeric values"
    df = pd.DataFrame({'value1': [10, 20, 'a'], 'value2': [5, 15, 25]})
    with pytest.raises(TypeError):
        differentiate_dataframe(df)