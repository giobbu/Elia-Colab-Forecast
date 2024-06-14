import pytest
import pandas as pd
from source.utils.file_read import filter_offshore, set_index_datetiemUTC, process_file, process_and_concat_files, filter_df

def test_filter_offshore_offshore(sample_df):
    """Test that the DataFrame is filtered correctly"""
    df = sample_df
    offshore_filter = 'offshore'
    result = filter_offshore(df, offshore_filter)
    expected_data = {
        'offshoreonshore': ['offshore', 'offshore'],
        'value': [10, 30]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_filter_offshore_missing_column(sample_df_no_column):
    """Test that an assertion is raised when the column is missing"""
    df_no_column = sample_df_no_column
    offshore_filter = 'offshore'
    with pytest.raises(AssertionError, match="The DataFrame must contain the 'offshoreonshore' column."):
        filter_offshore(df_no_column, offshore_filter)

def test_set_index_datetiemUTC_valid(df_datetime):
    "Test that the index is set to datetime and converted to UTC"
    result = set_index_datetiemUTC(df_datetime)
    # Check if the index is set to datetime and converted to UTC
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tzinfo is not None
    assert result.index.tzinfo.utcoffset(None) == pd.Timedelta(0)

def test_set_index_datetiemUTC_no_datetime_column(df_no_datetime):
    "Test that an assertion is raised when the datetime column is missing"
    with pytest.raises(AssertionError, match="The DataFrame must contain the 'datetime' column."):
        set_index_datetiemUTC(df_no_datetime)

def test_process_file_invalid_file_extension():
    "Test that an assertion is raised when the file extension is invalid"
    with pytest.raises(AssertionError, match="File must be a json file"):
        process_file("sample.csv")

def test_process_and_concat_files_empty_files():
    "Test that an assertion is raised when there are no files to process"
    with pytest.raises(AssertionError, match="No files to process"):
        process_and_concat_files([])


def test_filter_df_valid(forecasters_df):
    "Test that the DataFrame is filtered correctly"
    forecasts_col = ['forecast', 'confidence10', 'confidence90']
    measured_col = 'measured'
    result = filter_df(forecasters_df, forecasts_col, measured_col)
    # Check if the filtered DataFrame contains the expected columns
    expected_columns = ['measured', 'forecast', 'confidence10_temp', 'confidence90_temp']
    assert result.columns.tolist() == expected_columns


def test_filter_df_invalid_input_types():
    "Test that an assertion is raised when the input types are invalid"
    with pytest.raises(AssertionError, match="forecasts_col must be a list"):
        filter_df(pd.DataFrame(), forecasts_col='forecast_temp', measured_col='measured_temp')
        
    with pytest.raises(AssertionError, match="measured_col must be a string"):
        filter_df(pd.DataFrame(), forecasts_col=['forecast_temp'], measured_col=['measured_temp'])




