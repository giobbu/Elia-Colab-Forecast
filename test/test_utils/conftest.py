import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    " Return a sample DataFrame"
    data = {
        'offshoreonshore': ['offshore', 'onshore', 'offshore', 'onshore'],
        'value': [10, 20, 30, 40]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df_no_column():
    "Return a DataFrame without the required column"
    data = {
        'blabla': ['offshore', 'onshore', 'offshore', 'onshore'],
        'value': [10, 20, 30, 40]
    }
    return pd.DataFrame(data)

@pytest.fixture
def df_datetime():
    data = {
        'datetime': ['2024-01-01 12:00:00', '2024-01-02 12:00:00', '2024-01-03 12:00:00'],
        'value': [10, 20, 30]
    }
    return pd.DataFrame(data)

@pytest.fixture
def df_no_datetime():
    data = {
        'value': [10, 20, 30]
    }
    return pd.DataFrame(data)

@pytest.fixture
def forecasters_df():
    data = {
        'measured': [10, 20, 30],
        'forecast': [15, 25, 35],
        'confidence10_temp': [14, 24, 34],
        'confidence90_temp': [16, 26, 36],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_prediction_dict():
    " Return a sample dictionary with predictions"
    prediction = {
        'datetime': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [10, 20, 30]
    }
    return prediction

@pytest.fixture
def sample_quantile_prediction_dict():
    " Return a sample dictionary with quantile predictions"
    prediction = {
        'datetime': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [10, 20, 30]
    }
    return prediction

@pytest.fixture
def sample_data_preprocess():
    " Return a sample DataFrame for data preprocessing"
    data = {
        'values': [10, 20, 30, 40]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df_ramp_event():
    data = {
        'diff_norm_measured': [0.1, 0.4, -0.5, 0.2, -0.3]
    }
    return pd.DataFrame(data)



