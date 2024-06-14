import pytest
import pandas as pd
import numpy as np


## weighted average

@pytest.fixture
def data():
    return pd.DataFrame({
        'measured': [1.0, 2.0, 3.0],
        'forecast': [1.1, 2.1, 3.1],
        'confidence10': [0.9, 1.9, 2.9],
        'confidence90': [1.2, 2.2, 3.2]
    })

@pytest.fixture
def invalid_data():
    return pd.DataFrame({
        'measured': [1.0, 2.0, 3.0],
        'invalid': [1.1, 2.1, 3.1]
    })

@pytest.fixture
def single_weight():
    return [{'forecast': 2}]

@pytest.fixture
def multiple_weights():
    return [{'forecast1': 2}, {'forecast2': 3}, {'forecast3': 5}]

@pytest.fixture
def mock_data_forecasts_confidence10_confidence90():
    return pd.DataFrame({
        'forecast1': [1.1, 2.1, 3.1],
        'forecast2': [1.2, 2.2, 3.2],
        'forecast3': [1.3, 2.3, 3.3],
        'confidence101': [0.9, 1.9, 2.9],
        'confidence102': [0.8, 1.8, 2.8],
        'confidence901': [1.5, 2.5, 3.5],
        'confidence902': [1.4, 2.4, 3.4]
    })

@pytest.fixture
def mock_weights_forecasts_confidence10_confidence90():
    return (
        [{'forecast1': 0.3}, {'forecast2': 0.4}, {'forecast3': 0.3}],
        [{'confidence101': 0.6}, {'confidence102': 0.4}],
        [{'confidence901': 0.7}, {'confidence902': 0.3}]
    )

@pytest.fixture
def mock_data_for_weighted_avg():
    return pd.DataFrame({
        'diff_norm_measured': [0.1, 0.2, 0.3],
        'other_column': [1, 2, 3]  # Additional columns to mimic a real-world scenario
    })

@pytest.fixture
def mock_combination_forecast():
    return np.array([1.15, 2.25, 3.35])

@pytest.fixture
def mock_combination_quantile10():
    return np.array([0.85, 1.95, 2.85])

@pytest.fixture
def mock_combination_quantile90():
    return np.array([1.25, 2.35, 3.45])


@pytest.fixture
def mock_calculate_weighted_avg_data():
    # Mock data
    date_rng_train = pd.date_range(start='2022-01-01', end='2022-01-10', freq='D', tz='UTC')
    date_rng_test = pd.date_range(start='2022-01-11', end='2022-01-15', freq='D', tz='UTC')
    df_train_norm_diff = pd.DataFrame(date_rng_train, columns=['date'])
    df_train_norm_diff.set_index('date', inplace=True)
    df_train_norm_diff['diff_norm_measured'] = np.random.randn(len(date_rng_train))
    df_train_norm_diff['forecast'] = np.random.randn(len(date_rng_train))
    df_train_norm_diff['confidence10'] = np.random.randn(len(date_rng_train))
    df_train_norm_diff['confidence90'] = np.random.randn(len(date_rng_train))
    
    df_test_norm_diff = pd.DataFrame(date_rng_test, columns=['date'])
    df_test_norm_diff.set_index('date', inplace=True)
    df_test_norm_diff['diff_norm_measured'] = np.random.randn(len(date_rng_test))
    df_test_norm_diff['forecast'] = np.random.randn(len(date_rng_test))
    df_test_norm_diff['confidence10'] = np.random.randn(len(date_rng_test))
    df_test_norm_diff['confidence90'] = np.random.randn(len(date_rng_test))
    
    start_predictions = pd.Timestamp('2022-01-11', tz='UTC')
    return df_train_norm_diff, df_test_norm_diff, start_predictions


@pytest.fixture
def mock_calculate_equal_weights_data():
    data = {
        'diff_norm_weekaheadconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_dayaheadconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_dayahead11hconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_weekaheadforecast': [1.0, 1.1, 1.2],
        'diff_norm_dayaheadforecast': [1.0, 1.1, 1.2],
        'diff_norm_dayahead11hforecast': [1.0, 1.1, 1.2],
        'diff_norm_weekaheadconfidence90': [1.9, 2.0, 2.1],
        'diff_norm_dayaheadconfidence90': [1.9, 2.0, 2.1],
        'diff_norm_dayahead11hconfidence90': [1.9, 2.0, 2.1],
        'diff_norm_measured': [1.5, 1.6, 1.7]
    }
    df_test_norm_diff = pd.DataFrame(data)
    return df_test_norm_diff


@pytest.fixture
def mock_calculate_equal_weights_missing_data():
    data = {
        'diff_norm_weekaheadconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_dayaheadconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_dayahead11hconfidence10': [0.1, 0.2, 0.3],
        'diff_norm_weekaheadforecast': [1.0, 1.1, 1.2],
        'diff_norm_dayaheadforecast': [1.0, 1.1, 1.2],
        'diff_norm_dayahead11hforecast': [1.0, 1.1, 1.2],
        'diff_norm_weekaheadconfidence90': [1.9, 2.0, 2.1],
        'diff_norm_dayaheadconfidence90': [1.9, 2.0, 2.1],
        'diff_norm_dayahead11hconfidence90': [1.9, 2.0, 2.1]
    }
    df_test_norm_diff = pd.DataFrame(data)
    return df_test_norm_diff


@pytest.fixture
def dict_importance_weights():
    return {
        0.5: [
            {'diff_norm_feature1': 0.2, 'diff_norm_feature2': 0.8}
        ]
    }