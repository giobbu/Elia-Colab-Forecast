import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10

@pytest.fixture
def mock_train_test_split_data():
    " Mock data for testing train_test_split"
    date_rng_ensemble = pd.date_range(start='1/1/2020', end='1/14/2020', freq='D')
    df_ensemble = pd.DataFrame(date_rng_ensemble, columns=['date'])
    df_ensemble['data'] = range(len(date_rng_ensemble))
    df_ensemble.set_index('date', inplace=True)
    date_rng_val = pd.date_range(start='12/27/2019', end='1/09/2020', freq='D')
    df_val = pd.DataFrame(date_rng_val, columns=['date'])
    df_val['diff_norm_measured'] = range(len(date_rng_val))
    df_val.set_index('date', inplace=True)
    date_rng_test = pd.date_range(start='1/10/2020', end='1/14/2020', freq='D')
    df_test = pd.DataFrame(date_rng_test, columns=['date'])
    df_test['diff_norm_measured'] = range(len(date_rng_test))
    df_test.set_index('date', inplace=True)
    start_predictions = pd.Timestamp('2020-01-10')
    max_lag = 5
    return df_ensemble, df_val, df_test, start_predictions, max_lag

@pytest.fixture
def mock_data_for_get_numpy_Xy_train_test():
    df_train_ensemble = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [100, 200, 300, 400, 500]
    })
    df_test_ensemble = pd.DataFrame({
        'feature1': [6, 7, 8, 9, 10],
        'feature2': [60, 70, 80, 90, 100],
        'target': [600, 700, 800, 900, 1000]
    })
    return df_train_ensemble, df_test_ensemble

@pytest.fixture
def mock_data_optimize_gbr():
    " Mock data for testing optimize_gbr"
    X_train = np.random.random((10, 10))
    y_train = np.random.random(10)
    params = {
        'learning_rate': [0.01, 0.1],
        'max_features': [0.8, 1.0],
        'max_depth': [1, 2],
        'max_iter': [2, 3]
    }
    nr_cv_splits = 3
    quantile = 0.1
    return X_train, y_train, params, nr_cv_splits, quantile

@pytest.fixture
def mock_data_optimize_lr():
    " Mock data for testing optimize_lr"
    X_train = np.random.random((100, 10))
    y_train = np.random.random(100)
    params = {
        'alpha': [0.01, 0.1],
        'fit_intercept': [True, False]
    }
    return X_train, y_train, params

@pytest.fixture
def mock_data_score_func():
    " Mock data for testing score_func"
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    estimator = LinearRegression().fit(X,y)  # Mock estimator, could be any sklearn estimator
    return estimator, X, y

@pytest.fixture
def mock_data_evaluate():
    " Mock data for testing evaluate"
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    estimator = LinearRegression()
    cv, quantile = TimeSeriesSplit(n_splits=2), 0.1
    return estimator, X, y, cv, quantile

@pytest.fixture
def mock_data_optimization():
    " Mock data for testing optimization functions"
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    quantile = 0.1
    nr_cv_splits = 3
    gbr_config_params = {
        'learning_rate': [0.01, 0.1],
        'max_features': [0.5, 1.0],
        'max_depth': [3, 5],
        'max_iter': [100, 200]
    }
    lr_config_params = {
        'alpha': [0.01, 0.1],
        'fit_intercept': [True, False]
    }
    
    solver = "highs" 
    return X_train, y_train, quantile, nr_cv_splits, solver, gbr_config_params, lr_config_params

@pytest.fixture
def best_params_gbr():
    " Mock best parameters for Gradient Boosting Regressor"
    return {
        'learning_rate': 0.1,
        'max_features': 1.0,
        'max_depth': 5,
        'max_iter': 100
    }

@pytest.fixture
def best_params_lr():
    " Mock best parameters for Linear Regression"
    return {
        'fit_intercept': True,
        'alpha': 0.1
    }

@pytest.fixture
def mock_data_2stage_dataframe():
    " Mock data for testing create_2stage_dataframe"
    # Create sample df_train_ensemble
    date_rng_train = pd.date_range(start='1/1/2020', end='1/10/2020', freq='D')
    df_train_ensemble = pd.DataFrame(date_rng_train, columns=['date'])
    df_train_ensemble['data'] = range(len(date_rng_train))
    df_train_ensemble.set_index('date', inplace=True)
    # Create sample df_test_ensemble
    date_rng_test = pd.date_range(start='1/11/2020', end='1/20/2020', freq='D')
    df_test_ensemble = pd.DataFrame(date_rng_test, columns=['date'])
    df_test_ensemble['data'] = range(len(date_rng_test))
    df_test_ensemble.set_index('date', inplace=True)
    y_train = np.random.rand(10)
    y_test = np.random.rand(10)
    predictions_insample = np.random.rand(10)
    predictions_outsample = np.random.rand(10)
    return df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample

@pytest.fixture
def mock_data_var_ensemble_dataframe():
    " Mock data for testing create_var_ensemble_dataframe "
    quantiles = [0.1, 0.5, 0.9]
    date_rng = pd.date_range(start='1/1/2020', end='1/10/2020', freq='D')
    df_test = pd.DataFrame(date_rng, columns=['datetime'])
    df_test['targets'] = np.random.rand(len(date_rng))
    df_test.set_index('datetime', inplace=True)
    quantile_predictions_dict = {
        0.1: [(date, np.random.rand()) for date in date_rng],
        0.5: [(date, np.random.rand()) for date in date_rng],
        0.9: [(date, np.random.rand()) for date in date_rng],
    }
    return quantiles, quantile_predictions_dict, df_test

@pytest.fixture
def data_first_stage_importance():
    "Create data for testing first stage importance."
    num_permutations = 5
    quantile = 0.1
    num_samples = 100
    num_features = 5
    np.random.seed(42)  # For reproducibility
    X_test_augmented = np.random.rand(num_samples, num_features)  
    y_test = np.random.rand(num_samples)
    # Creating a dummy DataFrame to match the structure
    df_train_ensemble_augmented = pd.DataFrame(X_test_augmented, columns=[f'feature_{i}' for i in range(num_features)])
    df_train_ensemble_augmented['diff_norm_targ'] = y_test
    # Fitting a simple model for testing
    model = LinearRegression()
    fitted_model = model.fit(X_test_augmented, y_test)
    score_functions = {quantile: score_func_10}
    base_score = score_functions[quantile](fitted_model, X_test_augmented, y_test)['mean_pinball_loss']
    return num_permutations, X_test_augmented, y_test, df_train_ensemble_augmented, fitted_model, score_functions, base_score, quantile

@pytest.fixture
def data_second_stage_importance():
    "Mock data for testing second_stage_permutation_importance."
    num_permutations = 5
    quantile = 0.1
    num_samples_train = 50
    num_samples_test = 25
    num_features = 5
    np.random.seed(42)
    # Sample Data for testing
    X_train_augmented = np.random.rand(num_samples_train, num_features)  # 100 samples, 5 features for training
    y_train = np.random.rand(num_samples_train)
    X_test_augmented = np.random.rand(num_samples_test, num_features)
    y_test = np.random.rand(num_samples_test)
    df_train_ensemble = pd.DataFrame(X_train_augmented, columns=[f'feature_{i}' for i in range(num_features)])
    df_train_ensemble['diff_norm_targ'] = y_train
    df_train_ensemble_augmented = df_train_ensemble.copy()
    df_test_ensemble = pd.DataFrame(X_test_augmented, columns=[f'feature_{i}' for i in range(num_features)])
    df_test_ensemble['diff_norm_targ'] = y_test
    predictions_insample = np.random.rand(num_samples_train)
    permuted_predictions_outsample = np.random.rand(num_samples_test)
    start_prediction_timestamp = df_test_ensemble.index[0]
    order_diff = 1
    max_lags_var = 4
    augment_var = False
    # Fitting a simple model for testing wind power predictions
    model = LinearRegression()
    fitted_model = model.fit(X_train_augmented, y_train)
    # Fitting a simple model for testing wind power variability predictions
    model = LinearRegression()
    var_fitted_model = model.fit(X_train_augmented, y_train)
    return {
        "num_permutations": num_permutations,
        "quantile": quantile,
        "fitted_model": fitted_model,
        "var_fitted_model": var_fitted_model,
        "X_train_augmented": X_train_augmented,
        "y_train": y_train,
        "X_test_augmented": X_test_augmented,
        "y_test": y_test,
        "df_train_ensemble_augmented":df_train_ensemble_augmented,
        "df_train_ensemble": df_train_ensemble,
        "df_test_ensemble": df_test_ensemble,
        "predictions_insample": predictions_insample,
        "permuted_predictions_outsample": permuted_predictions_outsample,
        "start_prediction_timestamp": start_prediction_timestamp,
        "order_diff": order_diff,
        "max_lags_var": max_lags_var,
        "augment_var": augment_var
    }

