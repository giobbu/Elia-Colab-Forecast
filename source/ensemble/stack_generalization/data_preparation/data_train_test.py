import pandas as pd
import numpy as np

def create_pre_test_dataframe(df_buyer, df_ensemble, pre_start_prediction, buyer_name):
    """ Create test dataframes for buyer and ensemble predictions 
    args:
        df_buyer: pd.DataFrame, buyer predictions
        df_ensemble: pd.DataFrame, ensemble predictions
        pre_start_prediction: pd.Timestamp, start prediction timestamp
        buyer_name: str, buyer name
    returns:
        df_test_ensemble_pre: pd.DataFrame, test ensemble predictions"""
    # Ensure the DataFrame indices are datetime types
    if not pd.api.types.is_datetime64_any_dtype(df_buyer.index):
        raise TypeError("The df_buyer_norm index must be a datetime type.")
    if not pd.api.types.is_datetime64_any_dtype(df_ensemble.index):
        raise TypeError("The df_ensemble index must be a datetime type.")
    # Filter the DataFrames based on the start prediction timestamp
    df_test_targ_pre = df_buyer[df_buyer.index >= pre_start_prediction]
    df_test_ensemble_pre = df_ensemble[df_ensemble.index >= pre_start_prediction]
    # Assign the normalized target column to the ensemble DataFrame
    df_test_ensemble_pre.loc[:, 'norm_targ'] = df_test_targ_pre['norm_' + buyer_name].values
    return df_test_ensemble_pre

def prepare_pre_test_data(params, quantile, df_test_ensemble, df_test_ensemble_q10=pd.DataFrame([]), df_test_ensemble_q90=pd.DataFrame([])):
    """Prepare test set for 2-stage model.
    Args:
        params: dict, ensemble parameters
        quantile: float, quantile value
        df_test_ensemble: pd.DataFrame, test data
        df_test_ensemble_q10: pd.DataFrame, quantile 10% test data
        df_test_ensemble_q90: pd.DataFrame, quantile 90% test data
    Returns:
        X_test: np.array, test features
        y_test: np.array, test target
    """
    # Assertions for input validation
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    assert "norm_targ" in df_test_ensemble.columns, "'norm_targ' should be in df_test_ensemble columns"
    target_column = "norm_targ"
    # Get the test data (features and target)
    X_test = df_test_ensemble.drop(columns=[target_column]).values
    y_test = df_test_ensemble[target_column].values
    # If quantile is 0.5 and no need to augment, return the original test data
    if quantile == 0.5 and not params.get('augment_q50', False):
            return X_test, y_test
    # If quantile predictions need to be added
    if params['add_quantile_predictions']:
        # Extract quantile data, default to empty arrays if unavailable
        X_test_q10 = df_test_ensemble_q10.values if not df_test_ensemble_q10.empty else np.array([])
        X_test_q90 = df_test_ensemble_q90.values if not df_test_ensemble_q90.empty else np.array([])
        # Prepare quantile data dictionary
        quantile_data = {
            0.1: X_test_q10,
            0.9: X_test_q90,
        }
        # Add the median quantile data if available
        if not (df_test_ensemble_q10.empty or df_test_ensemble_q90.empty):
            quantile_data[0.5] = np.concatenate([X_test_q10, X_test_q90], axis=1)
        elif not df_test_ensemble_q10.empty:
            quantile_data[0.5] = X_test_q10
        elif not df_test_ensemble_q90.empty:
            quantile_data[0.5] = X_test_q90
        else:
            quantile_data[0.5] = np.array([])
        # Validate the requested quantile
        if quantile not in quantile_data:
            raise ValueError("Invalid quantile value. Must be 0.1, 0.5, or 0.9.")
        # Concatenate the selected quantile data if it's not empty
        if quantile_data[quantile].size != 0:
            X_test = np.concatenate([X_test, quantile_data[quantile]], axis=1)
    return X_test, y_test

def split_train_test_data(df, end_train, start_prediction): 
    """ Split the data into training and test sets 
    args:
        df: pd.DataFrame, dataframe
        end_train: pd.Timestamp, end training timestamp
        start_prediction: pd.Timestamp, start prediction timestamp"""
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"
    assert isinstance(end_train, pd.Timestamp), "end_training should be a Timestamp"
    assert isinstance(start_prediction, pd.Timestamp), "start_predictions should be a Timestamp"
    df_train = df[df.index < end_train]
    df_test = df[df.index >= start_prediction]
    return df_train, df_test

def concatenate_feat_targ_dataframes(buyer_resource_name, df_train_ensemble, df_test_ensemble, df_train, df_test,  max_lag):
    """ Prepare train and test data for ensemble model
    args:
        buyer_resource_name: str, buyer resource name
        df_train_ensemble: pd.DataFrame, ensemble training data
        df_test_ensemble: pd.DataFrame, ensemble testing data
        df_train: pd.DataFrame, training data
        df_test: pd.DataFrame, testing data
        max_lag: int, maximum lag value
    returns:
        df_train_ensemble: pd.DataFrame, ensemble training data
        df_test_ensemble: pd.DataFrame, ensemble testing data"""
    assert isinstance(df_train_ensemble, pd.DataFrame), "df_train_ensemble should be a DataFrame"
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    assert isinstance(df_train, pd.DataFrame), "df_train should be a DataFrame"
    assert isinstance(df_test, pd.DataFrame), "df_test should be a DataFrame"
    assert 'norm_' + buyer_resource_name in df_train.columns, "norm_measured should be in df_train columns"
    assert 'norm_' + buyer_resource_name in df_test.columns, "norm_measured should be in df_test columns"
    assert isinstance(max_lag, int), "max_lag should be an integer"
    col_name_buyer = 'norm_' + buyer_resource_name
    df_train_ensemble.loc[:, 'norm_targ'] = df_train[col_name_buyer].values[max_lag:]
    df_test_ensemble.loc[:, 'norm_targ'] = df_test[col_name_buyer].values
    return df_train_ensemble, df_test_ensemble

def get_numpy_Xy_train_test(df_train_ensemble, df_test_ensemble):
    """Get numpy arrays for X_train, y_train, X_test, y_test
    Args:
        df_train_ensemble: pd.DataFrame, ensemble training data
        df_test_ensemble: pd.DataFrame, ensemble testing data
    Returns:
        X_train: np.array, training features
        y_train: np.array, training target
        X_test: np.array, testing features
        y_test: np.array, testing target
    """
    assert isinstance(df_train_ensemble, pd.DataFrame), "df_train_ensemble should be a DataFrame"
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    X_train, y_train = df_train_ensemble.iloc[:, :-1].values, df_train_ensemble.iloc[:, -1].values
    X_test, y_test = df_test_ensemble.iloc[:, :-1].values, df_test_ensemble.iloc[:, -1].values
    return X_train, y_train, X_test, y_test
