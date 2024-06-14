import pandas as pd

def prepare_train_test_data(df_ensemble, df_val, df_test, start_predictions, max_lag):
    "Prepare train and test data for ensemble model"
    assert isinstance(df_ensemble, pd.DataFrame), "df_ensemble should be a DataFrame"
    assert isinstance(df_val, pd.DataFrame), "df_val should be a DataFrame"
    assert isinstance(df_test, pd.DataFrame), "df_test should be a DataFrame"
    assert isinstance(start_predictions, pd.Timestamp), "start_predictions should be a Timestamp"
    assert 'diff_norm_measured' in df_val.columns, "diff_norm_measured should be in df_val columns"
    assert 'diff_norm_measured' in df_test.columns, "diff_norm_measured should be in df_test columns"
    assert isinstance(max_lag, int), "max_lag should be an integer"
    assert max_lag > 0, "max_lag should be greater than 0"
    df_train_ensemble = df_ensemble[df_ensemble.index < start_predictions].copy()
    df_train_ensemble.loc[:, 'diff_norm_targ'] = df_val['diff_norm_measured'].values[max_lag:]
    df_test_ensemble = df_ensemble[df_ensemble.index >= start_predictions].copy()
    df_test_ensemble.loc[:, 'diff_norm_targ'] = df_test['diff_norm_measured'].values
    return df_train_ensemble, df_test_ensemble

def get_numpy_Xy_train_test(df_train_ensemble, df_test_ensemble):
    "Get numpy arrays for X_train, y_train, X_test, y_test"
    assert isinstance(df_train_ensemble, pd.DataFrame), "df_train_ensemble should be a DataFrame"
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    X_train, y_train = df_train_ensemble.iloc[:, :-1].values, df_train_ensemble.iloc[:, -1].values
    X_test, y_test = df_test_ensemble.iloc[:, :-1].values, df_test_ensemble.iloc[:, -1].values
    return X_train, y_train, X_test, y_test
