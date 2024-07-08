import pandas as pd
import numpy as np
 
def prepare_pre_test_data(params, quantile, df_test_ensemble, df_test_ensemble_q10=pd.DataFrame([]), df_test_ensemble_q90=pd.DataFrame([])):
    " Prepare test set for 2-stage model"
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    assert len(df_test_ensemble) == 192, "df_test_ensemble should have 192 rows"
    assert len(df_test_ensemble_q10) == 192 or df_test_ensemble_q10.empty, "df_test_ensemble_q10 should have 192 rows or be empty"
    assert len(df_test_ensemble_q90) == 192 or df_test_ensemble_q90.empty, "df_test_ensemble_q90 should have 192 rows or be empty"
    assert "diff_norm_targ" in df_test_ensemble.columns, "diff_norm_targ should be in df_test_ensemble columns"
    target_column = "diff_norm_targ"
    X_test = df_test_ensemble.drop(columns=[target_column]).values
    y_test = df_test_ensemble[target_column].values
    if params['add_quantile_predictions']:
        X_test_q10 = df_test_ensemble_q10.values if not df_test_ensemble_q10.empty else np.array([])
        X_test_q90 = df_test_ensemble_q90.values if not df_test_ensemble_q90.empty else np.array([])
        quantile_data = {
            0.1: X_test_q10 if not df_test_ensemble_q10.empty else np.array([]),
            0.5: np.concatenate([X_test_q10, X_test_q90], axis=1) if not (df_test_ensemble_q10.empty or df_test_ensemble_q90.empty) else (X_test_q10 if not df_test_ensemble_q10.empty else X_test_q90),
            0.9: X_test_q90 if not df_test_ensemble_q90.empty else np.array([])
        }
        if quantile not in quantile_data:
            raise ValueError('Invalid quantile value. Must be 0.1, 0.5, or 0.9.')
        X_test_part = quantile_data[quantile]
        if quantile == 0.5 and not params['augment_q50']:
            return X_test, y_test
        X_test = np.concatenate([X_test, X_test_part], axis=1) if X_test_part.size else X_test
    return X_test, y_test



def prepare_train_test_data(buyer_resource_name, df_ensemble, df_val, df_test, end_training, start_predictions, max_lag):
    "Prepare train and test data for ensemble model"
    assert isinstance(df_ensemble, pd.DataFrame), "df_ensemble should be a DataFrame"
    assert isinstance(df_val, pd.DataFrame), "df_val should be a DataFrame"
    assert isinstance(df_test, pd.DataFrame), "df_test should be a DataFrame"
    #assert isinstance(start_predictions, pd.Timestamp), "start_predictions should be a Timestamp"
    assert 'diff_norm_' + buyer_resource_name in df_val.columns, "diff_norm_measured should be in df_val columns"
    assert 'diff_norm_' + buyer_resource_name in df_test.columns, "diff_norm_measured should be in df_test columns"
    assert isinstance(max_lag, int), "max_lag should be an integer"
    assert max_lag > 0, "max_lag should be greater than 0"
    col_name_buyer = 'diff_norm_' + buyer_resource_name
    df_train_ensemble = df_ensemble[df_ensemble.index < end_training].copy()
    df_train_ensemble.loc[:, 'diff_norm_targ'] = df_val[col_name_buyer].values[max_lag:]
    df_test_ensemble = df_ensemble[df_ensemble.index >= start_predictions].copy()
    df_test_ensemble.loc[:, 'diff_norm_targ'] = df_test[col_name_buyer].values
    return df_train_ensemble, df_test_ensemble

def get_numpy_Xy_train_test(df_train_ensemble, df_test_ensemble):
    "Get numpy arrays for X_train, y_train, X_test, y_test"
    assert isinstance(df_train_ensemble, pd.DataFrame), "df_train_ensemble should be a DataFrame"
    assert isinstance(df_test_ensemble, pd.DataFrame), "df_test_ensemble should be a DataFrame"
    X_train, y_train = df_train_ensemble.iloc[:, :-1].values, df_train_ensemble.iloc[:, -1].values
    X_test, y_test = df_test_ensemble.iloc[:, :-1].values, df_test_ensemble.iloc[:, -1].values
    return X_train, y_train, X_test, y_test
