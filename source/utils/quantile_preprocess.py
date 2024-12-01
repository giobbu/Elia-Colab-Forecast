import pandas as pd
import numpy as np

def extract_quantile_columns(df, quantile):
    """Extract columns containing the specified quantile."""
    columns = [name for name in df.columns if quantile in name]
    if columns:
        return df[columns]
    else:
        print(f"No columns found for {quantile}")
        return pd.DataFrame()

def split_quantile_train_test_data(df, end_training_timestamp, start_prediction_timestamp):
    """Split the quantile data into training and test sets."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_train = df[df.index < end_training_timestamp]
    df_test = df[df.index >= start_prediction_timestamp]
    return df_train, df_test

def get_numpy_Xy_train_test_quantile(ens_params, df_train_ensemble_quantile10, df_test_ensemble_quantile10, df_train_ensemble_quantile90, df_test_ensemble_quantile90):
    "Make X-y train and test sets for quantile"
    if ens_params['add_quantile_predictions']:
        X_train_quantile10 = df_train_ensemble_quantile10.values if not df_train_ensemble_quantile10.empty else np.array([])
        X_test_quantile10 = df_test_ensemble_quantile10.values if not df_test_ensemble_quantile10.empty else np.array([])
        X_train_quantile90 = df_train_ensemble_quantile90.values if not df_train_ensemble_quantile90.empty else np.array([])
        X_test_quantile90 = df_test_ensemble_quantile90.values if not df_test_ensemble_quantile90.empty else np.array([])
    else:
        X_train_quantile10, X_test_quantile10 = np.array([]), np.array([])
        X_train_quantile90, X_test_quantile90 = np.array([]), np.array([])
    return X_train_quantile10, X_test_quantile10, X_train_quantile90, X_test_quantile90
