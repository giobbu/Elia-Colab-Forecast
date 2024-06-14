import pandas as pd
import numpy as np

def create_augmented_dataframe(df, max_lags, forecasters_diversity=False, lagged=False, augmented=False, differenciate=False):
    " Create feature engineering dataframe with forecasters diversity, lagged, augmented and differenciate features"
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"
    assert isinstance(max_lags, int), "max_lags should be an integer"
    assert max_lags > 0, "max_lags should be greater than 0"
    shifted_df_ensemble = pd.DataFrame()
    if lagged:
        " Create lagged features"
        for lag in range(1, max_lags + 1):
            for col in df.columns:
                shifted_df_ensemble[col+'_t-'+str(lag)] = df[col].shift(lag)  # lagged
    if augmented:
        " Create augmented features"
        for col in df.columns:
            shifted_df_ensemble[col + "_sqr"] = df[col]**2  # squared
            shifted_df_ensemble[col + "_std"] = df[col].rolling(max_lags).std()  # rolling standard deviation
            shifted_df_ensemble[col + "_var"] = df[col].rolling(max_lags).var()  # rolling variance
            if max_lags > 2:
                shifted_df_ensemble[col + "_lag-1_std"] = df[col].shift(1).rolling(max_lags-1).std()  # rolling standard deviation on lag-1
                shifted_df_ensemble[col + "_lag-1_var"] = df[col].shift(1).rolling(max_lags-1).var()  # rolling variance on lag-1
    if differenciate:
        " Create differenciate features"
        for col in df.columns:
            shifted_df_ensemble[col + "_diff"] = df[col].diff()  # difference
            shifted_df_ensemble[col + "_lag-1_diff"] = df[col].shift(1).diff()  # difference on lag-1
    if forecasters_diversity:
        " Create forecasters diversity features"
        forecast_cols = [name for name in df.columns if 'pred' in name]  # forecasters columns
        shifted_df_ensemble["forecasters_std"] = df[forecast_cols].std(axis=1)  # standard deviation among forecasters
        shifted_df_ensemble["forecasters_var"] = df[forecast_cols].var(axis=1)  # variance among forecasters
    " Concatenate the original dataframe with the shifted dataframe"
    df = pd.concat([df, shifted_df_ensemble], axis=1)
    df = df.iloc[max_lags:,:]
    return df


def augment_with_quantiles(X_train, X_test, df_train_ensemble, 
                            X_train_quantile10, X_test_quantile10, df_train_ensemble_quantile10,
                            X_train_quantile90, X_test_quantile90, df_train_ensemble_quantile90,  
                            quantile, augment_q50=False):
    " Augment the training and testing data with the quantiles predictions from forecasters"
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(df_train_ensemble, pd.DataFrame), "df_train_ensemble should be a DataFrame"
    assert isinstance(X_train_quantile10, np.ndarray), "X_train_quantile10 should be a numpy array"
    assert isinstance(X_test_quantile10, np.ndarray), "X_test_quantile10 should be a numpy array"
    assert isinstance(df_train_ensemble_quantile10, pd.DataFrame), "df_train_ensemble_quantile10 should be a DataFrame"
    assert isinstance(X_train_quantile90, np.ndarray), "X_train_quantile90 should be a numpy array"
    assert isinstance(X_test_quantile90, np.ndarray), "X_test_quantile90 should be a numpy array"
    assert isinstance(df_train_ensemble_quantile90, pd.DataFrame), "df_train_ensemble_quantile90 should be a DataFrame"
    assert quantile in [0.1, 0.5, 0.9], "Invalid quantile value. Must be 0.1, 0.5, or 0.9."
    # Create a dictionary with the quantile data
    quantile_data = {
        0.1: (X_train_quantile10, X_test_quantile10, df_train_ensemble_quantile10),
        0.5: (np.concatenate([X_train_quantile10, X_train_quantile90], axis=1),
                np.concatenate([X_test_quantile10, X_test_quantile90], axis=1),
                pd.concat([df_train_ensemble_quantile10, df_train_ensemble_quantile90], axis=1)),
        0.9: (X_train_quantile90, X_test_quantile90, df_train_ensemble_quantile90)
    }
    if quantile not in quantile_data:
        raise ValueError('Invalid quantile value. Must be 0.1, 0.5, or 0.9.')
    " Get the quantile data and augment the training and testing data with it"
    X_train_part, X_test_part, df_train_ensemble_part = quantile_data[quantile]
    if quantile == 0.5 and not augment_q50:
        " Do not augment with augmented quantiles"
        return X_train, X_test, df_train_ensemble
    X_train = np.concatenate([X_train, X_train_part], axis=1)
    X_test = np.concatenate([X_test, X_test_part], axis=1)
    df_train_ensemble = pd.concat([df_train_ensemble, df_train_ensemble_part], axis=1)
    return X_train, X_test, df_train_ensemble