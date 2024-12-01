import pandas as pd
import numpy as np

def create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample):
    " Create 2-stage ensemble dataframe."
    assert df_train_ensemble.shape[0] == len(predictions_insample), "Length mismatch between train data and in-sample predictions"
    assert df_test_ensemble.shape[0] == len(predictions_outsample), "Length mismatch between test data and out-sample predictions"
    assert len(y_train) == len(predictions_insample), "Length mismatch between targets and in-sample predictions"
    assert len(y_test) == len(predictions_outsample), "Length mismatch between targets and out-sample predictions"
    # Creating DataFrame for in-sample predictions
    df_insample = pd.DataFrame(predictions_insample, columns=['predictions'], index=df_train_ensemble.index)
    # # concatenate df_train_ensemble and df_insample
    # df_insample = pd.concat([df_train_ensemble, df_insample], axis=1)
    # # drop 'norm_targ' column
    # df_insample.drop(columns=['norm_targ'], inplace=True)
    df_insample['targets'] = y_train
    # Creating DataFrame for out-sample predictions
    df_outsample = pd.DataFrame(predictions_outsample, columns=['predictions'], index=df_test_ensemble.index)
    # # concatenate df_test_ensemble and df_outsample
    # df_outsample = pd.concat([df_test_ensemble, df_outsample], axis=1)
    # # drop 'norm_targ' column
    # df_outsample.drop(columns=['norm_targ'], inplace=True)
    df_outsample['targets'] = y_test
    # Concatenating in-sample and out-sample DataFrames
    df_2stage = pd.concat([df_insample, df_outsample], axis=0)
    return df_2stage

def create_augmented_dataframe_2stage(df, order_diff, max_lags, differentiate = False, add_lags=False, augment_with_poly=False, end_train=None, start_prediction=None):
    " Process 2-stage ensemble dataframe with lags."
    assert order_diff > 0, "Order of differentiation must be greater than 0"
    if add_lags:
        assert max_lags > 0, "max_lags should be greater than 0"
    else:
        assert max_lags == 0, "max_lags should be 0 when lagged is False"
    # Differentiate the targets
    for col in df.columns:
        if 'targets' in col:
            df[col] = df[col].diff(order_diff)
    # Differentiate the dataframe
    if differentiate:
        for col in df.columns:
            if 'targets' not in col:
                df[f'{col}_diff'] = df[col].diff(order_diff)
    # Create lagged features
    if add_lags:
        for col in df.columns:
            for lag in range(1, max_lags + 1):
                if 'targets' not in col:
                    df[col+'_t-'+str(lag)] = df[col].shift(lag)
    if augment_with_poly:
        for col in df.columns:
            if 'targets' not in col:
                df[f'{col}_sqr'] = df[col]**2
                df[f'{col}_cub'] = df[col]**3
    # Drop rows with NaNs resulting from the shift operation
    if add_lags:
        cut_ = max_lags + order_diff
    else:
        cut_ = order_diff
    df_ = df.iloc[cut_:, :]
    if add_lags:
        df_train = df_[df_.index < end_train]
        df_test = df_[df_.index >= start_prediction]
        for lag in range(1, max_lags + 1):
            lag_colunms = df_train.columns.str.contains('_t-'+str(lag))
            # set the first lag rows to nans
            df_train.loc[:df_train.index[lag - 1],df_train.columns[lag_colunms]] = np.nan
            df_test.loc[:df_test.index[lag - 1], df_test.columns[lag_colunms]] = np.nan
            # backfill the nans
            df_train = df_train.bfill()
            df_test = df_test.bfill()
            # concatenate the training and testing data
            df_ = pd.concat([df_train, df_test])
    return df_

def create_var_ensemble_dataframe(buyer_resource_name, quantiles, quantile_predictions_dict, df_test):
    " Create ensemble dataframe from quantile predictions."
    assert len(quantiles) == len(quantile_predictions_dict), "Length mismatch between quantiles and quantile predictions"
    assert df_test.shape[0] == len(quantile_predictions_dict[quantiles[0]]), "Length mismatch between test data and predictions"
    for i, quantile in enumerate(quantiles):
        if i == 0:
            df_pred_ensemble = pd.DataFrame(quantile_predictions_dict[quantile])
            df_pred_ensemble.columns = ['datetime', 'q' + str(int(quantile*100)) + '_' + buyer_resource_name]
            df_pred_ensemble.set_index('datetime', inplace=True)
        else:
            df_pred_quantile = pd.DataFrame(quantile_predictions_dict[quantile])
            df_pred_quantile.columns = ['datetime', 'q' + str(int(quantile*100)) + '_' + buyer_resource_name]
            df_pred_quantile.set_index('datetime', inplace=True)
            df_pred_ensemble = pd.concat([df_pred_ensemble, df_pred_quantile], axis=1)
    return df_pred_ensemble


def get_numpy_Xy_train_test_2stage(df_2stage_train, df_2stage_test):
    """
    Prepares training and testing data for a two-stage model by separating features and targets.
    """
    X_train_2stage = df_2stage_train.drop(columns=['targets']).values
    y_train_2stage = df_2stage_train['targets'].values
    X_test_2stage = df_2stage_test.drop(columns=['targets']).values
    return X_train_2stage, y_train_2stage, X_test_2stage