import pandas as pd

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

def create_augmented_dataframe_2stage(df_2stage, order_diff, max_lags, augment=False):
    " Process 2-stage ensemble dataframe with lags."
    assert order_diff > 0, "Order of differentiation must be greater than 0"
    assert max_lags > 0, "Maximum number of lags must be greater than 0"
    # Differentiate the dataframe
    df_2stage_diff = df_2stage.diff(order_diff)
    # Create lagged features
    for lag in range(1, max_lags + 1):
        df_2stage_diff[f'predictions_t-{lag}'] = df_2stage_diff['predictions'].shift(lag)
    if augment:
        for col in df_2stage_diff.columns:
            if 'targets' not in col:
                df_2stage_diff[f'{col}_sqr'] = df_2stage_diff[col]**2
    # Drop rows with NaNs resulting from the shift operation
    df_2stage_process = df_2stage_diff.iloc[max_lags+order_diff:, :]
    return df_2stage_process

# def create_augmented_dataframe_2stage(df_2stage, order_diff, max_lags, augment=False):
#     " Process 2-stage ensemble dataframe with lags."
#     assert order_diff > 0, "Order of differentiation must be greater than 0"
#     assert max_lags > 0, "Maximum number of lags must be greater than 0"
#     # Differentiate the dataframe
#     df_2stage_diff = df_2stage.diff(order_diff)
#     # Create lagged features
#     for lag in range(1, max_lags + 1):
#         df_2stage_diff[f'predictions_t-{lag}'] = df_2stage_diff['predictions'].shift(lag)
#         df_2stage_diff[f'predictions_t-{lag}_sqr'] = df_2stage_diff['predictions'].shift(lag)**2
#         df_2stage_diff[f'predictions_t-{lag}_cubic'] = df_2stage_diff['predictions'].shift(lag)**3
#     if augment:
#         for col in df_2stage_diff.columns:
#             if 'targets' not in col:
#                 df_2stage_diff[f'{col}_sqr'] = df_2stage_diff[col]**2
#                 df_2stage_diff[f'{col}_cubic'] = df_2stage_diff[col]**3
#     # Drop rows with NaNs resulting from the shift operation
#     df_2stage_process = df_2stage_diff.iloc[max_lags+order_diff:, :]
#     return df_2stage_process

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