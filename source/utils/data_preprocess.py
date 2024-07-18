import numpy as np
import pandas as pd

def scale(df, col_name, max_cap):
    " Scale a column by dividing by maximum capacity"
    assert max_cap > 0, "Maximum capacity must be greater than 0"
    df_ = df.copy()
    values = df_[col_name].values
    return values/max_cap

def detect_ramp_event(df, ramp_threshold):
    " Detect ramp event by comparing the absolute difference between consecutive values with a threshold"
    assert ramp_threshold > 0, "Ramp threshold must be greater than 0"
    df['ramp_event'] = (np.abs(df['diff_norm_measured']) >= ramp_threshold).astype(int)
    df['ramp_event_up'] = (df['diff_norm_measured'] >= ramp_threshold).astype(int)
    df['ramp_event_down'] = (df['diff_norm_measured'] <= -ramp_threshold).astype(int)
    return df

def rescale_normalized_predictions(predictions, quantile, maximum_capacity):
    " Rescale normalized predictions"
    assert maximum_capacity > 0, "Maximum capacity must be greater than 0"
    assert quantile in predictions.keys(), "Quantile must be in the predictions keys"
    assert isinstance(predictions[quantile], np.ndarray), "Predictions must be a numpy array"
    return predictions[quantile] * maximum_capacity

def rescale_normalized_targets(df, target_name, maximum_capacity):
    " Rescale normalized targets"
    assert target_name in df.columns, "Target name must be in the DataFrame columns"
    return df[target_name] * maximum_capacity

def get_maximum_values(df, end_train, buyer_resource_name=None):
    " Get the maximum values for the buyer resource and forecasters"
    assert isinstance(df, pd.DataFrame), 'df must be a DataFrame'
    assert buyer_resource_name is None or isinstance(buyer_resource_name, str), 'buyer_resource_name must be a string or None'
    # Check if the DataFrame indices are datetime types
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("The df index must be a datetime type.")
    if buyer_resource_name is not None:
        # get the maximum capacity for the buyer resource
        maximum_capacity_buyer = df[df.index < end_train][buyer_resource_name].max()
        return maximum_capacity_buyer
    else:
        # Get the maximum values for forecasters
        list_maximum_values_forecasters = df[df.index < end_train].max(axis=0).values
        return list_maximum_values_forecasters


def normalize_dataframe(df, axis=1, max_cap=None, max_cap_forecasters_list=None):
    " Normalize dataframe by dividing by maximum capacity"
    df_normalized = df.copy()
    if axis==1:
        assert max_cap is not None, "Maximum capacity must be provided"
        for col in df_normalized.columns:
            normalize_col = scale(df_normalized, col, max_cap)
            df_normalized[f'norm_{col}'] = normalize_col
        return df_normalized.filter(like='norm')
    elif axis==0:
        assert max_cap_forecasters_list is not None, "List of maximum capacities must be provided"
        for i, col in enumerate(df_normalized.columns):
            normalize_col = scale(df_normalized, col, max_cap_forecasters_list[i])
            df_normalized[f'norm_{col}'] = normalize_col
        return df_normalized.filter(like='norm')
    else:
        raise ValueError("Axis must be either 0 or 1")

def differentiate_dataframe(df):
    " Differentiate dataframe by computing the absolute difference between consecutive values"
    assert len(df) > 1, "Input DataFrame must have more than one row"
    df_differential = df.copy()
    for col in df_differential.columns:
        df_differential[f'diff_{col}'] = df_differential[col].diff()
    return df_differential.filter(like='diff').iloc[1:]