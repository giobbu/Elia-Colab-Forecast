import numpy as np

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