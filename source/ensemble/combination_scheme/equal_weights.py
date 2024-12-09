import pandas as pd

def calculate_equal_weights(df_test_norm, start_prediction_timestamp):
    """Calculate the mean prediction and quantiles using equal weights
    Args:
        df_test_norm: pd.DataFrame, test data
        start_prediction_timestamp: pd.Timestamp, start prediction timestamp
    Returns:
        df_equal_weights: pd.DataFrame, equal weights forecast"""
    assert 'norm_measured' in df_test_norm.columns, "norm_measured column is missing"
    df_test_norm = df_test_norm[df_test_norm.index >= start_prediction_timestamp]
    assert len(df_test_norm) == 96, "Dataframe must have 96 rows"
    list_q10 = [name for name in list(df_test_norm.columns) if 'confidence10' in name]  # list of columns with q10
    list_q50 = [name for name in list(df_test_norm.columns) if 'forecast' in name]  # list of columns with q50
    list_q90 = [name for name in list(df_test_norm.columns) if 'confidence90' in name]  # list of columns with q90
    q10_equal_weights = df_test_norm[list_q10].mean(axis=1)  # mean of q10
    q50_equal_weights = df_test_norm[list_q50].mean(axis=1)  # mean of q50
    q90_equal_weights = df_test_norm[list_q90].mean(axis=1)  # mean of q90
    # Create a DataFrame with the mean prediction and quantiles
    df_equal_weights = pd.DataFrame({
        'q10_equal_weights': q10_equal_weights,
        'q50_equal_weights': q50_equal_weights,
        'q90_equal_weights': q90_equal_weights
    }, index=df_test_norm.index)
    # Add the target column
    df_equal_weights['targets'] = df_test_norm['norm_measured']
    return df_equal_weights