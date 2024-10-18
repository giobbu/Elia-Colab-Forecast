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
    list_q10 = [name for name in list(df_test_norm.columns) if 'confidence10' in name]
    list_q50 = [name for name in list(df_test_norm.columns) if 'forecast' in name]
    list_q90 = [name for name in list(df_test_norm.columns) if 'confidence90' in name]
    Q10 = df_test_norm[list_q10].mean(axis=1)
    MEAN = df_test_norm[list_q50].mean(axis=1)
    Q90 = df_test_norm[list_q90].mean(axis=1)
    df_equal_weights = pd.DataFrame({
        'Q10': Q10,
        'mean_prediction': MEAN,
        'Q90': Q90
    }, index=df_test_norm.index)
    df_equal_weights['targets'] = df_test_norm['norm_measured']
    return df_equal_weights