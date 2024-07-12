import pandas as pd

def extract_quantile_columns(df, quantile):
    """Extract columns containing the specified quantile."""
    columns = [name for name in df.columns if quantile in name]
    if columns:
        return df[columns]
    else:
        print(f"No columns found for {quantile}")
        return pd.DataFrame()

def split_quantile_data(df, end_training_timestamp, start_prediction_timestamp, pre_start_prediction_timestamp):
    """Split the quantile data into training and test sets."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_train = df[df.index < end_training_timestamp]
    df_test = df[df.index >= start_prediction_timestamp]
    df_test_pre = df[df.index >= pre_start_prediction_timestamp]
    return df_train, df_test, df_test_pre