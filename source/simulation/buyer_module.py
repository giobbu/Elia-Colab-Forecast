import pandas as pd

def prepare_buyer_data(df_train, df_test, start_prediction_timestamp, end_prediction_timestamp):
    """ Prepare data for the buyer module
    Args:
        df_train: DataFrame with training data
        df_test: DataFrame with test data
        start_prediction_timestamp: Start timestamp for the prediction
        end_prediction_timestamp: End timestamp for the prediction
    Returns:
        df_buyer: DataFrame with the buyer data
        forecast_range: Forecast range for the prediction"""
    
    # Create DataFrame for train and test data
    df_train_buyer = pd.DataFrame(df_train['measured'])
    df_test_buyer = pd.DataFrame(df_test['measured'])
    # Create a forecast range (although it is not used in the rest of the code)
    forecast_range = pd.date_range(start=start_prediction_timestamp, end=end_prediction_timestamp, freq='15min')
    # Set 'measured' column in the test data to None
    df_test_buyer['measured'] = [None for _ in range(len(df_test_buyer))]
    # Concatenate train and test data
    df_buyer = pd.concat([df_train_buyer, df_test_buyer], axis=0)
    # Rename 'measured' to 'b1r1'
    df_buyer['b1r1'] = df_buyer['measured']
    # Drop the original 'measured' column
    df_buyer.drop(columns=['measured'], inplace=True)
    return df_buyer, forecast_range