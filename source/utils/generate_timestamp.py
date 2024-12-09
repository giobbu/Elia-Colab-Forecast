import pandas as pd
from loguru import logger

def generate_timestamps(start_training, i, window_size):
    # Generate timestamps for training and prediction
    assert window_size > 0, "Window size must be greater than 0"
    start_training_timestamp = pd.to_datetime(start_training, utc=True) + pd.Timedelta(days=i)
    end_training_timestamp = pd.to_datetime(start_training, utc=True) + pd.Timedelta(days=i + window_size)
    start_prediction_timestamp = pd.to_datetime(start_training, utc=True) + pd.Timedelta(days=1 + i + window_size)
    end_prediction_timestamp = pd.to_datetime(start_training, utc=True) + pd.Timedelta(days=2 + i + window_size)

    logger.info(' ')
    logger.opt(colors = True).info('<blue>-------------------------------------------------------------------------------------------</blue>')
    logger.opt(colors=True).info(f'<blue>Start training: {start_training_timestamp} - End training: {end_training_timestamp}</blue>')
    logger.opt(colors = True).info('<blue>-------------------------------------------------------------------------------------------</blue>')
    logger.opt(colors = True).info(f'<blue>Start prediction: {start_prediction_timestamp} - End prediction: {end_prediction_timestamp}</blue>')

    return start_training_timestamp, end_training_timestamp, start_prediction_timestamp, end_prediction_timestamp

