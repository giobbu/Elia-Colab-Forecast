import numpy as np
import pandas as pd

def process_ramp_events(measurements_df):
    """ Process ramp events in the measurements_df DataFrame.
    args:
        measurements_df: DataFrame with measurements
    returns:
        measurements_df: DataFrame with ramp events
        ramp_threshold: Threshold for ramp events
    """
    # Calculate the maximum measured value
    max_measured_value = measurements_df['measured'].max()
    # Calculate the threshold for ramp events
    ramp_threshold = 0.3 * max_measured_value
    # Identify ramp events
    measurements_df['ramp_events'] = (np.abs(measurements_df['measured'].diff()) >= ramp_threshold).astype(int)
    return measurements_df, ramp_threshold

def censore_ramp_values_with_max(df_insample, list_ramps_in_training):
    """ Censor the ramp values in df_insample to the maximum value in the remaining rows.
    args:
        df_insample: DataFrame with insample data
        list_ramps_in_training: List with ramp events in training data
    returns:
        df_insample: DataFrame with censored ramp values
    """
    # Identify rows corresponding to ramp events
    is_ramp_event = pd.Index(df_insample.index.date).isin(list_ramps_in_training)
    # Filter out the ramp events to get the remaining rows
    df_without_ramps = df_insample[~is_ramp_event]
    # Calculate the maximum value of the remaining rows
    max_value = df_without_ramps.max()
    # Replace the values in the ramp event rows with the maximum value
    df_insample.loc[is_ramp_event] = max_value
    return df_insample

def truncate_ramp_values(df_insample, list_ramps_in_training):
    """ Truncate the ramp values in df_insample to the maximum value in the remaining rows. 
    args:
        df_insample: DataFrame with insample data
        list_ramps_in_training: List with ramp events in training data
    returns:
        df_insample: DataFrame with truncated ramp values
    """
    # Identify rows corresponding to ramp events
    is_ramp_event = pd.Index(df_insample.index.date).isin(list_ramps_in_training)
    # Filter out the ramp events to get the remaining rows
    df_without_ramps = df_insample[~is_ramp_event]
    return df_without_ramps

def process_ramps_train_data(df_train, df_insample, method='truncate'):
    """
    Process ramp events in df_train and filter them out from df_insample.
    args:
        df_train: DataFrame with training data
        df_insample: DataFrame with insample data (variability predictions)
        method: Method to filter ramp events ('truncate' or 'censor')
    returns:
        df_insample_without_ramps: DataFrame with filtered ramp events
    """
    # Process ramp events
    df_train, _ = process_ramp_events(df_train)
    # Group by date and sum ramp events
    df_train_by_date = df_train.groupby(df_train.index.date).sum()['ramp_events']
    # Identify dates with a single ramp event
    list_ramps_in_training = df_train_by_date[df_train_by_date == 1].index
    if method == 'truncate':
        # Truncate ramp values in df_insample
        df_insample_without_ramps = truncate_ramp_values(df_insample, list_ramps_in_training)
        return df_insample_without_ramps
    elif method == 'censor':
        # Censor ramp values in df_insample
        df_insample_censored = censore_ramp_values_with_max(df_insample, list_ramps_in_training)
        return df_insample_censored
    else:
        raise ValueError('Invalid method. Choose between "truncate" and "censor".')

def filter_consecutive_ramps(df_anomalies, max_consecutive_points):
    """
    Filters the DataFrame to only include rows where 'is_consecutive' is True and
    'consecutive_count' is greater than or equal to max_consecutive_points.
    args:
        df_anomalies: DataFrame with anomalous events
        max_consecutive_points: Maximum number of consecutive points
    """
    return df_anomalies[(df_anomalies['is_consecutive'] == True) & 
                    (df_anomalies['consecutive_count'] >= max_consecutive_points)]

def detect_anomalous_clusters(df, max_consecutive_points, time_interval='15T'):
    """
    Detects and clusters consecutive anomalous events in a DataFrame based on a specified time interval.
    args:
        df: DataFrame with anomalous events
        max_consecutive_points: Maximum number of consecutive points
        time_interval: Time interval for consecutive events
    returns:
        df_ramps: DataFrame with ramp events (consecutive anomalous values)
    """
    # Filter out the rows where anomalous events occur
    df_anomalous = df[df['is_anomalous']].reset_index()
    # Calculate the time difference between consecutive events
    df_anomalous['time_diff'] = df_anomalous['datetime'].diff()
    # Determine if events are consecutive based on the provided time interval
    df_anomalous['is_consecutive'] = df_anomalous['time_diff'] == pd.Timedelta(time_interval)
    # Define clusters by identifying the start of a new cluster where events are not consecutive
    df_anomalous['cluster_id'] = (~df_anomalous['is_consecutive']).cumsum()
    # Group by cluster_id to calculate the number of consecutive events in each cluster
    df_cluster_info = df_anomalous.groupby('cluster_id')['is_consecutive'].sum().rename('consecutive_count')
    # Merge the original DataFrame with the cluster information
    df_anomalous = df_anomalous.merge(df_cluster_info, on='cluster_id')
    # Consecutive count + 1 to include the current event
    df_anomalous['consecutive_count'] += 1
    # Set ''datetime'' as the index for the final DataFrame
    df_anomalous.set_index('datetime', inplace=True)
    # Filter DataFrame based on consecutive conditions
    df_ramps = filter_consecutive_ramps(df_anomalous, max_consecutive_points)
    return df_ramps.drop(columns=['time_diff', 'is_consecutive', 'is_anomalous'])


def append_wind_ramps(df, list_ramp_alarm, i, target_variability, lst_wind_ramps_days, pred_var_outsample):
    """
    Detects wind ramp events within a specific day, updates prediction output, and appends results.
    args:
        df: DataFrame with measurements
        list_ramp_alarm: List with ramp alarms
        i: Index of the ramp alarm
        target_variability: Target variability
        lst_wind_ramps_days: List with wind ramp days
        pred_var_outsample: DataFrame with prediction output
    returns:
        lst_wind_ramps_days: Updated list with wind ramp days
    """
    # Get the datetime from the ramp alarm list
    datetime = list_ramp_alarm[i][0]
    # Process ramp events and get the updated dataframe and wind ramp threshold
    df, wind_ramp_threshold = process_ramp_events(df)
    # Filter the dataframe for measurements within the specific day
    df_day_measurements = df.loc[datetime : datetime + pd.Timedelta(days=1)]
    # Check if there are any wind ramp events for the day
    wind_ramp = df_day_measurements['ramp_events'].sum() > 0
    # Initialize list to store wind ramp indices
    list_wind_ramps = []
    # If wind ramp events exist, update the pred_var_outsample and append the results
    if wind_ramp:
        pred_var_outsample['TARG'] = target_variability
        list_wind_ramps = df_day_measurements[df_day_measurements['ramp_events'] == 1].index.tolist()
        lst_wind_ramps_days.append((pred_var_outsample.copy(), wind_ramp_threshold, list_wind_ramps))
    return lst_wind_ramps_days

def append_ramp_alarm_days(alarm_status, target_variability, pred_var_outsample, df_ramp_clusters, list_ramp_alarm_days):
    """
    Appends ramp alarm days to the list when an alarm is active.
    args:
        alarm_status: Status of the alarm (0 or 1)
        target_variability: Target variability
        pred_var_outsample: DataFrame with variability prediction output
        df_ramp_clusters: DataFrame with ramp clusters
        list_ramp_alarm_days: List with ramp alarm days
    returns:
        list_ramp_alarm_days: Updated list with ramp alarm days
    """
    # Append ramp alarm days if the alarm is active
    if alarm_status == 1:
        # add the observed target variability to the prediction output dataframe
        pred_var_outsample['TARG'] = target_variability
        # append the prediction output and ramp clusters to
        # the list of ramp alarm days
        list_ramp_alarm_days.append((pred_var_outsample.copy(), df_ramp_clusters))
    return list_ramp_alarm_days

