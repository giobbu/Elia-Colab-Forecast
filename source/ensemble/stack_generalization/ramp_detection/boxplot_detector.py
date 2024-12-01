import pandas as pd
import numpy as np
from loguru import logger
from source.ensemble.stack_generalization.ramp_detection.utils import process_ramps_train_data, detect_anomalous_clusters, log_ramp_alarm_status

def compute_IQW(df_insample, df_outsample):
    " Compute the Interquantile Width (IQW) for the insample and outsample predictions"
    # Insample Predictions
    df_insample['IQW'] = np.abs(df_insample.Q90 - df_insample.Q10)
    # Outsample Predictions
    df_outsample['IQW'] = np.abs(df_outsample.Q90 - df_outsample.Q10)
    # Compute the mean IQW for outsample predictions
    IQW_mean = df_outsample['IQW'].mean()
    return df_insample, df_outsample, IQW_mean

def group_IQW_by_date(df_insample, df_outsample):
    " Group the IQW by date for insample and outsample predictions"
    df_insample['date'] = pd.to_datetime(df_insample.index).date
    df_outsample['date'] = pd.to_datetime(df_outsample.index).date
    df_insample = df_insample.groupby('date').mean()
    df_outsample = df_outsample.groupby('date').mean()
    return df_insample, df_outsample

def compute_upper_bound_boxplot(df_insample, q1=0.25, q3=0.75, k=1.5):
    " Compute the Upper Bound for the  Boxplot Outlier Detection"
    # Compute Q1 and Q3 for each column
    Q1 = df_insample.IQW.quantile(q1)
    Q3 = df_insample.IQW.quantile(q3)
    # Compute IQR for each column
    IQR = Q3 - Q1
    # Lower and Upper Bounds
    Upper = Q3 + k*IQR
    return Upper

def anomaly_detection_boxplot(df_train, pred_insample, pred_outsample, preprocess_ramps, q1, q3, k):
    " Compute the Interquantile Width (IQW) for the insample and outsample predictions"
    # Compute the Interquantile Width (IQW) for the insample and outsample predictions
    df_insample, df_outsample, IQW_mean = compute_IQW(pred_insample, pred_outsample)
    # Compute the Upper Bound for the Boxplot Outlier Detection
    if preprocess_ramps:
        # Filter Ramp Events from the insample predictions
        df_insample_without_ramps = process_ramps_train_data(df_train, df_insample)
        upper_box_bound = compute_upper_bound_boxplot(df_insample_without_ramps, q1, q3, k)
    else:
        upper_box_bound = compute_upper_bound_boxplot(df_insample, q1, q3, k)
    # Determine initial alarm status based on IQW_mean
    alarm_status = int(IQW_mean >= upper_box_bound)
    # Identify anomalous events based on the IQW value
    df_outsample['is_anomalous'] = df_outsample['IQW'] > upper_box_bound  # formula: upper_bound = q3 + 1.5*IQR
    return alarm_status, df_outsample, upper_box_bound

def alarm_policy_rule(alarm_status, df_outsample, list_ramp_alarm, max_consecutive_points):
    """
    Check if a Ramp Alarm is triggered based on the number of outliers.
    """
    # If alarm is triggered and max_consecutive_points is specified
    df_ramp_clusters = pd.DataFrame()  # Default empty DataFrame
    if alarm_status == 1 and max_consecutive_points != 0:
        # Detect clusters of anomalous events
        df_ramp_clusters = detect_anomalous_clusters(df_outsample,  max_consecutive_points, time_interval='15T')
        # Update alarm status based on the presence of anomalous clusters
        alarm_status = int(not df_ramp_clusters.empty)
    # Log the forecast range and alarm status
    list_ramp_alarm.append((df_outsample.index[0], 
                            alarm_status))
    return list_ramp_alarm, alarm_status, df_ramp_clusters

def detect_wind_ramp_boxplot(pred_insample, pred_outsample, list_ramp_alarm, list_ramp_alarm_intraday, df_train, q1, q3, k, preprocess_ramps=True, max_consecutive_points=3):
    """
    This function processes forecast data to compute the Interquantile Width (IQW), 
    handles the upper bound calculation for boxplot outlier detection, 
    and checks if a ramp alarm is triggered based on the number of outliers.
    """
    # Compute the Interquantile Width (IQW) for the insample and outsample predictions
    pred_insample.columns = pred_insample.columns.map(lambda x: f'Q{x*100:.0f}') 
    pred_outsample.columns = pred_outsample.columns.map(lambda x: f'Q{x*100:.0f}')
    # wind ramp detection day-ahead
    alarm_status, df_outsample, upper_box_bound = anomaly_detection_boxplot(df_train, pred_insample, pred_outsample, preprocess_ramps, q1, q3, k)
    # wind ramp detection day-ahead
    list_ramp_alarm, alarm_status, df_ramp_clusters = alarm_policy_rule(alarm_status, df_outsample, list_ramp_alarm, max_consecutive_points)
    # Intraday Detection
    # Divide df_outsample into three DataFrames of 32 rows each
    df_outsample_list = [df_outsample.iloc[i:i + 32] for i in range(0, len(df_outsample), 32)]
    # wind ramp detection intraday
    alarm_status_list = []
    for df_ in df_outsample_list:
        list_ramp_alarm_ = []
        _, alarm_status_i, _ = alarm_policy_rule(alarm_status, df_, list_ramp_alarm_, max_consecutive_points)
        alarm_status_list.append(alarm_status_i)
    # log alarma status 1, 2, 3
    list_ramp_alarm_intraday.append(alarm_status_list)
    # Log the forecast range and alarm status
    logger.info(' ')
    logger.info(f"Ramp Alarm Status: {alarm_status}")
    if alarm_status:
        log_ramp_alarm_status(alarm_status_list, df_outsample_list)
    return list_ramp_alarm, list_ramp_alarm_intraday, alarm_status, upper_box_bound, df_ramp_clusters



