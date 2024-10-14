import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from source.ensemble.stack_generalization.ramp_detection.utils import process_ramps_train_data, detect_anomalous_clusters

def get_quantile(score_data, threshold):
    " Get the quantile threshold of the data. "
    return np.quantile(score_data, threshold)

def get_anomalies(data, threshold):
    " Get the anomalies based on the quantile threshold. "
    return data[data > threshold]  # Get the anomalies

def detect_anomalies(pred_var_outsample, quantile_anomaly_threshold, testing_pi_values):
    """
    Detect anomalies based on the quantile threshold and normalize probability alarms.
    """
    # Mark data points as anomalous based on the threshold
    pred_var_outsample['is_anomalous'] = is_anomalous( testing_pi_values, quantile_anomaly_threshold)
    return pred_var_outsample

def is_anomalous(testing_pi_values, quantile_anomaly_threshold):
    """
    Determine if data points are anomalous based on the KDE and threshold.
    """
    return testing_pi_values > quantile_anomaly_threshold  


def plot_hist_and_anomalies(preprocessed_training_eq, testing_eq, quantile_anomaly_threshold, threshold_quantile, max_value, anomalies_eq=None):
    """
    This function plots the KDE of the training data, testing data, and anomalies.
    """
    # Create the plot
    plt.figure(figsize=(8, 2))
    # Plot the Histogram of training data
    plt.scatter(preprocessed_training_eq, np.ones_like(preprocessed_training_eq), alpha=0.1, color='blue', label='Insample Data')
    # Plot scatter testing EQ points on x-axis
    plt.scatter(testing_eq, np.ones_like(testing_eq), color='green', label='Outsample Data', alpha=0.5)
    #Plot anomalies if any
    if len(anomalies_eq) > 0:
        for anomaly in range(len(anomalies_eq)):
            plt.scatter(anomalies_eq[anomaly], np.ones_like(anomalies_eq[anomaly]), color='red', 
                        alpha=1, marker='*')
    
    # # Plot the quantile anomaly threshold as a vertical line
    plt.axvline(quantile_anomaly_threshold, color='red', linestyle='--', label=f'Quantile {threshold_quantile}')
    # Set limits and labels
    plt.xlim(0, max_value)
    plt.ylim(0, 2)
    plt.xlabel('PI Width')
    plt.title('Wind Power Variability PI Width')
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def anomaly_detection_eq(df_train, df_insample, df_outsample, threshold_quantile, preprocess_ramps=True, max_value=1000):
    """
    Detect anomalies using the Kernel Density Estimation (KDE) technique.
    """
    # compute IQW column for df_insample and df_outsample
    df_insample['IQW'] = df_insample['Q90'] - df_insample['Q10']
    df_outsample['IQW'] = df_outsample['Q90'] - df_outsample['Q10']
    # Process Ramp Events if preprocessing is enabled
    if preprocess_ramps:
        df_insample_preprocessed = process_ramps_train_data(df_train, df_insample)
    else:
        df_insample_preprocessed = df_insample.copy()
    # get the IQW values for training and testing data
    preprocessed_training_eq = df_insample_preprocessed['IQW'].values
    testing_eq = df_outsample['IQW'].values 
    # get quantile for anomalies
    quantile_anomaly_threshold = get_quantile(preprocessed_training_eq, threshold_quantile)
    # get points with density above quantile threshold
    anomalies_eq = get_anomalies(testing_eq, quantile_anomaly_threshold)
    # detect anomalies 
    df_outsample = detect_anomalies(df_outsample, quantile_anomaly_threshold, testing_eq)
    # set alarm status
    alarm_status = 1 if np.any(df_outsample['is_anomalous'] == True) else 0
    # plot histogram and anomalies
    plot_hist_and_anomalies(preprocessed_training_eq, testing_eq, 
                            quantile_anomaly_threshold, 
                            threshold_quantile, max_value,
                            anomalies_eq)
    return df_outsample, alarm_status

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
    list_ramp_alarm.append((df_outsample.index[0], alarm_status))
    return list_ramp_alarm, alarm_status, df_ramp_clusters

def detect_wind_ramp_eq(df_train, df_insample, df_outsample, list_ramp_alarm, list_ramp_alarm_intraday, threshold_quantile, preprocess_ramps=True, max_value=1000, max_consecutive_points=3):
    """
    This function processes forecast data to compute the Interquantile Width (IQW), 
    handles the upper bound calculation for boxplot outlier detection, 
    and checks if a ramp alarm is triggered based on the number of outliers.
    """
    assert isinstance(df_train, pd.DataFrame), "df_train must be a DataFrame"
    assert isinstance(df_insample, pd.DataFrame), "df_insample must be a DataFrame"
    assert isinstance(df_outsample, pd.DataFrame), "df_outsample must be a DataFrame"
    assert isinstance(list_ramp_alarm, list), "list_ramp_alarm must be a list"
    assert isinstance(list_ramp_alarm_intraday, list), "list_ramp_alarm_intraday must be a list"
    assert isinstance(threshold_quantile, float), "threshold_quantile must be a float"
    assert isinstance(preprocess_ramps, bool), "preprocess_ramps must be a boolean"
    assert isinstance(max_value, int), "max_value must be an integer"
    assert isinstance(max_consecutive_points, int), "max_consecutive_points must be an integer"
    
    # Process variability forecasts variables names
    df_insample.columns = df_insample.columns.map(lambda x: f'Q{x*100:.0f}') 
    df_outsample.columns = df_outsample.columns.map(lambda x: f'Q{x*100:.0f}')
    # detect IQW anomalies for wind ramps using emprical quantile
    df_outsample, alarm_status = anomaly_detection_eq(df_train, df_insample, df_outsample, threshold_quantile, preprocess_ramps, max_value)
    # trigger alarm for ramp clusters
    list_ramp_alarm, alarm_status, df_ramp_clusters = alarm_policy_rule(alarm_status, df_outsample, list_ramp_alarm, max_consecutive_points)
    # intraday wind ramp detection
    # divide df_outsample in 3 dataframes of 32 rows each
    df_outsample_1 = df_outsample.iloc[:32]
    df_outsample_2 = df_outsample.iloc[32:64]
    df_outsample_3 = df_outsample.iloc[64:]
    # wind ramp detection intraday
    list_ramp_alarm_1, list_ramp_alarm_2, list_ramp_alarm_3 = [], [], []
    _, alarm_status_1, _ = alarm_policy_rule(alarm_status, df_outsample_1, list_ramp_alarm_1, max_consecutive_points)
    _, alarm_status_2, _ = alarm_policy_rule(alarm_status, df_outsample_2, list_ramp_alarm_2, max_consecutive_points)
    _, alarm_status_3, _ = alarm_policy_rule(alarm_status, df_outsample_3, list_ramp_alarm_3, max_consecutive_points)
    # log intraday ramp alarms
    list_ramp_alarm_intraday.append((alarm_status_1, alarm_status_2, alarm_status_3))
    logger.info(' ')
    logger.info(f"Ramp Alarm Status: {alarm_status}")
    if alarm_status:
        # log alarma status 1, 2, 3
        logger.info(' ')
        logger.info('Intraday Wind Ramp Detection')
        logger.info(' ')
        logger.info(f"Ramp Alarm Status 1: {alarm_status_1} - Datetime Range: {df_outsample_1.index[0]} - {df_outsample_1.index[-1]}")
        logger.info(f"Ramp Alarm Status 2: {alarm_status_2} - Datetime Range: {df_outsample_2.index[0]} - {df_outsample_2.index[-1]}")
        logger.info(f"Ramp Alarm Status 3: {alarm_status_3} - Datetime Range: {df_outsample_3.index[0]} - {df_outsample_3.index[-1]}")
    return list_ramp_alarm, list_ramp_alarm_intraday, alarm_status, df_ramp_clusters
