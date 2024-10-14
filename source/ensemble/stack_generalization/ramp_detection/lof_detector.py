import pandas as pd
import numpy as np
from loguru import logger
from source.ensemble.stack_generalization.ramp_detection.utils import process_ramps_train_data, detect_anomalous_clusters
from sklearn.neighbors import LocalOutlierFactor


def anomaly_detection_lof(df_train, df_insample, df_outsample, preprocess_ramps, n_neighbors, contamination):
    """
    Process ramp events, fit LocalOutlierFactor (LOF) model, and detect anomalies.
    """
    # Process Ramp Events if preprocessing is enabled
    df_train_processed = (
        process_ramps_train_data(df_train, df_insample) 
        if preprocess_ramps and process_ramps_train_data else df_insample
    )
    # Convert training and test data to numpy arrays
    X_train = df_train_processed.values
    X_test = df_outsample.values
    # Initialize LocalOutlierFactor model with provided parameters
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
    # Fit the model on training data and make predictions on test data
    fitted_anomaly_model = clf.fit(X_train)
    predicted_anomaly = fitted_anomaly_model.predict(X_test)
    # Initialize alarm status
    alarm_status = 1 if np.any(predicted_anomaly == -1) else 0
    df_outsample['anomaly'] = predicted_anomaly
    df_outsample['is_anomalous'] = df_outsample['anomaly'] == -1
    return alarm_status, df_outsample

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

def detect_wind_ramp_lof(pred_insample, pred_outsample, df_train_norm, df_test_norm, list_ramp_alarm, list_ramp_alarm_intraday, df_train,  n_neighbors, contamination, preprocess_ramps=True, max_consecutive_points=3):
    """
    This function processes forecast data to compute the Interquantile Width (IQW), 
    handles the upper bound calculation for boxplot outlier detection, 
    and checks if a ramp alarm is triggered based on the number of outliers.
    """
    try:
        pred_insample.columns = pred_insample.columns.map(lambda x: f'Q{x*100:.0f}') 
        pred_outsample.columns = pred_outsample.columns.map(lambda x: f'Q{x*100:.0f}')

        # compute IQW column for pred_insample and pred_outsample
        pred_insample['IQW'] = pred_insample['Q90'] - pred_insample['Q10']
        pred_outsample['IQW'] = pred_outsample['Q90'] - pred_outsample['Q10']

        alarm_status, df_outsample = anomaly_detection_lof(df_train, pred_insample, pred_outsample, preprocess_ramps, n_neighbors, contamination)
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

    except Exception as e:
        logger.error(f"Error processing forecast data: {e}")
        return None





