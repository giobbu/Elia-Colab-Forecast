import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from loguru import logger

from source.ensemble.stack_generalization.ramp_detection.utils import process_ramps_train_data, detect_anomalous_clusters

def generate_bandwidths(bandwidth_range, num_bandwidths):
    """
    Generate bandwidth values over a logarithmic scale.
    """
    start_exp, end_exp = bandwidth_range
    return 10 ** np.linspace(start_exp, end_exp, num_bandwidths)

def kde_mle_optimization(training_data, bandwidths, cv_folds):
    " Optimize the bandwidth of the KDE using grid search and cross-validation. "
    kde = KernelDensity()
    param_grid = {'bandwidth': bandwidths}
    grid_search = GridSearchCV(kde, param_grid, cv=cv_folds, n_jobs=-1)
    # Fit the grid search model
    grid_search.fit(training_data[:, None])
    return grid_search

def kde_fitting(training_data, bandwidth):
    " Fit the KDE model with the optimized bandwidth. "
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(training_data[:, None])
    return kde

def kde_predict(testing_data, fitted_kde):
    " Predict the probability densities of the testing data. "
    return np.exp(fitted_kde.score_samples(testing_data[:, None]))

def get_quantile(score_data, threshold):
    " Get the quantile threshold of the data. "
    return np.quantile(score_data, threshold)

def get_anomalies(data, threshold):
    " Get the anomalies based on the quantile threshold. "
    return data[data > threshold]

def detect_anomalies(pred_var_outsample, quantile_anomaly_threshold, testing_kde):
    """
    Detect anomalies based on the quantile threshold and normalize probability alarms.
    """
    # Mark data points as anomalous based on the threshold
    pred_var_outsample['is_anomalous'] = is_anomalous(testing_kde, quantile_anomaly_threshold)
    return pred_var_outsample

def is_anomalous(testing_kde_values, quantile_anomaly_threshold):
    """
    Determine if data points are anomalous based on the KDE and threshold.
    """
    return testing_kde_values > quantile_anomaly_threshold

def anomaly_detection_kde(df_train, df_insample, df_outsample, threshold_quantile, preprocess_ramps=True, max_value=1000, cv_folds=5):
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
    preprocessed_training_kde = df_insample_preprocessed['IQW'].values
    testing_kde = df_outsample['IQW'].values 

    # optimize bandwidth with grid search
    bandwidths = generate_bandwidths(bandwidth_range = (-1, 100), num_bandwidths=500)
    grid_search = kde_mle_optimization(training_data=preprocessed_training_kde, bandwidths=bandwidths, cv_folds=cv_folds)

    # employ optimized bandwidth on all training data
    fitted_kde = kde_fitting(preprocessed_training_kde, bandwidth=grid_search.best_params_['bandwidth'])

    # evaluate on training data as probability densities
    sorted_preprocessed_training_kde = np.sort(preprocessed_training_kde)
    kde_training_pdf = np.exp(fitted_kde.score_samples(sorted_preprocessed_training_kde[:, None]))

    # evaluate on testing data as probability densities
    kde_testing_pdf = kde_predict(testing_kde, fitted_kde) 

    # get quantile for anomalies
    quantile_anomaly_threshold_empirical = get_quantile(sorted_preprocessed_training_kde, threshold_quantile)

    # get quantile for anomalies
    # sample 1000 points from kde
    kde_samples = fitted_kde.sample(n_samples=10000, random_state=42)
    # concatenate the samples with the training data (should have same dimension)
    kde_samples = np.concatenate((preprocessed_training_kde, kde_samples[:, 0]))
    # get the quantile threshold
    quantile_anomaly_threshold_kde = get_quantile(kde_samples, threshold_quantile)

    # get points with density above quantile threshold
    anomalies_kde = get_anomalies(testing_kde, quantile_anomaly_threshold_kde)
    # get the probability densities of the anomalies
    if len(anomalies_kde) > 0:
        kde_anomalies_pdf = kde_predict(anomalies_kde, fitted_kde) 
    else:
        anomalies_kde = []
        kde_anomalies_pdf = []

    # detect anomalies 
    df_outsample = detect_anomalies(df_outsample, quantile_anomaly_threshold_kde, testing_kde)

    alarm_status = 1 if np.any(df_outsample['is_anomalous'] == True) else 0

    plot_kde_and_anomalies(sorted_preprocessed_training_kde, kde_training_pdf, 
                            testing_kde, kde_testing_pdf, 
                            quantile_anomaly_threshold_kde, threshold_quantile, max_value,
                            anomalies_kde, kde_anomalies_pdf, quantile_anomaly_threshold_empirical)

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

def detect_wind_ramp_kde(df_train, df_insample, df_outsample, list_ramp_alarm, list_ramp_alarm_intraday, threshold_quantile, preprocess_ramps=True, max_value=1000, cv_folds=5, max_consecutive_points=3):
    """
    This function processes forecast data to compute the Interquantile Width (IQW), 
    handles the upper bound calculation for boxplot outlier detection, 
    and checks if a ramp alarm is triggered based on the number of outliers.
    """

    # Process variability forecasts variables names
    df_insample.columns = df_insample.columns.map(lambda x: f'Q{x*100:.0f}') 
    df_outsample.columns = df_outsample.columns.map(lambda x: f'Q{x*100:.0f}')
    # detect IQW anomalies for wind ramps using KDE
    df_outsample, alarm_status = anomaly_detection_kde(df_train, df_insample, df_outsample, threshold_quantile, preprocess_ramps, max_value, cv_folds)
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

    
    
def plot_kde_and_anomalies(sorted_preprocessed_training_kde, kde_training_pdf, testing_kde, kde_testing_pdf, quantile_anomaly_threshold, threshold_quantile, max_value, anomalies_kde=None, kde_anomalies_pdf=None, quantile_anomaly_threshold_empirical=None):
    """
    This function plots the KDE of the training data, testing data, and anomalies.
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    # Plot the KDE of training data
    plt.plot(sorted_preprocessed_training_kde, kde_training_pdf, label='PDF', color='blue')
    # Plot testing KDE points
    plt.scatter(testing_kde, kde_testing_pdf, color='black', label='Testing KDE', alpha=0.1)
    # Plot anomalies if any
    if len(anomalies_kde) > 0:
        for anomaly in range(len(anomalies_kde)):
            plt.scatter(anomalies_kde[anomaly], kde_anomalies_pdf[anomaly], color='red', 
                        alpha=1 - kde_anomalies_pdf[anomaly], marker='*')
    #Plot the quantile anomaly threshold as a vertical line
    plt.axvline(quantile_anomaly_threshold, color='red', linestyle='--', label=f'KDE Quantile {threshold_quantile}')
    # Plot the empirical quantile anomaly threshold as a vertical line
    if quantile_anomaly_threshold_empirical is not None:
        plt.axvline(quantile_anomaly_threshold_empirical, color='blue', linestyle='--', label=f'Empirical Quantile {threshold_quantile}')
    # plot histogram of training and testing data
    plt.hist(sorted_preprocessed_training_kde, bins=100, density=True, alpha=0.5, color='blue', label='Data')
    # Set limits and labels
    plt.xlim(0, max_value)
    plt.xlabel('PI Width')
    plt.ylabel('PDF')
    plt.title('KDE of PI Width for Wind Ramp Events')
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()