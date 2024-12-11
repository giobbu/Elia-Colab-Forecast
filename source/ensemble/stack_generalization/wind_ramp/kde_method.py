import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

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


def anomaly_model_kde(df_insample, df_outsample, threshold_quantile, cv_folds=5):
    """
    Detect anomalies using the Kernel Density Estimation (KDE) technique.
    """
    # compute IQW column for df_insample and df_outsample
    df_insample['IQW'] = df_insample['Q90'] - df_insample['Q10']
    df_outsample['IQW'] = df_outsample['Q90'] - df_outsample['Q10']
    # get the IQW values for training and testing data
    preprocessed_training_kde = df_insample['IQW'].values
    testing_kde = df_outsample['IQW'].values 
    # optimize bandwidth with grid search
    bandwidths = generate_bandwidths(bandwidth_range = (-1, 100), num_bandwidths=500)
    grid_search = kde_mle_optimization(training_data=preprocessed_training_kde, bandwidths=bandwidths, cv_folds=cv_folds)
    # employ optimized bandwidth on all training data
    fitted_kde = kde_fitting(preprocessed_training_kde, bandwidth=grid_search.best_params_['bandwidth'])
    # get quantile for anomalies
    # sample 1000 points from kde
    kde_samples = fitted_kde.sample(n_samples=10000, random_state=42)
    # concatenate the samples with the training data (should have same dimension)
    kde_samples = np.concatenate((preprocessed_training_kde, kde_samples[:, 0]))
    # get the quantile threshold
    quantile_anomaly_threshold_kde = get_quantile(kde_samples, threshold_quantile)
    # detect anomalies 
    df_outsample = detect_anomalies(df_outsample, quantile_anomaly_threshold_kde, testing_kde)
    # determine alarm status
    alarm_status = 1 if np.any(df_outsample['is_anomalous'] == True) else 0
    return df_outsample, alarm_status

