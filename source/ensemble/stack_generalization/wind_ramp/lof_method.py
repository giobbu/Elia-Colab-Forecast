import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def anomaly_model_lof(df_insample, df_outsample, n_neighbors, contamination):
    """
    Process ramp events, fit LocalOutlierFactor (LOF) model, and detect anomalies.
    """
    # compute IQW column for pred_insample and pred_outsample
    df_insample['IQW'] = df_insample['Q90'] - df_insample['Q10']
    df_outsample['IQW'] = df_outsample['Q90'] - df_outsample['Q10']
    # Convert training and test data to numpy arrays
    X_train = df_insample.values
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
    return df_outsample, alarm_status