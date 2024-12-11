import pandas as pd
import numpy as np

def compute_IQW(df_insample, df_outsample):
    " Compute the Interquantile Width (IQW) for the insample and outsample predictions"
    # Insample Predictions
    df_insample['IQW'] = np.abs(df_insample.Q90 - df_insample.Q10)
    # Outsample Predictions
    df_outsample['IQW'] = np.abs(df_outsample.Q90 - df_outsample.Q10)
    return df_insample, df_outsample

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

def anomaly_model_box(pred_insample, pred_outsample, q1, q3, k):
    " Compute the Interquantile Width (IQW) for the insample and outsample predictions"
    # Compute the Interquantile Width (IQW) for the insample and outsample predictions
    df_insample, df_outsample = compute_IQW(pred_insample, pred_outsample)
    # Compute the Upper Bound for the Boxplot Outlier Detection
    upper_box_bound = compute_upper_bound_boxplot(df_insample, q1, q3, k)
    # Determine initial alarm status based on IQW_
    alarm_status = (df_outsample['IQW'] >= upper_box_bound).any()
    # Identify anomalous events based on the IQW value
    df_outsample['is_anomalous'] = df_outsample['IQW'] > upper_box_bound  # formula: upper_bound = q3 + 1.5*IQR
    return df_outsample, alarm_status