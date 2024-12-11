import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.wind_ramp.box_method import anomaly_model_box
from source.ensemble.stack_generalization.wind_ramp.emp_quantile_method import anomaly_model_eq
from source.ensemble.stack_generalization.wind_ramp.kde_method import anomaly_model_kde
from source.ensemble.stack_generalization.wind_ramp.alarm_policy import alarm_policy_rule

def wind_ramp_detector(ens_params, df_pred_variability_insample, df_pred_variability_outsample):
    """
    Detects wind ramp anomalies based on the Empirical Quantile technique.
    Args:
        ens_params (dict): ensemble parameters.
        df_pred_variability_insample (pd.DataFrame): DataFrame with variability predictions for the insample data.
        df_pred_variability_outsample (pd.DataFrame): DataFrame with variability predictions for the outsample data.
    Returns:
        alarm_status (int): alarm status.
        df_ramp_clusters (pd.DataFrame): DataFrame with ramp events (consecutive anomalous values).
    """
    assert isinstance(df_pred_variability_insample, pd.DataFrame), "df_pred_variability_insample must be a DataFrame"
    assert isinstance(df_pred_variability_outsample, pd.DataFrame), "df_pred_variability_outsample must be a DataFrame"

    # Process variability forecasts variables names
    df_pred_variability_insample.columns = df_pred_variability_insample.columns.map(lambda x: f'Q{x*100:.0f}') 
    df_pred_variability_outsample.columns = df_pred_variability_outsample.columns.map(lambda x: f'Q{x*100:.0f}')

    if ens_params['detector'] == 'eq':
        # detect IQW anomalies for wind ramps using emprical quantile
        logger.debug("Detecting wind ramp anomalies using Empirical Quantile method.")
        df_pred_variability_outsample, alarm_status = anomaly_model_eq(df_pred_variability_insample, 
                                                                    df_pred_variability_outsample, 
                                                                    threshold_quantile=ens_params['threshold_quantile_eq'])
    elif ens_params['detector'] == 'kde':
        # detect IQW anomalies for wind ramps using KDE
        logger.debug("Detecting wind ramp anomalies using KDE method.")
        df_pred_variability_outsample, alarm_status = anomaly_model_kde(df_pred_variability_insample, 
                                                                    df_pred_variability_outsample, 
                                                                    threshold_quantile=ens_params['threshold_quantile_kde'],
                                                                    cv_folds=ens_params['cv_folds_kde'])
    elif ens_params['detector'] == 'box':
        # detect IQW anomalies for wind ramps using Boxplot
        logger.debug("Detecting wind ramp anomalies using Boxplot method.")
        df_pred_variability_outsample, alarm_status = anomaly_model_box(df_pred_variability_insample, 
                                                                        df_pred_variability_outsample, 
                                                                        q1=ens_params['q1_box'], 
                                                                        q3=ens_params['q3_box'], 
                                                                        k=ens_params['k_box'])

    else:
        raise ValueError(f"Detector {ens_params['detector']} not supported.")
    
    # trigger alarm for wind ramps
    alarm_status, df_ramp_clusters = alarm_policy_rule(alarm_status, 
                                                        df_pred_variability_outsample, 
                                                        max_consecutive_points=ens_params['max_consecutive_points'])
    # if df_ramp_clusters is not empty, return alarm status and df_ramp_clusters
    if not df_ramp_clusters.empty:
        return alarm_status, df_ramp_clusters
    else:
        return alarm_status, None