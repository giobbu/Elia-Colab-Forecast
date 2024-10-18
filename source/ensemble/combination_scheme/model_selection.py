import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from source.ensemble.combination_scheme.utils import compute_weight

def calculate_weights(sim_params, df_val_norm, norm='sum'):
    " Calculate weights based on the pinball loss of the forecasts"
    assert len(df_val_norm) > 0, 'Dataframe is empty'
    if not sim_params['most_recent']:
        lst_cols = [name for name in list(df_val_norm.columns) if 'mostrecent' not in name]
    elif not sim_params['malicious']:
        lst_cols = [name for name in list(df_val_norm.columns) if 'malicious' not in name]
    else:
        lst_cols = list(df_val_norm.columns)
    targ_col = [name for name in lst_cols if 'measured' in name]
    targets =  df_val_norm[targ_col[0]]
    lst_cols_forecasts = [name for name in lst_cols if 'measured' not in name]
    lst_q10_weight = []
    lst_q50_weight = []
    lst_q90_weight = []
    for col in lst_cols_forecasts:
        if 'forecast' in col:
            forecast = df_val_norm[col]
            q50_pb_loss = mean_squared_error(targets.values,  forecast.values)  # 
            weight_q50 = compute_weight(q50_pb_loss, norm)
            lst_q50_weight.append({col : weight_q50})
        elif 'confidence10' in col:
            forecast = df_val_norm[col]
            q10_pb_loss = mean_pinball_loss(targets.values,  forecast.values, alpha=0.1)
            weight_q10 = compute_weight(q10_pb_loss, norm)
            lst_q10_weight.append({col : weight_q10})
        elif 'confidence90' in col:
            forecast = df_val_norm[col]
            q90_pb_loss = mean_pinball_loss(targets.values,  forecast.values, alpha=0.9)
            weight_q90 = compute_weight(q90_pb_loss, norm)
            lst_q90_weight.append({col : weight_q90})
        else:
            raise ValueError('Not a valid column')
    return lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight
    

def calculate_best_model(df_test_norm, lst_cols_forecasts, norm_lst_q50_pb_loss, norm_lst_q10_pb_loss, norm_lst_q90_pb_loss):
    " Calculate the combination forecast based on the pinball loss-based weights"
    best_forecast = np.zeros(len(df_test_norm))
    best_quantile10 = np.zeros(len(df_test_norm))
    best_quantile90 = np.zeros(len(df_test_norm))
    weight_50_final = 0
    weight_10_final = 0
    weight_90_final = 0
    for col in lst_cols_forecasts:
        if 'forecast' in col:
            forecast_50 = df_test_norm[col]
            weight_50 = [list(weight.values())[0] for weight in norm_lst_q50_pb_loss if col in weight][0]
            if weight_50 > weight_50_final:
                weight_50_final = weight_50
                best_forecast = forecast_50
        elif 'confidence10' in col:
            forecast_10 = df_test_norm[col]
            weight_10 = [list(weight.values())[0] for weight in norm_lst_q10_pb_loss if col in weight][0]
            if weight_10 > weight_10_final:
                weight_10_final = weight_10
                best_quantile10 = forecast_10
        elif 'confidence90' in col:
            forecast_90 = df_test_norm[col]
            weight_90 = [list(weight.values())[0] for weight in norm_lst_q90_pb_loss if col in weight][0]
            if weight_90 > weight_90_final:
                weight_90_final = weight_90
                best_quantile90 = forecast_90
        else:
            raise ValueError('Not a valid column')
    return best_forecast, best_quantile10, best_quantile90


def run_model_selection(sim_params, df_train_norm, df_test_norm, 
                        end_observations, start_predictions, window_size_valid=1, var=False, norm='sum'):
    " Calculate the weights based on the pinball loss of the forecasts "
    assert len(df_test_norm)==96*2, 'Length of test dataframe is not 96*2'
    if var:
        df = pd.concat([df_train_norm, df_test_norm], axis=0).diff().dropna()
        df_train_norm, df_test_norm = df[df.index < end_observations], df[df.index >= start_predictions]
        assert len(df_test_norm)==96, 'Length of test dataframe is not 96'
        window_validation =  pd.to_datetime(end_observations, utc=True) - pd.Timedelta(days=window_size_valid)
        df_val_norm = df_train_norm[df_train_norm.index.to_series().between(window_validation, end_observations)]
        lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight = calculate_weights(sim_params, df_val_norm, norm)
        best_forecast, _, _ = calculate_best_model(df_test_norm, lst_cols_forecasts, lst_q50_weight, lst_q10_weight, lst_q90_weight)
        df_best_model = pd.DataFrame({
                'mean_prediction': best_forecast,
            }, index=df_test_norm.index)
        df_best_model['targets'] = df_test_norm['norm_measured']
        return df_best_model
    df_test_norm = df_test_norm[df_test_norm.index >= start_predictions]
    window_validation =  pd.to_datetime(end_observations, utc=True) - pd.Timedelta(days=window_size_valid)
    df_val_norm = df_train_norm[df_train_norm.index.to_series().between(window_validation, end_observations)]
    lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight = calculate_weights(sim_params, df_val_norm, norm)
    best_forecast, best_quantile10, best_quantile90 = calculate_best_model(df_test_norm, lst_cols_forecasts, lst_q50_weight, lst_q10_weight, lst_q90_weight)
    df_best_model = create_best_model_df(df_test_norm, best_forecast, best_quantile10, best_quantile90)
    return df_best_model


def create_best_model_df(df_test_norm, best_forecast, best_quantile10, best_quantile90):
    " Create dataframe with the weighted average forecast"
    assert len(df_test_norm) == len(best_forecast) == len(best_quantile10) == len(best_quantile90), 'Length mismatch'
    df_best_model = pd.DataFrame({
        'Q10': best_quantile10,
        'mean_prediction': best_forecast,
        'Q90': best_quantile90
    }, index = df_test_norm.index)
    df_best_model['targets'] = df_test_norm['norm_measured']
    return df_best_model