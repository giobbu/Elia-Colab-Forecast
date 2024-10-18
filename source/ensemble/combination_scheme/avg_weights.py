import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from source.ensemble.combination_scheme.utils import compute_weight

def calculate_weights(sim_params, df_val_norm, norm='sum'):
    """ Calculate weights based on the pinball loss of the forecasts
    args:
        sim_params: dict, simulation parameters
        df_val_norm: pd.DataFrame, validation data
        norm: str, normalization method
    returns:
        lst_cols_forecasts: list, list of forecast columns
        lst_q10_weight: list, list of quantile 10 weights
        lst_q50_weight: list, list of quantile 50 weights
        lst_q90_weight: list, list of quantile 90 weights
    """
    assert len(df_val_norm) > 0, 'Dataframe is empty'
    if not sim_params['most_recent']:
        lst_cols = [name for name in list(df_val_norm.columns) if 'mostrecent' not in name]
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
            q50_pb_loss = mean_pinball_loss(targets.values,  forecast.values, alpha=0.5) #mean_squared_error(targets.values,  forecast.values)  # 
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
    
def normalize_weights(lst_weight):
    """ Normalize weights, the sum should be 1
    args:
        lst_weight: list, list of weights
    returns:
        norm_lst_weight: list, list of normalized weights
    """
    assert len(lst_weight) > 0, 'List of weights is empty'  
    total_sum = sum(list(loss.values())[0] for loss in lst_weight)
    norm_lst_weight = [{key: value/total_sum for key, value in d.items()} for d in lst_weight]
    return norm_lst_weight

# def softmax_normalize_weights(lst_weight):
#     """
#     Apply softmax normalization to a list of weights.
#     """
#     assert len(lst_weight) > 0, 'List of weights is empty'
#     # Compute the softmax for each weight value
#     lst_exp_weights = [{key: np.exp(value) for key, value in d.items()} for d in lst_weight]
#     total_sum = sum(list(loss.values())[0] for loss in lst_exp_weights)
#     # Normalize the weights
#     norm_lst_weight = [{key: value / total_sum for key, value in d.items()} for d in lst_exp_weights]
#     return norm_lst_weight

def calculate_combination_forecast(df_test_norm, lst_cols_forecasts, norm_lst_q50_pb_loss, norm_lst_q10_pb_loss, norm_lst_q90_pb_loss):
    """ Calculate the combination forecast based on the pinball loss-based weights 
    args:
        df_test_norm: pd.DataFrame, test data
        lst_cols_forecasts: list, list of forecast columns
        norm_lst_q50_pb_loss: list, list of normalized quantile 50 weights
        norm_lst_q10_pb_loss: list, list of normalized quantile 10 weights
        norm_lst_q90_pb_loss: list, list of normalized quantile 90 weights
    returns:
        combination_forecast: np.array, combination forecast
        combination_quantile10: np.array, combination quantile 10
        combination_quantile90: np.array, combination quantile 90
    """
    combination_forecast = np.zeros(len(df_test_norm))
    combination_quantile10 = np.zeros(len(df_test_norm))
    combination_quantile90 = np.zeros(len(df_test_norm))
    for col in lst_cols_forecasts:
        if 'forecast' in col:
            forecast_50 = df_test_norm[col]
            weight_50 = [list(weight.values())[0] for weight in norm_lst_q50_pb_loss if col in weight][0]
            combination_forecast += forecast_50 * weight_50
        elif 'confidence10' in col:
            forecast_10 = df_test_norm[col]
            weight_10 = [list(weight.values())[0] for weight in norm_lst_q10_pb_loss if col in weight][0]
            combination_quantile10 += forecast_10 * weight_10
        elif 'confidence90' in col:
            forecast_90 = df_test_norm[col]
            weight_90 = [list(weight.values())[0] for weight in norm_lst_q90_pb_loss if col in weight][0]
            combination_quantile90 += forecast_90 * weight_90
        else:
            raise ValueError('Not a valid column')
    return combination_forecast, combination_quantile10, combination_quantile90

def calculate_weighted_avg(sim_params, df_train_norm, df_test_norm, 
                            end_observations, start_predictions, window_size_valid=1, var=False, norm='sum'):
    """ Calculate the weights based on the pinball loss of the forecasts
    args:
        sim_params: dict, simulation parameters
        df_train_norm: pd.DataFrame, training data
        df_test_norm: pd.DataFrame, test data
        end_observations: pd.Timestamp, end observations timestamp
        start_predictions: pd.Timestamp, start predictions timestamp
        window_size_valid: int, window size for validation
        var: bool, variance
        norm: str, normalization method
    returns:
        df_weighted_avg: pd.DataFrame, weighted average forecast
        dict_weights: dict, dictionary of weights
    """
    assert len(df_test_norm)==96*2, 'Length of test dataframe is not 96*2'
    if var:
        df = pd.concat([df_train_norm, df_test_norm], axis=0).diff().dropna()
        df_train_norm, df_test_norm = df[df.index < end_observations], df[df.index >= start_predictions]
        assert len(df_test_norm)==96, 'Length of test dataframe is not 96'
        window_validation =  pd.to_datetime(end_observations, utc=True) - pd.Timedelta(days=window_size_valid)
        df_val_norm = df_train_norm[df_train_norm.index.to_series().between(window_validation, end_observations)]
        lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight = calculate_weights(sim_params, df_val_norm, norm)
        norm_lst_q50_weight = normalize_weights(lst_q50_weight) 
        norm_lst_q10_weight = normalize_weights(lst_q10_weight) 
        norm_lst_q90_weight = normalize_weights(lst_q90_weight) 
        combination_forecast, _, _ = calculate_combination_forecast(df_test_norm, lst_cols_forecasts, norm_lst_q50_weight, norm_lst_q10_weight, norm_lst_q90_weight)
        df_weighted_avg = pd.DataFrame({
                'mean_prediction': combination_forecast,
            }, index=df_test_norm.index)
        df_weighted_avg['targets'] = df_test_norm['norm_measured']
        dict_weights = {0.5: {key:value for d in norm_lst_q50_weight for key, value in d.items()}}
        return df_weighted_avg, dict_weights
    
    df_test_norm = df_test_norm[df_test_norm.index >= start_predictions]
    window_validation =  pd.to_datetime(end_observations, utc=True) - pd.Timedelta(days=window_size_valid)
    df_val_norm = df_train_norm[df_train_norm.index.to_series().between(window_validation, end_observations)]
    lst_cols_forecasts, lst_q10_weight, lst_q50_weight, lst_q90_weight = calculate_weights(sim_params, df_val_norm, norm)
    norm_lst_q50_weight = normalize_weights(lst_q50_weight) 
    norm_lst_q10_weight = normalize_weights(lst_q10_weight) 
    norm_lst_q90_weight = normalize_weights(lst_q90_weight)
    combination_forecast, combination_quantile10, combination_quantile90 = calculate_combination_forecast(df_test_norm, lst_cols_forecasts, norm_lst_q50_weight, norm_lst_q10_weight, norm_lst_q90_weight)
    df_weighted_avg = create_weighted_avg_df(df_test_norm, combination_forecast, combination_quantile10, combination_quantile90)
    dict_weights = {0.5: {key:value for d in norm_lst_q50_weight for key, value in d.items()},
                    0.1: {key:value for d in norm_lst_q10_weight for key, value in d.items()}, 
                    0.9: {key:value for d in norm_lst_q90_weight for key, value in d.items()}}
    return df_weighted_avg, dict_weights


def create_weighted_avg_df(df_test_norm, combination_forecast, combination_quantile10, combination_quantile90):
    """ Create dataframe with the weighted average forecast
    args:
        df_test_norm: pd.DataFrame, test data
        combination_forecast: np.array, combination forecast
        combination_quantile10: np.array, combination quantile 10
        combination_quantile90: np.array, combination quantile 90
    returns:
        df_weighted_avg: pd.DataFrame, weighted average forecast
    """
    assert len(df_test_norm) == len(combination_forecast) == len(combination_quantile10) == len(combination_quantile90), 'Length mismatch'
    df_weighted_avg = pd.DataFrame({
        'Q10': combination_quantile10,
        'mean_prediction': combination_forecast,
        'Q90': combination_quantile90
    }, index=df_test_norm.index)
    df_weighted_avg['targets'] = df_test_norm['norm_measured']
    return df_weighted_avg