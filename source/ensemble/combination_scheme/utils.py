import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_squared_error

def compute_weight(loss, norm):
    """ Compute the weight based on the pinball loss of the forecasts
    args:
        loss: float, pinball loss
        norm: str, normalization method
    returns:
        weight: float, weight"""
    if norm=='sum':
        weight = 1/loss
    elif norm=='softmax':
        weight = 1/np.exp(loss)
    else:
        raise ValueError('Not a valid normalization method')
    return weight

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