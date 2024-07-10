import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

def rmse(y_pred, y_targ):
    " Compute Root Mean Squared Error."
    assert len(y_pred) == len(y_targ), "Length mismatch between predictions and targets"
    return np.sqrt(np.mean((y_pred - y_targ)**2))

def mae(y_pred, y_targ):
    " Compute Mean Absolute Error."
    assert len(y_pred) == len(y_targ), "Length mismatch between predictions and targets"
    return np.mean(np.abs(y_pred - y_targ))

def calculate_rmse(df, pred_col, targ_col='target'):
    " Calculate RMSE Loss."
    assert pred_col in df.columns, "prediction column is missing"
    assert targ_col in df.columns, "target column is missing"
    rmse_loss = pd.DataFrame()
    rmse_loss['rmse'] = np.array([rmse(df[pred_col], df[targ_col])])
    return rmse_loss

def calculate_mae(df, pred_col, targ_col='target'):
    " Calculate MAE Loss."
    assert pred_col in df.columns, "prediction column is missing"
    assert targ_col in df.columns, "target column is missing"
    mae_loss = pd.DataFrame()
    mae_loss['mae'] = np.array([mae(df[pred_col], df[targ_col])])
    return mae_loss

def calculate_pinball_losses(df, confidence_10_col, confidence_90_col, targ_col='target'):
    " Calculate Pinball Losses for 10% and 90% quantiles."
    assert 'target' in df.columns, "target column is missing"
    pinball_losses = pd.DataFrame()
    score_10 = mean_pinball_loss( list(df[targ_col].values),  list(df[confidence_10_col].values), alpha=0.1)
    score_90 = mean_pinball_loss( list(df[targ_col].values),  list(df[confidence_90_col].values), alpha=0.9)
    pinball_losses['pb_loss_10'] =  np.array([score_10]) 
    pinball_losses['pb_loss_90'] = np.array([score_90]) 
    return pinball_losses