import pandas as pd

def dict2df_predictions(prediction, col_name):
    df_pred = pd.DataFrame.from_records(prediction)
    df_pred.set_index('datetime', inplace=True)
    df_pred.columns = [col_name + '_pred']
    return df_pred

def dict2df_quantiles10(prediction, col_name):
    df_pred = pd.DataFrame.from_records(prediction)
    df_pred.set_index('datetime', inplace=True)
    df_pred.columns = [col_name + '_quantile10']
    return df_pred

def dict2df_quantiles90(prediction, col_name):
    df_pred = pd.DataFrame.from_records(prediction)
    df_pred.set_index('datetime', inplace=True)
    df_pred.columns = [col_name + '_quantile90']
    return df_pred