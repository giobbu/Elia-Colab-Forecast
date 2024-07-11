from source.ensemble.utils.metrics import calculate_pinball_losses, calculate_rmse

def create_df_forecaster_first_stage(df, name):
    " Create a dataframe with the forecasters predictions for the first stage"
    assert type(name) == str
    df_forecaster = df[[f'norm_{name}forecast', f'norm_{name}confidence10', f'norm_{name}confidence90', 'norm_measured']]
    df_forecaster = df_forecaster.copy()
    df_forecaster.loc[:, 'target'] = df_forecaster['norm_measured']
    return df_forecaster

def create_df_forecaster_second_stage(df, name):
    " Create a dataframe with the forecasters predictions for the second stage"
    assert type(name) == str
    df_forecaster = df[[f'norm_{name}forecast', 'norm_measured']]
    df_forecaster = df_forecaster.copy()
    df_forecaster.loc[:, 'target'] = df_forecaster['norm_measured']
    return df_forecaster

def collect_rmse_result(df, forecast_col, lst_rmse):
    " Collect RMSE results"
    rmse = round(calculate_rmse(df, forecast_col).values[0][0], 3)
    lst_rmse.append(rmse)
    return lst_rmse, rmse


def collect_pb_result(df, name_q10, name_q90, lst_pb_q10, lst_pb_q90):
    " Collect pinball results"
    assert name_q10 in df.columns, f'{name_q10} not in columns'
    assert name_q90 in df.columns, f'{name_q90} not in columns'
    pinball = calculate_pinball_losses(df, name_q10, name_q90)
    pinball_q10 = round(pinball['pb_loss_10'].values[0], 3)
    pinball_q90 = round(pinball['pb_loss_90'].values[0], 3)
    lst_pb_q10.append(pinball_q10)
    lst_pb_q90.append(pinball_q90)
    return lst_pb_q10, lst_pb_q90, pinball_q10, pinball_q90