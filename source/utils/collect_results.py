from source.ensemble.utils.metrics import calculate_pinball_losses

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