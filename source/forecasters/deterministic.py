from source.utils.ensemble_predictions import dict2df_predictions

# day ahead
def create_day_ahead_predictions(df_val):
    " Create day ahead predictions "
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayaheadforecast' in df_val.columns, "dayaheadforecast column must be present"
    day_ahead_elia_prediction = []
    for i in range(len(df_val)):
        day_ahead_elia_prediction.append({'datetime': df_val.index[i],
                                            'predictions': df_val.dayaheadforecast.iloc[i]})
    df_day_ahead_pred = dict2df_predictions(day_ahead_elia_prediction, 'day_ahead')
    return df_day_ahead_pred

# day ahead 11
def create_day_ahead_11_predictions(df_val):
    " Create day ahead 11 predictions"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayahead11hforecast' in df_val.columns, "dayahead11hforecast column must be present"
    day_ahead_11_elia_prediction = []
    for i in range(len(df_val)):
        day_ahead_11_elia_prediction.append({'datetime': df_val.index[i],
                                                'predictions': df_val.dayahead11hforecast.iloc[i]})
    df_day_ahead11_pred = dict2df_predictions(day_ahead_11_elia_prediction, 'day_ahead11')
    return df_day_ahead11_pred

# week ahead
def create_week_ahead_predictions(df_val):
    " Create week ahead predictions"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'weekaheadforecast' in df_val.columns, "weekaheadforecast column must be present"
    week_ahead_elia_prediction = []
    for i in range(len(df_val)):
        week_ahead_elia_prediction.append({'datetime': df_val.index[i],
                                            'predictions': df_val.weekaheadforecast.iloc[i]})
    df_week_ahead_pred = dict2df_predictions(week_ahead_elia_prediction, 'week_ahead')
    return df_week_ahead_pred