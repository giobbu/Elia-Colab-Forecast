from source.utils.ensemble_predictions import dict2df_quantiles10, dict2df_quantiles90

# day ahead
def create_day_ahead_quantiles10(df_val):
    " Create day ahead quantiles 10"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayaheadconfidence10' in df_val.columns, "dayaheadconfidence10 column must be present"
    day_ahead_elia_quantile10 = []
    for i in range(len(df_val)):
        day_ahead_elia_quantile10.append({'datetime': df_val.index[i],
                                            'quantiles10': df_val.dayaheadconfidence10.iloc[i]})
    df_day_ahead_quantile10 = dict2df_quantiles10(day_ahead_elia_quantile10, 'day_ahead')
    return df_day_ahead_quantile10

# day ahead 11
def create_day_ahead_11_quantiles10(df_val):
    " Create day ahead 11 quantiles 10"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayahead11hconfidence10' in df_val.columns, "dayahead11hconfidence10 column must be present"
    day_ahead_11_elia_quantile10 = []
    for i in range(len(df_val)):
        day_ahead_11_elia_quantile10.append({'datetime': df_val.index[i],
                                                'quantiles10': df_val.dayahead11hconfidence10.iloc[i]})
    df_day_ahead11_quantile10 = dict2df_quantiles10(day_ahead_11_elia_quantile10, 'day_ahead11')
    return df_day_ahead11_quantile10

# week ahead
def create_week_ahead_quantiles10(df_val):
    " Create week ahead quantiles 10"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'weekaheadconfidence10' in df_val.columns, "weekaheadconfidence10 column must be present"
    week_ahead_elia_quantile10 = []
    for i in range(len(df_val)):
        week_ahead_elia_quantile10.append({'datetime': df_val.index[i],
                                            'quantiles10': df_val.weekaheadconfidence10.iloc[i]})
    df_week_ahead_quantile10 = dict2df_quantiles10(week_ahead_elia_quantile10, 'week_ahead')
    return df_week_ahead_quantile10

# day ahead
def create_day_ahead_quantiles90(df_val):
    " Create day ahead quantiles 90"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayaheadconfidence90' in df_val.columns, "dayaheadconfidence90 column must be present"
    day_ahead_elia_quantile90 = []
    for i in range(len(df_val)):
        day_ahead_elia_quantile90.append({'datetime': df_val.index[i],
                                            'quantiles90': df_val.dayaheadconfidence90.iloc[i]})
    df_day_ahead_quantile90 = dict2df_quantiles90(day_ahead_elia_quantile90, 'day_ahead')
    return df_day_ahead_quantile90

# day ahead 11
def create_day_ahead_11_quantiles90(df_val):
    " Create day ahead 11 quantiles 90"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'dayahead11hconfidence90' in df_val.columns, "dayahead11hconfidence90 column must be present"
    day_ahead_11_elia_quantile90 = []
    for i in range(len(df_val)):
        day_ahead_11_elia_quantile90.append({'datetime': df_val.index[i],
                                                'quantiles90': df_val.dayahead11hconfidence90.iloc[i]})
    df_day_ahead11_quantile90 = dict2df_quantiles90(day_ahead_11_elia_quantile90, 'day_ahead11')
    return df_day_ahead11_quantile90

# week ahead
def create_week_ahead_quantiles90(df_val):
    " Create week ahead quantiles 90"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'weekaheadconfidence90' in df_val.columns, "weekaheadconfidence90 column must be present"
    week_ahead_elia_quantile90 = []
    for i in range(len(df_val)):
        week_ahead_elia_quantile90.append({'datetime': df_val.index[i],
                                            'quantiles90': df_val.weekaheadconfidence90.iloc[i]})
    df_week_ahead_quantile90 = dict2df_quantiles90(week_ahead_elia_quantile90, 'week_ahead')
    return df_week_ahead_quantile90