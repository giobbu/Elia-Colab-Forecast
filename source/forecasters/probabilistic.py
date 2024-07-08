from source.utils.ensemble_predictions import dict2df_quantiles10, dict2df_quantiles90
import numpy as np
from config.simulation_setting import Simulation
sim_params = Simulation.testing_period

# noisy
def create_noisy_quantiles10(df, column):
    " Create most recent quantiles 10 with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    assert f'{column}confidence10' in df.columns, f"{column}confidence10 column must be present"
    noisy_prediction = []
    for i in range(len(df)):
        noisy_prediction.append({'datetime': df.index[i],
                                    'quantiles10': df[f'{column}confidence10'].sample(1).iloc[0]})
    dfnoisy_pred = dict2df_quantiles10(noisy_prediction, 'noisy')
    return dfnoisy_pred

# malicious
def create_malicious_quantiles10(df, column, cheat=False, df_train=None):
    " Create most recent quantiles 10 with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    assert f'{column}confidence10' in df.columns, f"{column}confidence10 column must be present"
    cheat_prediction = []
    for i in range(len(df)):
        if not cheat:
            cheat_prediction.append({'datetime': df.index[i],
                                        'quantiles10': df[f'{column}confidence10'].iloc[i] + sim_params['noise_degree']*np.random.normal(0, 1)})
        elif cheat:
            assert df_train is not None, "df_train must be provided"
            cheat_prediction.append({'datetime': df.index[i],
                                        'quantiles10': df_train[f'{column}confidence10'].sample(1).iloc[0] + sim_params['noise_degree']*np.random.normal(0, 1)})
        else:
            raise ValueError("cheat must be either True or False")
    dfcheat_pred = dict2df_quantiles10(cheat_prediction, 'malicious')
    return dfcheat_pred

# most recent
def create_most_recent_quantiles10(df_val):
    " Create most recent quantiles 10"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'mostrecentconfidence10' in df_val.columns, "mostrecentconfidence10 column must be present"
    most_recent_elia_quantile10 = []
    for i in range(len(df_val)):
        most_recent_elia_quantile10.append({'datetime': df_val.index[i],
                                            'quantiles10': df_val.mostrecentconfidence10.iloc[i]})
    df_most_recent_quantile10 = dict2df_quantiles10(most_recent_elia_quantile10, 'most_recent')
    return df_most_recent_quantile10

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


# noisy
def create_noisy_quantiles90(df, column):
    " Create most recent quantiles 90 with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    assert f'{column}confidence90' in df.columns, f"{column}confidence90 column must be present"
    noisy_prediction = []
    for i in range(len(df)):
        noisy_prediction.append({'datetime': df.index[i],
                                    'quantiles90': df[f'{column}confidence90'].sample(1).iloc[0]})
    dfnoisy_pred = dict2df_quantiles90(noisy_prediction, 'noisy')
    return dfnoisy_pred

# malicous
def create_malicious_quantiles90(df, column, cheat=False, df_train=None):
    " Create most recent quantiles 90 with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    assert f'{column}confidence90' in df.columns, f"{column}confidence90 column must be present"
    cheat_prediction = []
    for i in range(len(df)):
        if not cheat:
            cheat_prediction.append({'datetime': df.index[i],
                                        'quantiles90': df[f'{column}confidence90'].iloc[i] + sim_params['noise_degree']*np.random.normal(0, 1)})
        elif cheat:
            assert df_train is not None, "df_train must be provided"
            cheat_prediction.append({'datetime': df.index[i],
                                        'quantiles90': df_train[f'{column}confidence90'].sample(1).iloc[0] + sim_params['noise_degree']*np.random.normal(0, 1)})
        else:
            raise ValueError("cheat must be either True or False")
    dfcheat_pred = dict2df_quantiles90(cheat_prediction, 'malicious')
    return dfcheat_pred


# most recent
def create_most_recent_quantiles90(df_val):
    " Create most recent quantiles 90"
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'mostrecentconfidence90' in df_val.columns, "mostrecentconfidence90 column must be present"
    most_recent_elia_quantile90 = []
    for i in range(len(df_val)):
        most_recent_elia_quantile90.append({'datetime': df_val.index[i],
                                            'quantiles90': df_val.mostrecentconfidence90.iloc[i]})
    df_most_recent_quantile90 = dict2df_quantiles90(most_recent_elia_quantile90, 'most_recent')
    return df_most_recent_quantile90

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