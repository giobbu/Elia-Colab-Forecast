from source.utils.ensemble_predictions import dict2df_predictions
import numpy as np
from config.simulation_setting import Simulation
sim_params = Simulation.testing_period


# noisy
def create_noisy_predictions(df, column):
    " Create most recent predictions with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    # column string contained in column forecast
    assert f'{column}forecast' in df.columns, f"{column}forecast column must be present"
    noisy_prediction = []
    for i in range(len(df)):
        noisy_prediction.append({'datetime': df.index[i],
                                    'predictions': df[f'{column}forecast'].sample(1).iloc[0]})
    dfnoisy_pred = dict2df_predictions(noisy_prediction, 'noisy')
    return dfnoisy_pred

# malicious
def create_malicious_predictions(df, column, cheat=False, df_train=None):
    " Create most recent predictions with cheat option "
    assert df.index.name == 'datetime', "Index must be datetime"
    # column string contained in column forecast
    assert f'{column}forecast' in df.columns, f"{column}forecast column must be present"
    cheat_prediction = []
    for i in range(len(df)):
        if not cheat:
            cheat_prediction.append({'datetime': df.index[i],
                                        'predictions': df[f'{column}forecast'].iloc[i] + sim_params['noise_degree']*np.random.normal(0, 1) })
        elif cheat:
            assert df_train is not None, "df_train must be provided"
            cheat_prediction.append({'datetime': df.index[i],
                                        'predictions': df_train[f'{column}forecast'].sample(1).iloc[0] + sim_params['noise_degree']*np.random.normal(0, 1)})
        else:
            raise ValueError("cheat must be either True or False")
    dfcheat_pred = dict2df_predictions(cheat_prediction, 'malicious')
    return dfcheat_pred

# most recent
def create_most_recent_predictions(df_val):
    " Create most recent predictions "
    assert df_val.index.name == 'datetime', "Index must be datetime"
    assert 'mostrecentforecast' in df_val.columns, "dayaheadforecast column must be present"
    most_recent_elia_prediction = []
    for i in range(len(df_val)):
        most_recent_elia_prediction.append({'datetime': df_val.index[i],
                                            'predictions': df_val.mostrecentforecast.iloc[i]})
    df_most_recent_pred = dict2df_predictions(most_recent_elia_prediction, 'most_recent')
    return df_most_recent_pred

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