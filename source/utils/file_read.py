import pandas as pd

def filter_offshore(df, offshore_filter):
    assert 'offshoreonshore' in df.columns, "The DataFrame must contain the 'offshoreonshore' column."
    df = df[df.offshoreonshore == offshore_filter]  # filter by offshore/onshore
    return df

def set_index_datetiemUTC(df):
    assert 'datetime' in df.columns, "The DataFrame must contain the 'datetime' column."
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values(by='datetime').reset_index(drop=True)
    df.set_index('datetime', inplace=True)
    return df

def process_file(file, offshore_filter='Offshore'):
    assert file.endswith('.json'), 'File must be a json file'
    df = pd.read_json(file)
    df = filter_offshore(df, offshore_filter)
    df = set_index_datetiemUTC(df)
    df[df['measured'] < 0] = 0
    return df

def process_and_concat_files(files, offshore_filter='Offshore'):
    "Process and concatenate files"
    assert len(files) > 0, 'No files to process'
    dataframes = []
    for file in files:
        df = process_file(file, offshore_filter=offshore_filter)
        dataframes.append(df)
    concatenated_df = pd.concat(dataframes, axis=0)
    return concatenated_df

def filter_df(df, forecasts_col, measured_col):
    " Filter the columns of the dataframe"
    assert isinstance(forecasts_col, list), 'forecasts_col must be a list'
    assert isinstance(measured_col, str), 'measured_col must be a string'
    lst_cols = [measured_col]
    for forecast_col in forecasts_col:
        lst_cols.extend([name for name in df.columns if forecast_col in name])
    df_filtered = df[lst_cols]
    return df_filtered