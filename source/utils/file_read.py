import pandas as pd

def read_csv_file(csv_filename, columns, starting_period, ending_period):
    """
    Read csv file and return dataframe
    """
    df_csv = pd.read_csv(csv_filename)
    df_csv['datetime'] = pd.to_datetime(df_csv['datetime'])
    df_offshore = df_csv[df_csv['offshoreonshore'] == 'Offshore'].set_index('datetime')
    df_offshore_forecasters = df_offshore[columns]
    df_filtered = df_offshore_forecasters[df_offshore_forecasters.index.to_series().between(starting_period, ending_period)]
    return df_filtered


def set_index_datetiemUTC(df):
    assert 'datetime' in df.columns, "The DataFrame must contain the 'datetime' column."
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values(by='datetime').reset_index(drop=True)
    df.set_index('datetime', inplace=True)
    return df

def filter_offshore(df, offshore_filter):
    assert 'offshoreonshore' in df.columns, "The DataFrame must contain the 'offshoreonshore' column."
    return df[(df.offshoreonshore == offshore_filter)|(df.offshoreonshore == 0)]  # filter by offshore 

def process_file(file, offshore_filter):
    assert file.endswith('.json'), 'File must be a json file'
    df = pd.read_json(file)
    df = filter_offshore(df, offshore_filter)
    df = set_index_datetiemUTC(df)
    df[df['measured'] < 0] = 0
    return df

def process_and_concat_files(files, offshore_filter="Offshore"):
    "Process and concatenate files"
    assert len(files) > 0, 'No files to process'
    dataframes = []
    for file in files:
        df = process_file(file, offshore_filter)
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