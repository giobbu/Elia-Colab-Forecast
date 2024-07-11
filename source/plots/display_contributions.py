import pandas as pd

def weighted_avg_pivot_data(sim_params, avg_weighted_avg_contributions):
    """
    Collect data for all quantiles into a DataFrame, pivot it, and reorder the columns.
    """
    # Initialize an empty list to collect data
    all_data = []
    # Collect data for all quantiles into a DataFrame
    for key in avg_weighted_avg_contributions.keys():
        for quantile in avg_weighted_avg_contributions[key].keys():
            data = avg_weighted_avg_contributions[key][quantile]
            for series, value in data.items():
                q_str = str(int(quantile*100))
                all_data.append([key, f'Q{q_str}', series, value])
    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['Key', 'Quantile', 'Series', 'Value'])
    # Clean the Series column
    df.Series = df.Series.str.replace('norm_', '').str.replace('forecast', '').str.replace('confidence10', '').str.replace('confidence90', '')
    # Pivot the DataFrame to get the right structure for a stacked bar chart
    df_pivot = df.pivot_table(index=['Key', 'Quantile'], columns='Series', values='Value', fill_value=0)
    # Rename the columns
    df_pivot.rename(columns={'dayahead': 'Day-ahead', 
                                'dayahead11h': 'Day-ahead-11h', 
                                'weekahead': 'Week-ahead', 
                                'mostrecent': 'Most-recent',
                                'malicious': 'Malicious',
                                'noisy': 'Noisy'}, inplace=True)
    # Reorder the columns
    list_names = ['Day-ahead', 'Day-ahead-11h', 'Week-ahead']
    if sim_params.get('most_recent'): list_names.append('Most-recent')
    if sim_params.get('malicious'): list_names.append('Malicious')
    if sim_params.get('noisy'): list_names.append('Noisy')
    df_pivot = df_pivot[list_names]
    return df_pivot


def permutation_pivot_data(sim_params, avg_permutation_contributions):
    """
    Collect data for all quantiles into a DataFrame, pivot it, and reorder the columns.
    """
    # Initialize an empty DataFrame
    all_data = []
    # Collect data for all quantiles into a DataFrame
    for key in avg_permutation_contributions.keys():
        for quantile in avg_permutation_contributions[key].keys():
            data = avg_permutation_contributions[key][quantile]
            for series, value in data.items():
                q_str = str(int(quantile*100))
                all_data.append([key, f'Q{q_str}', series, value])

    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['Key', 'Quantile', 'Series', 'Value'])
    # Pivot the DataFrame to get the right structure for a stacked bar chart
    df_pivot = df.pivot_table(index=['Key', 'Quantile'], columns='Series', values='Value', fill_value=0)
    # Rename the columns
    df_pivot.rename(columns={'s1': 'Day-ahead',
                                's2': 'Day-ahead-11h', 
                                's3': 'Week-ahead', 
                                's4': 'Most-recent',
                                's5': 'Malicious',
                                's6': 'Noisy'}, 
                                inplace=True)
    list_names = ['Day-ahead', 'Day-ahead-11h', 'Week-ahead']
    if sim_params['most_recent']: list_names.append('Most-recent')
    if sim_params['malicious']: list_names.append('Malicious')
    if sim_params['noisy']: list_names.append('Noisy')
    df_pivot = df_pivot[list_names]
    return df_pivot


def lasso_coefs_pivot_data(sim_params, avg_coefficients_contributions):
    " Collect data for all quantiles into a DataFrame, pivot it, and reorder the columns."
    # Initialize an empty DataFrame
    all_data = []
    # Collect data for all quantiles into a DataFrame
    for key in avg_coefficients_contributions.keys():
        for quantile in avg_coefficients_contributions[key].keys():
            data = avg_coefficients_contributions[key][quantile]
            for series, value in data.items():
                q_str = str(int(quantile*100))
                all_data.append([key, f'Q{q_str}', series, value])
    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['Key', 'Quantile', 'Series', 'Value'])
    # Pivot the DataFrame to get the right structure for a stacked bar chart
    df_pivot = df.pivot_table(index=['Key', 'Quantile'], columns='Series', values='Value', fill_value=0)
    df_pivot.rename(columns={'s1': 'Day-ahead', 
                            's2': 'Day-ahead-11h', 
                            's3': 'Week-ahead', 
                            's4': 'Most-recent',
                            's5': 'Malicious',
                            's6': 'Noisy'}, 
                            inplace=True)
    list_names = ['Day-ahead', 'Day-ahead-11h', 'Week-ahead']
    if sim_params['most_recent']: list_names.append('Most-recent')
    if sim_params['malicious']: list_names.append('Malicious')
    if sim_params['noisy']: list_names.append('Noisy')
    df_pivot = df_pivot[list_names]
    return df_pivot