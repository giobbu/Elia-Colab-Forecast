import pandas as pd

def calculate_and_concatenate_stats(dfs, group_by_col, calc_col, prefixes):
    " Calculate mean and standard deviation for multiple DataFrames and concatenate the results"
    result_list = []
    for df, prefix in zip(dfs, prefixes):
        # Calculate the mean and standard deviation of the specified column grouped by the group_by_col
        df_stats = df.groupby(group_by_col)[calc_col].agg(['mean', 'std'])
        # Rename the columns to the desired names
        df_stats.columns = [f'mean_{calc_col}', 
                            f'std_{calc_col}']
        # Create a MultiIndex for the columns
        df_stats.columns = pd.MultiIndex.from_product([[prefix], df_stats.columns])
        result_list.append(df_stats)
    # Concatenate all results
    df_combined = pd.concat(result_list, axis=1)
    return df_combined

def highlight_min(s):
    "Highlight the minimum value in a Series."
    is_min = s == s.min()
    return ['background-color: green' if v else '' for v in is_min]

def highlight_max(s):
    "Highlight the minimum value in a Series."
    is_max = s == s.max()
    return ['background-color: red' if v else '' for v in is_max]

def display_table_metrics(dfs, prefixes):
    " Display the mean and standard deviation of the specified columns of quantiles"
    result = calculate_and_concatenate_stats(dfs, 'model', 'loss', prefixes)
    # Apply highlighting to the resulting DataFrame
    styled_result = result.style.apply(highlight_min, subset=pd.IndexSlice[:, :])
    styled_result = styled_result.apply(highlight_max, subset=pd.IndexSlice[:, :])
    return result, styled_result
