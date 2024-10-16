import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_append_results(file_paths, key_names):
    """
    Load data from the provided pickle file paths and extract the required keys.
    """
    result_list = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        result_list += [{key: entry[key] for key in key_names} for entry in data]
    return result_list

def plot_heatmap(data_list, value_col, title, cmap=sns.diverging_palette(10, 133, as_cmap=True)):
    """
    Converts a list of dictionaries to a DataFrame, pivots the table, and plots a heatmap.
    """
    # Convert list of dictionaries to DataFrame
    df_results = pd.DataFrame(data_list)
    # Determine column names based on value_col
    columns = df_results.columns.tolist()
    columns[columns.index(value_col)] = value_col
    df_results.columns = columns
    # Pivot table
    df_results_pivot = df_results.pivot(index='n_neighbors', columns='contamination', values=value_col)
    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_results_pivot, annot=True, fmt=".2f", cmap=cmap)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title(title)
    plt.show()