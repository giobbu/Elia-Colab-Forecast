from matplotlib import pyplot as plt
import scikit_posthocs as sp
import pandas as pd

def transform_loss_lists_to_df(model_type, lst_loss_ensemble, lst_loss_equal_weights, lst_loss_weighted_avg, lst_loss_baseline_dayahead, lst_loss_baseline_dayahead11h, lst_loss_baseline_weekahead):
    " Transform the loss lists into a DataFrame"
    assert len(lst_loss_ensemble) == len(lst_loss_equal_weights) == len(lst_loss_weighted_avg) == len(lst_loss_baseline_dayahead) == len(lst_loss_baseline_dayahead11h) == len(lst_loss_baseline_weekahead), 'Length mismatch'
        
    # Construct the dictionary from the input lists
    dict_data = {
        f'{model_type}_ensemble': lst_loss_ensemble,
        'eq_weights': lst_loss_equal_weights,
        'weighted_avg': lst_loss_weighted_avg,
        'dayahead': lst_loss_baseline_dayahead,
        'dayahead11h': lst_loss_baseline_dayahead11h,
        'weekahead': lst_loss_baseline_weekahead,
    }
    # Transform the dictionary into a DataFrame
    data = (
        pd.DataFrame(dict_data)
        .rename_axis('days')  # Set the index name to 'days'
        .melt(                # Melt the DataFrame to long format
            var_name='model',
            value_name='rmse',
            ignore_index=False,
        )
        .reset_index()        # Reset the index to include 'days' as a column
    )
    return data

def plot_statistical_comparison(pc, avg_rank, title1, title2):
    " Plot the statistical comparison"
    # Define the colormap and heatmap arguments
    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {
        'cmap': cmap,
        'linewidths': 0.25,
        'linecolor': '0.5',
        'clip_on': False,
        'square': True,
        'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]
    }
    # Plot the heatmap
    plt.title(title1)
    sp.sign_plot(pc, **heatmap_args)
    plt.show()
    # Plot the title for the critical difference diagram
    plt.title(title2)
    # Plot the critical difference diagram
    sp.critical_difference_diagram(avg_rank, pc)
    plt.show()