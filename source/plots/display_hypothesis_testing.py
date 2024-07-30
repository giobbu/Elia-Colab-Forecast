from matplotlib import pyplot as plt
import scikit_posthocs as sp
import pandas as pd

def transform_loss_lists_to_df(model_type, 
                                lst_loss_ensemble, 
                                lst_loss_equal_weights, 
                                lst_loss_weighted_avg, 
                                lst_loss_weighted_avg_soft,
                                lst_loss_baseline_dayahead, 
                                lst_loss_baseline_dayahead11h, 
                                lst_loss_baseline_weekahead, 
                                lst_loss_baseline_mostrecent=None,
                                lst_loss_baseline_cheat=None,
                                lst_loss_baseline_noisy=None):
    " Transform the loss lists into a DataFrame"
    assert len(lst_loss_ensemble) == len(lst_loss_equal_weights) == len(lst_loss_weighted_avg) == len(lst_loss_baseline_dayahead) == len(lst_loss_baseline_dayahead11h) == len(lst_loss_baseline_weekahead), 'Length mismatch'
        
    # Construct the dictionary from the input lists
    dict_data = {
        f'{model_type}_ensemble': lst_loss_ensemble,
        'eq_weights': lst_loss_equal_weights,
        'weighted_avg': lst_loss_weighted_avg,
        'weighted_avg_soft': lst_loss_weighted_avg_soft,
        'dayahead': lst_loss_baseline_dayahead,
        'dayahead11h': lst_loss_baseline_dayahead11h,
        'weekahead': lst_loss_baseline_weekahead,
    }
    if lst_loss_baseline_mostrecent is not None:
        dict_data['mostrecent'] = lst_loss_baseline_mostrecent
    if lst_loss_baseline_cheat is not None:
        dict_data['malicious'] = lst_loss_baseline_cheat
    if lst_loss_baseline_noisy is not None:
        dict_data['noisy'] = lst_loss_baseline_noisy
    
    # Transform the dictionary into a DataFrame
    data = (
        pd.DataFrame(dict_data)
        .rename_axis('days')  # Set the index name to 'days'
        .melt(                # Melt the DataFrame to long format
            var_name='model',
            value_name='loss',
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


def run_statistical_comparison_analysis(model_type,
                                        lst_gbr_ensemble, 
                                        lst_equal_weights, 
                                        lst_weighted_avg,
                                        lst_weighted_avg_soft, 
                                        lst_baseline_dayahead, 
                                        lst_baseline_dayahead11h, 
                                        lst_baseline_week_ahead,
                                        lst_baseline_most_recent,
                                        lst_baseline_malicious,
                                        lst_baseline_noisy,
                                        title1, title2):
    """
    Function to transform loss lists into a DataFrame, calculate average ranks, perform statistical tests,
    and plot the results.
    """
    # Transform loss lists into a DataFrame
    data = transform_loss_lists_to_df(model_type, 
                                            lst_gbr_ensemble, 
                                            lst_equal_weights, 
                                            lst_weighted_avg,
                                            lst_weighted_avg_soft,
                                            lst_baseline_dayahead, 
                                            lst_baseline_dayahead11h, 
                                            lst_baseline_week_ahead,
                                            lst_baseline_most_recent,
                                            lst_baseline_malicious,
                                            lst_baseline_noisy)
    # Calculate average ranks
    avg_rank = data.groupby('days').loss.rank(pct=True).groupby(data.model).mean()
    # Perform posthoc Nemenyi Friedman test
    pc = sp.posthoc_nemenyi_friedman(data, y_col='loss', block_col='days', group_col='model', melted=True)
    # Plot the statistical comparison
    plot_statistical_comparison(pc, avg_rank,
                                title1=title1,
                                title2=title2)
    return data, avg_rank