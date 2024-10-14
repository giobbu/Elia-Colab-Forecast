from matplotlib import pyplot as plt
import numpy as np

def plot_elia_forecasts(df, which = 'norm_dayahead'):
    " Plot ELIA forecasts "
    assert which in ['norm_dayahead', 'norm_weekahead', 'norm_dayahead11h'], 'which not in [norm_dayahead, norm_weekahead, norm_dayahead11h]'
    list_cols = [which + 'forecast', which + 'confidence10', which + 'confidence90']
    df[list_cols].plot(color='blue', linestyle='--')
    df['diff_norm_measured'].plot(color='red')

def plot_forecasts(df_pred, df_target, list_wind_ramps, title, color='blue'):
    " Plot ensemble forecasts "
    assert 'targets' in list(df_target), 'targets not in df_target'
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(15, 7))
    # Plot '10_predictions' and '90_predictions' on the same axes with blue dashed lines
    df_plot = df_pred[['10_predictions', '90_predictions']]
    df_plot.columns = ['Q10', 'Q90']
    ax.plot(df_plot.index, df_plot['Q10'], color='blue', linestyle='--', alpha=0.1, label='Q10')
    ax.plot(df_plot.index, df_plot['Q90'], color='blue', linestyle='--', alpha=0.1, label='Q90')
    # Plot '50_predictions' on the same axes with a solid blue line
    df_plot_mean = df_pred[['50_predictions']]
    df_plot_mean.columns = ['MEAN']
    ax.plot(df_plot_mean.index, df_plot_mean['MEAN'], color='blue', alpha=0.5, label='Mean')
    ax.plot(df_target.index, df_target['targets'], color='red', label='Target')
    # fill area between Q10 and Q90
    ax.fill_between(df_plot.index, df_plot['Q10'], df_plot['Q90'], color=color, alpha=0.05, label='80% prediction interval')
    if len(list_wind_ramps) != 0:
        for i, ramp in enumerate(list_wind_ramps):
            ax.axvline(ramp, color='black', alpha=0.75, label=f'Wind Ramp {i}')
    ax.grid(True)
    ax.legend()
    plt.title(title)
    plt.show()



def plot_var_forecasts(df_pred, df_target, list_wind_ramps, title):
    " Plot ensemble forecasts "
    assert 'targets' in list(df_target), 'targets not in df_target'
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(15, 7))
    # Plot '10_predictions' and '90_predictions' on the same axes with blue dashed lines
    df_plot = df_pred[['10_var_predictions', '90_var_predictions']]
    df_plot.columns = ['Q10', 'Q90']
    ax.plot(df_plot.index, df_plot['Q10'], color='blue', linestyle='--', alpha=0.1, label='Q10')
    ax.plot(df_plot.index, df_plot['Q90'], color='blue', linestyle='--', alpha=0.1, label='Q90')
    # Plot '50_predictions' on the same axes with a solid blue line
    df_plot_mean = df_pred[['50_var_predictions']]
    df_plot_mean.columns = ['MEAN']
    ax.plot(df_plot_mean.index, df_plot_mean['MEAN'], color='blue', alpha=0.5, label='Mean')
    ax.plot(df_target.index, df_target['targets'], color='red', label='Target')
    ax.fill_between(df_plot.index, df_plot['Q10'], df_plot['Q90'], color='blue', alpha=0.05, label='80% prediction interval')
    if len(list_wind_ramps) != 0:
        for i, ramp in enumerate(list_wind_ramps):
            ax.axvline(ramp, color='black', alpha=0.75, label=f'Wind Ramp {i}')
    ax.grid(True)
    ax.legend()
    plt.title(title)
    plt.show()






def plot_ramp_events(df_test_norm_diff, ABS_DIFFERENCIATE):
    "Plot ramp events"
    if ABS_DIFFERENCIATE:
        plt.axhline(0.3, color='black', linestyle='--')
    else:
        wind_power_changes = df_test_norm_diff['diff_norm_measured'].diff().fillna(0)
        df_test_norm_diff.loc[:, 'ramp'] = (np.abs(wind_power_changes) > 0.3).astype(int)
        # Add vertical lines for ramp events
        ramp_indices = df_test_norm_diff[df_test_norm_diff['ramp'] == 1].index.values
        for idx in ramp_indices:
            plt.axvline(idx, color='k', linestyle='--')

def plot_baseline_forecasts(df, model_name):
    df.columns = ['MEAN', 'Q10', 'Q90', 'Target']
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot 'Q10' and 'Q90' on the same axes with blue dashed lines
    df_plot = df[['Q10', 'Q90']]
    df_plot.plot(ax=ax, color='blue', linestyle='--', legend=False)
    # Plot 'MEAN' on the same axes with a solid blue line
    df_plot_mean = df[['MEAN']]
    df_plot_mean.plot(ax=ax, color='blue', legend=False)
    # Plot 'Target' on the same axes with a solid red line
    df_target = df[['Target']]
    df_target.plot(ax=ax, color='red', legend=False)
    # Add a title
    plt.title(model_name)
    # Show the plot
    plt.show()

def plot_weighted_avg_forecasts(df_weighted_avg):
    df_weighted_avg.columns = ['Q10', 'MEAN', 'Q90', 'Target']
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot 'Q10' and 'Q90' on the same axes with blue dashed lines
    df_plot = df_weighted_avg[['Q10', 'Q90']]
    df_plot.plot(ax=ax, color='blue', linestyle='--', legend=False)
    # Plot 'MEAN' on the same axes with a solid blue line
    df_plot_mean = df_weighted_avg[['MEAN']]
    df_plot_mean.plot(ax=ax, color='blue', legend=False)
    # Plot 'Target' on the same axes with a solid red line
    df_target = df_weighted_avg[['Target']]
    df_target.plot(ax=ax, color='red', legend=False)
    # Add a title
    plt.title('Weighted Average Scheme')
    # Show the plot
    plt.show()