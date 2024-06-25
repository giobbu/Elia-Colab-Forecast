from matplotlib import pyplot as plt
import numpy as np

def plot_elia_forecasts(df, which = 'norm_dayahead'):
    " Plot ELIA forecasts "
    assert which in ['norm_dayahead', 'norm_weekahead', 'norm_dayahead11h'], 'which not in [norm_dayahead, norm_weekahead, norm_dayahead11h]'
    list_cols = [which + 'forecast', which + 'confidence10', which + 'confidence90']
    df[list_cols].plot(color='blue', linestyle='--')
    df['diff_norm_measured'].plot(color='red')

def plot_ensemble_forecasts(df_pred_ensemble, df_ensemble):
    " Plot ensemble forecasts "
    assert 'target' in list(df_ensemble), 'target not in df_ensemble'
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    # Plot '10_predictions' and '90_predictions' on the same axes with blue dashed lines
    df_plot = df_pred_ensemble[['10_predictions', '90_predictions']]
    df_plot.columns = ['Q10', 'Q90']
    df_plot.plot(ax=ax, color='blue', linestyle='--')
    # Plot '50_predictions' on the same axes with a solid blue line
    df_plot_mean = df_pred_ensemble[['50_predictions']]
    df_plot_mean.columns = ['MEAN']
    df_plot_mean.plot(ax=ax, color='blue')
    df_target = df_ensemble[['target']]
    df_target.columns = ['target']
    df_target.plot(ax=ax, color='red')

def plot_var_ensemble_forecasts(df_pred_ensemble, df_ensemble):
    " Plot ensemble forecasts "
    assert 'target' in list(df_ensemble), 'target not in df_ensemble'
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    # Plot '10_predictions' and '90_predictions' on the same axes with blue dashed lines
    df_plot = df_pred_ensemble[['10_var_predictions', '90_var_predictions']]
    df_plot.columns = ['Q10_variability', 'Q90_variability']
    df_plot.plot(ax=ax, color='blue', linestyle='--')
    # Plot '50_predictions' on the same axes with a solid blue line
    df_plot_mean = df_pred_ensemble[['50_var_predictions']]
    df_plot_mean.columns = ['MEAN_variability']
    df_plot_mean.plot(ax=ax, color='blue')
    df_target = df_ensemble[['target']]
    df_target.columns = ['target_variability']
    df_target.plot(ax=ax, color='red')

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