import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn import metrics
from source.ensemble.stack_generalization.ramp_detection.utils import process_ramp_events

def plot_ramp_confusion_matrix(ramp_data):
    """
    Plots the confusion matrix for the actual and predicted ramp events.

    Parameters:
    - ramp_data: DataFrame containing actual ('actual_ramps') and predicted ('predicted_ramps') ramp events.
    """
    # Replace numbers higher than 1 with 1 in 'actual_ramps'
    ramp_data['ramp_events'] = ramp_data['ramp_events'].apply(lambda x: 1 if x > 1 else x)

    # Extract the actual and predicted values
    actual_values = ramp_data['ramp_events'].values
    predicted_values = ramp_data['predicted_ramps'].values

    # Calculate the ROC curve
    fpr, tpr, _ = metrics.roc_curve(actual_values, predicted_values)

    # Calculate the confusion matrix, F1 score, and ROC AUC score
    conf_matrix = confusion_matrix(actual_values, predicted_values)
    # Extract TN, FP, FN, TP from the confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    # Compute the Critical Success Index (CSI)
    csi = tp / (tp + fn + fp)
    # Compute the Bias Score (BS)
    bs = (tp + fp) / (tp + fn)
    # Compute the F1 score and ROC AUC score
    f1 = f1_score(actual_values, predicted_values)
    roc_auc = roc_auc_score(actual_values, predicted_values)

    # Output the CSI and BS values
    print(f"Critical Success Index (CSI): {csi:.2f}")
    print(f"Bias Score (BS): {bs:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 5))
    # increase font size
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='d', cmap='Blues', linewidth=1, xticklabels=['Non-Ramp', 'Ramp'], yticklabels=['Non-Ramp', 'Ramp']) 
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.title(f'Ramp Events Metrics - F1 Score: {f1:.2f} - AUC: {roc_auc:.2f} - CSI: {csi:.2f} - BS: {bs:.2f}', fontsize=15)
    plt.show()
    return f1, roc_auc, csi, bs, fpr, tpr


def plot_boxplot_with_outliers(df_insample, df_outsample, upper):
    """
    Plots a boxplot of the 'IQW' column from df_insample and highlights the outliers 
    from df_outsample as red points.
    """
    # Plot the boxplot
    plt.figure(figsize=(15, 5))
    sns.boxplot(x=df_insample['IQW'])
    # Plot the outliers as red points
    for i in range(len(df_outsample['IQW'])):
        if df_outsample['IQW'].iloc[i] >= upper:
            plt.scatter(df_outsample['IQW'].iloc[i], 0, color='red', zorder=5)
    # Add labels and title for better readability
    plt.xlabel('IQW')
    plt.title('Boxplot of IQW with Outliers Highlighted')
    return plt.show()


def calculate_ramp_events(measurements_df, alarm_list, intraday_alarm_list, intraday):
    """
    Analyzes the ramp events within the date range specified by alarm_list.
    """
    start_date = alarm_list[0][0].date()
    end_date = alarm_list[-1][0].date()
    measurements_df, ramp_threshold = process_ramp_events(measurements_df)
    ramp_events_within_simulation_range = measurements_df[(measurements_df.index.date >= start_date) & (measurements_df.index.date <= end_date)]['ramp_events']
    # value
    if intraday:
        ramp_events_by_date = ramp_events_within_simulation_range.groupby(pd.Grouper(freq='8h')).sum().reset_index(name='ramp_events')
        # create list of values from list of tuples by appending
        ramp_events_by_date['predicted_ramps'] = [item for sublist in [list(intraday_alarm_list[i]) for i in range(len(intraday_alarm_list))] for item in sublist]
    else:
        ramp_events_by_date = ramp_events_within_simulation_range.groupby(ramp_events_within_simulation_range.index.date).sum().reset_index(name='ramp_events')
        # Add predicted ramp events from alarm_list to ramp_events_by_date
        ramp_events_by_date['predicted_ramps'] = [alarm[1] for alarm in alarm_list]
    print('Actual ramp events num:', ramp_events_by_date['ramp_events'].sum())
    print('Predicted ramp events num:', ramp_events_by_date['predicted_ramps'].sum())
    print('Total dates:', len(ramp_events_by_date))
    return ramp_events_by_date, ramp_threshold


def plot_roc_curves(list_f1_score):
    """
    Plots the ROC curves and prints the F1 scores for each entry in the provided list.
    """
    for i in range(len(list_f1_score)):
        # Extracting the components from each dictionary
        f1_score = list_f1_score[i]["f1_score"]
        fpr = list_f1_score[i]["fpr"]
        tpr = list_f1_score[i]["tpr"]
        df_ramps = list_f1_score[i]["df_ramps"]
        _, fpr, tpr = plot_ramp_confusion_matrix(df_ramps)
        # Print the F1 Score with the corresponding threshold
        print(f'F1 Scoreis {round(f1_score, 3)}')
        # Calculate the ROC AUC
        roc_auc = metrics.auc(fpr, tpr)
        # Plotting the ROC Curve
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        # Show the plot
        plt.show()

def nearest(items, pivot):
    " Find the nearest value to the pivot in a list of items."
    return min(items, key=lambda x: abs(x - pivot))

def compute_distance_nearest_ramp(df_ramps, ramp_datetimes):
    """
    Calculates the nearest ramp datetime and the absolute distance to it, finally grouping by 'cluster_id'
    and returning the mean distance for each cluster.
    """
    
    # Reset index
    df_ramps = df_ramps.reset_index()
    
    # Find the nearest ramp datetime
    df_ramps['nearest_ramp'] = df_ramps['datetime'].apply(lambda x: nearest(ramp_datetimes.to_list(), x))
    
    # Calculate the absolute distance to the nearest ramp
    df_ramps['distance'] = (df_ramps['datetime'] - df_ramps['nearest_ramp']).abs()
    
    # Compute the mean distance grouped by 'cluster_id'
    mean_distance_by_cluster = df_ramps.groupby('cluster_id')['distance'].mean()

    return pd.DataFrame(mean_distance_by_cluster)

def TP_plot_anomalous_event(df_anomalous_event, df_ramp_clusters, ramp_threshold, ramp_datetime, max_consecutive_points=5, cluster_color=False, intraday=False):
        """
        Plots the Q10-Q90 IQW range with Q50, TARG line, and additional features.
        """

        # Compute distance to nearest ramp
        df_mean_distance_by_cluster = compute_distance_nearest_ramp(df_ramp_clusters, ramp_datetime)
        # Compute Mean Distance
        mean_distance = df_mean_distance_by_cluster['distance'].mean()
        # Compute Number of Clusters
        num_clusters = len(df_mean_distance_by_cluster)

        # Create a figure and axis
        _, ax = plt.subplots(figsize=(20, 10))
        # Fill the area between Q10 and Q90
        ax.fill_between(df_anomalous_event.index, 
                        df_anomalous_event['Q10'], 
                        df_anomalous_event['Q90'], 
                        color='blue', alpha=0.05, label='Q10-Q90 IQW')
        # Plot Q50 with a dashed blue line
        ax.plot(df_anomalous_event.index, df_anomalous_event['Q50'], 
                color='blue', linestyle='--', label='Q50')
        # Plot TARG with a solid red line
        ax.plot(df_anomalous_event.index, df_anomalous_event['TARG'], 
                color='red', label='TARG')
        
        # for each cluster, add fill_between
        i=1
        for cluster_id, df_cluster in df_ramp_clusters.groupby('cluster_id'):
            df_mean_distance_by_cluster['distance'] = df_mean_distance_by_cluster['distance'].astype(str)
            distance = df_mean_distance_by_cluster[df_mean_distance_by_cluster.index == cluster_id]['distance'].values
            if df_cluster['consecutive_count'].max() >= max_consecutive_points:
                if cluster_color:
                    ax.fill_between(df_cluster.index, 
                                    df_cluster['Q90'], 
                                    df_cluster['Q10'], 
                                    # different color for each cluster
                                    color = plt.get_cmap('tab20')(cluster_id), 
                                    alpha=0.5, label= f'{cluster_id} - Distance: {distance[0]}')
                else:
                    ax.fill_between(df_cluster.index, 
                                    df_cluster['Q90'], 
                                    df_cluster['Q10'], 
                                    color='red', alpha=0.3, label= f'Ramp Event {i} - Distance: {distance[0]}')
                    i+=1

        ax.set_title(f'Q10-Q90 IQW Range with Q50 and TARG Line - {df_anomalous_event.index.date[0]} - Mean Distance : {mean_distance} - Num Clusters : {num_clusters}')
        # Add a horizontal line with ramp threshold
        ax.axhline(ramp_threshold, color='black', linestyle='--', label='Ramp Threshold')
        ax.axhline(-ramp_threshold, color='black', linestyle='--')
        # Add a vertical line at the anomalous event
        if ramp_datetime.size > 0:
            for ramp in ramp_datetime:
                ax.axvline(ramp, color='black', label='Anomalous Event')
        
        if intraday:
            # plot # vertical blue  dashed lines every 8 hours (32 points) time period
            hours = 0
            num_clusters = 0
            mean_distance = pd.Timedelta(0)
            for i in range(0, len(df_anomalous_event), 32):
                ax.axvline(df_anomalous_event.index[i], color='blue', linestyle='--', label=f'{hours} hours')
                hours += 8
                # fill all the area between the vertical blue 8-hours lines if a ramp event is detected and a cluster is formed
                if ramp_datetime.size > 0:
                    for ramp in ramp_datetime:
                        if i < len(df_anomalous_event) - 32:
                            if ramp >= df_anomalous_event.index[i] and ramp <= df_anomalous_event.index[i+32]:
                                # cluster is present in the interval, fill the area between the vertical blue 8-hours lines
                                if df_ramp_clusters[(df_ramp_clusters.index >= df_anomalous_event.index[i]) & (df_ramp_clusters.index <= df_anomalous_event.index[i+32])].shape[0] > 0:
                                    ax.axvspan(df_anomalous_event.index[i], df_anomalous_event.index[i+32], color='green', alpha=0.1)

                                    # filter df_ramp_clusters
                                    df_ramp_clusters_intraday = df_ramp_clusters[(df_ramp_clusters.index >= df_anomalous_event.index[i]) & (df_ramp_clusters.index <= df_anomalous_event.index[i+32])]
                                    df_mean_distance_by_cluster_intraday = compute_distance_nearest_ramp(df_ramp_clusters_intraday, ramp_datetime)
                                    mean_distance = df_mean_distance_by_cluster_intraday['distance'].mean()
                                    num_clusters = len(df_mean_distance_by_cluster_intraday)

                                else:
                                    ax.axvspan(df_anomalous_event.index[i], df_anomalous_event.index[i+32], color='red', alpha=0.1)
                        else:
                            if ramp >= df_anomalous_event.index[i]:
                                # cluster is present in the interval, fill the area between the vertical blue 8-hours lines
                                if df_ramp_clusters[(df_ramp_clusters.index >= df_anomalous_event.index[i])].shape[0] > 0:
                                    ax.axvspan(df_anomalous_event.index[i], df_anomalous_event.index[-1], color='green', alpha=0.1)

                                    # filter df_ramp_clusters
                                    df_ramp_clusters_intraday = df_ramp_clusters[(df_ramp_clusters.index >= df_anomalous_event.index[i])]
                                    df_mean_distance_by_cluster_intraday = compute_distance_nearest_ramp(df_ramp_clusters_intraday, ramp_datetime)
                                    mean_distance = df_mean_distance_by_cluster_intraday['distance'].mean()
                                    num_clusters = len(df_mean_distance_by_cluster_intraday)

                                else:
                                    ax.axvspan(df_anomalous_event.index[i], df_anomalous_event.index[-1], color='red', alpha=0.1)
            ax.axvline(df_anomalous_event.index[-1], color='blue', linestyle='--', label=f'24 hours')

        # Set the x-axis and y-axis labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Wind Power Variability')
        # Add a title to the plot with date range
        # Add a legend to identify the lines
        ax.legend()
        # Add grid lines to the plot
        ax.grid(True)
        # Display the plot
        plt.show()
        plt.close()

        return df_ramp_clusters, mean_distance, num_clusters

def FP_plot_anomalous_event(df_anomalous_event, df_ramp_clusters, ramp_threshold, ramp_datetime, max_consecutive_points=5, cluster_color=False):
        """
        Plots the Q10-Q90 IQW range with Q50, TARG line, and additional features.
        """
        # Create a figure and axis
        _, ax = plt.subplots(figsize=(20, 10))

        # Fill the area between Q10 and Q90
        ax.fill_between(df_anomalous_event.index, 
                        df_anomalous_event['Q10'], 
                        df_anomalous_event['Q90'], 
                        color='blue', alpha=0.05, label='Q10-Q90 IQW')
        
        # Plot Q50 with a dashed blue line
        ax.plot(df_anomalous_event.index, df_anomalous_event['Q50'], 
                color='blue', linestyle='--', label='Q50')
        
        # Plot TARG with a solid red line
        ax.plot(df_anomalous_event.index, df_anomalous_event['TARG'], 
                color='red', label='TARG')


        # for each cluster, add fill_between
        i=1
        for cluster_id, df_cluster in df_ramp_clusters.groupby('cluster_id'):

            if df_cluster['consecutive_count'].max() >= max_consecutive_points:

                if cluster_color:
                    ax.fill_between(df_cluster.index, 
                                    df_cluster['Q90'], 
                                    df_cluster['Q10'], 
                                    # different color for each cluster
                                    color = plt.get_cmap('tab20')(cluster_id), 
                                    alpha=0.5, label= f'{cluster_id}')
                else:
                    ax.fill_between(df_cluster.index, 
                                    df_cluster['Q90'], 
                                    df_cluster['Q10'], 
                                    color='red', alpha=0.3, label= f'Ramp Event {i}')
                    i+=1
                
        ax.set_title(f'Q10-Q90 IQW Range with Q50 and TARG Line - {df_anomalous_event.index.date[0]}')

        # Add a horizontal line with ramp threshold
        ax.axhline(ramp_threshold, color='black', linestyle='--', label='Ramp Threshold')
        ax.axhline(-ramp_threshold, color='black', linestyle='--')
        # Add a vertical line at the anomalous event
        if ramp_datetime.size > 0:
            ax.axvline(ramp_datetime, color='black', label='Anomalous Event')
        # Set the x-axis and y-axis labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Wind Power Variability')
        # Add a title to the plot with date range
        # Add a legend to identify the lines
        ax.legend()
        # Add grid lines to the plot
        ax.grid(True)
        # Display the plot
        plt.show()
        plt.close()
        return df_ramp_clusters

def plot_iqw_with_bound(df_anomalous_event, upper_box_bound, ramp_datetime):
        """
        Plots the IQW data with an upper bound and highlights anomalies.
        """
        # Create a figure and axis
        _, ax = plt.subplots(figsize=(20, 10))
        # Plot the 'IQW' data
        ax.plot(df_anomalous_event.index, df_anomalous_event['IQW'], label='IQW')
        # Add a horizontal line at the upper_box_bound with a red dashed line
        ax.axhline(upper_box_bound, color='r', linestyle='--', label='Upper Bound')
        # Fill the area above the horizontal line (upper_box_bound)
        ax.fill_between(df_anomalous_event.index, 
                        df_anomalous_event['IQW'], 
                        upper_box_bound, 
                        where=(df_anomalous_event['IQW'] > upper_box_bound),
                        color='red', alpha=0.3, label='Above Upper Bound')
        # Add a vertical line at the anomalous event
        ax.axvline(ramp_datetime, color='black', label='Anomalous Event')
        # Set the x-axis and y-axis labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Inter Quantile Width')
        # Add a title to the plot
        ax.set_title('IQW vs. Upper Bound with Highlighted Anomalies')
        # Add a legend to identify the lines
        ax.legend()
        # Add grid lines to the plot
        ax.grid(True)
        # Display the plot
        plt.show()

def plot_ramp_events(ramp_events_by_date, list_ramp_alarm_days, ramp_threshold, plot_results='TP', max_consecutive_points=5, plot_prediction=True, plot_iqw=True, cluster_color=False, intraday=False):
        " Plot Ramp Events"
        # True Positive and False Positive Ramp Events
        if plot_results == 'TP' :
            _ramp_events_by_date = ramp_events_by_date[(ramp_events_by_date.ramp_events == 1) & (ramp_events_by_date.predicted_ramps == 1)]
        elif plot_results == 'FP':
            _ramp_events_by_date = ramp_events_by_date[(ramp_events_by_date.ramp_events == 0) & (ramp_events_by_date.predicted_ramps == 1)]
        else:
            raise ValueError('Invalid plot_results value. Choose between TP or FP')
        list_df_anomalous = []
        list_mean_distance = []
        for i in range(len(_ramp_events_by_date)):
            date_ramp = pd.to_datetime(_ramp_events_by_date['index'].iloc[i], utc=True)
            df_anomalous_event, df_ramp_clusters = next(
                                                    (df[0], df[1]) for df in list_ramp_alarm_days if date_ramp in df[0].index
                                                    )
            ramp_datetime = df_anomalous_event[df_anomalous_event['TARG'].abs() >= ramp_threshold].index
            num_ramps = len(ramp_datetime.to_list())
            if plot_prediction:
                if plot_results == 'TP' :
                    df_anomalous, mean_distance, num_clusters = TP_plot_anomalous_event(df_anomalous_event, 
                                                                                        df_ramp_clusters, 
                                                                                        ramp_threshold, 
                                                                                        ramp_datetime, 
                                                                                        max_consecutive_points=max_consecutive_points, 
                                                                                        cluster_color=cluster_color,
                                                                                        intraday=intraday)
                    list_df_anomalous.append((df_anomalous, ramp_datetime))
                    list_mean_distance.append((mean_distance, num_ramps, num_clusters))                
                else:
                    df_anomalous = FP_plot_anomalous_event(df_anomalous_event, 
                                                            df_ramp_clusters, 
                                                            ramp_threshold, 
                                                            ramp_datetime, 
                                                            max_consecutive_points=max_consecutive_points, 
                                                            cluster_color=cluster_color)
                    list_df_anomalous.append(df_anomalous)
        if plot_iqw:
            plot_iqw_with_bound(df_anomalous_event, df_ramp_clusters, ramp_datetime)
        return list_df_anomalous, list_mean_distance

        