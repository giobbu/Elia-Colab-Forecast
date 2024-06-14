from matplotlib import pyplot as plt
import pandas as pd

def plot_weight_avg_contributions(dict_weights, quantile, stage, days):
    "Plot the normalized contributions of the predictors."
    # Flatten the list into a single dictionary
    weight_data = {k: v for d in dict_weights[quantile] for k, v in d.items()}
    # Convert the dictionary into a DataFrame
    results_df = pd.DataFrame(list(weight_data.items()), columns=['predictor', 'norm_contribution'])
    results_df['predictor'] = results_df['predictor'].str.replace('diff_norm_', '')
    # Plot the normalized contributions
    results_df.plot.bar(x='predictor', y='norm_contribution', rot=45)
    plt.xlabel('Predictor')
    plt.ylabel('Normalized Contribution')
    plt.title(f'Quantile: {quantile} - Stage: {stage} - Performance Days: {days}')
    plt.suptitle('Weighted Avg Scheme - Contributions based on past performance')
    plt.show()
