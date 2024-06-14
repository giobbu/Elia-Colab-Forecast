from matplotlib import pyplot as plt

def plot_normalized_contributions(num_permutations, quantile, results_df, stage):
    "Plot the normalized contributions of the predictors."
    assert 'predictor' in results_df.columns
    assert 'contribution' in results_df.columns
    assert num_permutations > 0, "Number of permutations must be greater than 0"
    assert 0 <= quantile <= 1, "Quantile must be between 0 and 1"
    assert 0 < results_df['contribution'].sum() , "Sum of contributions must be greater than 0 "
    assert results_df['contribution'].sum()<=1.01 , "Sum of contributions must be lower or equal to 1"
    # Replace 'norm_' in predictor names
    results_df['predictor'] = results_df['predictor'].str.replace('norm_', '')
    results_df.columns = ['predictor', 'norm_contribution']
    # Plot the normalized contributions
    results_df.plot.bar(x='predictor', y='norm_contribution', rot=45)
    plt.xlabel('Predictor')
    plt.ylabel('Normalized Contribution')
    if quantile == 0.5:
        plt.title(f'Num of Permutations: {num_permutations} - Stage: {stage}')
        plt.suptitle(f'Normalized Contributions for {stage} for mean forecast')
        return plt.show()
    plt.title(f'Num of Permutations: {num_permutations} - Quantile: {quantile} - Stage: {stage}')
    plt.suptitle(f'Contributions for Predicted Day')
    return plt.show()