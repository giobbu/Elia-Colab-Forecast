import pandas as pd
from unittest.mock import patch
from source.ensemble.combination_scheme.weight_avg_plot_importance import plot_weight_avg_contributions


def test_plot_weight_avg_contributions(dict_importance_weights):
    quantile = 0.5
    stage = 'Stage 1'
    days = 30
    with patch('source.ensemble.combination_scheme.weight_avg_plot_importance.plt.show') as mock_show:
        # Call the function
        plot_weight_avg_contributions(dict_importance_weights, quantile, stage, days)
        # Check that plt.show() was called
        mock_show.assert_called_once()
    # Additional checks can include verifying the DataFrame content if necessary
    weight_data = {k: v for d in dict_importance_weights[quantile] for k, v in d.items()}
    results_df = pd.DataFrame(list(weight_data.items()), columns=['predictor', 'norm_contribution'])
    results_df['predictor'] = results_df['predictor'].str.replace('diff_norm_', '')
    assert results_df['predictor'].tolist() == ['feature1', 'feature2']
    assert results_df['norm_contribution'].tolist() == [0.2, 0.8]  # This is based on the sample data given

