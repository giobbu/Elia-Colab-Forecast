import pandas as pd
from source.ensemble.stack_generalization.test_importance.first_stage_importance import first_stage_compute_permuted_score, first_stage_permutation_importance

def test_first_stage_compute_permuted_score(data_first_stage_importance):
    "Test first_stage_compute_permuted_score function."
    _, X_test_augmented, y_test, _, fitted_model, score_functions, base_score, quantile = data_first_stage_importance
    predictor_index = 2  # Arbitrary feature index to permute
    permuted_score_diff = first_stage_compute_permuted_score(
        predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, base_score
    )
    assert isinstance(permuted_score_diff, float), "Output should be a float"
    assert permuted_score_diff >= 0, "Permuted score difference should be non-negative"

def test_first_stage_permutation_importance(data_first_stage_importance):
    "Test first_stage_permutation_importance function."
    num_permutations, X_test_augmented, y_test, df_train_ensemble_augmented, fitted_model, _, _, quantile = data_first_stage_importance
    results_df = first_stage_permutation_importance(
        num_permutations, quantile, fitted_model, X_test_augmented, y_test, df_train_ensemble_augmented
    )
    assert isinstance(results_df, pd.DataFrame), "Output should be a DataFrame"
    assert 'predictor' in results_df.columns, "DataFrame should contain 'predictor' column"
    assert 'contribution' in results_df.columns, "DataFrame should contain 'contribution' column"
    assert results_df.shape[0] == X_test_augmented.shape[1], "DataFrame should have the same number of rows as predictors"
    assert results_df['contribution'].min() >= 0, "Contribution scores should be non-negative"
    assert all(results_df['contribution'].sort_values(ascending=False) == results_df['contribution']), "Contribution scores should be in ascending order"
