from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90

def first_stage_compute_permuted_score(predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, base_score):
    " Compute the permuted score for a single predictor."
    X_test_permuted = X_test_augmented.copy()
    X_test_permuted[:, predictor_index] = np.random.permutation(X_test_augmented[:, predictor_index])
    permuted_score = score_functions[quantile](fitted_model, X_test_permuted, y_test)['mean_pinball_loss']
    return max(0.0, permuted_score - base_score)

def first_stage_permutation_importance(num_permutations, quantile, fitted_model, X_test_augmented, y_test, df_train_ensemble_augmented):
    " Compute permutation importances for the first stage model."
    assert num_permutations > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    # Define the score functions for different quantiles
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    # Compute the original score
    base_score = score_functions[quantile](fitted_model, X_test_augmented, y_test)['mean_pinball_loss']
    importance_scores = []
    # Loop through each predictor
    for predictor_index in range(X_test_augmented.shape[1]):
        predictor_name = df_train_ensemble_augmented.drop(columns=['diff_norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        permuted_scores = Parallel(n_jobs=4)(delayed(first_stage_compute_permuted_score)(
            predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, base_score
        ) for _ in range(num_permutations))
        # Calculate mean contribution for the predictor
        mean_contribution = np.mean(permuted_scores)
        importance_scores.append({'predictor': predictor_name, 'contribution': mean_contribution})
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores).sort_values(by='contribution', ascending=False)
    return results_df