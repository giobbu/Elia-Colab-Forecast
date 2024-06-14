from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_augmented_dataframe_2stage

def second_stage_permuted_score(predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, base_score, df_train_ensemble, df_test_ensemble, y_train, predictions_insample, order_diff, max_lags_var, augment_var, start_prediction_timestamp, var_fitted_model):
    "Compute the permuted score for a single predictor in the second stage model."
    X_test_permuted = X_test_augmented.copy()
    X_test_permuted[:, predictor_index] = np.random.permutation(X_test_augmented[:, predictor_index])
    permuted_predictions_outsample = fitted_model.predict(X_test_permuted)
    df_2stage_permuted = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, permuted_predictions_outsample)
    df_2stage_processed_permuted = create_augmented_dataframe_2stage(df_2stage_permuted, order_diff, max_lags=max_lags_var, augment=augment_var)
    df_2stage_test_permuted = df_2stage_processed_permuted[df_2stage_processed_permuted.index >= start_prediction_timestamp]
    X_test_2stage_permuted, y_test_2stage_permuted = df_2stage_test_permuted.drop(columns=['targets']).values, df_2stage_test_permuted['targets'].values
    permutation_score = score_functions[quantile](var_fitted_model, X_test_2stage_permuted, y_test_2stage_permuted)['mean_pinball_loss']
    return max(0.0, permutation_score - base_score)

def second_stage_permutation_importance(num_permutations, quantile, var_fitted_model, fitted_model, X_test_augmented, y_test, df_train_ensemble_augmented,
                                        X_train_augmented, df_train_ensemble, df_test_ensemble, y_train, order_diff, max_lags_var, augment_var, start_prediction_timestamp):
    "Compute permutation importances for the second stage model."
    # Define the score functions for different quantiles
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    # Generate predictions from the first-stage model
    predictions_insample = fitted_model.predict(X_train_augmented)
    predictions_outsample = fitted_model.predict(X_test_augmented)
    # Create and preprocess the two-stage dataframe
    df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble, y_train, y_test, predictions_insample, predictions_outsample)
    df_2stage_processed = create_augmented_dataframe_2stage(df_2stage, order_diff, max_lags=max_lags_var, augment=augment_var)
    # Split the processed dataframe into test sets
    df_2stage_test = df_2stage_processed[df_2stage_processed.index >= start_prediction_timestamp]
    X_test_2stage, y_test_2stage = df_2stage_test.drop(columns=['targets']).values, df_2stage_test['targets'].values
    # Compute the original score
    base_score = score_functions[quantile](var_fitted_model, X_test_2stage, y_test_2stage)['mean_pinball_loss']
    importance_scores = []
    # Loop through each predictor
    for predictor_index in range(X_test_augmented.shape[1]):
        predictor_name = df_train_ensemble_augmented.drop(columns=['diff_norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        permuted_scores = Parallel(n_jobs=-1)(delayed(second_stage_permuted_score)(
            predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile, base_score,
            df_train_ensemble, df_test_ensemble, y_train, predictions_insample, order_diff, max_lags_var,
            augment_var, start_prediction_timestamp, var_fitted_model
        ) for _ in range(num_permutations))
        # Calculate mean contribution for the predictor
        mean_contribution = np.mean(permuted_scores)
        importance_scores.append({'predictor': predictor_name, 'contribution': mean_contribution})
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores).sort_values(by='contribution', ascending=False)
    return results_df