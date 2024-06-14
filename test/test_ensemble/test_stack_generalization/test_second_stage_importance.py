import pandas as pd
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10
from source.ensemble.stack_generalization.test_importance.second_stage_importance import second_stage_permuted_score, second_stage_permutation_importance

def test_second_stage_permuted_score(data_second_stage_importance):
    "Test second_stage_permuted_score function."
    data = data_second_stage_importance
    quantile = data["quantile"]
    y_train = data["y_train"]
    X_test_augmented = data["X_test_augmented"]
    y_test = data["y_test"]
    df_train_ensemble = data["df_train_ensemble"]
    df_test_ensemble = data["df_test_ensemble"]
    predictions_insample = data["predictions_insample"]
    start_prediction_timestamp = data["start_prediction_timestamp"]
    order_diff = data["order_diff"]
    max_lags_var = data["max_lags_var"]
    augment_var = data["augment_var"]
    var_fitted_model = data["var_fitted_model"]
    fitted_model = data["fitted_model"]
    # Compute the base score
    score_functions = {
        quantile: score_func_10
    }
    base_score = score_functions[0.1](var_fitted_model, X_test_augmented, y_test)['mean_pinball_loss']
    # Test the second_stage_permuted_score function
    permutation_score = second_stage_permuted_score(
        predictor_index=0, X_test_augmented=X_test_augmented, y_test=y_test, fitted_model=fitted_model,
        score_functions=score_functions, quantile=quantile, base_score=base_score, 
        df_train_ensemble=df_train_ensemble, df_test_ensemble=df_test_ensemble, 
        y_train=y_train, predictions_insample=predictions_insample, 
        order_diff=order_diff, max_lags_var=max_lags_var, augment_var=augment_var, 
        start_prediction_timestamp=start_prediction_timestamp, var_fitted_model=var_fitted_model
    )
    assert isinstance(permutation_score, float), "Output should be a float"
    assert permutation_score >= 0, "Permutation score should be non-negative"

def test_second_stage_permutation_importance(data_second_stage_importance):
    "Test second_stage_permutation_importance function."
    data = data_second_stage_importance
    num_permutations = data["num_permutations"]
    quantile = data["quantile"]
    var_fitted_model = data["var_fitted_model"]
    fitted_model = data["fitted_model"]
    X_test_augmented = data["X_test_augmented"]
    y_test = data["y_test"]
    df_train_ensemble_augmented = data["df_train_ensemble_augmented"]
    X_train_augmented = data["X_train_augmented"]
    df_train_ensemble = data["df_train_ensemble"]
    df_test_ensemble = data["df_test_ensemble"]
    y_train = data["y_train"]
    order_diff = data["order_diff"]
    max_lags_var = data["max_lags_var"]
    augment_var = data["augment_var"]
    start_prediction_timestamp = data["start_prediction_timestamp"]
    # Test the second_stage_permutation_importance function
    results_df = second_stage_permutation_importance( 
        num_permutations = num_permutations,
        quantile=quantile,
        var_fitted_model=var_fitted_model,
        fitted_model=fitted_model,
        X_test_augmented=X_test_augmented,
        y_test=y_test,
        df_train_ensemble_augmented=df_train_ensemble_augmented,
        X_train_augmented=X_train_augmented,
        df_train_ensemble=df_train_ensemble,
        df_test_ensemble=df_test_ensemble,
        y_train=y_train,
        order_diff=order_diff,
        max_lags_var=max_lags_var,
        augment_var=augment_var,
        start_prediction_timestamp=start_prediction_timestamp
    )
    assert isinstance(results_df, pd.DataFrame), "Result should be a DataFrame"
    assert 'predictor' in results_df.columns, "DataFrame should contain 'predictor' column"
    assert 'contribution' in results_df.columns, "DataFrame should contain 'contribution' column"
    assert all(results_df['contribution'] >= 0), "All contributions should be non-negative"
