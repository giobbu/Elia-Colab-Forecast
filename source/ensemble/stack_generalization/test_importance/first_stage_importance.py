from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90

def decrease_performance(base_score, permuted_scores):
    " Decrease performance."
    return max(0, np.mean(permuted_scores) - base_score)

def permute_predictor(X, index, seed):
    " Permute the predictor."
    rng = np.random.default_rng(seed)
    X[:, index] = rng.permutation(X[:, index])
    return X

def validate_inputs(params_model, quantile, y_test, X_test):
    " Validate the inputs."
    assert params_model['nr_permutations'] > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    assert len(y_test) == len(X_test), "The length of y_test_prev and X_test_augmented_prev must be the same"

def get_score_function(quantile):
    " Get the score function for the quantile."
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    return score_functions[quantile]

def extract_data(info, quantile):
    " Extract data from the info dictionary."
    return (
            info[quantile]['fitted_model'],
            info[quantile]['X_test_augmented'],
            info[quantile]['df_train_ensemble_augmented']
        )

def normalize_contributions(df):
    " Normalize the contributions."
    total_contribution = df['contribution'].sum()
    df['contribution'] = df['contribution'] / total_contribution
    return df

def compute_first_stage_score(seed, X_test_augm, y_test, fitted_model, score_function, permutate=False, predictor_index=None):
    " Compute  score for a single predictor."
    # Generate predictions from the first-stage model
    X_test = X_test_augm.copy()
    if permutate:
        # Permute the predictor if permute is True
        X_test = permute_predictor(X_test, predictor_index, seed)
    score = score_function(fitted_model, X_test, y_test)['mean_loss']
    return score

def first_stage_permutation_importance(y_test, params_model, quantile, info_previous_day_first_stage):
    " Compute permutation importances for the first stage model."
    # get info previous day
    fitted_model, X_test_augm, df_train_ens_augm = extract_data(info_previous_day_first_stage, quantile)
    # Validate inputs
    validate_inputs(params_model, quantile, y_test, X_test_augm)
    # Define the score functions for different quantiles
    score_function = get_score_function(quantile)
    # Compute the original score
    seed=42
    base_score = compute_first_stage_score(seed, X_test_augm, y_test, fitted_model, score_function)
    # Initialize the list to store the importance scores
    importance_scores = []
    # Loop through each predictor
    for predictor_index in range(X_test_augm.shape[1]):
        # Get the predictor name
        predictor_name = df_train_ens_augm.drop(columns=['norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        permuted_scores = Parallel(n_jobs=4)(delayed(compute_first_stage_score)(seed, X_test_augm, 
                                                                                y_test, fitted_model, score_function,
                                                                                permutate=True, predictor_index=predictor_index) 
                                                                                for seed in range(params_model['nr_permutations']))
        # Compute the mean contribution
        mean_contribution = decrease_performance(base_score, permuted_scores)
        # Append the importance score to the list
        importance_scores.append({'predictor': predictor_name, 
                                'contribution': mean_contribution})
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores)
    results_df = results_df.sort_values(by='contribution', ascending=False)
    # Drop the forecasters standard deviation and variance rows
    results_df = results_df[~results_df.predictor.isin(['forecasters_var', 'forecasters_std'])]
    # Normalize contributions
    results_df = normalize_contributions(results_df)
    return results_df

def wind_power_importance(results_challenge_dict, ens_params, y_test, results_contributions):
    " Get the importance of the wind power"
    assert 'wind_power' in results_challenge_dict.keys(), 'The key wind_power_variability is not present in the results_challenge_dict'
    assert 'info_contributions' in results_challenge_dict['wind_power'].keys(), 'The key info_contributions is not present in the results_challenge_dict'
    assert 'quantiles' in ens_params.keys(), 'The key quantiles is not present in the ens_params'
    assert 'nr_permutations' in ens_params.keys(), 'The key nr_permutations is not present in the ens_params'
    logger.opt(colors=True).info(f'<blue>--</blue>' * 79)
    logger.opt(colors=True).info(f'<blue>Wind Power</blue>')
    # Get the info from the previous day
    info_previous_day_first_stage = results_challenge_dict['wind_power']['info_contributions']
    num_permutations = ens_params['nr_permutations']
    logger.info(f'Number of permutations: {num_permutations}')
    # Get the contributions per quantile
    for quantile in ens_params['quantiles']:
        logger.opt(colors=True).info(f'<blue>Quantile: {quantile}</blue>')
        # Get the contributions
        df_contributions = first_stage_permutation_importance(
            y_test=y_test, 
            params_model=ens_params, 
            quantile=quantile, 
            info_previous_day_first_stage=info_previous_day_first_stage
        )
        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions