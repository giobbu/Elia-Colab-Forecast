from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90

def normalize_contributions(df):
    total_contribution = df['contribution'].sum()
    df['contribution'] = df['contribution'] / total_contribution
    return df

def first_stage_compute_permuted_score(predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile):
    " Compute the permuted score for a single predictor."
    X_test_permuted = X_test_augmented.copy()
    X_test_permuted[:, predictor_index] = np.random.permutation(X_test_augmented[:, predictor_index])
    permuted_score = score_functions[quantile](fitted_model, X_test_permuted, y_test)['mean_loss']
    return permuted_score

def first_stage_permutation_importance(y_test, parameters_model, quantile, info_previous_day_first_stage):
    " Compute permutation importances for the first stage model."
    num_permutations = parameters_model['nr_permutations']
    assert num_permutations > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    # Define the score functions for different quantiles
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    # get the model
    fitted_model = info_previous_day_first_stage[quantile]['fitted_model']
    # get the data
    X_test_augmented = info_previous_day_first_stage[quantile]['X_test_augmented']
    df_train_ensemble_augmented = info_previous_day_first_stage[quantile]['df_train_ensemble_augmented']
    # Compute the original score
    base_score = score_functions[quantile](fitted_model, X_test_augmented, y_test)['mean_loss']
    importance_scores = []
    # Loop through each predictor
    for predictor_index in range(X_test_augmented.shape[1]):
        predictor_name = df_train_ensemble_augmented.drop(columns=['norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        permuted_scores = Parallel(n_jobs=4)(delayed(first_stage_compute_permuted_score)(
            predictor_index, X_test_augmented, y_test, fitted_model, score_functions, quantile) for _ in range(num_permutations))
        # Calculate mean contribution for the predictor
        mean_contribution = max(0, np.mean(permuted_scores) - base_score)
        importance_scores.append({'predictor': predictor_name, 'contribution': mean_contribution})
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
            parameters_model=ens_params, 
            quantile=quantile, 
            info_previous_day_first_stage=info_previous_day_first_stage
        )
        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions