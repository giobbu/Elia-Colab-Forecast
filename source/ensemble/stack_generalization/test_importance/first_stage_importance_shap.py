from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90
import matplotlib.pyplot as plt
import seaborn as sns

############################################################################################################ Utils

def extract_data(info, quantile):
    """Extract data from the info dictionary.
    Args:
        info: The info dictionary
        quantile: The quantile
    Returns:
        fitted_model: The fitted model
        X_test_augmented: The augmented test set
        df_train_ensemble_augmented: The augmented training set
        buyer_scaler_stats: The buyer scaler statistics
    """
    return (
            info[quantile]['fitted_model'],
            info[quantile]['X_test_augmented'],
            info[quantile]['df_train_ensemble_augmented'],
            info[quantile]["buyer_scaler_stats"]
        )

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


def normalize_contributions(df):
    " Normalize the contributions."
    total_contribution = df['contribution'].sum()
    df['contribution'] = df['contribution']/total_contribution
    return df

def create_norm_import_scores_df(importance_scores):
    """
    Create a DataFrame with the importance scores, sort it, drop specific rows, 
    and normalize the contributions.
    """
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores)
    results_df = results_df.sort_values(by='contribution', ascending=False)
    # Drop the forecasters standard deviation and variance rows
    results_df = results_df[~results_df.predictor.isin(['forecasters_var', 'forecasters_std'])]
    # Normalize contributions
    results_df = normalize_contributions(results_df)
    return results_df

############################################################################################################ Shapley

def run_col_permutation(seed, nr_features):
    " Run column permutation."
    rng = np.random.default_rng(seed)
    return rng.permutation(nr_features)

def run_row_permutation_predictor(seed, X_test, predictor_index):
    " Run row permutation predictor."
    rng = np.random.default_rng(seed)
    predictor_permutated = rng.permutation(X_test[: , predictor_index])
    return predictor_permutated

def run_row_permutation_set_features(seed, X_test, set_feat2permutate):
    " Run row permutation set features."
    rng = np.random.default_rng(seed)
    X_set_permutated = rng.permutation(X_test[:, set_feat2permutate])
    return X_set_permutated

def compute_row_perm_score(seed, fitted_model, set_feat2permutate, predictor_index, X_test_augm, y_test, score_function, X_test_perm_with, X_test_perm_without):
    " Compute row permutation score."
    # compute error by PERMUTATING WITHOUT feature of interest
    X_test_perm_without[:, set_feat2permutate] = run_row_permutation_set_features(seed, X_test_augm, set_feat2permutate)
    score_without_permutation = score_function(fitted_model, X_test_perm_without, y_test)['mean_loss']
    # compute error by PERMUTATING WITH feature of interest
    X_test_perm_with[:, set_feat2permutate] = run_row_permutation_set_features(seed, X_test_augm, set_feat2permutate)
    X_test_perm_with[:, predictor_index] = run_row_permutation_predictor(seed, X_test_augm, predictor_index)
    score_with_permutation = score_function(fitted_model, X_test_perm_with, y_test)['mean_loss']
    # return the difference in error
    return max(0, score_with_permutation - score_without_permutation)

def compute_col_perm_score(seed, params_model, nr_features, X_test_augm, y_test, fitted_model, score_function, predictor_index):
    " Compute  score for a single predictor."
    col_perm = run_col_permutation(seed, nr_features)
    while col_perm[0] == predictor_index:
        seed = seed + 1
        col_perm = run_col_permutation(seed, nr_features)
    set_feat2permutate = col_perm[np.arange(0, np.where(col_perm == predictor_index)[0][0])]
    X_test_perm_with, X_test_perm_without = X_test_augm.copy(), X_test_augm.copy()
    row_scores = Parallel(n_jobs=2)(delayed(compute_row_perm_score)(seed,
                                                                    fitted_model,
                                                                    set_feat2permutate,
                                                                    predictor_index,
                                                                    X_test_augm,
                                                                    y_test,
                                                                    score_function,
                                                                    X_test_perm_with,
                                                                    X_test_perm_without
                                                                    ) for seed in range(params_model['nr_row_permutations']))
    col_score = np.mean(row_scores)
    return col_score

def first_stage_shapley_importance(y_test, params_model, quantile, info_previous_day_first_stage):
    " Compute permutation importances for the first stage model."
    # get info previous day
    fitted_model, X_test_augm, df_train_ens_augm, buyer_scaler_stats = extract_data(info_previous_day_first_stage, quantile)
    y_test = (y_test - buyer_scaler_stats['mean_buyer']) / buyer_scaler_stats['std_buyer']
    # Validate inputs
    validate_inputs(params_model, quantile, y_test, X_test_augm)
    # Define the score functions for different quantiles
    score_function = get_score_function(quantile)
    # Initialize the list to store the importance scores
    importance_scores = []
    # Loop through each predictor
    nr_features = X_test_augm.shape[1]
    for predictor_index in range(nr_features):
        # Get the predictor name
        predictor_name = df_train_ens_augm.drop(columns=['norm_targ']).columns[predictor_index]
        col_scores = []
        list_set_feat2permutate = []
        for seed in range(params_model['nr_col_permutations']):
            col_score, set_feat2permutate = compute_col_perm_score(seed, params_model, nr_features, X_test_augm, y_test, fitted_model, score_function, predictor_index, list_set_feat2permutate)
            col_scores.append(col_score)
            list_set_feat2permutate.append(set_feat2permutate)

        # # Compute the permuted scores in parallel
        # col_scores = Parallel(n_jobs=4)(delayed(compute_col_perm_score)(seed, 
        #                                                                 params_model,
        #                                                                     nr_features, 
        #                                                                     X_test_augm, 
        #                                                                     y_test, 
        #                                                                     fitted_model, 
        #                                                                     score_function, 
        #                                                                     predictor_index) 
        #                                                                     for seed in range(params_model['nr_col_permutations']))


        shapley_score = np.mean(col_scores)
        # Append the importance score to the list
        importance_scores.append({'predictor': predictor_name, 
                                'contribution': shapley_score})
    # Create a DataFrame with the importance scores, sort, and normalize it
    results_df = create_norm_import_scores_df(importance_scores)
    return results_df

############################################################################################################ Permutation

def decrease_performance(base_score, permuted_scores):
    " Decrease performance."
    return max(0, np.mean(permuted_scores) - base_score)

def permute_predictor(X, index, seed):
    " Permute the predictor."
    rng = np.random.default_rng(seed)
    X[:, index] = rng.permutation(X[:, index])
    return X

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
    fitted_model, X_test_augm, df_train_ens_augm, buyer_scaler_stats = extract_data(info_previous_day_first_stage, quantile)
    # Standardize the target variable
    y_test = (y_test - buyer_scaler_stats['mean_buyer'])/buyer_scaler_stats['std_buyer']
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
    # Create a DataFrame with the importance scores, sort, and normalize it
    results_df = create_norm_import_scores_df(importance_scores)
    return results_df

############################################################################################################

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
        if ens_params['contribution_method'] == 'shapley':
            col_permutation = ens_params['nr_col_permutations']
            row_permutation = ens_params['nr_row_permutations']
            logger.info(f'Number of column permutations: {col_permutation}')
            logger.info(f'Number of row permutations: {row_permutation}')
            # Compute the contributions using the Shapley method
            df_contributions = first_stage_shapley_importance(
                y_test=y_test, 
                params_model=ens_params, 
                quantile=quantile, 
                info_previous_day_first_stage=info_previous_day_first_stage
            )
        elif ens_params['contribution_method'] == 'permutation':
            num_permutations = ens_params['nr_permutations']
            logger.info(f'Number of permutations: {num_permutations}')
            # Compute the contributions using the permutation method
            df_contributions = first_stage_permutation_importance(
                                                                    y_test=y_test, 
                                                                    params_model=ens_params, 
                                                                    quantile=quantile, 
                                                                    info_previous_day_first_stage=info_previous_day_first_stage
                                                                )
        else:
            raise ValueError('The contribution method is not implemented')
        
        if ens_params['plot_importance_permutation_first_stage']:
            plot_permutation_importance(df_contributions, quantile)

        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions


def plot_permutation_importance(df_contributions, quantile, top_n=10, figsize=(10, 5), palette='viridis'):
    """
    Plot the top N permutation feature importances using a seaborn bar plot.
    """
    # Select top N contributions
    df_top_contributions = df_contributions.head(top_n)
    # Create the plot
    plt.figure(figsize=figsize)
    sns.barplot(y='contribution', x='predictor', data=df_top_contributions, palette=palette)
    # Customize plot
    plt.xlabel('Predictor')
    plt.ylabel('Contribution')
    plt.xticks(rotation=45)
    plt.title(f'Wind Power - Top {top_n} Feature Importances - Quantile {quantile}')
    # Display the plot
    plt.show()