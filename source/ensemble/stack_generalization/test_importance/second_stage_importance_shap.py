from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_augmented_dataframe_2stage
import matplotlib.pyplot as plt
import seaborn as sns

############################################################################################################ Utils
def extract_data(info, quantile):
    " Extract info previous day"
    fitted_model = info[quantile]['fitted_model']
    y_train = info[quantile]['y_train'] 
    var_fitted_model = info[quantile]['var_fitted_model']
    X_test_augm_prev = info[quantile]['X_test_augmented_prev'] 
    df_test_ens_prev = info[quantile]['df_test_ensemble_prev'] 
    df_train_ens = info[quantile]['df_train_ensemble']
    df_train_ens_augm = info[quantile]['df_train_ensemble_augmented']  
    X_train_augmented = info[quantile]['X_train_augmented']
    buyer_scaler_stats = info[quantile]["buyer_scaler_stats"]
    return fitted_model, y_train, var_fitted_model, X_test_augm_prev, df_test_ens_prev, df_train_ens, df_train_ens_augm, X_train_augmented, buyer_scaler_stats


def validate_inputs(params_model, quantile, y_test_prev, X_test_augmented_prev):
    " Validate the inputs."
    assert parameters_model['nr_permutations'] > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    assert len(y_test_prev) == len(X_test_augmented_prev), "The length of y_test_prev and X_test_augmented_prev must be the same"

def prepare_second_stage_data(parameters_model, df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample):
    " Prepare the second stage data."
    df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample)
    df_2stage_processed = create_augmented_dataframe_2stage(df_2stage, parameters_model['order_diff'], max_lags=parameters_model['max_lags_var'], augment=parameters_model['augment_var'])
    return df_2stage_processed

def normalize_contributions(df):
    " Normalize the contributions."
    total_contribution = abs(df['contribution']).sum()
    df['contribution'] = abs(df['contribution'])/total_contribution
    return df

def get_score_function(quantile):
    " Get the score function for the quantile."
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    return score_functions[quantile]

def create_norm_import_scores_df(importance_scores):
    """
    Create a DataFrame with the importance scores, sort it, drop specific rows, 
    and normalize the contributions.
    """
    # Create a DataFrame with the importance scores and sort it
    results_df = pd.DataFrame(importance_scores)
    # Drop the forecasters standard deviation and variance rows
    results_df = results_df[~results_df.predictor.isin(['forecasters_var', 'forecasters_std'])]
    # Normalize contributions
    results_df = normalize_contributions(results_df)
    # Sort the DataFrame by the contributions
    results_df = results_df.sort_values(by='contribution', ascending=False)
    return results_df

############################################################################################################ Permutation

def decrease_performance(base_score, permuted_scores):
    " Decrease performance."
    decrease_score = max(0, np.mean(permuted_scores) - base_score)
    return decrease_score

def permute_predictor(X, index, seed):
    " Permute the predictor."
    rng = np.random.default_rng(seed)
    X[:, index] = rng.permutation(X[:, index])
    return X

def compute_second_stage_score(seed, parameters_model, 
                                    fitted_model, var_fitted_model, X_test_augmented_prev, df_train_ensemble, df_test_ensemble_prev, y_train, 
                                    y_test_prev, score_function, predictions_insample, forecast_range, permutate=False, predictor_index=None):
    "Compute the permuted score for a single predictor in the second stage model."
    # Generate predictions from the first-stage model
    X_test = X_test_augmented_prev.copy()
    if permutate:
        # Permute the predictor if permute is True
        X_test = permute_predictor(X_test, predictor_index, seed)
    predictions_outsample = fitted_model.predict(X_test)
    # Prepare second stage data
    df_2stage_processed = prepare_second_stage_data(parameters_model, df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample)
    df_2stage_test = df_2stage_processed[(df_2stage_processed.index >= forecast_range[0]) & (df_2stage_processed.index <= forecast_range[-1])]
    X_test_2stage, y_test_2stage = df_2stage_test.drop(columns=['targets']).values, df_2stage_test['targets'].values
    # Compute and return the score
    score = score_function(var_fitted_model, X_test_2stage, y_test_2stage)['mean_loss']
    return score

def second_stage_permutation_importance(y_test_prev, parameters_model, quantile, info, forecast_range):
    """
    Compute permutation importances for the second stage model.
    """
    # Initial validations 
    validate_inputs(parameters_model, quantile, y_test_prev, info[quantile]['X_test_augmented_prev'])
    # Get the score function
    score_function = get_score_function(quantile)
    # Compute the base score
    predictions_insample = info[quantile]['fitted_model'].predict(info[quantile]['X_train_augmented'])
    seed=42
    base_score = compute_second_stage_score(seed, parameters_model,  
                                                info[quantile]['fitted_model'], 
                                                info[quantile]['var_fitted_model'], 
                                                info[quantile]['X_test_augmented_prev'], 
                                                info[quantile]['df_train_ensemble'], 
                                                info[quantile]['df_test_ensemble_prev'], 
                                                info[quantile]['y_train'], 
                                                y_test_prev, score_function, predictions_insample, forecast_range)
    # Compute importance scores for each predictor
    importance_scores = []
    for predictor_index in range(info[quantile]['X_test_augmented_prev'].shape[1]):
        # Get the predictor name
        predictor_name = info[quantile]['df_train_ensemble_augmented'].drop(columns=['norm_targ']).columns[predictor_index]
        # Compute permuted scores in parallel
        permuted_scores = Parallel(n_jobs=4)(delayed(compute_second_stage_score)(seed, 
                                                                                        parameters_model, 
                                                                                        info[quantile]['fitted_model'], 
                                                                                        info[quantile]['var_fitted_model'], 
                                                                                        info[quantile]['X_test_augmented_prev'], 
                                                                                        info[quantile]['df_train_ensemble'], 
                                                                                        info[quantile]['df_test_ensemble_prev'], 
                                                                                        info[quantile]['y_train'],
                                                                                        y_test_prev, score_function, predictions_insample, forecast_range, 
                                                                                        permutate=True, predictor_index=predictor_index) 
                                                                                        for seed in range(parameters_model['nr_permutations']))
        # Compute the mean contribution
        mean_contribution = decrease_performance(base_score, permuted_scores)
        # Append the importance score
        importance_scores.append({'predictor': predictor_name, 
                                    'contribution': mean_contribution})

    # Create a DataFrame and normalize contributions
    results_df = create_norm_import_scores_df(importance_scores)
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

def compute_row_perm_score(seed, params_model, set_feat2perm, predictor_index, y_test_prev, fit_model, y_train, var_fit_model, X_test_augm_prev, df_test_ens_prev, df_train_ens_augm, pred_insample, score_function, X_test_perm_with, X_test_perm_without, forecast_range):
    " Compute row permutation score."
    # compute error by PERMUTATING WITHOUT feature of interest
    X_test_perm_without[:, set_feat2perm] = run_row_permutation_set_features(seed, X_test_augm_prev, set_feat2perm)
    pred_outsample_perm_without = fit_model.predict(X_test_perm_without)
    df_2stage_without_perm = prepare_second_stage_data(params_model, df_train_ens_augm, df_test_ens_prev, y_train, y_test_prev, pred_insample, pred_outsample_perm_without)
    df_2stage_test_without_perm = df_2stage_without_perm[(df_2stage_without_perm.index >= forecast_range[0]) & (df_2stage_without_perm.index <= forecast_range[-1])]
    X_test_2stage_without_perm, y_test_2stage_without_perm = df_2stage_test_without_perm.drop(columns=['targets']).values, df_2stage_test_without_perm['targets'].values
    score_without_perm = score_function(var_fit_model, X_test_2stage_without_perm, y_test_2stage_without_perm)['mean_loss']
    
    # compute error by PERMUTATING WITH feature of interest
    X_test_perm_with[:, set_feat2perm] = run_row_permutation_set_features(seed, X_test_augm_prev, set_feat2perm)
    X_test_perm_with[:, predictor_index] = run_row_permutation_predictor(seed, X_test_augm_prev, predictor_index)
    pred_outsample_perm_with = fit_model.predict(X_test_perm_with)
    df_2stage_with_perm = prepare_second_stage_data(params_model, df_train_ens_augm, df_test_ens_prev, y_train, y_test_prev, pred_insample, pred_outsample_perm_with)
    df_2stage_test_with_perm = df_2stage_with_perm[(df_2stage_with_perm.index >= forecast_range[0]) & (df_2stage_with_perm.index <= forecast_range[-1])]
    X_test_2stage_with_perm, y_test_2stage_with_perm = df_2stage_test_with_perm.drop(columns=['targets']).values, df_2stage_test_with_perm['targets'].values
    score_with_perm = score_function(var_fit_model, X_test_2stage_with_perm, y_test_2stage_with_perm)['mean_loss']

    # return the difference in error
    return max(0, score_with_perm - score_without_perm)

def compute_col_perm_score(seed, params_model, nr_features, y_test_prev, fitted_model, y_train, var_fitted_model, X_test_augm_prev, df_test_ens_prev, df_train_ens_augm, predictions_insample, score_function, predictor_index, forecast_range):
    " Compute  score for a single predictor."
    col_perm = run_col_permutation(seed, nr_features)
    while col_perm[0] == predictor_index:
        seed = seed + 1
        col_perm = run_col_permutation(seed, nr_features)
    set_feat2perm = col_perm[np.arange(0, np.where(col_perm == predictor_index)[0][0])]
    X_test_perm_with, X_test_perm_without = X_test_augm_prev.copy(), X_test_augm_prev.copy()
    row_scores = Parallel(n_jobs=2)(delayed(compute_row_perm_score)(seed,
                                                                    params_model,
                                                                    set_feat2perm,
                                                                    predictor_index,
                                                                    y_test_prev,
                                                                    fitted_model, y_train, var_fitted_model, X_test_augm_prev, df_test_ens_prev, df_train_ens_augm,
                                                                    predictions_insample,
                                                                    score_function,
                                                                    X_test_perm_with,
                                                                    X_test_perm_without,
                                                                    forecast_range
                                                                    ) for seed in range(params_model['nr_row_permutations']))
    col_score = np.mean(row_scores)
    return col_score


def second_stage_shapley_importance(y_test_prev, parameters_model, quantile, info, forecast_range):
    " Compute permutation importances for the first stage model."
    # get info previous day
    fitted_model, y_train, var_fitted_model, X_test_augm_prev, df_test_ens_prev, df_train_ens_augm = extract_data(info, quantile)
    # Get In-sample Predictions
    X_train_augm = info[quantile]['X_train_augmented']
    predictions_insample = fitted_model.predict(X_train_augm)
    # Validate inputs
    validate_inputs(parameters_model, quantile, y_test_prev, X_test_augm_prev)
    # Define the score functions for different quantiles
    score_function = get_score_function(quantile)
    # Initialize the list to store the importance scores
    importance_scores = []
    # Loop through each predictor
    nr_features = X_test_augm_prev.shape[1]
    for predictor_index in range(nr_features):
        # Get the predictor name
        predictor_name = df_train_ens_augm.drop(columns=['norm_targ']).columns[predictor_index]
        # Compute the permuted scores in parallel
        col_scores = Parallel(n_jobs=4)(delayed(compute_col_perm_score)(seed, 
                                                                        parameters_model,
                                                                        nr_features, 
                                                                        y_test_prev, 
                                                                        fitted_model, y_train, var_fitted_model, X_test_augm_prev, df_test_ens_prev, df_train_ens_augm,
                                                                        predictions_insample, 
                                                                        score_function, 
                                                                        predictor_index,
                                                                        forecast_range) 
                                                                        for seed in range(parameters_model['nr_col_permutations']))
        shapley_score = np.mean(col_scores)
        # Append the importance score to the list
        importance_scores.append({'predictor': predictor_name, 
                                'contribution': shapley_score})
    # Create a DataFrame with the importance scores, sort, and normalize it
    results_df = create_norm_import_scores_df(importance_scores)
    return results_df

############################################################################################################

def wind_power_ramp_importance(results_challenge_dict, ens_params, y_test, forecast_range, results_contributions):
    " Get the importance of the wind power ramp"
    assert 'wind_power_ramp' in results_challenge_dict.keys(), 'The key wind_power_variability is not present in the results_challenge_dict'
    assert 'info_contributions' in results_challenge_dict['wind_power_ramp'].keys(), 'The key info_contributions is not present in the results_challenge_dict'
    assert 'quantiles' in ens_params.keys(), 'The key quantiles is not present in the ens_params'
    assert 'nr_permutations' in ens_params.keys(), 'The key nr_permutations is not present in the ens_params'
    logger.opt(colors=True).info(f'<blue>--</blue>' * 79)
    logger.opt(colors=True).info(f'<blue>Wind Power Ramp</blue>')
    # Get the info from the previous day
    info_previous_day_second_stage = results_challenge_dict['wind_power_ramp']['info_contributions']
    # Loop through each quantile
    for quantile in ens_params['quantiles']:
        logger.opt(colors=True).info(f'<blue>Quantile: {quantile}</blue>')
        # Get the contributions
        if ens_params['contribution_method'] == 'shapley':
            col_permutation = ens_params['nr_col_permutations']
            row_permutation = ens_params['nr_row_permutations']
            logger.info(f'Number of column permutations: {col_permutation}')
            logger.info(f'Number of row permutations: {row_permutation}')
            # Get the contributions using the SHAPLEY method
            df_contributions = second_stage_shapley_importance(
                                                                y_test_prev=y_test, 
                                                                params_model=ens_params, 
                                                                quantile=quantile, 
                                                                info=info_previous_day_second_stage, 
                                                                forecast_range = forecast_range
                                                            )
        elif ens_params['contribution_method'] == 'permutation':
            num_permutations = ens_params['nr_permutations']
            logger.info(f'Number of permutations: {num_permutations}')
            # Get the contributions using the PERMUTATION method
            df_contributions = second_stage_permutation_importance(
                                                                    y_test_prev=y_test, 
                                                                    params_model=ens_params, 
                                                                    quantile=quantile, 
                                                                    info=info_previous_day_second_stage, 
                                                                    forecast_range = forecast_range
                                                                )
        else:
            raise ValueError('The contribution method is not valid')
        # Plot the importance scores
        if ens_params['plot_importance_second_stage']:
            plot_importance(df_contributions, quantile)
        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power_ramp'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions

def plot_importance(df_contributions, quantile, top_n=10, figsize=(10, 5), palette='viridis'):
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