from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from loguru import logger
from source.ensemble.stack_generalization.hyperparam_optimization.models.utils.cross_validation import score_func_10, score_func_50, score_func_90
from source.ensemble.stack_generalization.second_stage.create_data_second_stage import create_2stage_dataframe, create_augmented_dataframe_2stage

def extract_data(info, quantile):
        return (
            info[quantile]['fitted_model'],
            info[quantile]['var_fitted_model'],
            info[quantile]['X_test_augmented_prev'],
            info[quantile]['df_train_ensemble'],
            info[quantile]['df_test_ensemble_prev'],
            info[quantile]['y_train']
        )
    
def permute_predictor(X, index, seed):
    rng = np.random.default_rng(seed)
    X[:, index] = rng.permutation(X[:, index])
    return X

def prepare_second_stage_data(parameters_model, df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample):
    df_2stage = create_2stage_dataframe(df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample)
    df_2stage_processed = create_augmented_dataframe_2stage(df_2stage, parameters_model['order_diff'], max_lags=parameters_model['max_lags_var'], augment=parameters_model['augment_var'])
    return df_2stage_processed

def compute_second_stage_score_perm(seed, parameters_model,
                                    fitted_model, var_fitted_model, X_test_augmented_prev, df_train_ensemble, df_test_ensemble_prev, y_train, 
                                    y_test_prev, score_function, predictions_insample, forecast_range, predictor_index=None):
    "Compute the permuted score for a single predictor in the second stage model."
    X_test = X_test_augmented_prev.copy()
    # Permute the predictor if permute is True
    X_test = permute_predictor(X_test, predictor_index, seed) 
    # Generate predictions from the first-stage model
    predictions_outsample = fitted_model.predict(X_test)
    # Prepare second stage data
    df_2stage_processed = prepare_second_stage_data(parameters_model, df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample)
    df_2stage_test = df_2stage_processed[(df_2stage_processed.index >= forecast_range[0]) & (df_2stage_processed.index <= forecast_range[-1])]
    X_test_2stage, y_test_2stage = df_2stage_test.drop(columns=['targets']).values, df_2stage_test['targets'].values
    # Compute and return the score
    score = score_function(var_fitted_model, X_test_2stage, y_test_2stage)['mean_loss']
    return score

def compute_second_stage_score_base(parameters_model, 
                                    fitted_model, var_fitted_model, X_test_augmented_prev, df_train_ensemble, df_test_ensemble_prev, y_train, 
                                    y_test_prev, score_function, predictions_insample, forecast_range):
    "Compute the permuted score for a single predictor in the second stage model."
    # Generate predictions from the first-stage model
    predictions_outsample = fitted_model.predict(X_test_augmented_prev)
    # Prepare second stage data
    df_2stage_processed = prepare_second_stage_data(parameters_model, df_train_ensemble, df_test_ensemble_prev, y_train, y_test_prev, predictions_insample, predictions_outsample)
    df_2stage_test = df_2stage_processed[(df_2stage_processed.index >= forecast_range[0]) & (df_2stage_processed.index <= forecast_range[-1])]
    X_test_2stage, y_test_2stage = df_2stage_test.drop(columns=['targets']).values, df_2stage_test['targets'].values
    # Compute and return the score
    score = score_function(var_fitted_model, X_test_2stage, y_test_2stage)['mean_loss']
    return score

def validate_inputs(parameters_model, quantile, y_test_prev, X_test_augmented_prev):
    assert parameters_model['nr_permutations'] > 0, "Number of permutations must be positive"
    assert quantile in [0.1, 0.5, 0.9], "Quantile must be one of 0.1, 0.5, 0.9"
    assert len(y_test_prev) == len(X_test_augmented_prev), "The length of y_test_prev and X_test_augmented_prev must be the same"

def normalize_contributions(df):
    total_contribution = df['contribution'].sum()
    df['contribution'] = df['contribution'] / total_contribution
    return df

def get_score_function(quantile):
    score_functions = {
        0.1: score_func_10,
        0.5: score_func_50,
        0.9: score_func_90
    }
    return score_functions[quantile]
    
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
    base_score = compute_second_stage_score_base(parameters_model,  
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
        permuted_scores = Parallel(n_jobs=-1)(delayed(compute_second_stage_score_perm)(seed, 
                                                                                        parameters_model, 
                                                                                        info[quantile]['fitted_model'], 
                                                                                        info[quantile]['var_fitted_model'], 
                                                                                        info[quantile]['X_test_augmented_prev'], 
                                                                                        info[quantile]['df_train_ensemble'], 
                                                                                        info[quantile]['df_test_ensemble_prev'], 
                                                                                        info[quantile]['y_train'],
                                                                                        y_test_prev, score_function, predictions_insample, forecast_range, 
                                                                                        predictor_index=predictor_index) 
                                                                                        for seed in range(num_permutations = parameters_model['nr_permutations']))
        # Calculate mean contribution for the predictor
        mean_contribution = max(0, np.mean(permuted_scores) - base_score)
        importance_scores.append({'predictor': predictor_name, 
                                    'contribution': mean_contribution})
    # Create a DataFrame and normalize contributions
    results_df = pd.DataFrame(importance_scores)
    results_df = results_df.sort_values(by='contribution', ascending=False)
    results_df = results_df[~results_df['predictor'].isin(['forecasters_var', 'forecasters_std'])]
    results_df = normalize_contributions(results_df)
    return results_df

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
    num_permutations = ens_params['nr_permutations']
    logger.info(f'Number of permutations: {num_permutations}')
    for quantile in ens_params['quantiles']:
        logger.opt(colors=True).info(f'<blue>Quantile: {quantile}</blue>')
        # Get the contributions
        df_contributions = second_stage_permutation_importance(
            y_test_prev=y_test, 
            parameters_model=ens_params, 
            quantile=quantile, 
            info_previous_day_second_stage=info_previous_day_second_stage, 
            forecast_range = forecast_range
        )
        # Get the predictor name
        df_contributions['predictor'] = df_contributions['predictor'].apply(lambda x: x.split('_')[1])
        # Save the contributions
        results_contributions['wind_power_ramp'][quantile] = dict(df_contributions.groupby('predictor')['contribution'].sum())
    return results_contributions