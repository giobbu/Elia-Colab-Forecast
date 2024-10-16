from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def process_combination_scheme(df_train, df_test, end_training_timestamp, start_prediction_timestamp):
    " Process data for the combination scheme"

    assert isinstance(df_train, pd.DataFrame), 'df_train must be a pandas DataFrame'
    assert isinstance(df_test, pd.DataFrame), 'df_test must be a pandas DataFrame'
    assert isinstance(end_training_timestamp, pd.Timestamp), 'end_training_timestamp must be a pandas Timestamp'
    assert isinstance(start_prediction_timestamp, pd.Timestamp), 'start_prediction_timestamp must be a pandas Timestamp'
    
    # Concatenate train and test dataframes
    df_comb_scheme = pd.concat([df_train, df_test], axis=0)
    df_comb_scheme_norm = df_comb_scheme.copy()
    df_comb_scheme_norm= df_comb_scheme_norm.add_prefix('norm_')
    # Split train and test dataframes
    df_train_norm = df_comb_scheme_norm[df_comb_scheme_norm.index < end_training_timestamp]
    df_test_norm = df_comb_scheme_norm[df_comb_scheme_norm.index >= start_prediction_timestamp]
    assert len(df_test_norm)==96*2, 'Length of test dataframe is not 96*2'
    # concatenate last training row with test data
    df_test_norm_var = df_test.diff().iloc[-96*2:, :]
    df_test_norm_var = df_test_norm_var.add_prefix('norm_')
    return df_train_norm, df_test_norm, df_test_norm_var

def update_dict_weights(mu, observation, iteration):
    " Update the contributions of the forecasters" 

    assert isinstance(mu, dict), 'mu must be a dictionary'
    assert isinstance(observation, dict), 'observation must be a dictionary'
    assert isinstance(iteration, int), 'iteration must be an integer'

    n = iteration + 1   
    if n == 1:
        mu = observation.copy()
    else:
        # Sum the values
        for key in observation.keys():  # Union of keys from both dictionaries
            for subkey in observation[key].keys():  # Union of subkeys
                mu[key][subkey] = {
                    k: mu[key].get(subkey, {}).get(k, 0) + (observation[key].get(subkey, {}).get(k, 0) - mu[key].get(subkey, {}).get(k, 0)) / n
                    for k in set(mu[key].get(subkey, {}).keys()) | set(observation[key].get(subkey, {}).keys())
                }
    return mu


def normalize_coefficients(df_coefs):
    """Normalize the coefficients by their absolute sum."""
    coef_sum = abs(df_coefs['coef']).sum()
    df_coefs['coef'] = abs(df_coefs['coef']) / coef_sum if coef_sum != 0 else 0
    return df_coefs

def plot_top_contributions(df_coefs, quantile, top_n=10, figsize=(10, 5)):
    """Plot the top N contributions of the predictors."""
    df_top_contributions = df_coefs.head(top_n)
    plt.figure(figsize=figsize)
    sns.barplot(y='coef', x='predictor', data=df_top_contributions, palette='magma')
    plt.xlabel('Predictor')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45)
    plt.title(f'Wind Power - Top {top_n} LASSO coefs - Quantile {quantile}')
    plt.show()

def compute_coefficients(ens_params, previous_day, quantiles=[0.1, 0.5, 0.9], stages=['wind_power', 'wind_power_ramp'], p_values=False):
    """Compute the coefficients of the forecasters for different stages."""

    assert isinstance(ens_params, dict), 'ens_params must be a dictionary'
    assert isinstance(previous_day, dict), 'previous_day must be a dictionary'
    assert isinstance(quantiles, list), 'quantiles must be a list'
    assert isinstance(stages, list), 'stages must be a list'
    assert isinstance(p_values, bool), 'p_values must be a boolean'

    iter_coefficients = defaultdict(dict)
    for stage in stages:
        for quantile in quantiles:
            # Handle 'wind_power' stage
            if stage == 'wind_power':
                df_summary = previous_day[stage]['info_contributions'][quantile]['model-summary']
                if p_values:
                    # Set coefficients to 0 where significance is False
                    df_summary['Coefs'] = df_summary.apply(
                        lambda x: 0 if not x['significant'] else x['Coefs'], axis=1
                    )
                coefs = df_summary['Coefs'].values
                df_coefs = pd.DataFrame({'predictor': df_summary['Predictor'], 'coef': abs(coefs)})
                # Filter out unwanted predictors
                df_coefs = df_coefs[~df_coefs['predictor'].isin(['forecasters_var', 'forecasters_std'])]
                df_coefs = df_coefs.sort_values(by='coef', ascending=False)
                # Plot if requested
                if ens_params.get('plot_importance_lasso_coefs', False):
                    plot_top_contributions(df_coefs, quantile)
                # Normalize coefficients
                df_coefs = normalize_coefficients(df_coefs)
                df_coefs['predictor'] = df_coefs['predictor'].apply(lambda x: x.split('_')[1])
                # Summarize coefficients by predictor
                iter_coefficients[stage][quantile] = df_coefs.groupby('predictor')['coef'].sum().sort_values(ascending=False).to_dict()
            # Handle 'wind_power_ramp' stage (only for quantile 0.5)
            elif stage == 'wind_power_ramp' and quantile == 0.5:
                iter_coefficients[stage][quantile] = iter_coefficients['wind_power'][0.5].copy()
    return iter_coefficients