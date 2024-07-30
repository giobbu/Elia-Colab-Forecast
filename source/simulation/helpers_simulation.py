from collections import defaultdict
import pandas as pd 

def process_combination_scheme(df_train, df_test, end_training_timestamp, start_prediction_timestamp):
    " Process data for the combination scheme"
    # Concatenate train and test dataframes
    df_comb_scheme = pd.concat([df_train, df_test], axis=0)
    df_comb_scheme_norm = df_comb_scheme.copy()
    df_comb_scheme_norm= df_comb_scheme_norm.add_prefix('norm_')
    # Split train and test dataframes
    df_train_norm = df_comb_scheme_norm[df_comb_scheme_norm.index < end_training_timestamp]
    df_test_norm = df_comb_scheme_norm[df_comb_scheme_norm.index >= start_prediction_timestamp]
    assert len(df_test_norm)==96*2, 'Length of test dataframe is not 96'
    # concatenate last training row with test data
    df_test_norm_var = df_test.diff().iloc[-96*2:, :]
    df_test_norm_var = df_test_norm_var.add_prefix('norm_')
    return df_train_norm, df_test_norm, df_test_norm_var

def update_dict_weights(mu, observation, iteration):
    " Update the contributions of the forecasters" 
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

def compute_coefficients(previous_day, quantiles=[0.1, 0.5, 0.9], stages=['wind_power', 'wind_power_ramp']):
    " Compute the coefficients of the forecasters for the different stages"
    iter_coefficients = defaultdict(dict)
    for stage in stages:
        for quantile in quantiles:
            # Extracting the necessary data
            if stage == 'wind_power_ramp':
                quantile = 0.5
            df_train_ensemble = previous_day[stage]['info_contributions'][quantile]['df_train_ensemble_augmented'].drop(['norm_targ'], axis=1)
            fitted_model = previous_day[stage]['info_contributions'][quantile]['fitted_model']
            # Creating DataFrame of coefficients
            df_coefs = pd.DataFrame(fitted_model.coef_, 
                                    index=df_train_ensemble.columns, 
                                    columns=['coef']).reset_index()
            df_coefs.columns = ['predictor', 'coef']
            # Filtering out specific predictors
            df_coefs = df_coefs[~df_coefs.predictor.isin(['forecasters_var', 'forecasters_std'])]
            # Modifying predictors and normalizing coefficients
            df_coefs['predictor'] = df_coefs['predictor'].apply(lambda x: x.split('_')[1])
            df_coefs['coef'] = abs(df_coefs['coef'])/abs(df_coefs['coef']).sum()
            # Summarizing coefficients by predictor
            iter_coefficients[stage][quantile] = df_coefs.groupby('predictor').coef.sum().sort_values(ascending=False).to_dict()
    return iter_coefficients