import pandas as pd
from source.forecasters.deterministic import create_day_ahead_predictions, create_day_ahead_11_predictions, create_week_ahead_predictions, create_most_recent_predictions, create_malicious_predictions, create_noisy_predictions
from source.forecasters.probabilistic import create_day_ahead_quantiles10, create_day_ahead_11_quantiles10, create_week_ahead_quantiles10, create_most_recent_quantiles10, create_malicious_quantiles10, create_noisy_quantiles10
from source.forecasters.probabilistic import create_day_ahead_quantiles90, create_day_ahead_11_quantiles90, create_week_ahead_quantiles90, create_most_recent_quantiles90, create_malicious_quantiles90, create_noisy_quantiles90

def submission_forecasters(sim_params, df_train, df_test):
            " Concatenate forecasters predictions for the submission"

            # forecaster - day ahead forecast
            df_day_ahead_pred_train = create_day_ahead_predictions(df_train)
            df_day_ahead_pred_test = create_day_ahead_predictions(df_test)

            # forecaster - day ahead 11 forecast
            df_day_ahead11_pred_train = create_day_ahead_11_predictions(df_train)
            df_day_ahead11_pred_test = create_day_ahead_11_predictions(df_test)

            # forecaster - week ahead forecast
            df_week_ahead_pred_train = create_week_ahead_predictions(df_train)
            df_week_ahead_pred_test = create_week_ahead_predictions(df_test)

            # forecaster - day ahead quantile-10
            df_day_ahead_q10_train = create_day_ahead_quantiles10(df_train)
            df_day_ahead_q10_test = create_day_ahead_quantiles10(df_test)

            # forecaster - day ahead 11 quantile-10
            df_day_ahead11_q10_train = create_day_ahead_11_quantiles10(df_train)
            df_day_ahead11_q10_test = create_day_ahead_11_quantiles10(df_test)

            # forecaster - week ahead quantile-10
            df_week_ahead_q10_train = create_week_ahead_quantiles10(df_train)
            df_week_ahead_q10_test = create_week_ahead_quantiles10(df_test)

            # forecaster - day ahead quantile-90
            df_day_ahead_q90_train = create_day_ahead_quantiles90(df_train)
            df_day_ahead_q90_test = create_day_ahead_quantiles90(df_test)

            # forecaster - day ahead 11 quantile-90
            df_day_ahead11_q90_train = create_day_ahead_11_quantiles90(df_train)
            df_day_ahead11_q90_test = create_day_ahead_11_quantiles90(df_test)

            # forecaster - week ahead quantile-90
            df_week_ahead_q90_train = create_week_ahead_quantiles90(df_train)
            df_week_ahead_q90_test = create_week_ahead_quantiles90(df_test)

            # forecaster - most recent forecast (intra-day market)
            if sim_params['most_recent']:
                # mean forecasts
                df_most_recent_pred_train = create_most_recent_predictions(df_train)
                df_most_recent_pred_test = create_most_recent_predictions(df_test)
                # q10 forecasts
                df_most_recent_q10_train = create_most_recent_quantiles10(df_train)
                df_most_recent_q10_test = create_most_recent_quantiles10(df_test)
                # q90 forecasts
                df_most_recent_q90_train = create_most_recent_quantiles90(df_train)
                df_most_recent_q90_test = create_most_recent_quantiles90(df_test)
            
            # forecaster - malicious forecast
            if sim_params['malicious']:
                # mean forecasts
                df_malicious_pred_train = create_malicious_predictions(df_train, column= sim_params['malicious_name'])
                df_malicious_pred_test = create_malicious_predictions(df=df_test, column= sim_params['malicious_name'], cheat=True, df_train=df_train)
                # q10 forecasts
                df_malicious_q10_train = create_malicious_quantiles10(df_train, column= sim_params['malicious_name'])
                df_malicious_q10_test = create_malicious_quantiles10(df=df_test, column= sim_params['malicious_name'], cheat=True, df_train=df_train)
                # q90 forecasts
                df_malicious_q90_train = create_malicious_quantiles90(df_train, column= sim_params['malicious_name'])
                df_malicious_q90_test = create_malicious_quantiles90(df=df_test, column= sim_params['malicious_name'], cheat=True, df_train=df_train)

            # forecaster - noisy forecast
            if sim_params['noisy']:
                # mean forecasts
                df_noisy_pred_train = create_noisy_predictions(df_train, column= sim_params['noisy_name'])
                df_noisy_pred_test = create_noisy_predictions(df_test, column= sim_params['noisy_name'])
                # q10 forecasts
                df_noisy_q10_train = create_noisy_quantiles10(df_train, column= sim_params['noisy_name'])
                df_noisy_q10_test = create_noisy_quantiles10(df_test, column= sim_params['noisy_name'])
                # q90 forecasts
                df_noisy_q90_train = create_noisy_quantiles90(df_train, column= sim_params['noisy_name'])
                df_noisy_q90_test = create_noisy_quantiles90(df_test, column= sim_params['noisy_name'])

        # # ----------------------------> SELLERS DATA <----------------------------
            # sellers data
            # q50 forecasts
            df_train_ensemble_quantile50 = pd.concat([df_day_ahead_pred_train, df_day_ahead11_pred_train, df_week_ahead_pred_train], axis=1)
            df_test_ensemble_quantile50 = pd.concat([df_day_ahead_pred_test, df_day_ahead11_pred_test, df_week_ahead_pred_test], axis=1)
            if sim_params['malicious']:
                df_train_ensemble_quantile50 = pd.concat([df_train_ensemble_quantile50, df_malicious_pred_train], axis=1)
                df_test_ensemble_quantile50 = pd.concat([df_test_ensemble_quantile50, df_malicious_pred_test], axis=1)
            if sim_params['most_recent']:
                df_train_ensemble_quantile50 = pd.concat([df_train_ensemble_quantile50, df_most_recent_pred_train], axis=1)
                df_test_ensemble_quantile50 = pd.concat([df_test_ensemble_quantile50, df_most_recent_pred_test], axis=1)
            if sim_params['noisy']:
                df_train_ensemble_quantile50 = pd.concat([df_train_ensemble_quantile50, df_noisy_pred_train], axis=1)
                df_test_ensemble_quantile50 = pd.concat([df_test_ensemble_quantile50, df_noisy_pred_test], axis=1)
            df_ensemble_quantile50 = pd.concat([df_train_ensemble_quantile50, df_test_ensemble_quantile50], axis=0)

            # q10 forecasts
            df_train_ensemble_quantile10 = pd.concat([df_day_ahead_q10_train, df_day_ahead11_q10_train, df_week_ahead_q10_train], axis=1)
            df_test_ensemble_quantile10 = pd.concat([df_day_ahead_q10_test, df_day_ahead11_q10_test, df_week_ahead_q10_test], axis=1)
            if sim_params['malicious']:
                df_train_ensemble_quantile10 = pd.concat([df_train_ensemble_quantile10, df_malicious_q10_train], axis=1)
                df_test_ensemble_quantile10 = pd.concat([df_test_ensemble_quantile10, df_malicious_q10_test], axis=1)
            if sim_params['most_recent']:
                df_train_ensemble_quantile10 = pd.concat([df_train_ensemble_quantile10, df_most_recent_q10_train], axis=1)
                df_test_ensemble_quantile10 = pd.concat([df_test_ensemble_quantile10, df_most_recent_q10_test], axis=1)
            if sim_params['noisy']:
                df_train_ensemble_quantile10 = pd.concat([df_train_ensemble_quantile10, df_noisy_q10_train], axis=1)
                df_test_ensemble_quantile10 = pd.concat([df_test_ensemble_quantile10, df_noisy_q10_test], axis=1)
            df_ensemble_quantile10 = pd.concat([df_train_ensemble_quantile10, df_test_ensemble_quantile10], axis=0)

            # q90 forecasts
            df_train_ensemble_quantile90 = pd.concat([df_day_ahead_q90_train, df_day_ahead11_q90_train, df_week_ahead_q90_train], axis=1)
            df_test_ensemble_quantile90 = pd.concat([df_day_ahead_q90_test, df_day_ahead11_q90_test, df_week_ahead_q90_test], axis=1)
            if sim_params['malicious']:
                df_train_ensemble_quantile90 = pd.concat([df_train_ensemble_quantile90, df_malicious_q90_train], axis=1)
                df_test_ensemble_quantile90 = pd.concat([df_test_ensemble_quantile90, df_malicious_q90_test], axis=1)
            if sim_params['most_recent']:
                df_train_ensemble_quantile90 = pd.concat([df_train_ensemble_quantile90, df_most_recent_q90_train], axis=1)
                df_test_ensemble_quantile90 = pd.concat([df_test_ensemble_quantile90, df_most_recent_q90_test], axis=1)
            if sim_params['noisy']:
                df_train_ensemble_quantile90 = pd.concat([df_train_ensemble_quantile90, df_noisy_q90_train], axis=1)
                df_test_ensemble_quantile90 = pd.concat([df_test_ensemble_quantile90, df_noisy_q90_test], axis=1)
            df_ensemble_quantile90 = pd.concat([df_train_ensemble_quantile90, df_test_ensemble_quantile90], axis=0)

            lst_cols_name_q50 = ['s1_q50_b1r1', 's2_q50_b1r1', 's3_q50_b1r1']
            lst_cols_name_q10 = ['s1_q10_b1r1', 's2_q10_b1r1', 's3_q10_b1r1']
            lst_cols_name_q90 = ['s1_q90_b1r1', 's2_q90_b1r1', 's3_q90_b1r1']
            
            if sim_params['malicious']:
                lst_cols_name_q50.append('s5_q50_b1r1')
                lst_cols_name_q10.append('s5_q10_b1r1')
                lst_cols_name_q90.append('s5_q90_b1r1')

            if sim_params['most_recent']:
                lst_cols_name_q50.append('s4_q50_b1r1')
                lst_cols_name_q10.append('s4_q10_b1r1')
                lst_cols_name_q90.append('s4_q90_b1r1')

            if sim_params['noisy']:
                lst_cols_name_q50.append('s6_q50_b1r1')
                lst_cols_name_q10.append('s6_q10_b1r1')
                lst_cols_name_q90.append('s6_q90_b1r1')

            df_ensemble_quantile50.columns = lst_cols_name_q50
            df_ensemble_quantile10.columns = lst_cols_name_q10
            df_ensemble_quantile90.columns = lst_cols_name_q90

            df_market = pd.concat([df_ensemble_quantile50, df_ensemble_quantile10, df_ensemble_quantile90], axis=1)

            if sim_params['malicious']:
                # mean forecasts
                df_train['maliciousforecast'] = df_malicious_pred_train.values
                df_test['maliciousforecast'] = df_malicious_pred_test.values
                # q10 forecasts
                df_train['maliciousconfidence10'] = df_malicious_q10_train.values
                df_test['maliciousconfidence10'] = df_malicious_q10_test.values
                # q90 forecasts
                df_train['maliciousconfidence90'] = df_malicious_q90_train.values
                df_test['maliciousconfidence90'] = df_malicious_q90_test.values

            if sim_params['noisy']:
                # mean forecasts
                df_train['noisyforecast'] = df_noisy_pred_train.values
                df_test['noisyforecast'] = df_noisy_pred_test.values
                # q10 forecasts
                df_train['noisyconfidence10'] = df_noisy_q10_train.values
                df_test['noisyconfidence10'] = df_noisy_q10_test.values
                # q90 forecasts
                df_train['noisyconfidence90'] = df_noisy_q90_train.values
                df_test['noisyconfidence90'] = df_noisy_q90_test.values

            return df_market, df_train, df_test