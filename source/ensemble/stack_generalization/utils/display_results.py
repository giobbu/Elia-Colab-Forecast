from loguru import logger

def display_forecasting_metrics(sim_params, ens_params, dict_metrics):
    
    # RMSE metric for MEAN
    rmse_ensemble = dict_metrics['ensemble']['rmse']
    rmse_weighted_avg = dict_metrics['weighted_avg']['rmse']
    rmse_equal_weights = dict_metrics['equal_weights']['rmse']
    rmse_dayahead = dict_metrics['day_ahead']['rmse'] 
    rmse_dayahead_11h = dict_metrics['day_ahead_11h']['rmse'] 
    rmse_week_ahead = dict_metrics['week_ahead']['rmse']

    # Set the model type
    model_type = ens_params['model_type']
    var_model_type = ens_params['var_model_type']

    logger.info(' ')
    logger.info('------------- Wind Power Forecasting ------------------------------')
    logger.info('----------------- RMSE -----------------')
    logger.info(f'{model_type} ensemble {rmse_ensemble}')
    logger.info(f'Weighted Average {rmse_weighted_avg}')
    logger.info(f'Equal Weights {rmse_equal_weights}')
    logger.info(f'Day-Ahead {rmse_dayahead}')
    logger.info(f'Day-Ahead-11h {rmse_dayahead_11h}')
    logger.info(f'Week-Ahead {rmse_week_ahead}')
    if sim_params['most_recent']:
        rmse_most_recent = dict_metrics['most_recent']['rmse']
        logger.info(f'Most-Recent {rmse_most_recent}')
    if sim_params['malicious']:
        rmse_malicious = dict_metrics['malicious']['rmse']
        logger.info(f'Malicious {rmse_malicious}')
    if sim_params['noisy']:
        rmse_noisy = dict_metrics['noisy']['rmse']
        logger.info(f'Noisy {rmse_noisy}')
    logger.info(' ')

    if not ens_params['compute_abs_difference']:
        # PB metric for Q10
        pinball_ensemble_q10 = dict_metrics['ensemble']['pb10']
        pinball_weighted_avg_q10 = dict_metrics['weighted_avg']['pb10']
        pinball_equal_weights_q10 = dict_metrics['equal_weights']['pb10']
        pinball_dayahead_q10 = dict_metrics['day_ahead']['pb10']
        pinball_dayahead_11h_q10 = dict_metrics['day_ahead_11h']['pb10']
        pinball_week_ahead_q10 = dict_metrics['week_ahead']['pb10']
        # PB metric for Q90
        pinball_ensemble_q90 = dict_metrics['ensemble']['pb90']
        pinball_weighted_avg_q90 = dict_metrics['weighted_avg']['pb90']
        pinball_equal_weights_q90 = dict_metrics['equal_weights']['pb90']
        pinball_dayahead_q90 = dict_metrics['day_ahead']['pb90']
        pinball_dayahead_11h_q90 = dict_metrics['day_ahead_11h']['pb90']
        pinball_week_ahead_q90 = dict_metrics['week_ahead']['pb90']
        # RMSE metric for MEAN VAR
        rmse_var_ensemble = dict_metrics['ensemble']['rmse_var']
        rmse_var_weighted_avg = dict_metrics['weighted_avg']['rmse_var']
        rmse_var_equal_weights = dict_metrics['equal_weights']['rmse_var']
        rmse_var_dayahead = dict_metrics['day_ahead']['rmse_var'] 
        rmse_var_dayahead_11h = dict_metrics['day_ahead_11h']['rmse_var'] 
        rmse_var_week_ahead = dict_metrics['week_ahead']['rmse_var']
        
        logger.info('----------------- PB Q10 -----------------')
        logger.info(f'{model_type} ensemble {pinball_ensemble_q10}')
        logger.info(f'Weig Avg {pinball_weighted_avg_q10}')
        logger.info(f'Eq Weig {pinball_equal_weights_q10}')
        logger.info(f'Day-Ahead {pinball_dayahead_q10}')
        logger.info(f'Day-Ahead-11h {pinball_dayahead_11h_q10}')
        logger.info(f'Week-Ahead {pinball_week_ahead_q10}')
        if sim_params['most_recent']:
            pinball_most_recent_q10 = dict_metrics['most_recent']['pb10']
            logger.info(f'Most-Recent {pinball_most_recent_q10}')
        if sim_params['malicious']:
            pinball_malicious_q10 = dict_metrics['malicious']['pb10']
            logger.info(f'Malicious {pinball_malicious_q10}')
        if sim_params['noisy']:
            pinball_noisy_q10 = dict_metrics['noisy']['pb10']
            logger.info(f'Noisy {pinball_noisy_q10}')
        logger.info(' ')
        logger.info('----------------- PB Q90 -----------------')
        logger.info(f'{model_type} ensemble {pinball_ensemble_q90}')
        logger.info(f'Weig Avg {pinball_weighted_avg_q90}')
        logger.info(f'Eq Weig {pinball_equal_weights_q90}')
        logger.info(f'Day-Ahead {pinball_dayahead_q90}')
        logger.info(f'Day-Ahead-11h {pinball_dayahead_11h_q90}')
        logger.info(f'Week-Ahead {pinball_week_ahead_q90}')
        if sim_params['most_recent']:
            pinball_most_recent_q90 = dict_metrics['most_recent']['pb90']
            logger.info(f'Most-Recent {pinball_most_recent_q90}')
        if sim_params['malicious']:
            pinball_malicious_q90 = dict_metrics['malicious']['pb90']
            logger.info(f'Malicious {pinball_malicious_q90}')
        if sim_params['noisy']:
            pinball_noisy_q90 = dict_metrics['noisy']['pb90']
            logger.info(f'Noisy {pinball_noisy_q90}')
        logger.info(' ')
        logger.info('----------------- Wind Power Variability Forecast -----------------')
        logger.info('----------------- RMSE -----------------')
        logger.info(f'{var_model_type} ensemble {rmse_var_ensemble}')
        logger.info(f'Weighted Average {rmse_var_weighted_avg}')
        logger.info(f'Equal Weights {rmse_var_equal_weights}')
        logger.info(f'Day-Ahead {rmse_var_dayahead}')
        logger.info(f'Day-Ahead-11h {rmse_var_dayahead_11h}')
        logger.info(f'Week-Ahead {rmse_var_week_ahead}')
        if sim_params['most_recent']:
            rmse_var_most_recent = dict_metrics['most_recent']['rmse_var']
            logger.info(f'Most-Recent {rmse_var_most_recent}')
        if sim_params['malicious']:
            rmse_var_malicious = dict_metrics['malicious']['rmse_var']
            logger.info(f'Malicious {rmse_var_malicious}')
        if sim_params['noisy']:
            rmse_var_noisy = dict_metrics['noisy']['rmse_var']
            logger.info(f'Noisy {rmse_var_noisy}')
        logger.info(' ')