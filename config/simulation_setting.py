from dataclasses import dataclass

@dataclass(frozen=True)
class Simulation:
    testing_period = dict( 
        file_0 = '/Users/gio/Desktop/elia_group/dataset_elia/2023/01.json',
        file_1 = '/Users/gio/Desktop/elia_group/dataset_elia/2023/02.json',
        file_2 = '/Users/gio/Desktop/elia_group/dataset_elia/2023/03.json',
        file_3 = '/Users/gio/Desktop/elia_group/dataset_elia/2023/04.json',
        file_4 = '/Users/gio/Desktop/elia_group/dataset_elia/2023/05.json',
        window_size = 30,
        start_training = '2023-02-25',
        num_test_days = 80,
        forecasts_col = ['forecast', 'confidence10', 'confidence90'],
        measured_col = 'measured',
        display_metrics=True)
    
@dataclass(frozen=True)
class WeightedAvg:
    params = dict(window_size_valid = 5)

@dataclass(frozen=True)
class Stack:
    params = dict(

        save_info = './info_model/',
        save_file = 'previous_day.pickle',
        
        normalize = True,
        compute_abs_difference = False,
        add_quantile_predictions = False,
        augment_q50 = False,

        # prediction pipeline
        nr_cv_splits = 5,
        quantiles = [0.1, 0.9, 0.5],

        # params for 1st stage
        max_lags = 3,
        forecasters_diversity = False,
        lagged = True,
        augment = False,
        differenciate = False,

        # params for 2nd stage
        max_lags_var = 3,
        augment_var=False,

        baseline_model = 'diff_norm_dayahead',

        # Ensemble Learning
        model_type = 'LR',  # 'GBR' or 'LR'
        var_model_type = 'LR',  # 'GBR' or 'LR'

        gbr_update_every_days = 30,

        # forecasts model parameters
        gbr_config_params = {'learning_rate': [0.001, 0.005, 0.01],
                                'max_features' : [.85, .95, 1.0],
                                'max_depth': [3, 4],
                                'max_iter': [250, 500]},
        lr_config_params = {'alpha': [0.00001, 0.0001, 0.001, 0.005],
                            'fit_intercept' : [True, False]},

        # variability forecasts model parameters
        order_diff = 1,
        var_gbr_config_params = {'learning_rate': [0.001, 0.005, 0.01],
                                    'max_features' : [.85, .95, 1.0],
                                    'max_depth': [3, 4],
                                    'max_iter': [150, 300]},
        var_lr_config_params = {'alpha': [0.00001, 0.0001, 0.001, 0.005],
                                'fit_intercept' : [True, False]},

        nr_permutations=100,
        compute_second_stage = True,  # activate the second stage of the ensemble learning

        # plot settings

        plot_baseline = False,  # plot the baseline model
        plot_weighted_avg = False,  # plot the weighted average predictions
        plot_importance_gbr = False,  # plot the feature importances for the GBR model
        plot_importance_permutation_first_stage = False,  # plot the permutation importances first stage
        plot_importance_permutation_second_stage = True,  # plot the permutation importances second stage
        plot_importance_weighted_avg = False,  # plot the feature importances for the weighted average model
        zoom_in_variability = True, # zoom in the variability forecasts
    )