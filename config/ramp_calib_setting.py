from dataclasses import dataclass
import os
from dotenv import load_dotenv
from sklearn.utils.fixes import parse_version, sp_version

# load the environment variables
load_dotenv()
current_path = os.getenv("PATH_CURRENT")
# set solver for quantile regression
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

@dataclass(frozen=True)
class Simulation:
    testing_period = dict(
        file_1 = current_path + '/dataset_elia/2023/01.json',
        file_2 = current_path + '/dataset_elia/2023/02.json',
        file_3 = current_path + '/dataset_elia/2023/03.json',
        file_4 = current_path + '/dataset_elia/2023/04.json',
        file_5 = current_path + '/dataset_elia/2023/05.json',
        file_6 = current_path + '/dataset_elia/2023/06.json',
        file_7 = current_path + '/dataset_elia/2023/07.json',
        file_8 = current_path + '/dataset_elia/2023/08.json',
        file_9 = current_path + '/dataset_elia/2023/09.json',
        file_10 = current_path + '/dataset_elia/2023/10.json',
        file_11 = current_path + '/dataset_elia/2023/11.json',
        file_12 = current_path + '/dataset_elia/2023/12.json',
        replace_nan = True,
        random_seed = 42,
        window_size = 30,
        start_training = '2023-02-01',
        num_test_days = 300,
        forecasts_col = ['forecast', 'confidence10', 'confidence90'],
        measured_col = 'measured',
        most_recent = True,
        malicious = False,
        malicious_name= 'mostrecent',
        noise_degree = 10,
        noisy = False,
        noisy_name = 'weekahead',
        scenario = 'malicious',
        save_scenario_contributions = False,
        display_metrics=True,
        boxplot = True,
        lof = True,
        kde = True)
    
@dataclass(frozen=True)
class WeightedAvg:
    params = dict(window_size_valid = 1)

@dataclass(frozen=True)
class Stack:
    params = dict(

        save_info = './info_model/',
        save_file = 'previous_day.pickle',
        
        # scaling with normalization or standardization
        scale_features = True,
        axis = 0,
        normalize = False,
        standardize = True,

        # add quantile predictions
        add_quantile_predictions = True,
        augment_q50 = True,

        # prediction pipeline
        nr_cv_splits = 3,
        quantiles = [0.1, 0.9, 0.5],

        # params for 1st stage
        max_lags = 3,
        forecasters_diversity = False,
        lagged = True,
        augment = True,
        differenciate = True,

        # params for 2nd stage
        max_lags_var = 3,
        augment_var=True,

        baseline_model = 'diff_norm_dayahead',

        # Ensemble Learning
        model_type = 'LR',  # 'GBR' or 'LR'
        var_model_type = 'LR',  # 'GBR' or 'LR'
        solver = solver,

        gbr_update_every_days = 15,

        conformalized_qr = False,
        day_calibration = 2,

        # forecasts model parameters
        gbr_config_params = {'learning_rate': [0.00001, 0.0001, 0.001, 0.005, 0.01],
                                'max_features' : [.98, 1.0],
                                'max_depth': [2, 3, 4],
                                'max_iter': [150]},

        lr_config_params = {'alpha': [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.0075, 0.01],
                            'fit_intercept' : [True, False]},

        # variability forecasts model parameters
        order_diff = 1,
        var_gbr_config_params = {'learning_rate': [0.0001, 0.001, 0.005, 0.01],
                                    'max_features' : [.98, 1.0],
                                    'max_depth': [2, 3, 4],
                                    'max_iter': [25]},
        var_lr_config_params = {'alpha': [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.0075, 0.01],
                                'fit_intercept' : [True, False]},

        contribution_method = 'permutation',  # 'shapley' or 'permutation'

        # Lasso coefs importance
        alpha = 0.01,  # significance level for the permutation test
        nr_pvalues_permutations = 1,  # number of permutations for the p-values

        # permutation importance
        nr_permutations = 50,

        # shapeley importance
        nr_row_permutations = 10,
        nr_col_permutations = 5,

        # plot settings
        plt_wind_power_ensemble = True,  # plot the wind power ensemble forecasts
        plt_wind_power_variability_ensemble = False,  # plot the wind power variability forecasts
        plot_baseline = False,  # plot the baseline model
        plot_weighted_avg = False,  # plot the weighted average predictions
        plot_importance_gbr = False,  # plot the feature importances for the GBR model
        
        plot_importance_first_stage = False,  # plot the permutation importances first stage
        plot_importance_second_stage = False,  # plot the permutation importances second stage

        plot_importance_lasso_coefs = False,  # plot the lasso coefficients
        
        plot_importance_weighted_avg = False,  # plot the feature importances for the weighted average model
        zoom_in_variability = True, # zoom in the variability forecasts

        # wind ramp detection
        detector = 'eq',  # 'eq', 'kde', 'box', 'lof'

        preprocess_ramps = True,  # preprocess training data (get rid of wind ramps)
        max_IQW = 1000,
        max_consecutive_points = 3,  # max number of consecutive anomalies for ramp alarm

        # params for eq detector
        threshold_quantile_eq = 0.95,

        # params for kde detector
        threshold_quantile_kde = 0.95,
        cv_folds_kde = 5,

        # params for box detector
        q1_box = 0.5, 
        q3_box = 0.75,
        k_box = 1.5,

        # params for lof detector
        n_neighbors_lof = 25,
        contamination_lof = 0.075,

    )