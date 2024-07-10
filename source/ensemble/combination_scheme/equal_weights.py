import pandas as pd

def calculate_equal_weights(df_test_norm_diff):
    " Calculate the mean prediction and quantiles using equal weights"
    assert 'norm_measured' in df_test_norm_diff.columns, "norm_measured column is missing"
    Q10 = df_test_norm_diff[['norm_weekaheadconfidence10', 
                                'norm_dayaheadconfidence10', 
                                'norm_dayahead11hconfidence10']].mean(axis=1)
    MEAN = df_test_norm_diff[['norm_weekaheadforecast', 
                                'norm_dayaheadforecast', 
                                'norm_dayahead11hforecast']].mean(axis=1)
    Q90 = df_test_norm_diff[['norm_weekaheadconfidence90', 
                                'norm_dayaheadconfidence90', 
                                'norm_dayahead11hconfidence90']].mean(axis=1)
    df_equal_weights = pd.DataFrame({
        'Q10': Q10,
        'mean_prediction': MEAN,
        'Q90': Q90
    }, index=df_test_norm_diff.index)
    df_equal_weights['target'] = df_test_norm_diff['norm_measured']
    return df_equal_weights