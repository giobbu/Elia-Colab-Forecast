import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(feature_importance, df_train_ensemble):
    "Plot feature importance"
    assert  len(feature_importance) == len(list(df_train_ensemble)), 'feature_importance and df_train_ensemble have different lengths'
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    _ = plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(df_train_ensemble.columns)[sorted_idx])
    plt.title("Feature Importance (training set)")
    plt.show()

