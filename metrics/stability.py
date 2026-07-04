"""
Bootstrap stability of feature-importance rankings.
"""
import numpy as np
from scipy.stats import kendalltau


def bootstrap_importance_ci(shap_or_lime_arr, mean_abs_fn, cls_idx=None,
                             n_bootstrap=200, ci=0.90, random_state=42):
    """
    Bootstrap over the sample axis (axis 0) of a (n_samples, n_features[, n_classes])
    importance array to get a confidence interval on each feature's importance share.

    Returns dict with:
        point_estimate : (n_features,) array, importance on the full sample
        lower, upper    : (n_features,) arrays, the ci-level bootstrap interval
        boot_matrix     : (n_bootstrap, n_features) array of all bootstrap draws
    """
    rng = np.random.default_rng(random_state)
    n_samples = shap_or_lime_arr.shape[0]

    point_estimate = mean_abs_fn(shap_or_lime_arr, cls_idx=cls_idx)
    n_features = point_estimate.shape[0]

    boot_matrix = np.zeros((n_bootstrap, n_features))
    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        boot_matrix[b] = mean_abs_fn(shap_or_lime_arr[idx], cls_idx=cls_idx)

    alpha = (1 - ci) / 2
    lower = np.quantile(boot_matrix, alpha, axis=0)
    upper = np.quantile(boot_matrix, 1 - alpha, axis=0)

    return {
        "point_estimate": point_estimate,
        "lower": lower,
        "upper": upper,
        "boot_matrix": boot_matrix,
    }


def bootstrap_rank_stability(shap_or_lime_arr, mean_abs_fn, cls_idx=None,
                              n_bootstrap=200, random_state=42):
    """
    Measures how often the top-k feature ranking is preserved across
    bootstrap resamples, by computing Kendall's tau between the
    full-sample ranking and each bootstrap resample's ranking.

    Returns dict with:
        tau_values : (n_bootstrap,) array of tau(full_sample_ranking, bootstrap_ranking)
        mean_tau, std_tau : summary stats -- higher mean_tau / lower std_tau means
                             the ranking is more stable under resampling
    """
    rng = np.random.default_rng(random_state)
    n_samples = shap_or_lime_arr.shape[0]

    point_estimate = mean_abs_fn(shap_or_lime_arr, cls_idx=cls_idx)

    tau_values = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        boot_imp = mean_abs_fn(shap_or_lime_arr[idx], cls_idx=cls_idx)
        tau_values[b], _ = kendalltau(point_estimate, boot_imp)

    return {
        "tau_values": tau_values,
        "mean_tau": float(np.mean(tau_values)),
        "std_tau": float(np.std(tau_values)),
    }
