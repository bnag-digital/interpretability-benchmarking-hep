"""
Rank-agreement metrics between any two (or more) feature-importance vectors,
regardless of which explainer/model produced them. 
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata


def normalize_to_share(imp: np.ndarray) -> np.ndarray:
    return imp / imp.sum()


def kendall_tau_between(imp_a: np.ndarray, imp_b: np.ndarray):
    return kendalltau(imp_a, imp_b)


def ranks_from_importance(imp: np.ndarray) -> np.ndarray:
    return rankdata(-imp, method='average')


def agreement_table(imp_a, imp_b, feature_names, name_a="A", name_b="B", tol=1.0):
    """
    Side-by-side rank table for two importance vectors, with an "Agree?"
    column when ranks differ by <= tol.
    """
    ranks_a = ranks_from_importance(imp_a)
    ranks_b = ranks_from_importance(imp_b)
    agree = np.abs(ranks_a - ranks_b) <= tol
    return pd.DataFrame({
        "feature": feature_names,
        f"{name_a}_rank": ranks_a,
        f"{name_b}_rank": ranks_b,
        "agree": agree,
    })


def rank_table(imps_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Given {method_name: importance_array}, build a DataFrame with each
    method's normalized share, its rank, and the average rank across
    methods -- sorted by average rank. Works for 2 methods (SHAP vs LIME)
    or 3+ (SHAP vs LIME vs XGB native).
    """
    ranks = {method: ranks_from_importance(imp) for method, imp in imps_dict.items()}

    rows = []
    for j, fname in enumerate(feature_names):
        row = {"feature": fname}
        for method, imp in imps_dict.items():
            row[f"{method}_share"] = imp[j]
            row[f"{method}_rank"] = ranks[method][j]
        row["avg_rank"] = np.mean([ranks[m][j] for m in imps_dict])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
    df.index += 1
    return df


def per_class_tau(shap_or_lime_arr_a, shap_or_lime_arr_b, class_names, mean_abs_fn):
    """
    Kendall's tau per class between two importance arrays of shape
    (n_samples, n_features, n_classes), using mean_abs_fn (e.g.
    explainers.shap_explainer.mean_abs_importance) to collapse each to a
    per-feature vector before comparing.
    """
    results = []
    for i, cls in enumerate(class_names):
        imp_a = mean_abs_fn(shap_or_lime_arr_a, cls_idx=i)
        imp_b = mean_abs_fn(shap_or_lime_arr_b, cls_idx=i)
        tau, p = kendalltau(imp_a, imp_b)
        results.append((cls, tau, p))
    return results
