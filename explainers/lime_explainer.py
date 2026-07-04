"""
LimeTabularExplainer wrapper. Returns arrays in the same
(n_samples, n_features, n_classes) shape convention as shap_explainer.py,
so metrics/rank_correlation.py can treat SHAP and LIME outputs identically.
Callers typically explain a smaller subsample
than the SHAP subsample (e.g. 4000 of the 20000 SHAP rows).
"""
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def build_lime_explainers(X_train_raw, X_train_scaled, feature_names, class_names, random_state=42):
    """
    One LimeTabularExplainer per model, each trained on the scale that
    model's predict_proba actually expects (raw for BDT, standardized for DNN).
    Returns (explainer_bdt, explainer_dnn).
    """
    explainer_bdt = LimeTabularExplainer(
        training_data=X_train_raw,
        feature_names=feature_names,
        class_names=list(class_names),
        mode='classification',
        random_state=random_state,
    )
    explainer_dnn = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=list(class_names),
        mode='classification',
        random_state=random_state,
    )
    return explainer_bdt, explainer_dnn


def explain_with_lime(explainer, X_to_explain, predict_proba_fn, n_features, n_classes,
                       save_path=None, log_every=10, verbose=True):
    """
    Run LIME row-by-row over X_to_explain against predict_proba_fn.
    """
    n_samples = len(X_to_explain)
    lime_arr = np.zeros((n_samples, n_features, n_classes))

    for row in range(n_samples):
        exp = explainer.explain_instance(
            X_to_explain[row],
            predict_proba_fn,
            labels=list(range(n_classes)),
            num_features=n_features,
        )
        for cls_idx in range(n_classes):
            for feat_idx, coef in exp.local_exp[cls_idx]:
                lime_arr[row, feat_idx, cls_idx] = coef
        if verbose and (row + 1) % log_every == 0:
            print(f"  {row + 1}/{n_samples} rows done")

    if save_path:
        np.save(save_path, lime_arr)
    return lime_arr


def mean_abs_importance(lime_arr, cls_idx=None):
    """Same convention as shap_explainer.mean_abs_importance."""
    if cls_idx is None:
        return np.mean(np.abs(lime_arr), axis=(0, 2))
    return np.mean(np.abs(lime_arr[:, :, cls_idx]), axis=0)
