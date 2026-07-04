"""
Unified SHAP interface for the BDT (TreeExplainer) and DNN (DeepExplainer).

Both explain_* functions return an array of shape
(n_samples, n_features, n_classes), so downstream metrics code never needs
to know which model or explainer produced a given array.
"""
import numpy as np
import pandas as pd
import shap
import torch


def explain_bdt(bdt, X_shap_raw, feature_names, n_classes, save_path=None):
    """
    Calculates exact SHAP values for an XGBClassifier via TreeExplainer
    """
    X_df = pd.DataFrame(X_shap_raw, columns=feature_names)
    explainer = shap.TreeExplainer(bdt)
    sv = explainer(X_df)

    shap_arr = np.asarray(sv.values)
    expected_shape = (X_shap_raw.shape[0], len(feature_names), n_classes)
    assert shap_arr.shape == expected_shape, (
        f"Unexpected BDT SHAP shape: {shap_arr.shape}, expected {expected_shape}"
    )
    if save_path:
        np.save(save_path, shap_arr)
    return shap_arr


def explain_dnn(model, device, X_shap_scaled, X_train_scaled, feature_names, n_classes,
                 background_size=500, rng=None, save_path=None):
    """
    Calculates approximate SHAP values for a torch model via DeepExplainer.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    background_idx = rng.choice(len(X_train_scaled), background_size, replace=False)
    background = torch.tensor(X_train_scaled[background_idx], dtype=torch.float32).to(device)
    X_tensor = torch.tensor(X_shap_scaled, dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(model, background)
    shap_vals = explainer.shap_values(X_tensor)

    if isinstance(shap_vals, list):
        shap_arr = np.stack(shap_vals, axis=-1)
    else:
        shap_arr = np.array(shap_vals)
        # some shap versions return (n_samples, n_classes, n_features); transpose to match
        if shap_arr.shape[1] == n_classes and shap_arr.shape[2] == len(feature_names):
            shap_arr = np.transpose(shap_arr, (0, 2, 1))

    expected_shape = (X_shap_scaled.shape[0], len(feature_names), n_classes)
    assert shap_arr.shape == expected_shape, (
        f"Unexpected DNN SHAP shape: {shap_arr.shape}, expected {expected_shape}"
    )
    if save_path:
        np.save(save_path, shap_arr)
    return shap_arr


def mean_abs_importance(shap_arr, cls_idx=None):
    """
    Collapse a (n_samples, n_features, n_classes) SHAP array into a
    per-feature importance vector: mean(|SHAP|) over samples, and over
    classes too if cls_idx is None.
    """
    if cls_idx is None:
        return np.mean(np.abs(shap_arr), axis=(0, 2))
    return np.mean(np.abs(shap_arr[:, :, cls_idx]), axis=0)
