"""
Plotting functions: scatter of raw SHAP magnitude for BDT-vs-DNN, 
grouped bar charts of normalized importance share across
methods/classes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau


def scatter_grid_by_class(imp_a_per_class, imp_b_per_class, class_names, feature_names,
                           label_a="A", label_b="B", save_path=None):
    """
    imp_a_per_class, imp_b_per_class : list/array of shape (n_classes, n_features)
    One scatter subplot per class, with a y=x reference line and per-point
    feature-name annotations.
    """
    n_classes = len(class_names)
    ncols = 3
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for i, cls in enumerate(class_names):
        ax = axes[i]
        a, b = imp_a_per_class[i], imp_b_per_class[i]
        ax.scatter(a, b, s=60, color='steelblue', zorder=3)
        for j, fname in enumerate(feature_names):
            ax.annotate(fname, (a[j], b[j]), fontsize=7, xytext=(3, 3), textcoords='offset points')

        lims = [0, max(a.max(), b.max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1, zorder=1)
        ax.set_xlim(lims); ax.set_ylim(lims)

        tau_cls, _ = kendalltau(a, b)
        ax.set_title(f"Class {cls}  (τ = {tau_cls:.3f})", fontsize=12)
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.grid(alpha=0.3)

    for k in range(n_classes, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def grouped_bar_by_class(imp_dict_per_class, class_names, feature_names, save_path=None,
                          ylabel="Normalized share", title_prefix=""):
    """
    imp_dict_per_class : {method_name: array of shape (n_classes, n_features)}
    One horizontal row of bars per class, grouped by method, sorted by
    combined importance for readability.
    """
    methods = list(imp_dict_per_class.keys())
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(11, 5 * n_classes))
    if n_classes == 1:
        axes = [axes]

    width = 0.8 / len(methods)
    for i, cls in enumerate(class_names):
        ax = axes[i]
        per_method = {m: imp_dict_per_class[m][i] for m in methods}
        combined = sum(per_method.values())
        order = np.argsort(combined)[::-1]

        x = np.arange(len(feature_names))
        for k, m in enumerate(methods):
            offset = (k - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, per_method[m][order], width, label=m)

        ax.set_xticks(x)
        ax.set_xticklabels([feature_names[j] for j in order], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}Class {cls}")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def grouped_bar_overall(imp_dict, feature_names, save_path=None,
                         ylabel="Normalized share", title=""):
    """
    imp_dict : {method_name: array of shape (n_features,)}
    Single grouped bar chart (e.g. SHAP vs LIME vs XGB-native, overall).
    """
    methods = list(imp_dict.keys())
    combined = sum(imp_dict.values())
    order = np.argsort(combined)[::-1]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(feature_names))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        offset = (k - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, imp_dict[m][order], width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[j] for j in order], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
