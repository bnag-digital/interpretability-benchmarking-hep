# interpretability-benchmarking-hep

Model-agnostic benchmarking of post-hoc explanation methods (SHAP, LIME) and
rank-agreement metrics for jet classifiers (BDT, DNN, and eventually
attention-based architectures) trained on HEP tagging datasets.

## Structure

```
interpretability-benchmarking-hep/
├── data/loader.py                  # fetch_openml wrapper
├── models/bdt.py                   # XGBClassifier wrapper with early stopping
├── models/dnn.py                   # JetDNN + training loop
├── explainers/shap_explainer.py    # SHAP TreeExplainer + DeepExplainer
├── explainers/lime_explainer.py    # LimeTabularExplainer
├── metrics/rank_correlation.py     # Kendall's tau, normalized share, rank tables
├── metrics/stability.py            # measure uncertainty on importance/rankings
├── plots/comparison_plots.py       # scatter + grouped bar plotting helpers
├── notebooks/full_analysis.ipynb   # example usage
```

## Quickstart

```python
from data.loader import load_hls4ml_jets, make_scaled_copies
from models.bdt import train_bdt, evaluate_bdt
from models.dnn import train_dnn, evaluate_dnn, dnn_predict_proba_factory
from explainers.shap_explainer import explain_bdt, explain_dnn, mean_abs_importance
from metrics.rank_correlation import kendall_tau_between, rank_table

d = load_hls4ml_jets()
scaler, X_train_sc, X_val_sc, X_test_sc = make_scaled_copies(d["X_train"], d["X_val"], d["X_test"])

bdt = train_bdt(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
model, device = train_dnn(X_train_sc, d["y_train"], X_val_sc, d["y_val"])

# ... compute SHAP for both models on a shared subsample, then:
# tau, p = kendall_tau_between(bdt_importance, dnn_importance)
```

## Environment

Managed with [pixi](https://pixi.sh).

## Data

Uses the `hls4ml_lhc_jets_hlf` OpenML dataset (16 substructure features, 5
jet classes: g, q, t, w, z).
## Roadmap

