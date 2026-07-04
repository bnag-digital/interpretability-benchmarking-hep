"""
Data loading for the hls4ml LHC jets dataset via fetch_openml.

Verifies column order against a known-good expected order before returning
anything, since a silent column reshuffle would mislabel every downstream
SHAP/LIME importance array without raising an error.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

EXPECTED_FEATURE_ORDER = [
    'zlogz', 'c1_b0_mmdt', 'c1_b1_mmdt', 'c1_b2_mmdt', 'c2_b1_mmdt',
    'c2_b2_mmdt', 'd2_b1_mmdt', 'd2_b2_mmdt', 'd2_a1_b1_mmdt',
    'd2_a1_b2_mmdt', 'm2_b1_mmdt', 'm2_b2_mmdt', 'n2_b1_mmdt',
    'n2_b2_mmdt', 'mass_mmdt', 'multiplicity',
]


def load_hls4ml_jets(random_state: int = 42, verbose: bool = True):
    """
    Fetch the hls4ml_lhc_jets_hlf dataset and split it 60/20/20 into
    train/val/test. The split is stratified and shared: the BDT and DNN
    are trained/evaluated on identical rows, so downstream comparisons
    between them aren't confounded by different test sets.
    """
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X_raw = data['data']
    y = data['target']

    feature_names = list(X_raw.columns)
    if feature_names != EXPECTED_FEATURE_ORDER:
        mismatches = [
            f"position {i}: expected '{a}', got '{b}'"
            for i, (a, b) in enumerate(zip(EXPECTED_FEATURE_ORDER, feature_names))
            if a != b
        ]
        raise ValueError(
            "Column order from fetch_openml does not match the expected order. "
            "Any code that hardcodes feature_names instead of reading it from "
            "the DataFrame would mislabel SHAP/LIME rankings. Mismatches:\n"
            + "\n".join(mismatches)
        )
    if verbose:
        print("Column order matches expected order.")

    X = X_raw.values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if verbose:
        print("Classes (in label-encoder order):", list(le.classes_))

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )

    if verbose:
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": le,
    }


def make_scaled_copies(X_train, X_val, X_test):
    """
    Fit a StandardScaler on X_train only, and apply it to val/test.
    The DNN needs standardized inputs; the BDT uses the raw arrays directly.
    Returns (scaler, X_train_sc, X_val_sc, X_test_sc).
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    return scaler, X_train_sc, X_val_sc, X_test_sc
