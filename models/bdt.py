"""
XGBoost multiclass BDT wrapper, with early stopping mirroring the DNN's
scheduler patience so the two models are tuned on comparable terms.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_bdt(X_train, y_train, X_val, y_val, num_class=5, random_state=42, verbose=True):
    """
    Train an XGBClassifier with early stopping on (X_val, y_val).
    Returns the fitted classifier.
    """
    bdt = xgb.XGBClassifier(
        n_estimators=2000,          
        max_depth=4,
        objective='multi:softprob',
        num_class=num_class,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=10,  
        eval_metric='mlogloss',
    )
    bdt.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    if verbose:
        print(f"BDT stopped at {bdt.best_iteration + 1} trees "
              f"(best val mlogloss = {bdt.best_score:.4f})")
    return bdt


def evaluate_bdt(bdt, X_test, y_test, class_names, verbose=True):
    """
    Returns (y_pred, y_prob, per_class_auc_dict).
    """
    y_pred = bdt.predict(X_test)
    y_prob = bdt.predict_proba(X_test)

    if verbose:
        print(f"BDT Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

    per_class_auc = {}
    for i, cls in enumerate(class_names):
        auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        per_class_auc[cls] = auc
        if verbose:
            print(f"  {cls} tagger: AUC = {auc:.3f}")

    return y_pred, y_prob, per_class_auc
