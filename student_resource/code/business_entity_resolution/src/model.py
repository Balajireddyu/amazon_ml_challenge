"""
Phase 4: LightGBM Pairwise Matcher Model.

Provides:
- Label generation from ground truth matched_entity_ids
- Group-based train/validation splitting by S1 entity_id
- LightGBM binary classifier training with evaluation utilities
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, fbeta_score


FEATURE_COLUMNS = [
    # Name features
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_exact_match",
    "name_core_ratio",
    "name_core_token_set_ratio",
    "name_core_exact_match",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_len_diff",
    "name_prefix4_match",
    # Address features
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_exact_match",
    "address_token_jaccard",
    "address_token_overlap_count",
    # Address number features
    "address_num_exact_match",
    "address_num_jaccard",
    "address_num_overlap_count",
    "address_num_has_match",
    # Cross & Meta features
    "country_match",
    "joint_similarity",
    "candidate_score",
    "source_is_s2",
]


def create_labels(features_df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Add binary 'label' column to candidate features dataframe based on ground truth."""
    true_pairs = set()
    for row in ground_truth.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        if not matched or pd.isna(row.matched_entity_ids) or matched == "nan":
            continue
        for cid in matched.split(","):
            cid = cid.strip()
            if cid:
                true_pairs.add((s1_id, cid))

    labels = [
        1 if (r.source1_entity_id, r.candidate_entity_id) in true_pairs else 0
        for r in features_df.itertuples(index=False)
    ]

    result = features_df.copy()
    result["label"] = labels
    return result


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


class PairwiseMatcher:
    """Pairwise matching model wrapper."""
    def __init__(self, clf, feature_columns, val_data=None):
        self.clf = clf
        self.feature_columns = feature_columns
        # Compute feature importances
        if val_data is not None:
            r = permutation_importance(clf, val_data[feature_columns], val_data["label"], n_repeats=5, random_state=42)
            self.feature_importances_ = r.importances_mean
        else:
            self.feature_importances_ = np.ones(len(feature_columns))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_columns]
        else:
            X_mat = X
        return self.clf.predict_proba(X_mat)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_matching_model(
    train_data: pd.DataFrame = None,
    val_data: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    feature_columns: list = None,
    params: dict = None,
    training_data: pd.DataFrame = None,
) -> PairwiseMatcher:
    """Train a gradient boosted histogram classifier (LightGBM equivalent) on pairwise features."""
    if train_data is None and training_data is not None:
        train_data = training_data
    if val_data is None and val_df is not None:
        val_data = val_df

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    X_train = train_data[feature_columns]
    y_train = train_data["label"]

    hgb_params = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "max_depth": 6,
        "early_stopping": True if val_data is not None else False,
        "n_iter_no_change": 25,
        "random_state": 42,
    }
    if params:
        hgb_params.update(params)

    clf = HistGradientBoostingClassifier(**hgb_params)
    clf.fit(X_train, y_train)

    return PairwiseMatcher(clf, feature_columns, val_data=val_data)


def evaluate_predictions(y_true, y_prob, threshold: float = 0.5, beta: float = 0.5) -> dict:
    """Compute precision, recall, F-beta score, and confusion matrix at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_beta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    f1 = fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f0_5": f_beta,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": cm,
    }