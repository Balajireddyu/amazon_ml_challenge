"""
Phase 5: Post-Processing, Matching Logic & Submission Formatting.

Provides:
- Prediction thresholding and source-wise bipartite/1-to-N matching logic
- Final competition submission formatting (source1_entity_id -> comma-separated matched_entity_ids)
- End-to-end evaluation metrics (Precision, Recall, F0.5, F1) against ground truth
"""

import pandas as pd
import numpy as np
from src.model import FEATURE_COLUMNS, PairwiseMatcher


def predict_matches(
    features_df: pd.DataFrame,
    model: PairwiseMatcher,
    threshold: float = 0.45,
    max_matches_per_source: int = None,
) -> pd.DataFrame:
    """
    Predict matching pairs from features using model probability threshold.
    
    Args:
        features_df: DataFrame containing the 27 pairwise features.
        model: Trained PairwiseMatcher model.
        threshold: Probability cutoff for positive match.
        max_matches_per_source: If set (e.g. 1), keeps only top-1 match per S1 from each source (S2, S3).
    """
    probs = model.predict_proba(features_df)[:, 1]
    df = features_df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    df["probability"] = probs

    # Filter by probability threshold
    matches = df[df["probability"] >= threshold].copy()

    # Sort by S1 and descending probability
    matches = matches.sort_values(
        ["source1_entity_id", "candidate_source", "probability"],
        ascending=[True, True, False],
    )

    if max_matches_per_source is not None:
        # Keep at most `max_matches_per_source` candidates per source per S1
        matches = matches.groupby(["source1_entity_id", "candidate_source"]).head(max_matches_per_source)

    return matches


def create_submission_dataframe(
    matches_df: pd.DataFrame,
    all_s1_ids: list,
) -> pd.DataFrame:
    """
    Aggregate predicted matches per S1 into comma-separated matched_entity_ids.
    Ensures all S1 records are present in the final output (empty string if no matches).
    """
    # Group matches by source1_entity_id
    grouped = matches_df.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(ids)
    ).to_dict()

    rows = []
    for s1_id in all_s1_ids:
        matched_str = grouped.get(s1_id, "")
        rows.append((s1_id, matched_str))

    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def evaluate_submission(
    submission_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    beta: float = 0.5,
) -> dict:
    """
    Compute pairwise Precision, Recall, F0.5, and F1 between submission and ground truth.
    """
    # Extract true pairs
    gt_pairs = set()
    for row in ground_truth_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        if not matched or pd.isna(row.matched_entity_ids) or matched == "nan":
            continue
        for cid in matched.split(","):
            cid = cid.strip()
            if cid:
                gt_pairs.add((s1_id, cid))

    # Extract predicted pairs
    pred_pairs = set()
    for row in submission_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        if not matched or pd.isna(row.matched_entity_ids) or matched == "nan":
            continue
        for cid in matched.split(","):
            cid = cid.strip()
            if cid:
                pred_pairs.add((s1_id, cid))

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    b2 = beta ** 2
    f_beta = ((1 + b2) * precision * recall / (b2 * precision + recall)) if (precision + recall) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f0_5": f_beta,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(pred_pairs),
        "gt_count": len(gt_pairs),
    }
