import pandas as pd
import lightgbm as lgb


def create_labels(candidates, ground_truth):
    true_pairs = set()

    for row in ground_truth.itertuples(index=False):
        source1_id = row.source1_entity_id
        matched_ids = row.matched_entity_ids

        if pd.isna(matched_ids):
            continue

        matched_ids = str(matched_ids).strip()

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):
            candidate_id = candidate_id.strip()

            if candidate_id:
                true_pairs.add((source1_id, candidate_id))

    labels = []

    for row in candidates.itertuples(index=False):
        pair = (
            row.source1_entity_id,
            row.candidate_entity_id
        )

        labels.append(1 if pair in true_pairs else 0)

    result = candidates.copy()
    result["label"] = labels

    return result


def train_matching_model(training_data):

    feature_columns = [
        "name_similarity",
        "name_partial_similarity",
        "name_token_sort_similarity",
        "name_token_set_similarity",

        "address_similarity",
        "address_partial_similarity",
        "address_token_sort_similarity",
        "address_token_set_similarity",

        "exact_name",
        "exact_address",
        "country_match",

        # New features
        "address_number_overlap",
        "combined_similarity"
    ]

    X = training_data[feature_columns]
    y = training_data["label"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        verbosity=-1
    )

    model.fit(X, y)

    return model