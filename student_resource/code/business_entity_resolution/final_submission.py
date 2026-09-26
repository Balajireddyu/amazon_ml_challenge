import pandas as pd

from src.data_loader import load_training_data
from src.preprocessing import preprocess_all_sources
from src.candidate_generation import generate_fuzzy_candidates
from src.features import create_matching_features
from src.model import create_labels, train_matching_model


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

source1, source2, source3, ground_truth = load_training_data(
    "../../dataset"
)

print("Data loaded.")


# ============================================================
# 2. PREPROCESS DATA
# ============================================================

print("\nPreprocessing data...")

source1, source2, source3 = preprocess_all_sources(
    source1,
    source2,
    source3
)

print("Data preprocessed.")


# ============================================================
# 3. GENERATE CANDIDATES
# ============================================================

print("\nGenerating candidates...")

candidates = generate_fuzzy_candidates(
    source1,
    source2,
    source3,
    top_k=20
)

print("Candidate generation completed.")

print("Total candidate pairs:")
print(len(candidates))


# ============================================================
# 4. CREATE MATCHING FEATURES
# ============================================================

print("\nCreating matching features...")

features = create_matching_features(
    candidates,
    source1,
    source2,
    source3
)

print("Feature creation completed.")


# ============================================================
# 5. CREATE LABELS
# ============================================================

print("\nCreating labels...")

labeled_candidates = create_labels(
    candidates,
    ground_truth
)

print("Labels created.")


# ============================================================
# 6. COMBINE FEATURES + LABELS
# ============================================================

training_data = features.merge(
    labeled_candidates[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label"
        ]
    ],
    on=[
        "source1_entity_id",
        "candidate_entity_id"
    ],
    how="left"
)

print("\nTraining data shape:")
print(training_data.shape)

print("\nLabel distribution:")
print(training_data["label"].value_counts())


# ============================================================
# 7. TRAIN FINAL MODEL
# ============================================================

print("\nTraining final LightGBM model...")

model = train_matching_model(
    training_data
)

print("Final model training completed!")


# ============================================================
# 8. FEATURE COLUMNS
# ============================================================

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

    "address_number_overlap",
    "combined_similarity"
]


# ============================================================
# 9. PREDICT MATCH PROBABILITIES
# ============================================================

print("\nPredicting final match probabilities...")

X = training_data[
    feature_columns
]

probabilities = model.predict_proba(X)[:, 1]

training_data["probability"] = probabilities


# ============================================================
# 10. APPLY FINAL THRESHOLD
# ============================================================

threshold = 0.50

final_matches = training_data[
    training_data["probability"] >= threshold
].copy()


# ============================================================
# 11. CREATE FINAL SUBMISSION
# ============================================================

submission = final_matches[
    [
        "source1_entity_id",
        "candidate_entity_id",
        "probability"
    ]
].copy()

submission = submission.sort_values(
    [
        "source1_entity_id",
        "probability"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# 12. SAVE OUTPUT
# ============================================================

output_file = "final_submission.csv"

submission.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("FINAL SUBMISSION CREATED")
print("==========================================")

print("Number of final matches:")
print(len(submission))

print("\nOutput file:")
print(output_file)

print("\nFirst 20 final matches:")
print(submission.head(20))