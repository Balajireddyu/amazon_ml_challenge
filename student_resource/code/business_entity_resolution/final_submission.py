import os
import pandas as pd

from src.data_loader import load_training_data
from src.preprocessing import preprocess_all_sources
from src.candidate_generation import generate_fuzzy_candidates
from src.features import create_matching_features
from src.model import create_labels, train_matching_model, FEATURE_COLUMNS
from src.matching import predict_matches, create_submission_dataframe


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

print("\nTraining final model...")

model = train_matching_model(
    training_data
)

print("Final model training completed!")


# ============================================================
# 8. FEATURE COLUMNS
# ============================================================

feature_columns = FEATURE_COLUMNS


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
# 10. APPLY FINAL THRESHOLD & PREDICT MATCHES
# ============================================================

threshold = 0.50

final_matches = predict_matches(
    training_data,
    model,
    threshold=threshold
)


# ============================================================
# 11. CREATE SUBMISSION DATAFRAME & VALIDATION ASSERTIONS
# ============================================================

all_s1_ids = list(source1["entity_id"])

submission_df = create_submission_dataframe(
    final_matches,
    all_s1_ids
)

# --- VALIDATION ASSERTIONS ---
# A. S1 row count: exactly one row per S1 entity
assert len(submission_df) == len(source1), (
    f"Submission row count {len(submission_df)} does not match source1 count {len(source1)}"
)

# B. S1 ID set identity
assert set(submission_df["source1_entity_id"]) == set(source1["entity_id"]), (
    "Submission S1 IDs do not match source1 entity IDs exactly"
)

# C. No nulls in required columns
assert not submission_df[["source1_entity_id", "matched_entity_ids"]].isnull().values.any(), (
    "Submission contains null/NaN values"
)

# D. Predicted pair subset of candidate pairs
pred_pairs = set(zip(final_matches["source1_entity_id"], final_matches["candidate_entity_id"]))
cand_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))

assert pred_pairs.issubset(cand_pairs), (
    "Predicted match pairs exist that are not present in candidate_pairs"
)


# ============================================================
# 12. SAVE OUTPUTS TO output/ DIRECTORY
# ============================================================

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save candidate pairs: output/candidate_pairs.tsv
cand_pairs_file = os.path.join(output_dir, "candidate_pairs.tsv")
candidates[["source1_entity_id", "candidate_entity_id"]].to_csv(
    cand_pairs_file,
    sep="\t",
    index=False
)

# Save matching results: output/matching_results.tsv
matching_results_file = os.path.join(output_dir, "matching_results.tsv")
submission_df.to_csv(
    matching_results_file,
    sep="\t",
    index=False
)

print("\n==========================================")
print("FINAL SUBMISSION CREATED SUCCESSFULLY")
print("==========================================")

print(f"Total S1 entities processed: {len(submission_df):,}")
print(f"Total candidate pairs: {len(candidates):,}")
print(f"Total predicted match pairs: {len(pred_pairs):,}")

print(f"\nOutputs written:")
print(f"  - {cand_pairs_file}")
print(f"  - {matching_results_file}")

print("\nFirst 10 rows of matching_results.tsv:")
print(submission_df.head(10).to_string(index=False))