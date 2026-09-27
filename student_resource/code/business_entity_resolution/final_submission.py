import os
import gc
import pandas as pd

from src.data_loader import load_test_data
from src.preprocessing import preprocess_all_sources
from src.candidate_generation import (
    InvertedCandidateIndex,
    generate_candidates_from_index,
)
from src.features import create_matching_features
from src.model import create_labels, train_matching_model, FEATURE_COLUMNS
from src.matching import predict_matches, create_submission_dataframe


# ============================================================
# CONSTANTS  (do not change)
# ============================================================

TEST_CHUNK_SIZE   = 100_000  # S1 entities processed per inference chunk
TOP_K             = 20       # candidates per S1
THRESHOLD         = 0.50     # match probability threshold
MAX_BUCKET_SIZE   = 500      # inverted-index bucket pruning size


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(preferred_paths):
    for p in preferred_paths:
        if os.path.exists(p):
            return p
    return preferred_paths[0]


VAL_DATA_DIR = resolve_path([
    os.environ.get("VAL_DATA_DIR", "C:/am/val_10k_data"),
    "C:/am/val_10k_data",
    "../../val_10k_data",
    "val_10k_data",
])

TEST_DATA_DIR = resolve_path([
    os.environ.get("DATASET_DIR", "../../dataset"),
    "../../dataset",
    "dataset",
    "C:/am/dataset",
])


# ============================================================
# 1. LOAD TRAINING DATA (val_10k_data)
# ============================================================

print(f"Loading training data from {VAL_DATA_DIR}...")

train_source1 = pd.read_csv(os.path.join(VAL_DATA_DIR, "s1_10k.tsv"), sep="\t")
train_source2 = pd.read_csv(os.path.join(VAL_DATA_DIR, "s2_10k.tsv"), sep="\t")
train_source3 = pd.read_csv(os.path.join(VAL_DATA_DIR, "s3_10k.tsv"), sep="\t")
ground_truth  = pd.read_csv(os.path.join(VAL_DATA_DIR, "gt_10k.tsv"), sep="\t")

print(
    f"Training data loaded: S1={len(train_source1):,}  S2={len(train_source2):,}"
    f"  S3={len(train_source3):,}  GT={len(ground_truth):,}"
)


# ============================================================
# 2. PREPROCESS TRAINING DATA
# ============================================================

print("\nPreprocessing training data...")

train_source1, train_source2, train_source3 = preprocess_all_sources(
    train_source1, train_source2, train_source3
)

print("Training data preprocessed.")


# ============================================================
# 3. BUILD TRAINING INDEX & GENERATE CANDIDATES
# ============================================================

print("\nBuilding training index over S2+S3...")

train_index = InvertedCandidateIndex(max_bucket_size=MAX_BUCKET_SIZE)
train_index.add_source(train_source2, "S2")
train_index.add_source(train_source3, "S3")
train_index.finalize()

print("Training index built.")

print(
    f"\nGenerating training candidates (K={TOP_K}) for {len(train_source1):,} S1..."
)

train_candidates = generate_candidates_from_index(
    train_source1, train_index, top_k=TOP_K
)

print(f"Training candidate pairs: {len(train_candidates):,}")

# Index no longer needed — free it before feature extraction
del train_index
gc.collect()
print("Training index freed.")


# ============================================================
# 4. CREATE TRAINING FEATURES (27 FEATURES)
# ============================================================

print("\nCreating training matching features (27 features)...")

train_features = create_matching_features(
    train_candidates, train_source1, train_source2, train_source3
)

print(f"Training features: {len(train_features):,} rows x {len(train_features.columns)} cols")


# ============================================================
# 5. CREATE LABELS & BUILD TRAINING DATASET
# ============================================================

print("\nCreating training labels...")

labeled_candidates = create_labels(train_candidates, ground_truth)

training_data = train_features.merge(
    labeled_candidates[["source1_entity_id", "candidate_entity_id", "label"]],
    on=["source1_entity_id", "candidate_entity_id"],
    how="left",
)
training_data["label"] = training_data["label"].fillna(0).astype(int)

print(f"Training data shape: {training_data.shape}")
print("\nLabel distribution:")
print(training_data["label"].value_counts())

# Free intermediate DataFrames before model fit
del train_candidates, train_features, labeled_candidates, train_source1
del train_source2, train_source3, ground_truth
gc.collect()


# ============================================================
# 6. TRAIN FINAL MODEL (27 FEATURES, HistGradientBoosting)
# ============================================================

print("\nTraining final model with 27 features...")

model = train_matching_model(training_data, feature_columns=FEATURE_COLUMNS)

print("Model training completed!")

del training_data
gc.collect()


# ============================================================
# 7. LOAD & PREPROCESS TEST DATA
# ============================================================

print(f"\nLoading test data from {TEST_DATA_DIR}...")

test_source1, test_source2, test_source3 = load_test_data(TEST_DATA_DIR)

print(
    f"Test data loaded: S1={len(test_source1):,}  S2={len(test_source2):,}"
    f"  S3={len(test_source3):,}"
)

print("\nPreprocessing test data...")

test_source1, test_source2, test_source3 = preprocess_all_sources(
    test_source1, test_source2, test_source3
)

print("Test data preprocessed.")


# ============================================================
# 8. BUILD TEST INDEX ONCE (full test S2+S3)
# ============================================================

print("\nBuilding test index over full test S2+S3...")

test_index = InvertedCandidateIndex(max_bucket_size=MAX_BUCKET_SIZE)
test_index.add_source(test_source2, "S2")
test_index.add_source(test_source3, "S3")
test_index.finalize()

print("Test index built.")


# ============================================================
# 9. PREPARE OUTPUT FILES (write headers once)
# ============================================================

os.makedirs("output", exist_ok=True)

cand_out_path  = "output/candidate_pairs.tsv"
match_out_path = "output/matching_results.tsv"

with open(cand_out_path, "w") as f:
    f.write("source1_entity_id\tcandidate_entity_id\n")
with open(match_out_path, "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")


# ============================================================
# 10. STREAM TEST INFERENCE — chunk by chunk, no full materialisation
# ============================================================

test_s1_ids = test_source1["entity_id"].tolist()
n_test      = len(test_source1)
n_chunks    = (n_test + TEST_CHUNK_SIZE - 1) // TEST_CHUNK_SIZE

print(
    f"\nStreaming test inference: {n_test:,} S1s in {n_chunks} chunk(s)"
    f" of up to {TEST_CHUNK_SIZE:,}..."
)

all_written_s1_ids = []   # track every written S1 for final assertion
total_cand_pairs   = 0
total_pred_pairs   = 0

for chunk_idx in range(n_chunks):
    chunk_start = chunk_idx * TEST_CHUNK_SIZE
    chunk_end   = min(chunk_start + TEST_CHUNK_SIZE, n_test)
    chunk_s1    = test_source1.iloc[chunk_start:chunk_end].copy()
    chunk_s1_ids = chunk_s1["entity_id"].tolist()

    print(
        f"  Chunk {chunk_idx + 1}/{n_chunks}: rows {chunk_start:,}–{chunk_end:,}"
        f" ({len(chunk_s1):,} S1 records)...",
        flush=True,
    )

    # Candidate generation (index unchanged, K=20)
    chunk_cands = generate_candidates_from_index(chunk_s1, test_index, top_k=TOP_K)

    # 27-feature extraction (test_source2/3 full pools, filtered internally by needed IDs)
    chunk_feats = create_matching_features(
        chunk_cands, chunk_s1, test_source2, test_source3
    )

    # Model prediction (threshold=0.50)
    chunk_matches = predict_matches(chunk_feats, model, threshold=THRESHOLD)

    # Per-chunk assertion: every predicted pair must exist in candidate pairs
    chunk_cand_set = set(
        zip(chunk_cands["source1_entity_id"], chunk_cands["candidate_entity_id"])
    )
    chunk_pred_set = set(
        zip(chunk_matches["source1_entity_id"], chunk_matches["candidate_entity_id"])
    )
    assert chunk_pred_set.issubset(chunk_cand_set), (
        f"Chunk {chunk_idx + 1}: {len(chunk_pred_set - chunk_cand_set)} predicted"
        " pairs not present in candidate_pairs!"
    )

    # Append candidate_pairs.tsv (no header — already written)
    chunk_cands[["source1_entity_id", "candidate_entity_id"]].to_csv(
        cand_out_path, sep="\t", index=False, header=False, mode="a"
    )

    # Format and append matching_results.tsv (every S1 in chunk appears)
    chunk_submission = create_submission_dataframe(chunk_matches, chunk_s1_ids)

    assert len(chunk_submission) == len(chunk_s1_ids), (
        f"Chunk {chunk_idx + 1}: submission row count {len(chunk_submission)}"
        f" != expected {len(chunk_s1_ids)}"
    )

    chunk_submission.to_csv(
        match_out_path, sep="\t", index=False, header=False, mode="a"
    )

    total_cand_pairs += len(chunk_cands)
    total_pred_pairs += len(chunk_matches)
    all_written_s1_ids.extend(chunk_s1_ids)

    del chunk_s1, chunk_cands, chunk_feats, chunk_matches, chunk_submission
    gc.collect()

del test_index
del test_source1, test_source2, test_source3
gc.collect()


# ============================================================
# 11. FINAL ASSERTIONS
# ============================================================

print("\nRunning final assertions...")

# Every test S1 written exactly once
assert len(all_written_s1_ids) == len(test_s1_ids), (
    f"S1 count mismatch: expected {len(test_s1_ids)}, got {len(all_written_s1_ids)}"
)
assert set(all_written_s1_ids) == set(test_s1_ids), (
    "Test S1 entity ID set mismatch in output!"
)

# Output files readable with correct schemas
df_check_cp = pd.read_csv(cand_out_path, sep="\t", nrows=5)
df_check_mr = pd.read_csv(match_out_path, sep="\t", nrows=5)

assert list(df_check_cp.columns) == ["source1_entity_id", "candidate_entity_id"], (
    f"candidate_pairs.tsv column error: {list(df_check_cp.columns)}"
)
assert list(df_check_mr.columns) == ["source1_entity_id", "matched_entity_ids"], (
    f"matching_results.tsv column error: {list(df_check_mr.columns)}"
)

print("All final assertions passed!")


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL SUBMISSION OUTPUTS CREATED")
print("==========================================")
print(f"candidate_pairs.tsv  : {cand_out_path}  ({total_cand_pairs:,} pairs)")
print(f"matching_results.tsv : {match_out_path}  ({len(test_s1_ids):,} S1 records)")
print(f"Total predicted matches: {total_pred_pairs:,}")
print("\nFirst 10 rows of matching_results.tsv:")
print(pd.read_csv(match_out_path, sep="\t").head(10).to_string(index=False))