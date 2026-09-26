import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

from src.data_loader import load_training_data
from src.preprocessing import preprocess_all_sources
from src.candidate_generation import generate_fuzzy_candidates
from src.features import create_matching_features
from src.model import create_labels, train_matching_model


# ============================================================
# 1. LOAD DATA
# ============================================================

source1, source2, source3, ground_truth = load_training_data(
    "../../dataset"
)

print("Data loaded.")


# ============================================================
# 2. PREPROCESS
# ============================================================

source1, source2, source3 = preprocess_all_sources(
    source1,
    source2,
    source3
)

print("Data preprocessed.")


# ============================================================
# 3. SMALL SOURCE-1 DATASET
# ============================================================

source1_sample = source1.head(100)

sample_ids = source1_sample["entity_id"].tolist()

print("\nSmall test dataset:")
print("Source 1:", len(source1_sample))


# ============================================================
# 4. GROUND TRUTH FOR THE 100 S1 RECORDS
# ============================================================

ground_truth_sample = ground_truth[
    ground_truth["source1_entity_id"].isin(sample_ids)
].copy()


# ============================================================
# 5. EXTRACT TRUE MATCHES
# ============================================================

true_match_rows = []

for row in ground_truth_sample.itertuples(index=False):

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
            true_match_rows.append(
                (source1_id, candidate_id)
            )


true_matches = pd.DataFrame(
    true_match_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
).drop_duplicates()


print("\nTotal true matches:")
print(len(true_matches))


# ============================================================
# 6. GET TRUE S2 / S3 IDs
# ============================================================

true_s2_ids = set(
    true_matches[
        true_matches["candidate_entity_id"].str.startswith("S2-")
    ]["candidate_entity_id"]
)

true_s3_ids = set(
    true_matches[
        true_matches["candidate_entity_id"].str.startswith("S3-")
    ]["candidate_entity_id"]
)


print("\nTrue Source 2 matches:")
print(len(true_s2_ids))

print("\nTrue Source 3 matches:")
print(len(true_s3_ids))


# ============================================================
# 7. CREATE SMALL S2 / S3 DATASETS
# ============================================================

source2_true = source2[
    source2["entity_id"].isin(true_s2_ids)
].copy()

source2_random = source2[
    ~source2["entity_id"].isin(true_s2_ids)
].sample(
    n=2000,
    random_state=42
)

source2_validation = pd.concat(
    [source2_true, source2_random],
    ignore_index=True
)


source3_true = source3[
    source3["entity_id"].isin(true_s3_ids)
].copy()

source3_random = source3[
    ~source3["entity_id"].isin(true_s3_ids)
].sample(
    n=2000,
    random_state=42
)

source3_validation = pd.concat(
    [source3_true, source3_random],
    ignore_index=True
)


print("\nValidation dataset sizes:")
print("Source 1:", len(source1_sample))
print("Source 2:", len(source2_validation))
print("Source 3:", len(source3_validation))


# ============================================================
# 8. GENERATE CANDIDATES
# ============================================================

print("\nGenerating fuzzy candidates...")

candidates = generate_fuzzy_candidates(
    source1_sample,
    source2_validation,
    source3_validation,
    top_k=20
)

print("\nCandidate generation completed.")

print("Number of candidate pairs:")
print(len(candidates))


# ============================================================
# 9. CANDIDATE RECALL
# ============================================================

found_matches = candidates.merge(
    true_matches,
    on=[
        "source1_entity_id",
        "candidate_entity_id"
    ],
    how="inner"
)

total_true_matches = len(true_matches)
found_true_matches = len(found_matches)

candidate_recall = (
    found_true_matches / total_true_matches
    if total_true_matches > 0
    else 0
)

print("\nOverall Candidate Recall:")
print(f"{candidate_recall:.2%}")

# ============================================================
# FIND MISSED TRUE MATCHES
# ============================================================

missed_matches = true_matches.merge(
    candidates[
        ["source1_entity_id", "candidate_entity_id"]
    ],
    on=[
        "source1_entity_id",
        "candidate_entity_id"
    ],
    how="left",
    indicator=True
)

missed_matches = missed_matches[
    missed_matches["_merge"] == "left_only"
].drop(columns=["_merge"])

print("\n==========================================")
print("MISSED TRUE MATCHES")
print("==========================================")

print("Number of missed true matches:")
print(len(missed_matches))

print("\nMissed matches:")
print(missed_matches)
# ============================================================
# INSPECT MISSED MATCH DETAILS
# ============================================================

print("\n==========================================")
print("MISSED MATCH DETAILS")
print("==========================================")

for _, match in missed_matches.iterrows():

    s1_id = match["source1_entity_id"]
    candidate_id = match["candidate_entity_id"]

    s1_row = source1_sample[
        source1_sample["entity_id"] == s1_id
    ]

    if candidate_id.startswith("S2-"):
        s2_row = source2_validation[
            source2_validation["entity_id"] == candidate_id
        ]
        source_name = "S2"

    else:
        s2_row = source3_validation[
            source3_validation["entity_id"] == candidate_id
        ]
        source_name = "S3"

    print("\n------------------------------------------")
    print(f"Source: {source_name}")

    print("S1 ID:", s1_id)
    print("Candidate ID:", candidate_id)

    if not s1_row.empty and not s2_row.empty:

        s1 = s1_row.iloc[0]
        s2 = s2_row.iloc[0]

        print("S1 Country :", s1["country_normalized"])
        print("S2/S3 Country :", s2["country_normalized"])

        print("S1 Name :", s1["business_name_normalized"])
        print("S2/S3 Name :", s2["business_name_normalized"])

        print("S1 Address :", s1["business_address_normalized"])
        print("S2/S3 Address :", s2["business_address_normalized"])
# ============================================================
# 10. CREATE FEATURES
# ============================================================

print("\nCreating matching features...")

features = create_matching_features(
    candidates,
    source1_sample,
    source2_validation,
    source3_validation
)

print("Feature creation completed.")


# ============================================================
# 11. CREATE LABELS
# ============================================================

print("\nCreating labels...")

labeled_candidates = create_labels(
    candidates,
    ground_truth_sample
)

print("Labels created.")


# ============================================================
# 12. COMBINE FEATURES + LABELS
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
# 13. SPLIT SOURCE-1 ENTITIES
# ============================================================

train_ids, validation_ids = train_test_split(
    sample_ids,
    test_size=0.20,
    random_state=42
)

train_ids = set(train_ids)
validation_ids = set(validation_ids)


print("\nEntity-level split:")
print("Training S1 entities:", len(train_ids))
print("Validation S1 entities:", len(validation_ids))


# ============================================================
# 14. CREATE TRAINING / VALIDATION DATA
# ============================================================

train_data = training_data[
    training_data["source1_entity_id"].isin(train_ids)
].copy()

validation_data = training_data[
    training_data["source1_entity_id"].isin(validation_ids)
].copy()


print("\nTraining candidate pairs:")
print(len(train_data))

print("Validation candidate pairs:")
print(len(validation_data))


print("\nTraining labels:")
print(train_data["label"].value_counts())

print("\nValidation labels:")
print(validation_data["label"].value_counts())


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\nTraining LightGBM model...")

model = train_matching_model(
    train_data
)

print("Model training completed!")


# ============================================================
# 16. PREDICT VALIDATION PROBABILITIES
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

X_validation = validation_data[
    feature_columns
]

y_validation = validation_data["label"]


probabilities = model.predict_proba(
    X_validation
)[:, 1]


validation_data = validation_data.copy()

validation_data["probability"] = probabilities


print("\nPrediction completed.")

print("\nProbability statistics:")

print(
    pd.Series(probabilities).describe()
)

print("\nFirst 30 probabilities:")

print(
    probabilities[:30]
)
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (probabilities >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, y_pred))

print("\nClassification Report:")
print(classification_report(y_validation, y_pred))
# ============================================================
# INSPECT FALSE POSITIVES AND FALSE NEGATIVES
# ============================================================

print("\n==========================================")
print("MISCLASSIFIED PAIRS")
print("==========================================")


# Add predictions to validation data
validation_data = validation_data.copy()
validation_data["prediction"] = y_pred


# ------------------------------------------------------------
# FALSE POSITIVES
# ------------------------------------------------------------

false_positives = validation_data[
    (validation_data["label"] == 0) &
    (validation_data["prediction"] == 1)
].copy()


print("\n==========================================")
print("FALSE POSITIVES")
print("==========================================")

print("Number of false positives:")
print(len(false_positives))


# ------------------------------------------------------------
# FALSE NEGATIVES
# ------------------------------------------------------------

false_negatives = validation_data[
    (validation_data["label"] == 1) &
    (validation_data["prediction"] == 0)
].copy()


print("\n==========================================")
print("FALSE NEGATIVES")
print("==========================================")

print("Number of false negatives:")
print(len(false_negatives))


# ============================================================
# CREATE LOOKUPS
# ============================================================

source1_lookup = source1_sample.set_index(
    "entity_id"
).to_dict("index")


source2_lookup = source2_validation.set_index(
    "entity_id"
).to_dict("index")


source3_lookup = source3_validation.set_index(
    "entity_id"
).to_dict("index")


# ============================================================
# FUNCTION TO DISPLAY A MISCLASSIFIED PAIR
# ============================================================

def print_pair_details(row, error_type):

    s1_id = row["source1_entity_id"]
    candidate_id = row["candidate_entity_id"]
    candidate_source = row["source"]

    s1 = source1_lookup.get(s1_id)

    if candidate_source == "S2":
        candidate = source2_lookup.get(candidate_id)
    else:
        candidate = source3_lookup.get(candidate_id)


    print("\n------------------------------------------")
    print(error_type)
    print("------------------------------------------")

    print("Source       :", candidate_source)
    print("S1 ID        :", s1_id)
    print("Candidate ID :", candidate_id)

    print(
        "Probability  :",
        f"{row['probability']:.6f}"
    )

    print(
        "Actual label :",
        int(row["label"])
    )

    print(
        "Prediction   :",
        int(row["prediction"])
    )

    if s1 is not None:

        print(
            "\nS1 Country  :",
            s1["country_normalized"]
        )

        print(
            "S1 Name     :",
            s1["business_name_normalized"]
        )

        print(
            "S1 Address  :",
            s1["business_address_normalized"]
        )


    if candidate is not None:

        print(
            "\nCandidate Country :",
            candidate["country_normalized"]
        )

        print(
            "Candidate Name    :",
            candidate["business_name_normalized"]
        )

        print(
            "Candidate Address :",
            candidate["business_address_normalized"]
        )


    print(
        "\nName similarity           :",
        f"{row['name_similarity']:.2f}"
    )

    print(
        "Name partial similarity  :",
        f"{row['name_partial_similarity']:.2f}"
    )

    print(
        "Name token sort          :",
        f"{row['name_token_sort_similarity']:.2f}"
    )

    print(
        "Name token set           :",
        f"{row['name_token_set_similarity']:.2f}"
    )

    print(
        "Address similarity       :",
        f"{row['address_similarity']:.2f}"
    )

    print(
        "Address partial          :",
        f"{row['address_partial_similarity']:.2f}"
    )

    print(
        "Address token sort       :",
        f"{row['address_token_sort_similarity']:.2f}"
    )

    print(
        "Address token set        :",
        f"{row['address_token_set_similarity']:.2f}"
    )

    print(
        "Exact name               :",
        row["exact_name"]
    )

    print(
        "Exact address            :",
        row["exact_address"]
    )

    print(
        "Country match             :",
        row["country_match"]
    )


# ============================================================
# PRINT FALSE POSITIVES
# ============================================================

for _, row in false_positives.iterrows():

    print_pair_details(
        row,
        "FALSE POSITIVE"
    )


# ============================================================
# PRINT FALSE NEGATIVES
# ============================================================

for _, row in false_negatives.iterrows():

    print_pair_details(
        row,
        "FALSE NEGATIVE"
    )
# ============================================================
# 17. CALCULATE PAIR-LEVEL METRICS
# ============================================================

print("\nPair-level metrics:")

pair_predictions = (
    probabilities >= 0.5
).astype(int)

precision = precision_score(
    y_validation,
    pair_predictions,
    zero_division=0
)

recall = recall_score(
    y_validation,
    pair_predictions,
    zero_division=0
)

f05 = fbeta_score(
    y_validation,
    pair_predictions,
    beta=0.5,
    zero_division=0
)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F0.5      : {f05:.4f}")


# ============================================================
# 18. MACRO F0.5 AT ENTITY LEVEL
# ============================================================

def calculate_macro_f05(
    validation_data,
    true_matches,
    threshold,
    validation_ids
):

    scores = []

    # Convert ground truth to dictionary
    truth_dict = {}

    for row in true_matches.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id

        if s1_id not in validation_ids:
            continue

        if s1_id not in truth_dict:
            truth_dict[s1_id] = set()

        truth_dict[s1_id].add(candidate_id)


    # Calculate score for every validation S1
    for s1_id in validation_ids:

        # True matches
        true_set = truth_dict.get(
            s1_id,
            set()
        )

        # Predicted matches
        rows = validation_data[
            validation_data["source1_entity_id"] == s1_id
        ]

        predicted_set = set(
            rows[
                rows["probability"] >= threshold
            ]["candidate_entity_id"]
        )

        # Correct predictions
        true_positive = len(
            true_set.intersection(predicted_set)
        )

        false_positive = len(
            predicted_set - true_set
        )

        false_negative = len(
            true_set - predicted_set
        )

        # Precision
        if (
            true_positive + false_positive
            == 0
        ):
            precision_value = 0.0
        else:
            precision_value = (
                true_positive /
                (true_positive + false_positive)
            )

        # Recall
        if (
            true_positive + false_negative
            == 0
        ):
            recall_value = 0.0
        else:
            recall_value = (
                true_positive /
                (true_positive + false_negative)
            )

        # F0.5
        if precision_value == 0 and recall_value == 0:
            f05_value = 0.0

        else:
            f05_value = (
                1.25 *
                precision_value *
                recall_value
            ) / (
                0.25 * precision_value +
                recall_value
            )

        scores.append(f05_value)

    return sum(scores) / len(scores)


# ============================================================
# 19. TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


threshold_results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_validation,
        predictions,
        beta=0.5,
        zero_division=0
    )

    macro_f05 = calculate_macro_f05(
        validation_data,
        true_matches,
        threshold,
        validation_ids
    )

    threshold_results.append([
        threshold,
        precision,
        recall,
        f05,
        macro_f05
    ])


# ============================================================
# 20. DISPLAY THRESHOLD RESULTS
# ============================================================

results = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "precision",
        "recall",
        "pair_f05",
        "macro_f05"
    ]
)


print("\n==========================================")
print("THRESHOLD RESULTS")
print("==========================================")

print(
    results.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "pair_f05": "{:.4f}".format,
            "macro_f05": "{:.4f}".format
        }
    )
)


print("\n==========================================")
print("VALIDATION COMPLETED")
print("==========================================")