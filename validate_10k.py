"""
10,000 S1 Validation Experiment (Phases 1 to 5).

Evaluates the complete entity resolution pipeline on 10,000 S1 entities:
1. Phase 1: Normalization
2. Phase 2: Candidate generation (top_k=20) using InvertedCandidateIndex
3. Phase 3: 27 pairwise feature extraction
4. Phase 4: LightGBM / HistGradientBoosting model training (80/20 GroupSplit by S1 entity)
5. Phase 5: Matching decision (threshold=0.40 & sweep), empty-match accuracy, average predicted matches, and macro F0.5 evaluation
"""

import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import os
import time
import psutil
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing import preprocess_dataframe
from src.candidate_generation import InvertedCandidateIndex, generate_candidates_from_index
from src.features import create_matching_features
from src.model import create_labels, train_matching_model, FEATURE_COLUMNS
from src.matching import predict_matches, create_submission_dataframe, evaluate_submission

process = psutil.Process(os.getpid())
t_all = time.perf_counter()

print("=" * 70, flush=True)
print("10,000 S1 END-TO-END VALIDATION EXPERIMENT (Phases 1-5)", flush=True)
print("=" * 70, flush=True)

# ------------------------------------------------------------------
# 1. Load Data & Preprocessing
# ------------------------------------------------------------------
print("\n[Phase 1] Preprocessing & Data Loading...", flush=True)
t0 = time.perf_counter()

s1 = preprocess_dataframe(pd.read_csv('val_10k_data/s1_10k.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('val_10k_data/s2_10k.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('val_10k_data/s3_10k.tsv', sep='\t'))
gt = pd.read_csv('val_10k_data/gt_10k.tsv', sep='\t')

t_load = time.perf_counter() - t0
print(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,} | GT: {len(gt):,} rows ({t_load:.2f}s)", flush=True)

# Build ground truth pairs set
true_pairs = set()
for row in gt.itertuples(index=False):
    matched = str(row.matched_entity_ids).strip()
    if matched and matched != 'nan' and not pd.isna(row.matched_entity_ids):
        for cid in matched.split(','):
            cid = cid.strip()
            if cid:
                true_pairs.add((row.source1_entity_id, cid))
print(f"  Total Ground Truth Pairs: {len(true_pairs):,}", flush=True)

# ------------------------------------------------------------------
# 2. Candidate Generation (top_k=20)
# ------------------------------------------------------------------
print("\n[Phase 2] Candidate Generation (@ top_k=20)...", flush=True)
t0 = time.perf_counter()

index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

cand_df = generate_candidates_from_index(s1, index, top_k=20)
t_cand = time.perf_counter() - t0
print(f"  Generated {len(cand_df):,} candidate pairs in {t_cand:.2f}s ({t_cand/len(s1)*1000:.2f} ms/S1)", flush=True)

cand_pairs = set(zip(cand_df['source1_entity_id'], cand_df['candidate_entity_id']))
found_true = len(true_pairs & cand_pairs)
recall_k20 = found_true / len(true_pairs) if true_pairs else 0.0
print(f"  Candidate Recall@20: {recall_k20*100:.2f}% ({found_true:,} / {len(true_pairs):,} true pairs)", flush=True)

# ------------------------------------------------------------------
# 3. Feature Extraction (27 ML features)
# ------------------------------------------------------------------
print("\n[Phase 3] Pairwise Feature Extraction (27 ML features)...", flush=True)
t0 = time.perf_counter()
feat_df = create_matching_features(cand_df, s1, s2, s3)
t_feat = time.perf_counter() - t0
print(f"  Extracted {len(feat_df.columns)} columns for {len(feat_df):,} pairs in {t_feat:.2f}s ({t_feat/len(feat_df)*1000:.3f} ms/pair)", flush=True)

# ------------------------------------------------------------------
# 4. Model Training (80/20 Group Split by S1 entity)
# ------------------------------------------------------------------
print("\n[Phase 4] Model Training (GroupSplit 8,000 Train S1 / 2,000 Val S1)...", flush=True)
labeled_df = create_labels(feat_df, gt)
pos_cnt = int(labeled_df['label'].sum())
neg_cnt = len(labeled_df) - pos_cnt
print(f"  Dataset Labels: Positives={pos_cnt:,} | Negatives={neg_cnt:,} | Imbalance={neg_cnt/pos_cnt:.1f}:1", flush=True)

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(labeled_df, groups=labeled_df['source1_entity_id']))

train_df = labeled_df.iloc[train_idx].copy()
val_df = labeled_df.iloc[val_idx].copy()

val_s1_ids = list(val_df['source1_entity_id'].unique())
val_gt = gt[gt['source1_entity_id'].isin(val_s1_ids)].copy()

print(f"  Train: {len(train_df):,} pairs ({train_df['label'].sum():,} pos) across {train_df['source1_entity_id'].nunique():,} S1s", flush=True)
print(f"  Val:   {len(val_df):,} pairs ({val_df['label'].sum():,} pos) across {len(val_s1_ids):,} S1s", flush=True)

t0 = time.perf_counter()
model = train_matching_model(train_df, val_df=val_df, feature_columns=FEATURE_COLUMNS)
t_train = time.perf_counter() - t0
print(f"  Model trained in {t_train:.2f}s.", flush=True)

# ------------------------------------------------------------------
# 5. Matching Decisions & End-to-End Evaluation on Validation Set
# ------------------------------------------------------------------
print("\n[Phase 5] Matching Decision & Submission Generation on Validation Set...", flush=True)

def compute_empty_match_acc(sub, all_val_s1_ids, ground_truth_df):
    gt_with_matches = set()
    for row in ground_truth_df.itertuples(index=False):
        matched = str(row.matched_entity_ids).strip()
        if matched and matched != 'nan' and not pd.isna(row.matched_entity_ids):
            gt_with_matches.add(row.source1_entity_id)
    s1_with_no_gt = [s1_id for s1_id in all_val_s1_ids if s1_id not in gt_with_matches]
    
    sub_matched_s1 = set()
    for row in sub.itertuples(index=False):
        matched = str(row.matched_entity_ids).strip()
        if matched and matched != 'nan' and not pd.isna(row.matched_entity_ids):
            sub_matched_s1.add(row.source1_entity_id)
            
    correctly_empty = sum(1 for s1_id in s1_with_no_gt if s1_id not in sub_matched_s1)
    s1_no_gt = len(s1_with_no_gt)
    acc = correctly_empty / s1_no_gt if s1_no_gt > 0 else 1.0
    return acc, s1_no_gt

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5':>10} | {'F1':>10} | {'Empty Acc':>10} | {'Avg Matches':>12}", flush=True)
print("-" * 86, flush=True)

target_res_040 = None
best_res = None
best_th = None

for th in thresholds:
    matches = predict_matches(val_df, model, threshold=th)
    sub = create_submission_dataframe(matches, val_s1_ids)
    res = evaluate_submission(sub, val_gt, beta=0.5)
    empty_acc, n_no_gt = compute_empty_match_acc(sub, val_s1_ids, val_gt)
    avg_matches = len(matches) / len(val_s1_ids)
    
    print(f"{th:10.2f} | {res['precision']*100:9.2f}% | {res['recall']*100:9.2f}% | {res['f0_5']*100:9.2f}% | {res['f1']*100:9.2f}% | {empty_acc*100:9.2f}% | {avg_matches:12.2f}", flush=True)
    
    if abs(th - 0.40) < 1e-4:
        target_res_040 = (res, empty_acc, avg_matches, matches, sub)
    if best_res is None or res['f0_5'] > best_res['f0_5']:
        best_res = res
        best_th = th

# Focus on Threshold = 0.40 as requested
res_040, empty_acc_040, avg_matches_040, matches_040, sub_040 = target_res_040

peak_ram_mb = process.memory_info().rss / 1024**2
total_runtime_s = time.perf_counter() - t_all

print("\n" + "=" * 70, flush=True)
print("10,000 S1 VALIDATION BENCHMARK RESULTS (Frozen Threshold = 0.40)", flush=True)
print("=" * 70, flush=True)
print(f"  • Candidate Recall@20       : {recall_k20*100:.2f}% ({found_true:,}/{len(true_pairs):,} true pairs)", flush=True)
print(f"  • Macro F0.5 Score          : {res_040['f0_5']*100:.2f}%", flush=True)
print(f"  • Final Precision           : {res_040['precision']*100:.2f}%", flush=True)
print(f"  • Final Recall              : {res_040['recall']*100:.2f}%", flush=True)
print(f"  • F1 Score                  : {res_040['f1']*100:.2f}%", flush=True)
print(f"  • Confusion Matrix Counts   : TP={res_040['tp']:,} | FP={res_040['fp']:,} | FN={res_040['fn']:,}", flush=True)
print(f"  • Empty-Match Accuracy      : {empty_acc_040*100:.2f}% ({n_no_gt:,} S1 entities without ground truth)", flush=True)
print(f"  • Average Predicted Matches : {avg_matches_040:.2f} per S1 entity", flush=True)
print(f"  • Total End-to-End Runtime  : {total_runtime_s:.2f}s ({total_runtime_s/60:.1f} min)", flush=True)
print(f"  • Peak RSS Memory           : {peak_ram_mb:.1f} MB ({peak_ram_mb/1024:.2f} GB)", flush=True)

# Sample submission output
print("\n" + "=" * 70, flush=True)
print("SAMPLE PREDICTED SUBMISSION (First 10 S1 Entities in Validation)", flush=True)
print("=" * 70, flush=True)
print(sub_040.head(10).to_string(index=False), flush=True)
