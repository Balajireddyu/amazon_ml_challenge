"""
End-to-End Pipeline Smoke Test (Phases 1 to 5).
Validates the complete entity resolution pipeline on 1000-S1 smoke data:
1. Phase 1: Normalization
2. Phase 2: High-speed candidate generation (top_k=20)
3. Phase 3: 27 pairwise feature extraction
4. Phase 4: LightGBM training on 80% train split (Grouped by S1)
5. Phase 5: Matching decision, submission generation, and end-to-end metric evaluation on 20% validation split
"""

import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import time
import os
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

print("="*70, flush=True)
print("END-TO-END PIPELINE SMOKE TEST (1,000 S1 RECORDS)", flush=True)
print("="*70, flush=True)

# ------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------
print("\n[Phase 1] Preprocessing & Data Loading...", flush=True)
t0 = time.perf_counter()
s1 = preprocess_dataframe(pd.read_csv('smoke_data/s1_1000.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('smoke_data/s2_smoke.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('smoke_data/s3_smoke.tsv', sep='\t'))
gt = pd.read_csv('smoke_data/gt_1000.tsv', sep='\t')
print(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,} | GT: {len(gt):,} rows ({time.perf_counter()-t0:.2f}s)", flush=True)

# ------------------------------------------------------------------
# 2. Candidate Generation
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

# ------------------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------------------
print("\n[Phase 3] Pairwise Feature Extraction (27 ML features)...", flush=True)
t0 = time.perf_counter()
feat_df = create_matching_features(cand_df, s1, s2, s3)
t_feat = time.perf_counter() - t0
print(f"  Extracted {len(feat_df.columns)} columns for {len(feat_df):,} pairs in {t_feat:.2f}s ({t_feat/len(feat_df)*1000:.3f} ms/pair)", flush=True)

# ------------------------------------------------------------------
# 4. Model Training (80/20 Group Split by S1)
# ------------------------------------------------------------------
print("\n[Phase 4] Model Training (GroupSplit 800 Train S1 / 200 Val S1)...", flush=True)
labeled_df = create_labels(feat_df, gt)

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(labeled_df, groups=labeled_df['source1_entity_id']))

train_df = labeled_df.iloc[train_idx].copy()
val_df = labeled_df.iloc[val_idx].copy()

val_s1_ids = list(val_df['source1_entity_id'].unique())
val_gt = gt[gt['source1_entity_id'].isin(val_s1_ids)].copy()

t0 = time.perf_counter()
model = train_matching_model(train_df, val_df=val_df, feature_columns=FEATURE_COLUMNS)
t_train = time.perf_counter() - t0
print(f"  Model trained in {t_train:.2f}s.", flush=True)

# ------------------------------------------------------------------
# 5. Matching Decisions & End-to-End Evaluation on Validation Set
# ------------------------------------------------------------------
print("\n[Phase 5] Matching Decision & Submission Generation on Validation Set...", flush=True)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5':>10} | {'F1':>10} | {'Pred Pairs':>11} | {'GT Pairs':>9}", flush=True)
print("-" * 78, flush=True)

best_res = None
best_th = None
for th in thresholds:
    matches = predict_matches(val_df, model, threshold=th)
    sub = create_submission_dataframe(matches, val_s1_ids)
    res = evaluate_submission(sub, val_gt, beta=0.5)
    print(f"{th:10.2f} | {res['precision']*100:9.2f}% | {res['recall']*100:9.2f}% | {res['f0_5']*100:9.2f}% | {res['f1']*100:9.2f}% | {res['pred_count']:11d} | {res['gt_count']:9d}", flush=True)
    if best_res is None or res['f0_5'] > best_res['f0_5']:
        best_res = res
        best_th = th

print("\n" + "="*70, flush=True)
print(f"OPTIMAL END-TO-END VALIDATION RESULT (Threshold = {best_th:.2f})", flush=True)
print("="*70, flush=True)
print(f"  • Precision : {best_res['precision']*100:.2f}%", flush=True)
print(f"  • Recall    : {best_res['recall']*100:.2f}%", flush=True)
print(f"  • F0.5 Score: {best_res['f0_5']*100:.2f}%", flush=True)
print(f"  • F1 Score  : {best_res['f1']*100:.2f}%", flush=True)
print(f"  • TP: {best_res['tp']:,} | FP: {best_res['fp']:,} | FN: {best_res['fn']:,}", flush=True)

# Sample Submission Output
sample_matches = predict_matches(val_df, model, threshold=best_th)
sample_sub = create_submission_dataframe(sample_matches, val_s1_ids[:10])
print("\n" + "="*70, flush=True)
print("SAMPLE SUBMISSION FORMAT (First 10 S1 Entities in Validation)", flush=True)
print("="*70, flush=True)
print(sample_sub.to_string(index=False), flush=True)

print(f"\nTotal smoke test runtime: {time.perf_counter()-t_all:.2f}s | Peak RAM: {process.memory_info().rss/1024**2:.1f} MB", flush=True)
