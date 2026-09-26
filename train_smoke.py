"""
Phase 4 Training & Validation on 1000-S1 Smoke Test.
- Generates candidates & 27 features
- Labels ground truth
- Splits train/validation by S1 entity_id (Group split - zero leakage)
- Trains LightGBM Binary Classifier
- Reports Precision, Recall, F0.5, F1, Confusion Matrix, and Threshold Sweep
"""

import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import time
import os
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing import preprocess_dataframe
from src.candidate_generation import InvertedCandidateIndex, generate_candidates_from_index
from src.features import create_matching_features
from src.model import create_labels, train_matching_model, evaluate_predictions, FEATURE_COLUMNS

process = psutil.Process(os.getpid())
t0 = time.perf_counter()

print("1. Loading smoke datasets...", flush=True)
s1 = preprocess_dataframe(pd.read_csv('smoke_data/s1_1000.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('smoke_data/s2_smoke.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('smoke_data/s3_smoke.tsv', sep='\t'))
gt = pd.read_csv('smoke_data/gt_1000.tsv', sep='\t')

print("2. Building index and generating candidates @ top_k=20...", flush=True)
index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

cand_df = generate_candidates_from_index(s1, index, top_k=20)
print(f"   Generated {len(cand_df):,} candidates.", flush=True)

print("3. Extracting 27 pairwise features...", flush=True)
t_feat = time.perf_counter()
feat_df = create_matching_features(cand_df, s1, s2, s3)
print(f"   Feature extraction done in {time.perf_counter()-t_feat:.2f}s.", flush=True)

print("4. Generating ground-truth labels...", flush=True)
labeled_df = create_labels(feat_df, gt)
pos_cnt = (labeled_df['label'] == 1).sum()
neg_cnt = (labeled_df['label'] == 0).sum()
print(f"   Total labeled pairs: {len(labeled_df):,} (Positives: {pos_cnt:,} ({pos_cnt/len(labeled_df)*100:.1f}%), Negatives: {neg_cnt:,})", flush=True)

print("5. Group-based Train/Validation Split by S1 entity...", flush=True)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(labeled_df, groups=labeled_df['source1_entity_id']))

train_df = labeled_df.iloc[train_idx].copy()
val_df = labeled_df.iloc[val_idx].copy()

n_train_s1 = train_df['source1_entity_id'].nunique()
n_val_s1 = val_df['source1_entity_id'].nunique()
print(f"   Train set: {len(train_df):,} candidate pairs across {n_train_s1} S1 entities (Positives: {(train_df['label']==1).sum():,})", flush=True)
print(f"   Val set  : {len(val_df):,} candidate pairs across {n_val_s1} S1 entities (Positives: {(val_df['label']==1).sum():,})", flush=True)

print("6. Training LightGBM Pairwise Matcher...", flush=True)
t_train = time.perf_counter()
model = train_matching_model(train_df, val_df=val_df, feature_columns=FEATURE_COLUMNS)
t_train_dur = time.perf_counter() - t_train
print(f"   Model training completed in {t_train_dur:.2f}s.", flush=True)

# Validation Predictions
val_probs = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
val_y = val_df['label'].values

# Initial Threshold Evaluation (e.g. 0.5)
init_eval = evaluate_predictions(val_y, val_probs, threshold=0.5, beta=0.5)

print("\n" + "="*70, flush=True)
print("PHASE 4: LIGHTGBM MODEL VALIDATION REPORT (Threshold = 0.50)", flush=True)
print("="*70, flush=True)
print(f"Validation Precision : {init_eval['precision']*100:.2f}%", flush=True)
print(f"Validation Recall    : {init_eval['recall']*100:.2f}%", flush=True)
print(f"Validation F0.5 Score: {init_eval['f0_5']*100:.2f}%", flush=True)
print(f"Validation F1 Score  : {init_eval['f1']*100:.2f}%", flush=True)
print("\nConfusion Matrix (at threshold 0.50):", flush=True)
print(f"  True Positives  (TP) : {init_eval['tp']:,}", flush=True)
print(f"  False Positives (FP) : {init_eval['fp']:,}", flush=True)
print(f"  False Negatives (FN) : {init_eval['fn']:,}", flush=True)
print(f"  True Negatives  (TN) : {init_eval['tn']:,}", flush=True)

print("\n" + "="*70, flush=True)
print("THRESHOLD SWEEP (Validation Set)", flush=True)
print("="*70, flush=True)
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5':>10} | {'F1':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}", flush=True)
print("-" * 75, flush=True)
for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
    ev = evaluate_predictions(val_y, val_probs, threshold=th, beta=0.5)
    print(f"{th:10.2f} | {ev['precision']*100:9.2f}% | {ev['recall']*100:9.2f}% | {ev['f0_5']*100:9.2f}% | {ev['f1']*100:9.2f}% | {ev['tp']:6d} | {ev['fp']:6d} | {ev['fn']:6d}", flush=True)

print("\n" + "="*70, flush=True)
print("TOP 10 MOST IMPORTANT FEATURES (LightGBM)", flush=True)
print("="*70, flush=True)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:2d}. {FEATURE_COLUMNS[idx]:<30} : {importances[idx]:8.4f}", flush=True)

print(f"\nTotal pipeline runtime: {time.perf_counter()-t0:.2f}s | Peak RAM: {process.memory_info().rss/1024**2:.1f} MB", flush=True)
