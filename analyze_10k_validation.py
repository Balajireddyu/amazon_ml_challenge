"""
Detailed Analysis of 10,000 S1 Validation Results.
- Sweeps thresholds 0.40 to 0.70 in 0.05 steps for best macro F0.5.
- Error breakdowns by:
  1. Country (US vs India)
  2. S1 match cardinality (0 true matches, 1 true match, >1 true matches)
  3. Candidate Source (S2 vs S3)
"""

import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing import preprocess_dataframe
from src.candidate_generation import InvertedCandidateIndex, generate_candidates_from_index
from src.features import create_matching_features
from src.model import create_labels, train_matching_model, FEATURE_COLUMNS
from src.matching import predict_matches, create_submission_dataframe, evaluate_submission

print("=" * 75)
print("DETAILED 10K VALIDATION ERROR ANALYSIS & THRESHOLD OPTIMIZATION")
print("=" * 75)

# 1. Load Data
t0 = time.perf_counter()
s1 = preprocess_dataframe(pd.read_csv('val_10k_data/s1_10k.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('val_10k_data/s2_10k.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('val_10k_data/s3_10k.tsv', sep='\t'))
gt = pd.read_csv('val_10k_data/gt_10k.tsv', sep='\t')
print(f"Data loaded in {time.perf_counter()-t0:.2f}s")

# 2. Candidate Generation
index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

cand_df = generate_candidates_from_index(s1, index, top_k=20)
print(f"Generated {len(cand_df):,} candidate pairs.")

# 3. Features
feat_df = create_matching_features(cand_df, s1, s2, s3)
labeled_df = create_labels(feat_df, gt)

# 4. Train / Val Split (GroupShuffleSplit on S1)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(labeled_df, groups=labeled_df['source1_entity_id']))

train_df = labeled_df.iloc[train_idx].copy()
val_df = labeled_df.iloc[val_idx].copy()

val_s1_ids = list(val_df['source1_entity_id'].unique())
val_gt = gt[gt['source1_entity_id'].isin(val_s1_ids)].copy()
val_s1_df = s1[s1['entity_id'].isin(val_s1_ids)].set_index('entity_id')

model = train_matching_model(train_df, val_df=val_df, feature_columns=FEATURE_COLUMNS)

# Extract Ground Truth pairs mapping
gt_dict = {}
for row in val_gt.itertuples(index=False):
    s1_id = row.source1_entity_id
    matched = str(row.matched_entity_ids).strip()
    if matched and matched != 'nan' and not pd.isna(row.matched_entity_ids):
        gt_dict[s1_id] = set(cid.strip() for cid in matched.split(',') if cid.strip())
    else:
        gt_dict[s1_id] = set()

# -------------------------------------------------------------
# 5. Threshold Sweep 0.40 - 0.70 (0.05 steps)
# -------------------------------------------------------------
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
print("\n" + "=" * 75)
print("THRESHOLD SWEEP (0.40 to 0.70 in 0.05 steps)")
print("=" * 75)
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'Macro F0.5':>12} | {'F1':>10} | {'TP':>7} | {'FP':>6} | {'FN':>6}")
print("-" * 85)

best_th = None
best_f05 = -1
sweep_results = {}

for th in thresholds:
    matches = predict_matches(val_df, model, threshold=th)
    sub = create_submission_dataframe(matches, val_s1_ids)
    res = evaluate_submission(sub, val_gt, beta=0.5)
    sweep_results[th] = (res, matches, sub)
    
    print(f"{th:10.2f} | {res['precision']*100:9.2f}% | {res['recall']*100:9.2f}% | {res['f0_5']*100:11.2f}% | {res['f1']*100:9.2f}% | {res['tp']:7d} | {res['fp']:6d} | {res['fn']:6d}")
    
    if res['f0_5'] > best_f05:
        best_f05 = res['f0_5']
        best_th = th

print(f"\n>>> Best Macro F0.5 is at Threshold = {best_th:.2f} with F0.5 = {best_f05*100:.2f}% <<<\n")

# -------------------------------------------------------------
# 6. Detailed Error Breakdown at Recommended Threshold
# -------------------------------------------------------------
def run_breakdown(eval_th):
    res, matches, sub = sweep_results[eval_th]
    
    # Build predicted dict
    pred_dict = {}
    for row in sub.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        if matched and matched != 'nan' and not pd.isna(row.matched_entity_ids):
            pred_dict[s1_id] = set(cid.strip() for cid in matched.split(',') if cid.strip())
        else:
            pred_dict[s1_id] = set()
            
    print("=" * 75)
    print(f"DETAILED ERROR BREAKDOWN AT THRESHOLD = {eval_th:.2f}")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # A. Breakdown by Country (US vs India)
    # ---------------------------------------------------------
    print("\n--- 1. Breakdown by Country ---")
    country_stats = {}
    for c in ['us', 'india']:
        c_s1_ids = [s1_id for s1_id in val_s1_ids if val_s1_df.loc[s1_id, 'country_normalized'] == c]
        
        tp, fp, fn = 0, 0, 0
        for s1_id in c_s1_ids:
            gt_set = gt_dict.get(s1_id, set())
            pred_set = pred_dict.get(s1_id, set())
            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
            
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        country_stats[c] = (len(c_s1_ids), tp, fp, fn, prec, rec, f05, f1)
        
    print(f"{'Country':>10} | {'S1 Count':>9} | {'Precision':>10} | {'Recall':>10} | {'Macro F0.5':>12} | {'F1':>10} | {'TP':>7} | {'FP':>6} | {'FN':>6}")
    print("-" * 90)
    for c, stats in country_stats.items():
        cnt, tp, fp, fn, prec, rec, f05, f1 = stats
        print(f"{c.upper():>10} | {cnt:9d} | {prec*100:9.2f}% | {rec*100:9.2f}% | {f05*100:11.2f}% | {f1*100:9.2f}% | {tp:7d} | {fp:6d} | {fn:6d}")

    # ---------------------------------------------------------
    # B. Breakdown by S1 Match Cardinality (0, 1, >1 matches)
    # ---------------------------------------------------------
    print("\n--- 2. Breakdown by S1 Ground Truth Cardinality ---")
    s1_0 = [s1_id for s1_id in val_s1_ids if len(gt_dict.get(s1_id, set())) == 0]
    s1_1 = [s1_id for s1_id in val_s1_ids if len(gt_dict.get(s1_id, set())) == 1]
    s1_multi = [s1_id for s1_id in val_s1_ids if len(gt_dict.get(s1_id, set())) > 1]
    
    card_groups = [
        ("0 matches (Empty)", s1_0),
        ("1 match (Single)", s1_1),
        (">1 matches (Multi)", s1_multi)
    ]
    
    print(f"{'GT Category':>20} | {'S1 Count':>9} | {'Precision':>10} | {'Recall':>10} | {'Macro F0.5':>12} | {'TP':>7} | {'FP':>6} | {'FN':>6}")
    print("-" * 96)
    for name, group in card_groups:
        tp, fp, fn = 0, 0, 0
        for s1_id in group:
            gt_set = gt_dict.get(s1_id, set())
            pred_set = pred_dict.get(s1_id, set())
            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fn == 0 else 0.0)
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{name:>20} | {len(group):9d} | {prec*100:9.2f}% | {rec*100:9.2f}% | {f05*100:11.2f}% | {tp:7d} | {fp:6d} | {fn:6d}")
        
    # S1 Empty match accuracy specifically
    correct_empty = sum(1 for s1_id in s1_0 if len(pred_dict.get(s1_id, set())) == 0)
    print(f"\n  * Empty Ground Truth Clean Rate: {correct_empty}/{len(s1_0)} ({correct_empty/len(s1_0)*100:.2f}%) S1 records correctly generated 0 predictions.")

    # ---------------------------------------------------------
    # C. Breakdown by Target Source (S2 vs S3)
    # ---------------------------------------------------------
    print("\n--- 3. Breakdown by Target Candidate Source (S2 vs S3) ---")
    source_stats = {}
    for src in ['S2', 'S3']:
        tp, fp, fn = 0, 0, 0
        for s1_id in val_s1_ids:
            gt_set = {cid for cid in gt_dict.get(s1_id, set()) if cid.startswith(src + '-')}
            pred_set = {cid for cid in pred_dict.get(s1_id, set()) if cid.startswith(src + '-')}
            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
            
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        source_stats[src] = (tp, fp, fn, prec, rec, f05, f1)
        
    print(f"{'Target Source':>15} | {'Precision':>10} | {'Recall':>10} | {'Macro F0.5':>12} | {'F1':>10} | {'TP':>7} | {'FP':>6} | {'FN':>6}")
    print("-" * 88)
    for src, stats in source_stats.items():
        tp, fp, fn, prec, rec, f05, f1 = stats
        print(f"{src:>15} | {prec*100:9.2f}% | {rec*100:9.2f}% | {f05*100:11.2f}% | {f1*100:9.2f}% | {tp:7d} | {fp:6d} | {fn:6d}")

run_breakdown(best_th)
