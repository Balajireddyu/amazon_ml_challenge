"""
Test script for Phase 3 Feature Pipeline on 1000-S1 Smoke Test.
Measures: feature count, extraction runtime, memory consumption, and displays sample output.
"""

import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import time
import os
import psutil
import pandas as pd

from src.preprocessing import preprocess_dataframe
from src.candidate_generation import InvertedCandidateIndex, generate_candidates_from_index
from src.features import create_matching_features

process = psutil.Process(os.getpid())

print("1. Loading smoke datasets...", flush=True)
t0 = time.perf_counter()
s1 = preprocess_dataframe(pd.read_csv('smoke_data/s1_1000.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('smoke_data/s2_smoke.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('smoke_data/s3_smoke.tsv', sep='\t'))

print("2. Building index and generating candidates @ top_k=20...", flush=True)
index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

cand_df = generate_candidates_from_index(s1, index, top_k=20)
print(f"   Generated {len(cand_df):,} candidate pairs for {len(s1):,} S1 records.", flush=True)

# Measure feature extraction
mem_before = process.memory_info().rss / 1024**2
t_feat_start = time.perf_counter()

print("3. Running Phase 3 Feature Pipeline...", flush=True)
feat_df = create_matching_features(cand_df, s1, s2, s3)

t_feat = time.perf_counter() - t_feat_start
mem_after = process.memory_info().rss / 1024**2

# Metadata columns vs feature columns
id_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
feature_cols = [c for c in feat_df.columns if c not in id_cols]

print("\n" + "="*70, flush=True)
print("PHASE 3 FEATURE EXTRACTION BENCHMARK REPORT", flush=True)
print("="*70, flush=True)
print(f"Total Candidate Pairs Processed : {len(feat_df):,}", flush=True)
print(f"Total Columns Output            : {len(feat_df.columns)} ({len(id_cols)} IDs + {len(feature_cols)} ML features)", flush=True)
print(f"Feature Extraction Runtime      : {t_feat:.3f}s ({t_feat / len(feat_df) * 1000:.3f} ms / pair)", flush=True)
print(f"Peak Process RSS Memory         : {mem_after:.1f} MB (Delta during features: {mem_after - mem_before:.1f} MB)", flush=True)
print(f"\nFeature Columns ({len(feature_cols)} total):", flush=True)
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}", flush=True)

print("\n" + "="*70, flush=True)
print("SAMPLE FEATURE MATRIX OUTPUT (First 5 candidate pairs)", flush=True)
print("="*70, flush=True)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
print(feat_df.head(5).to_string(index=False), flush=True)

print("\n" + "="*70, flush=True)
print("SUMMARY STATISTICS OF EXTRACTED FEATURES", flush=True)
print("="*70, flush=True)
print(feat_df[feature_cols].describe().round(2).to_string(), flush=True)
