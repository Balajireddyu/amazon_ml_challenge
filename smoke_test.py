"""
1000-S1 Smoke Test for improved candidate_generation.py
Reports: runtime, peak RSS memory, candidate count, Recall@20, Recall@30, Recall@50
"""
import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')

import time
import tracemalloc
import psutil
import os
import pandas as pd

from src.preprocessing import preprocess_dataframe
from src.candidate_generation import (
    InvertedCandidateIndex,
    generate_candidates_from_index,
    generate_fuzzy_candidates,
)

process = psutil.Process(os.getpid())

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading smoke datasets...")
s1 = preprocess_dataframe(pd.read_csv('smoke_data/s1_1000.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('smoke_data/s2_smoke.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('smoke_data/s3_smoke.tsv', sep='\t'))
gt = pd.read_csv('smoke_data/gt_1000.tsv', sep='\t')

print(f"  S1: {len(s1):,}  |  S2: {len(s2):,}  |  S3: {len(s3):,}")

# Build ground-truth pairs
true_pairs = set()
for row in gt.itertuples(index=False):
    for m in str(row.matched_entity_ids).split(','):
        m = m.strip()
        if m and m != 'nan':
            true_pairs.add((row.source1_entity_id, m))
print(f"  True pairs in GT: {len(true_pairs):,}\n")

# ------------------------------------------------------------------
# Build index (time separately)
# ------------------------------------------------------------------
mem_before = process.memory_info().rss / 1024**2

t_idx_start = time.perf_counter()
index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()
t_idx = time.perf_counter() - t_idx_start

mem_after_idx = process.memory_info().rss / 1024**2
print(f"Index build time : {t_idx:.2f}s")
print(f"Index RAM delta  : {mem_after_idx - mem_before:.1f} MB  (RSS now: {mem_after_idx:.0f} MB)")

# ------------------------------------------------------------------
# Generate candidates @ K=20
# ------------------------------------------------------------------
t_gen_start = time.perf_counter()
cand_df = generate_candidates_from_index(s1, index, top_k=20)
t_gen = time.perf_counter() - t_gen_start

mem_peak = process.memory_info().rss / 1024**2
print(f"\nCandidate gen time : {t_gen:.3f}s  ({t_gen/len(s1)*1000:.2f} ms/record)")
print(f"Peak RSS after gen : {mem_peak:.0f} MB")
print(f"Candidate pairs    : {len(cand_df):,}  ({len(cand_df)/len(s1):.1f} per S1)")

# ------------------------------------------------------------------
# Recall evaluation
# ------------------------------------------------------------------
def recall_at_k(s1_df, idx, k, true_pairs):
    c = generate_candidates_from_index(s1_df, idx, top_k=k)
    found = set(zip(c['source1_entity_id'], c['candidate_entity_id'])) & true_pairs
    return len(found) / len(true_pairs) if true_pairs else 0.0, len(found)

cand_pairs_20 = set(zip(cand_df['source1_entity_id'], cand_df['candidate_entity_id']))
found_20 = len(cand_pairs_20 & true_pairs)
r20 = found_20 / len(true_pairs)

r30_val, found_30 = recall_at_k(s1, index, 30, true_pairs)
r50_val, found_50 = recall_at_k(s1, index, 50, true_pairs)

print(f"\n{'='*48}")
print(f"  Recall@20 : {r20:.4f}  ({found_20}/{len(true_pairs)})")
print(f"  Recall@30 : {r30_val:.4f}  ({found_30}/{len(true_pairs)})")
print(f"  Recall@50 : {r50_val:.4f}  ({found_50}/{len(true_pairs)})")
print(f"{'='*48}")

# ------------------------------------------------------------------
# Quick breakdown of still-missed pairs
# ------------------------------------------------------------------
missed = [p for p in true_pairs if p not in cand_pairs_20]
print(f"\nStill missed @K=20: {len(missed)}")

s2_lkp = s2.set_index('entity_id')
s3_lkp = s3.set_index('entity_id')
s1_lkp = s1.set_index('entity_id')

not_in_pool = sum(
    1 for _, cid in missed
    if cid not in (s2_lkp.index if cid.startswith('S2-') else s3_lkp.index)
)
print(f"  Not in smoke pool (can't be found): {not_in_pool}")
print(f"  Genuinely hard (in pool but missed): {len(missed) - not_in_pool}")
