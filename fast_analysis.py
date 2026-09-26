import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')
import time
import pandas as pd
from collections import Counter, defaultdict
from src.preprocessing import preprocess_dataframe
from src.candidate_generation import (
    InvertedCandidateIndex,
    generate_candidates_from_index,
    GENERIC_NAME_WORDS, GENERIC_ADDRESS_WORDS,
    extract_name_core, extract_address_numbers, extract_tokens
)

t0 = time.perf_counter()
print("1. Loading smoke datasets...", flush=True)
s1 = preprocess_dataframe(pd.read_csv('smoke_data/s1_1000.tsv', sep='\t'))
s2 = preprocess_dataframe(pd.read_csv('smoke_data/s2_smoke.tsv', sep='\t'))
s3 = preprocess_dataframe(pd.read_csv('smoke_data/s3_smoke.tsv', sep='\t'))
gt = pd.read_csv('smoke_data/gt_1000.tsv', sep='\t')

true_pairs = set()
for row in gt.itertuples(index=False):
    for m in str(row.matched_entity_ids).split(','):
        m = m.strip()
        if m and m != 'nan':
            true_pairs.add((row.source1_entity_id, m))

print(f"2. Building index... ({time.perf_counter()-t0:.1f}s)", flush=True)
index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

print(f"3. Generating candidates @ K=100... ({time.perf_counter()-t0:.1f}s)", flush=True)
cand_df_100 = generate_candidates_from_index(s1, index, top_k=100)

print(f"4. Mapping ranks... ({time.perf_counter()-t0:.1f}s)", flush=True)
# Vectorized rank computation
s1_col = cand_df_100['source1_entity_id'].values
cand_col = cand_df_100['candidate_entity_id'].values

# Compute rank by counting occurrences of s1_id
rank_map = {}
curr_s1 = None
curr_rank = 0
for s1_id, cid in zip(s1_col, cand_col):
    if s1_id != curr_s1:
        curr_s1 = s1_id
        curr_rank = 1
    else:
        curr_rank += 1
    rank_map[(s1_id, cid)] = curr_rank

s1_lkp = s1.set_index('entity_id')
s2_lkp = s2.set_index('entity_id')
s3_lkp = s3.set_index('entity_id')

all_missed_20 = [p for p in true_pairs if rank_map.get(p, 999999) > 20]
genuinely_missed = []
not_in_pool = []
for s1_id, cid in all_missed_20:
    lkp = s2_lkp if cid.startswith('S2-') else s3_lkp
    if cid in lkp.index:
        genuinely_missed.append((s1_id, cid))
    else:
        not_in_pool.append((s1_id, cid))

print(f"\n=======================================================", flush=True)
print(f"Total true pairs in GT : {len(true_pairs)}", flush=True)
print(f"Total missed @ K=20    : {len(all_missed_20)}", flush=True)
print(f"Not in smoke pool      : {len(not_in_pool)} (unretrievable)", flush=True)
print(f"Genuinely missed       : {len(genuinely_missed)} (present in S2/S3 pool)", flush=True)
print(f"=======================================================", flush=True)

categories = defaultdict(list)

for s1_id, cid in genuinely_missed:
    s1r = s1_lkp.loc[s1_id]
    lkp = s2_lkp if cid.startswith('S2-') else s3_lkp
    cr = lkp.loc[cid]

    c1 = str(s1r['country_normalized']).strip().lower()
    c2 = str(cr['country_normalized']).strip().lower()
    n1 = str(s1r['business_name_normalized']).strip()
    n2 = str(cr['business_name_normalized']).strip()
    a1 = str(s1r['business_address_normalized']).strip()
    a2 = str(cr['business_address_normalized']).strip()
    
    rnk = rank_map.get((s1_id, cid), None)
    
    tok1 = set(extract_tokens(n1, GENERIC_NAME_WORDS, 3))
    tok2 = set(extract_tokens(n2, GENERIC_NAME_WORDS, 3))
    addr1 = set(extract_tokens(a1, GENERIC_ADDRESS_WORDS, 4))
    addr2 = set(extract_tokens(a2, GENERIC_ADDRESS_WORDS, 4))
    num1 = set(extract_address_numbers(a1))
    num2 = set(extract_address_numbers(a2))
    
    shared_name_tokens = tok1 & tok2
    shared_addr_tokens = addr1 & addr2
    shared_nums = num1 & num2
    
    pair_info = {
        's1_id': s1_id, 'cid': cid,
        's1_name': n1, 'cand_name': n2,
        's1_addr': a1, 'cand_addr': a2,
        's1_country': c1, 'cand_country': c2,
        'rank': rnk,
        'shared_name': shared_name_tokens,
        'shared_addr': shared_addr_tokens,
        'shared_nums': shared_nums
    }

    if c1 != c2:
        categories['Country Mismatch (Blocked by Country Key)'].append(pair_info)
    elif rnk is not None and 21 <= rnk <= 30:
        categories['Ranking / Top-K Fallout (Retrieved at Rank 21-30)'].append(pair_info)
    elif rnk is not None and 31 <= rnk <= 50:
        categories['Ranking / Top-K Fallout (Retrieved at Rank 31-50)'].append(pair_info)
    elif rnk is not None and 51 <= rnk <= 100:
        categories['Ranking / Top-K Fallout (Retrieved at Rank 51-100)'].append(pair_info)
    else:
        # Not in top 100:
        if shared_name_tokens:
            categories['Low Score / Pruned Name Token (Rank > 100)'].append(pair_info)
        elif shared_nums and shared_addr_tokens:
            categories['Address Match Only (Numbers + Street, Rank > 100)'].append(pair_info)
        elif shared_nums:
            categories['Address Number Only (Different Name & Street)'].append(pair_info)
        elif not n1 or not n2 or not a1 or not a2:
            categories['Missing / Empty Field Data in Source'].append(pair_info)
        else:
            categories['Zero Shared Keys / Severe Name Variation'].append(pair_info)

print("\n" + "="*70, flush=True)
print("BREAKDOWN BY CATEGORY FOR 262 GENUINELY MISSED PAIRS", flush=True)
print("="*70, flush=True)

sorted_cats = sorted(categories.items(), key=lambda x: -len(x[1]))
for cat, items in sorted_cats:
    pct = len(items) / len(genuinely_missed) * 100
    print(f"  • {cat:<58} : {len(items):3d} pairs ({pct:5.1f}%)", flush=True)

print("\n" + "="*70, flush=True)
print("10 REPRESENTATIVE EXAMPLES ACROSS CATEGORIES", flush=True)
print("="*70, flush=True)

selected_examples = []
for cat, items in sorted_cats:
    for item in items[:2]:
        selected_examples.append((cat, item))
        if len(selected_examples) == 10:
            break
    if len(selected_examples) == 10:
        break

for i, (cat, ex) in enumerate(selected_examples, 1):
    rnk_str = f"Rank={ex['rank']}" if ex['rank'] is not None else "Unranked / Rank>100"
    print(f"\n[Example {i}] Category: {cat} ({rnk_str})", flush=True)
    print(f"  S1   ({ex['s1_id']}): Name='{ex['s1_name']}' | Addr='{ex['s1_addr']}' | Ctry='{ex['s1_country']}'", flush=True)
    print(f"  Cand ({ex['cid']}): Name='{ex['cand_name']}' | Addr='{ex['cand_addr']}' | Ctry='{ex['cand_country']}'", flush=True)
    print(f"  Diagnostics: Shared Name Tokens={list(ex['shared_name'])}, Shared Addr Tokens={list(ex['shared_addr'])}, Shared Numbers={list(ex['shared_nums'])}", flush=True)

print(f"\nAnalysis completed in {time.perf_counter()-t0:.1f}s", flush=True)
