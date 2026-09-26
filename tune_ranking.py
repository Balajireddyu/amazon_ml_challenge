import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')
import time, heapq, psutil, os
import pandas as pd
from collections import defaultdict
from src.preprocessing import preprocess_dataframe
from src.candidate_generation import (
    InvertedCandidateIndex,
    GENERIC_NAME_WORDS, GENERIC_ADDRESS_WORDS,
    extract_name_core, extract_address_numbers, extract_tokens, extract_name_bigrams
)

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

index = InvertedCandidateIndex(max_bucket_size=500)
index.add_source(s2, 'S2')
index.add_source(s3, 'S3')
index.finalize()

def test_scoring_weights(s1_df, idx, weights, top_k=20):
    t0 = time.perf_counter()
    candidates = []
    
    w_exact = weights.get('exact', 100.0)
    w_core = weights.get('core', 80.0)
    w_num_name1 = weights.get('num_name1', 65.0)
    w_prefix_num = weights.get('prefix_num', 55.0)
    w_num_addr1 = weights.get('num_addr1', 75.0)
    w_name_bigram = weights.get('name_bigram', 40.0)
    w_prefix2_addr = weights.get('prefix2_addr', 25.0)
    w_name_tok = weights.get('name_tok', 8.0)
    w_addr_tok = weights.get('addr_tok', 15.0)
    w_num_only = weights.get('num_only', 12.0)
    boost_addr_combo = weights.get('boost_addr_combo', 50.0)
    
    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        country = str(row.country_normalized).strip().lower() if not pd.isna(row.country_normalized) else ""
        name = str(row.business_name_normalized).strip() if not pd.isna(row.business_name_normalized) else ""
        addr = str(row.business_address_normalized).strip() if not pd.isna(row.business_address_normalized) else ""

        core = extract_name_core(name) if name else ""
        ntoks = extract_tokens(name, GENERIC_NAME_WORDS, min_len=3) if name else []
        bigrams = extract_name_bigrams(ntoks)
        first_w = ntoks[0] if ntoks else ""
        name_prefix_4 = name[:4] if len(name) >= 4 else name
        name_prefix_2 = name[:2] if len(name) >= 2 else name

        nums = extract_address_numbers(addr) if addr else []
        atoks = extract_tokens(addr, GENERIC_ADDRESS_WORDS, min_len=4) if addr else []
        first_a = atoks[0] if atoks else ""

        scores = {}
        hit_num = set()
        hit_addr_tok = defaultdict(int)

        def _add(keys_list, score):
            for cand_idx in keys_list:
                scores[cand_idx] = scores.get(cand_idx, 0.0) + score

        # 1. Exact normalized name
        if name:
            _add(idx.exact_name_idx.get((country, name), ()), w_exact)

        # 2. Sorted core name
        if core:
            _add(idx.core_name_idx.get((country, core), ()), w_core)

        # 3. Compound: num + first name token
        if first_w:
            for num in nums:
                _add(idx.num_name1_idx.get((country, num, first_w), ()), w_num_name1)

        # 4. Compound: name prefix4 + num
        if name_prefix_4:
            for num in nums:
                _add(idx.prefix_num_idx.get((country, name_prefix_4, num), ()), w_prefix_num)

        # 5. Compound: num + first addr token
        if first_a:
            for num in nums:
                for cand_idx in idx.num_addr1_idx.get((country, num, first_a), ()):
                    scores[cand_idx] = scores.get(cand_idx, 0.0) + w_num_addr1
                    hit_num.add(cand_idx)

        # 6. Name bigrams
        for bg in bigrams:
            _add(idx.name_bigram_idx.get((country, bg), ()), w_name_bigram)

        # 7. Prefix2 + addr token
        if name_prefix_2:
            for tok in atoks:
                _add(idx.prefix2_name_idx.get((country, name_prefix_2, tok), ()), w_prefix2_addr)

        # 8. Address number only
        for num in nums:
            for cand_idx in idx.num_only_idx.get((country, num), ()):
                scores[cand_idx] = scores.get(cand_idx, 0.0) + w_num_only
                hit_num.add(cand_idx)

        # 9. Individual address tokens
        for tok in atoks:
            for cand_idx in idx.addr_token_idx.get((country, tok), ()):
                scores[cand_idx] = scores.get(cand_idx, 0.0) + w_addr_tok
                hit_addr_tok[cand_idx] += 1

        # 10. Individual name tokens
        for tok in ntoks:
            _add(idx.name_token_idx.get((country, tok), ()), w_name_tok)

        # Apply address synergy boost: if candidate has both address number AND address token(s)
        if boost_addr_combo > 0:
            for cand_idx in hit_num:
                tok_cnt = hit_addr_tok.get(cand_idx, 0)
                if tok_cnt >= 1:
                    scores[cand_idx] += boost_addr_combo + (tok_cnt * 12.0)

        if not scores:
            continue

        top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        for cand_idx, sc in top:
            candidates.append((s1_id, idx.candidate_ids[cand_idx]))

    cand_pairs = set(candidates)
    found = len(cand_pairs & true_pairs)
    recall = found / len(true_pairs)
    t_sec = time.perf_counter() - t0
    return recall, found, t_sec

configs = [
    ("Tuned 1: w_name_tok=6, boost=55", {
        'exact': 100.0, 'core': 80.0, 'num_name1': 65.0, 'prefix_num': 55.0,
        'num_addr1': 80.0, 'name_bigram': 40.0, 'prefix2_addr': 25.0,
        'name_tok': 6.0, 'addr_tok': 16.0, 'num_only': 10.0,
        'boost_addr_combo': 55.0
    }),
    ("Tuned 2: w_name_tok=5, boost=65, num_addr1=90", {
        'exact': 100.0, 'core': 80.0, 'num_name1': 65.0, 'prefix_num': 55.0,
        'num_addr1': 90.0, 'name_bigram': 40.0, 'prefix2_addr': 25.0,
        'name_tok': 5.0, 'addr_tok': 18.0, 'num_only': 10.0,
        'boost_addr_combo': 65.0
    }),
    ("Tuned 3: w_name_tok=4, boost=75, num_addr1=95", {
        'exact': 100.0, 'core': 80.0, 'num_name1': 70.0, 'prefix_num': 60.0,
        'num_addr1': 95.0, 'name_bigram': 40.0, 'prefix2_addr': 30.0,
        'name_tok': 4.0, 'addr_tok': 20.0, 'num_only': 10.0,
        'boost_addr_combo': 75.0
    }),
]

for name, cfg in configs:
    rec, cnt, dur = test_scoring_weights(s1, index, cfg, top_k=20)
    print(f"{name:<50} -> Recall@20: {rec*100:.2f}% ({cnt}/{len(true_pairs)}) in {dur:.2f}s", flush=True)
