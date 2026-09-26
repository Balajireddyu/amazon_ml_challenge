import sys
sys.path.insert(0, 'student_resource/code/business_entity_resolution')
import re, heapq, time, psutil, pandas as pd
from collections import defaultdict, Counter
from src.preprocessing import preprocess_dataframe
from src.candidate_generation import (
    GENERIC_NAME_WORDS, GENERIC_ADDRESS_WORDS,
    extract_name_core, extract_address_numbers, extract_tokens,
    InvertedCandidateIndex, generate_candidates_from_index
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
cand_df = generate_candidates_from_index(s1, index, top_k=20)

cand_pairs = set(zip(cand_df['source1_entity_id'], cand_df['candidate_entity_id']))
missed = [(s, c) for s, c in true_pairs if (s, c) not in cand_pairs]
print(f'Total missed: {len(missed)} / {len(true_pairs)}')

s1_lkp = s1.set_index('entity_id')
s2_lkp = s2.set_index('entity_id')
s3_lkp = s3.set_index('entity_id')

category = Counter()
SHOW = 40

print('\n=== MISSED PAIR ANALYSIS (first 40) ===')
for s1_id, cid in missed[:SHOW]:
    lkp = s2_lkp if cid.startswith('S2-') else s3_lkp
    s1r = s1_lkp.loc[s1_id]
    if cid not in lkp.index:
        category['not_in_smoke_s2s3'] += 1
        continue
    cr = lkp.loc[cid]
    c1 = str(s1r['country_normalized']).strip().lower()
    c2 = str(cr['country_normalized']).strip().lower()
    n1 = str(s1r['business_name_normalized']).strip()
    n2 = str(cr['business_name_normalized']).strip()
    a1 = str(s1r['business_address_normalized']).strip()
    a2 = str(cr['business_address_normalized']).strip()
    if c1 != c2:
        category['country_mismatch'] += 1
        print(f'CTRY_MISMATCH: S1=[{n1}|{a1}|{c1}] C=[{n2}|{a2}|{c2}]')
    elif not n1 and not n2 and not a1 and not a2:
        category['both_empty'] += 1
    else:
        category['fell_out_topk'] += 1
        tok1 = set(extract_tokens(n1, GENERIC_NAME_WORDS, 3))
        tok2 = set(extract_tokens(n2, GENERIC_NAME_WORDS, 3))
        addr1 = set(extract_tokens(a1, GENERIC_ADDRESS_WORDS, 4))
        addr2 = set(extract_tokens(a2, GENERIC_ADDRESS_WORDS, 4))
        num1 = set(extract_address_numbers(a1))
        num2 = set(extract_address_numbers(a2))
        core1 = extract_name_core(n1)
        core2 = extract_name_core(n2)
        print(f'TOPK: S1=[{n1}|{a1}]')
        print(f'   C=[{n2}|{a2}]')
        print(f'  tok_common={tok1 & tok2}, core_match={core1==core2}, num_common={num1 & num2}, addr_tok_common={addr1 & addr2}')

# Analyze all missed
print('\n=== FULL ANALYSIS OF ALL 305 MISSED ===')
category_full = Counter()
country_mismatch_count = 0
not_in_pool = 0
addr_only_match = 0
name_empty_both = 0
name_token_count = Counter()

for s1_id, cid in missed:
    lkp = s2_lkp if cid.startswith('S2-') else s3_lkp
    s1r = s1_lkp.loc[s1_id]
    if cid not in lkp.index:
        not_in_pool += 1
        category_full['not_in_smoke_pool'] += 1
        continue
    cr = lkp.loc[cid]
    c1 = str(s1r['country_normalized']).strip().lower()
    c2 = str(cr['country_normalized']).strip().lower()
    n1 = str(s1r['business_name_normalized']).strip()
    n2 = str(cr['business_name_normalized']).strip()
    a1 = str(s1r['business_address_normalized']).strip()
    a2 = str(cr['business_address_normalized']).strip()
    if c1 != c2:
        country_mismatch_count += 1
        category_full['country_mismatch'] += 1
    else:
        tok1 = set(extract_tokens(n1, GENERIC_NAME_WORDS, 3))
        tok2 = set(extract_tokens(n2, GENERIC_NAME_WORDS, 3))
        addr1 = set(extract_tokens(a1, GENERIC_ADDRESS_WORDS, 4))
        addr2 = set(extract_tokens(a2, GENERIC_ADDRESS_WORDS, 4))
        num1 = set(extract_address_numbers(a1))
        num2 = set(extract_address_numbers(a2))
        common_tok = tok1 & tok2
        common_addr = addr1 & addr2
        common_num = num1 & num2
        if not n1 or not n2:
            category_full['name_empty'] += 1
        elif common_tok:
            name_token_count[len(common_tok)] += 1
            category_full['fell_out_topk_has_common_name_tok'] += 1
        elif common_num and common_addr:
            category_full['fell_out_topk_num_addr_tok'] += 1
        elif common_num:
            category_full['fell_out_topk_num_only'] += 1
        elif common_addr:
            category_full['fell_out_topk_addr_tok_only'] += 1
        else:
            category_full['fell_out_completely_no_common_key'] += 1
            print(f'NO_KEY: S1=[{n1}|{a1}] C=[{n2}|{a2}]')

print('Category breakdown:')
for k, v in sorted(category_full.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
print(f'\nNot in smoke pool: {not_in_pool}')
print(f'Country mismatch: {country_mismatch_count}')
print(f'\nName token common count distribution: {dict(name_token_count)}')
