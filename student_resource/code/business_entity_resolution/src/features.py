"""
Pairwise Feature Extraction for Business Entity Resolution (Phase 3).

Features computed:
- Name similarities: ratio, partial_ratio, token_sort_ratio, token_set_ratio, exact match
- Name core similarities: core_ratio, core_token_set_ratio, core_exact_match
- Name token overlaps: token_jaccard, token_overlap_count, len_diff_ratio, prefix4_match
- Address similarities: ratio, partial_ratio, token_sort_ratio, token_set_ratio, exact match
- Address token overlaps: token_jaccard, token_overlap_count
- Address numbers: num_exact_match, num_jaccard, num_overlap_count, num_has_match
- Country & Cross features: country_match, candidate_score, joint_similarity, source_is_s2
"""

import re
import pandas as pd
from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio,
)

from src.candidate_generation import (
    GENERIC_NAME_WORDS,
    GENERIC_ADDRESS_WORDS,
    extract_name_core,
    extract_address_numbers,
    extract_tokens,
    EntityRecord,
    build_entity_record,
)


def _precompute_entity_record(
    df: pd.DataFrame,
    needed_ids: set = None,
    s1_records_map: dict = None,
) -> dict:
    """Precompute tokens and cores once per entity to maximize extraction speed."""
    record_map = {}
    if needed_ids is not None and len(needed_ids) == 0:
        return record_map

    for row in df.itertuples(index=False):
        eid = row.entity_id
        if needed_ids is not None and eid not in needed_ids:
            continue

        rec = s1_records_map.get(eid) if s1_records_map else None
        if rec is None:
            rec = build_entity_record(row)

        record_map[eid] = {
            "name": rec.name,
            "core": rec.core,
            "addr": rec.addr,
            "country": rec.country,
            "ntoks": set(rec.ntoks),
            "atoks": set(rec.atoks),
            "nums": set(rec.nums),
            "pfx4": rec.name_prefix_4,
            "name_len": len(rec.name),
        }
    return record_map


def create_matching_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    s1_records_map: dict = None,
) -> pd.DataFrame:
    """
    Generate rich pairwise matching features for candidate pairs.
    """
    # Filter needed IDs to avoid processing unused entities from large sources
    s1_needed = set(candidates["source1_entity_id"])
    cand_needed = set(candidates["candidate_entity_id"])

    # 1. Precompute record representations for fast lookup
    s1_map = _precompute_entity_record(source1, needed_ids=s1_needed, s1_records_map=s1_records_map)
    s2_map = _precompute_entity_record(source2, needed_ids=cand_needed, s1_records_map=s1_records_map)
    s3_map = _precompute_entity_record(source3, needed_ids=cand_needed, s1_records_map=s1_records_map)


    has_score_col = "score" in candidates.columns
    has_source_col = "candidate_source" in candidates.columns

    feature_rows = []

    for row in candidates.itertuples(index=False):
        s1_id = row.source1_entity_id
        c_id = row.candidate_entity_id
        c_src = row.candidate_source if has_source_col else ("S2" if c_id.startswith("S2-") else "S3")
        gen_score = float(row.score) if has_score_col else 0.0

        r1 = s1_map.get(s1_id)
        r2 = s2_map.get(c_id) if c_src == "S2" else s3_map.get(c_id)

        if not r1 or not r2:
            continue

        n1, n2 = r1["name"], r2["name"]
        a1, a2 = r1["addr"], r2["addr"]
        c1, c2 = r1["country"], r2["country"]
        core1, core2 = r1["core"], r2["core"]

        # ====================================================
        # 1. NAME SIMILARITY FEATURES
        # ====================================================
        name_rat = ratio(n1, n2)
        name_part = partial_ratio(n1, n2)
        name_tsort = token_sort_ratio(n1, n2)
        name_tset = token_set_ratio(n1, n2)
        name_exact = 1.0 if (n1 and n1 == n2) else 0.0

        # Core name features (order & generic-word invariant)
        core_rat = ratio(core1, core2) if (core1 and core2) else 0.0
        core_tset = token_set_ratio(core1, core2) if (core1 and core2) else 0.0
        core_exact = 1.0 if (core1 and core1 == core2) else 0.0

        # Name token overlap & Jaccard
        ntoks1, ntoks2 = r1["ntoks"], r2["ntoks"]
        n_inter = len(ntoks1 & ntoks2)
        n_union = len(ntoks1 | ntoks2)
        name_tok_jaccard = (n_inter / n_union) if n_union > 0 else 0.0
        name_tok_overlap_count = float(n_inter)

        # Name structural / length features
        l1, l2 = r1["name_len"], r2["name_len"]
        name_len_diff = abs(l1 - l2) / max(l1, l2, 1)
        name_prefix4_match = 1.0 if (r1["pfx4"] and r1["pfx4"] == r2["pfx4"]) else 0.0

        # ====================================================
        # 2. ADDRESS SIMILARITY FEATURES
        # ====================================================
        addr_rat = ratio(a1, a2) if (a1 and a2) else 0.0
        addr_part = partial_ratio(a1, a2) if (a1 and a2) else 0.0
        addr_tsort = token_sort_ratio(a1, a2) if (a1 and a2) else 0.0
        addr_tset = token_set_ratio(a1, a2) if (a1 and a2) else 0.0
        addr_exact = 1.0 if (a1 and a1 == a2) else 0.0

        # Address token overlap & Jaccard
        atoks1, atoks2 = r1["atoks"], r2["atoks"]
        a_inter = len(atoks1 & atoks2)
        a_union = len(atoks1 | atoks2)
        addr_tok_jaccard = (a_inter / a_union) if a_union > 0 else 0.0
        addr_tok_overlap_count = float(a_inter)

        # ====================================================
        # 3. ADDRESS NUMBER FEATURES
        # ====================================================
        nums1, nums2 = r1["nums"], r2["nums"]
        num_inter = len(nums1 & nums2)
        num_union = len(nums1 | nums2)
        num_jaccard = (num_inter / num_union) if num_union > 0 else 0.0
        num_overlap_count = float(num_inter)
        num_has_match = 1.0 if num_inter > 0 else 0.0
        num_exact_match = 1.0 if (nums1 and nums1 == nums2) else 0.0

        # ====================================================
        # 4. CROSS, COUNTRY & META FEATURES
        # ====================================================
        country_match = 1.0 if (c1 and c1 == c2) else 0.0
        joint_sim = 0.5 * name_tset + 0.5 * addr_tset
        source_is_s2 = 1.0 if c_src == "S2" else 0.0

        feature_rows.append((
            s1_id,
            c_id,
            c_src,
            # Name
            name_rat,
            name_part,
            name_tsort,
            name_tset,
            name_exact,
            core_rat,
            core_tset,
            core_exact,
            name_tok_jaccard,
            name_tok_overlap_count,
            name_len_diff,
            name_prefix4_match,
            # Address
            addr_rat,
            addr_part,
            addr_tsort,
            addr_tset,
            addr_exact,
            addr_tok_jaccard,
            addr_tok_overlap_count,
            # Address numbers
            num_exact_match,
            num_jaccard,
            num_overlap_count,
            num_has_match,
            # Country & Meta
            country_match,
            joint_sim,
            gen_score,
            source_is_s2,
        ))

    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        # Name
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_exact_match",
        "name_core_ratio",
        "name_core_token_set_ratio",
        "name_core_exact_match",
        "name_token_jaccard",
        "name_token_overlap_count",
        "name_len_diff",
        "name_prefix4_match",
        # Address
        "address_ratio",
        "address_partial_ratio",
        "address_token_sort_ratio",
        "address_token_set_ratio",
        "address_exact_match",
        "address_token_jaccard",
        "address_token_overlap_count",
        # Numbers
        "address_num_exact_match",
        "address_num_jaccard",
        "address_num_overlap_count",
        "address_num_has_match",
        # Country & Meta
        "country_match",
        "joint_similarity",
        "candidate_score",
        "source_is_s2",
    ]

    return pd.DataFrame(feature_rows, columns=columns)
