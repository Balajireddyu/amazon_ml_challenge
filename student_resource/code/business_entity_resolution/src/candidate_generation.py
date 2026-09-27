"""
High-Performance Scalable Candidate Generation for Business Entity Resolution.

Design:
  - Country blocking (exact country match)
  - Inverted index with 8 blocking key types, weighted scoring
  - Pruned high-frequency token buckets to control noise
  - Configurable top-K per S1 record, no per-row fuzzy calls
  - finalize() must be called after all add_source() calls before retrieval
"""

import re
import heapq
from collections import defaultdict
import pandas as pd


# ============================================================
# STOPWORDS & FILTER LISTS
# ============================================================

GENERIC_NAME_WORDS = {
    "the", "and", "of", "for", "in", "on", "at", "to", "a", "an",
    "co", "company", "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "group", "services", "service", "solutions", "solution", "holdings", "holding",
    "international", "intl", "enterprises", "enterprise", "consulting", "technologies",
    "technology", "tech", "management", "global", "system", "systems", "associates",
    "associate", "center", "centre", "care", "health", "industries", "industry",
    "products", "studio", "studios", "agency", "firm", "pvt", "private", "usa", "us",
    "l", "c", "trading", "import", "export", "retail", "wholesale", "real", "estate",
    "construction", "food", "media", "digital"
}

GENERIC_ADDRESS_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln",
    "blvd", "boulevard", "highway", "hwy", "floor", "fl", "suite", "ste", "unit",
    "building", "bldg", "parkway", "pkwy", "square", "sq", "plaza", "plz", "court",
    "ct", "circle", "cir", "route", "rt", "place", "pl", "block", "blk", "alley",
    "north", "south", "east", "west", "n", "s", "e", "w", "null", "po", "box",
    "tx", "fl", "ca", "ny", "il", "pa", "oh", "ga", "nc", "mi", "nj", "va", "wa",
    "az", "ma", "tn", "in", "mo", "md", "wi", "co", "mn", "sc", "al", "la", "ky",
    "or", "ok", "ct", "ut", "ia", "nv", "ar", "ms", "ks", "nm", "ne", "wv", "id",
    "hi", "nh", "me", "mt", "ri", "de", "sd", "nd", "ak", "vt", "wy"
}


# ============================================================
# HELPER EXTRACTORS
# ============================================================

def extract_name_core(name: str) -> str:
    """Sorted bag-of-words ignoring generic tokens (order-invariant match)."""
    if not name or pd.isna(name):
        return ""
    tokens = [
        t for t in str(name).split()
        if t not in GENERIC_NAME_WORDS and len(t) > 1
    ]
    tokens.sort()
    return " ".join(tokens)


def extract_address_numbers(address: str) -> list:
    """Extract distinct normalized integer strings from address."""
    if not address or pd.isna(address):
        return []
    nums = re.findall(r"\d+", str(address))
    res = []
    for n in nums:
        try:
            res.append(str(int(n)))
        except ValueError:
            res.append(n)
    return list(dict.fromkeys(res))


def extract_tokens(text: str, stopwords: set, min_len: int = 4) -> list:
    """Alphanumeric tokens with length >= min_len not in stopwords."""
    if not text or pd.isna(text):
        return []
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) >= min_len and t not in stopwords]


def extract_name_bigrams(ntoks: list) -> list:
    """Adjacent pairs of name tokens (order-sensitive bigrams)."""
    return [f"{ntoks[i]}_{ntoks[i+1]}" for i in range(len(ntoks) - 1)]


# ============================================================
# INVERTED INDEX (all defaultdict; pruning via finalize())
# ============================================================

class InvertedCandidateIndex:
    """
    Multi-key inverted index for blocking candidate entity pairs.

    Call add_source() for each candidate source (S2, S3 ...), then
    call finalize() once before generating candidates.
    """

    def __init__(self, max_bucket_size: int = 500):
        self.max_bucket_size = max_bucket_size
        self.candidate_ids: list = []
        self.candidate_sources: list = []
        self._finalized = False

        # Blocking indexes (defaultdict during build phase)
        self.exact_name_idx   = defaultdict(list)  # (country, norm_name)
        self.core_name_idx    = defaultdict(list)  # (country, sorted_core)
        self.num_name1_idx    = defaultdict(list)  # (country, num, first_name_tok)
        self.num_addr1_idx    = defaultdict(list)  # (country, num, first_addr_tok)
        self.prefix_num_idx   = defaultdict(list)  # (country, name_prefix4, num)
        self.name_token_idx   = defaultdict(list)  # (country, name_tok)
        self.addr_token_idx   = defaultdict(list)  # (country, addr_tok)
        self.name_bigram_idx  = defaultdict(list)  # (country, "tok1_tok2")
        self.num_only_idx     = defaultdict(list)  # (country, num) — addr-number-only
        self.prefix2_name_idx = defaultdict(list)  # (country, name_prefix2, name_tok)

    @staticmethod
    def _append_prunable(index_dict: dict, key: tuple, cand_idx: int, limit: int):
        """
        Inline bucket pruning helper for defaultdict(list) buckets.
        If a bucket exceeds `limit` during index build, it is set to None sentinel.
        Subsequent insertions for disabled keys (None) are ignored immediately,
        preventing memory accumulation for high-frequency generic tokens.
        """
        val = index_dict[key]
        if val is not None:
            val.append(cand_idx)
            if len(val) > limit:
                index_dict[key] = None

    def add_source(self, df: pd.DataFrame, source_label: str):
        """Index all records from a candidate dataframe. May be called multiple times."""
        assert not self._finalized, "Cannot add_source after finalize()."
        offset = len(self.candidate_ids)

        for row in df.itertuples(index=False):
            cand_idx = offset
            offset += 1

            self.candidate_ids.append(row.entity_id)
            self.candidate_sources.append(source_label)

            country = (
                str(row.country_normalized).strip().lower()
                if not pd.isna(row.country_normalized) else ""
            )
            name = (
                str(row.business_name_normalized).strip()
                if not pd.isna(row.business_name_normalized) else ""
            )
            addr = (
                str(row.business_address_normalized).strip()
                if not pd.isna(row.business_address_normalized) else ""
            )

            # ---- Name indexing ----
            if name:
                self.exact_name_idx[(country, name)].append(cand_idx)

                core = extract_name_core(name)
                if core and core != name:
                    self.core_name_idx[(country, core)].append(cand_idx)

                ntoks = extract_tokens(name, GENERIC_NAME_WORDS, min_len=3)
                for tok in ntoks:
                    self._append_prunable(self.name_token_idx, (country, tok), cand_idx, self.max_bucket_size)

                for bg in extract_name_bigrams(ntoks):
                    self._append_prunable(self.name_bigram_idx, (country, bg), cand_idx, self.max_bucket_size)

                first_w = ntoks[0] if ntoks else ""
                name_prefix_4 = name[:4]
                name_prefix_2 = name[:2]
            else:
                first_w = ""
                name_prefix_4 = ""
                name_prefix_2 = ""
                ntoks = []

            # ---- Address indexing ----
            if addr:
                nums = extract_address_numbers(addr)
                atoks = extract_tokens(addr, GENERIC_ADDRESS_WORDS, min_len=4)
                for tok in atoks:
                    self._append_prunable(self.addr_token_idx, (country, tok), cand_idx, self.max_bucket_size)

                first_a = atoks[0] if atoks else ""

                for num in nums:
                    # Addr-number-only (catches same-address matches with different names)
                    self._append_prunable(self.num_only_idx, (country, num), cand_idx, self.max_bucket_size // 5)

                    # Compound: num + first name token
                    if first_w:
                        self.num_name1_idx[(country, num, first_w)].append(cand_idx)

                    # Compound: num + first address token
                    if first_a:
                        self.num_addr1_idx[(country, num, first_a)].append(cand_idx)

                    # Compound: name prefix4 + num
                    if name_prefix_4:
                        self.prefix_num_idx[(country, name_prefix_4, num)].append(cand_idx)

                # 2-char name prefix + each addr token (light cross signal)
                if name_prefix_2:
                    for tok in atoks:
                        self._append_prunable(self.prefix2_name_idx, (country, name_prefix_2, tok), cand_idx, self.max_bucket_size)

    def finalize(self):
        """
        Prune oversized token buckets (high-frequency generic terms) and
        freeze the index. Must be called once after all add_source() calls.
        Safe and idempotent.
        """
        mb = self.max_bucket_size

        self.name_token_idx = {
            k: v for k, v in self.name_token_idx.items() if v is not None and len(v) <= mb
        }
        self.addr_token_idx = {
            k: v for k, v in self.addr_token_idx.items() if v is not None and len(v) <= mb
        }
        self.name_bigram_idx = {
            k: v for k, v in self.name_bigram_idx.items() if v is not None and len(v) <= mb
        }
        self.num_only_idx = {
            k: v for k, v in self.num_only_idx.items() if v is not None and len(v) <= mb // 5
        }
        self.prefix2_name_idx = {
            k: v for k, v in self.prefix2_name_idx.items() if v is not None and len(v) <= mb
        }
        self._finalized = True


# ============================================================
# ============================================================
# CANDIDATE GENERATION ENGINE
# ============================================================

SCORE_EXACT_NAME   = 100.0
SCORE_CORE_NAME    = 80.0
SCORE_NUM_ADDR1    = 85.0
SCORE_NUM_NAME1    = 65.0
SCORE_PREFIX_NUM   = 55.0
SCORE_NAME_BIGRAM  = 40.0
SCORE_PREFIX2_ADDR = 25.0
SCORE_ADDR_TOK     = 16.0
SCORE_NUM_ONLY     = 12.0
SCORE_NAME_TOK     = 6.0


def _add_scores(scores: dict, keys_list, score: float):
    for cand_idx in keys_list:
        scores[cand_idx] = scores.get(cand_idx, 0.0) + score


def generate_candidates_from_index(
    source1: pd.DataFrame,
    index: InvertedCandidateIndex,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    For each S1 record, score all candidate entities reached via index lookups,
    then return the top-K by score. No fuzzy string calls.
    """
    assert index._finalized, "Call index.finalize() before generating candidates."

    candidates = []

    for row in source1.itertuples(index=False):
        s1_id = row.entity_id

        country = (
            str(row.country_normalized).strip().lower()
            if not pd.isna(row.country_normalized) else ""
        )
        name = (
            str(row.business_name_normalized).strip()
            if not pd.isna(row.business_name_normalized) else ""
        )
        addr = (
            str(row.business_address_normalized).strip()
            if not pd.isna(row.business_address_normalized) else ""
        )

        core = extract_name_core(name) if name else ""
        ntoks = extract_tokens(name, GENERIC_NAME_WORDS, min_len=3) if name else []
        bigrams = extract_name_bigrams(ntoks)
        first_w = ntoks[0] if ntoks else ""
        name_prefix_4 = name[:4] if len(name) >= 4 else name
        name_prefix_2 = name[:2] if len(name) >= 2 else name

        nums = extract_address_numbers(addr) if addr else []
        atoks = extract_tokens(addr, GENERIC_ADDRESS_WORDS, min_len=4) if addr else []
        first_a = atoks[0] if atoks else ""

        scores: dict = {}
        hit_num = set()
        hit_addr_tok = defaultdict(int)

        # 1. Exact normalized name
        if name:
            _add_scores(scores, index.exact_name_idx.get((country, name), ()), SCORE_EXACT_NAME)

        # 2. Sorted core name (order-invariant)
        if core:
            _add_scores(scores, index.core_name_idx.get((country, core), ()), SCORE_CORE_NAME)

        # 3. Compound: address number + first address token (strong address signal)
        if first_a:
            for num in nums:
                for cand_idx in index.num_addr1_idx.get((country, num, first_a), ()):
                    scores[cand_idx] = scores.get(cand_idx, 0.0) + SCORE_NUM_ADDR1
                    hit_num.add(cand_idx)

        # 4. Compound: address number + first name token
        if first_w:
            for num in nums:
                _add_scores(scores, index.num_name1_idx.get((country, num, first_w), ()), SCORE_NUM_NAME1)

        # 5. Compound: name prefix-4 + address number
        if name_prefix_4:
            for num in nums:
                _add_scores(scores, index.prefix_num_idx.get((country, name_prefix_4, num), ()), SCORE_PREFIX_NUM)

        # 6. Name bigrams (adjacent token pairs, order-sensitive)
        for bg in bigrams:
            _add_scores(scores, index.name_bigram_idx.get((country, bg), ()), SCORE_NAME_BIGRAM)

        # 7. Name prefix2 + each address token (cross-field signal)
        if name_prefix_2:
            for tok in atoks:
                _add_scores(scores, index.prefix2_name_idx.get((country, name_prefix_2, tok), ()), SCORE_PREFIX2_ADDR)

        # 8. Address number only
        for num in nums:
            for cand_idx in index.num_only_idx.get((country, num), ()):
                scores[cand_idx] = scores.get(cand_idx, 0.0) + SCORE_NUM_ONLY
                hit_num.add(cand_idx)

        # 9. Individual address tokens
        for tok in atoks:
            for cand_idx in index.addr_token_idx.get((country, tok), ()):
                scores[cand_idx] = scores.get(cand_idx, 0.0) + SCORE_ADDR_TOK
                hit_addr_tok[cand_idx] += 1

        # 10. Individual name tokens (downweighted to prevent generic name blowout)
        for tok in ntoks:
            _add_scores(scores, index.name_token_idx.get((country, tok), ()), SCORE_NAME_TOK)

        # 11. Address Synergy Boost: address number + address token(s)
        for cand_idx in hit_num:
            tok_cnt = hit_addr_tok.get(cand_idx, 0)
            if tok_cnt >= 1:
                scores[cand_idx] += 55.0 + (tok_cnt * 10.0)

        if not scores:
            continue

        top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        for cand_idx, sc in top:
            candidates.append((
                s1_id,
                index.candidate_ids[cand_idx],
                index.candidate_sources[cand_idx],
                sc,
            ))

    return pd.DataFrame(
        candidates,
        columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score"],
    )


# ============================================================
# HIGH-LEVEL PUBLIC API
# ============================================================

def generate_fuzzy_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    top_k: int = 20,
    max_bucket_size: int = 500,
) -> pd.DataFrame:
    """
    Build index over S2+S3, generate top-K candidates for every S1 entity.

    Parameters
    ----------
    source1, source2, source3 : preprocessed DataFrames
    top_k : candidates to return per S1 entity (configurable)
    max_bucket_size : max inverted list size before a token key is pruned
    """
    index = InvertedCandidateIndex(max_bucket_size=max_bucket_size)
    index.add_source(source2, "S2")
    index.add_source(source3, "S3")
    index.finalize()

    return generate_candidates_from_index(source1, index, top_k=top_k)


# Alias for backward compat with original student code
generate_candidates = generate_fuzzy_candidates