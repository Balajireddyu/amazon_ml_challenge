import pandas as pd
import re
from rapidfuzz import process, fuzz


# ============================================================
# BUILD FIELD INDEX
# ============================================================

def build_field_index(df, field, prefix_lengths=(2, 3)):
    """
    Build an index using country + field prefix.

    Example:
        country = us
        name = microsoft corporation

    Keys:
        (us, mi)
        (us, mic)
    """

    index = {}

    for row in df.itertuples(index=False):

        value = getattr(row, field)

        if pd.isna(value):
            continue

        value = str(value).strip().lower()

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        for prefix_length in prefix_lengths:

            prefix = value[:prefix_length]

            if not prefix:
                continue

            key = (
                country,
                prefix
            )

            if key not in index:
                index[key] = []

            index[key].append(
                (
                    value,
                    row.entity_id
                )
            )

    return index


# ============================================================
# EXTRACT ADDRESS NUMBERS
# ============================================================

def extract_address_numbers(value):
    """
    Extract numeric tokens from an address.

    Example:
        '26 stone street unit 3 beverly ma'

    returns:
        {'26', '3'}
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    return set(
        re.findall(r'\d+', value)
    )


# ============================================================
# EXTRACT ADDRESS TOKENS
# ============================================================

def extract_address_tokens(value):
    """
    Extract useful word/number tokens from an address.

    Very short tokens are ignored.

    Example:
        '1216 preston avenue charlottesville va'

    returns approximately:
        {'1216', 'preston', 'avenue', 'charlottesville'}
    """

    if pd.isna(value):
        return set()

    value = str(value).strip().lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        value
    )

    return {
        token
        for token in tokens
        if len(token) >= 4
    }


# ============================================================
# GENERATE FUZZY CANDIDATES
# ============================================================

def generate_fuzzy_candidates(
    source1,
    source2,
    source3,
    top_k=20
):
    """
    Generate candidate entity pairs using multiple blocking
    strategies:

    1. Business-name prefix blocking
    2. Address prefix blocking
    3. Address-number blocking
    4. Address-token blocking

    The generated candidates are later used by the
    entity-resolution model.
    """

    candidates = []


    # ========================================================
    # BUILD NAME INDEXES
    # ========================================================

    name_index_s2 = build_field_index(
        source2,
        "business_name_normalized"
    )

    name_index_s3 = build_field_index(
        source3,
        "business_name_normalized"
    )


    # ========================================================
    # BUILD ADDRESS INDEXES
    # ========================================================

    address_index_s2 = build_field_index(
        source2,
        "business_address_normalized"
    )

    address_index_s3 = build_field_index(
        source3,
        "business_address_normalized"
    )


    # ========================================================
    # BUILD ADDRESS NUMBER INDEXES
    # ========================================================

    number_index_s2 = {}
    number_index_s3 = {}


    # -------------------------
    # Source 2
    # -------------------------

    for row in source2.itertuples(index=False):

        address = row.business_address_normalized

        if pd.isna(address):
            continue

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        numbers = extract_address_numbers(
            address
        )

        for number in numbers:

            # Normalize leading zeros
            try:
                normalized_number = str(
                    int(number)
                )
            except ValueError:
                normalized_number = number

            key = (
                country,
                normalized_number
            )

            if key not in number_index_s2:
                number_index_s2[key] = []

            number_index_s2[key].append(
                (
                    str(address).strip(),
                    row.entity_id
                )
            )


    # -------------------------
    # Source 3
    # -------------------------

    for row in source3.itertuples(index=False):

        address = row.business_address_normalized

        if pd.isna(address):
            continue

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        numbers = extract_address_numbers(
            address
        )

        for number in numbers:

            # Normalize leading zeros
            try:
                normalized_number = str(
                    int(number)
                )
            except ValueError:
                normalized_number = number

            key = (
                country,
                normalized_number
            )

            if key not in number_index_s3:
                number_index_s3[key] = []

            number_index_s3[key].append(
                (
                    str(address).strip(),
                    row.entity_id
                )
            )


    # ========================================================
    # BUILD ADDRESS TOKEN INDEXES
    # ========================================================

    token_index_s2 = {}
    token_index_s3 = {}


    # -------------------------
    # Source 2 token index
    # -------------------------

    for row in source2.itertuples(index=False):

        address = row.business_address_normalized

        if pd.isna(address):
            continue

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        tokens = extract_address_tokens(
            address
        )

        for token in tokens:

            key = (
                country,
                token
            )

            if key not in token_index_s2:
                token_index_s2[key] = []

            token_index_s2[key].append(
                (
                    str(address).strip(),
                    row.entity_id
                )
            )


    # -------------------------
    # Source 3 token index
    # -------------------------

    for row in source3.itertuples(index=False):

        address = row.business_address_normalized

        if pd.isna(address):
            continue

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        tokens = extract_address_tokens(
            address
        )

        for token in tokens:

            key = (
                country,
                token
            )

            if key not in token_index_s3:
                token_index_s3[key] = []

            token_index_s3[key].append(
                (
                    str(address).strip(),
                    row.entity_id
                )
            )


    # ========================================================
    # PROCESS EACH SOURCE 1 RECORD
    # ========================================================

    for row in source1.itertuples(index=False):

        source1_id = row.entity_id

        country = row.country_normalized

        if pd.isna(country):
            continue

        country = str(country).strip().lower()


        # ====================================================
        # NAME BLOCKING
        # ====================================================

        name = row.business_name_normalized

        if not pd.isna(name):

            name = str(name).strip()

            for prefix_length in [2, 3]:

                prefix = name[:prefix_length]

                if not prefix:
                    continue

                key = (
                    country,
                    prefix
                )


                # -------------------------
                # Source 2
                # -------------------------

                possible_records = name_index_s2.get(
                    key,
                    []
                )

                if possible_records:

                    names = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        name,
                        names,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_name,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S2",
                                "name",
                                score
                            ]
                        )


                # -------------------------
                # Source 3
                # -------------------------

                possible_records = name_index_s3.get(
                    key,
                    []
                )

                if possible_records:

                    names = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        name,
                        names,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_name,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S3",
                                "name",
                                score
                            ]
                        )


        # ====================================================
        # ADDRESS PREFIX BLOCKING
        # ====================================================

        address = row.business_address_normalized

        if not pd.isna(address):

            address = str(address).strip()

            for prefix_length in [2, 3]:

                prefix = address[:prefix_length]

                if not prefix:
                    continue

                key = (
                    country,
                    prefix
                )


                # -------------------------
                # Source 2
                # -------------------------

                possible_records = address_index_s2.get(
                    key,
                    []
                )

                if possible_records:

                    addresses = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        address,
                        addresses,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_address,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S2",
                                "address",
                                score
                            ]
                        )


                # -------------------------
                # Source 3
                # -------------------------

                possible_records = address_index_s3.get(
                    key,
                    []
                )

                if possible_records:

                    addresses = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        address,
                        addresses,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_address,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S3",
                                "address",
                                score
                            ]
                        )


        # ====================================================
        # ADDRESS NUMBER BLOCKING
        # ====================================================

        if not pd.isna(address):

            address_numbers = extract_address_numbers(
                address
            )

            for number in address_numbers:

                # Normalize leading zeros
                try:
                    normalized_number = str(
                        int(number)
                    )
                except ValueError:
                    normalized_number = number

                key = (
                    country,
                    normalized_number
                )


                # -------------------------
                # Source 2
                # -------------------------

                possible_records = number_index_s2.get(
                    key,
                    []
                )

                if possible_records:

                    addresses = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        address,
                        addresses,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_address,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S2",
                                "address_number",
                                score
                            ]
                        )


                # -------------------------
                # Source 3
                # -------------------------

                possible_records = number_index_s3.get(
                    key,
                    []
                )

                if possible_records:

                    addresses = [
                        record[0]
                        for record in possible_records
                    ]

                    matches = process.extract(
                        address,
                        addresses,
                        scorer=fuzz.ratio,
                        limit=top_k
                    )

                    for (
                        matched_address,
                        score,
                        position
                    ) in matches:

                        candidate_id = (
                            possible_records[position][1]
                        )

                        candidates.append(
                            [
                                source1_id,
                                candidate_id,
                                "S3",
                                "address_number",
                                score
                            ]
                        )


        # ====================================================
        # ADDRESS TOKEN BLOCKING
        # ====================================================

        if not pd.isna(address):

            address = str(address).strip()

            if address:

                tokens = extract_address_tokens(
                    address
                )

                for token in tokens:

                    key = (
                        country,
                        token
                    )


                    # -------------------------
                    # Source 2
                    # -------------------------

                    possible_records = token_index_s2.get(
                        key,
                        []
                    )

                    if possible_records:

                        addresses = [
                            record[0]
                            for record in possible_records
                        ]

                        matches = process.extract(
                            address,
                            addresses,
                            scorer=fuzz.ratio,
                            limit=top_k
                        )

                        for (
                            matched_address,
                            score,
                            position
                        ) in matches:

                            candidate_id = (
                                possible_records[position][1]
                            )

                            candidates.append(
                                [
                                    source1_id,
                                    candidate_id,
                                    "S2",
                                    "address_token",
                                    score
                                ]
                            )


                    # -------------------------
                    # Source 3
                    # -------------------------

                    possible_records = token_index_s3.get(
                        key,
                        []
                    )

                    if possible_records:

                        addresses = [
                            record[0]
                            for record in possible_records
                        ]

                        matches = process.extract(
                            address,
                            addresses,
                            scorer=fuzz.ratio,
                            limit=top_k
                        )

                        for (
                            matched_address,
                            score,
                            position
                        ) in matches:

                            candidate_id = (
                                possible_records[position][1]
                            )

                            candidates.append(
                                [
                                    source1_id,
                                    candidate_id,
                                    "S3",
                                    "address_token",
                                    score
                                ]
                            )


    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    candidates_df = pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "match_field",
            "score"
        ]
    )


    # ========================================================
    # REMOVE DUPLICATE ENTITY PAIRS
    # ========================================================

    candidates_df = candidates_df.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )


    candidates_df = candidates_df.reset_index(
        drop=True
    )


    return candidates_df