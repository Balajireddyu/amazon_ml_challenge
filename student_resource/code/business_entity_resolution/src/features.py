import pandas as pd
import re

from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio
)


# ============================================================
# EXTRACT NORMALIZED ADDRESS NUMBERS
# ============================================================

def extract_normalized_numbers(value):

    if pd.isna(value):
        return set()

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    normalized = set()

    for number in numbers:

        try:
            normalized.add(
                str(int(number))
            )

        except ValueError:
            normalized.add(
                number
            )

    return normalized


def create_matching_features(
    candidates,
    source1,
    source2,
    source3
):

    source1_lookup = source1.set_index(
        "entity_id"
    ).to_dict("index")

    source2_lookup = source2.set_index(
        "entity_id"
    ).to_dict("index")

    source3_lookup = source3.set_index(
        "entity_id"
    ).to_dict("index")

    features = []

    for row in candidates.itertuples(index=False):

        # ----------------------------------------------------
        # Get Source-1 record
        # ----------------------------------------------------

        s1 = source1_lookup[
            row.source1_entity_id
        ]

        # ----------------------------------------------------
        # Get candidate record
        # ----------------------------------------------------

        if row.candidate_source == "S2":

            candidate = source2_lookup[
                row.candidate_entity_id
            ]

        else:

            candidate = source3_lookup[
                row.candidate_entity_id
            ]

        # ----------------------------------------------------
        # Get normalized values
        # ----------------------------------------------------

        name1 = str(
            s1["business_name_normalized"]
        )

        name2 = str(
            candidate["business_name_normalized"]
        )

        address1 = str(
            s1["business_address_normalized"]
        )

        address2 = str(
            candidate["business_address_normalized"]
        )

        country1 = str(
            s1["country_normalized"]
        )

        country2 = str(
            candidate["country_normalized"]
        )

        # ====================================================
        # NAME FEATURES
        # ====================================================

        name_similarity = ratio(
            name1,
            name2
        )

        name_partial_similarity = partial_ratio(
            name1,
            name2
        )

        name_token_sort_similarity = token_sort_ratio(
            name1,
            name2
        )

        name_token_set_similarity = token_set_ratio(
            name1,
            name2
        )

        exact_name = int(
            name1 == name2
        )

        # ====================================================
        # ADDRESS FEATURES
        # ====================================================

        address_similarity = ratio(
            address1,
            address2
        )

        address_partial_similarity = partial_ratio(
            address1,
            address2
        )

        address_token_sort_similarity = token_sort_ratio(
            address1,
            address2
        )

        address_token_set_similarity = token_set_ratio(
            address1,
            address2
        )

        exact_address = int(
            address1 == address2
        )

        # ====================================================
        # COUNTRY FEATURE
        # ====================================================

        country_match = int(
            country1 == country2
        )

        # ====================================================
        # ADDRESS NUMBER OVERLAP
        # ====================================================

        numbers1 = extract_normalized_numbers(
            address1
        )

        numbers2 = extract_normalized_numbers(
            address2
        )

        if numbers1 and numbers2:

            address_number_overlap = (
                len(
                    numbers1.intersection(numbers2)
                )
                /
                len(
                    numbers1.union(numbers2)
                )
            )

        else:

            address_number_overlap = 0.0

        # ====================================================
        # COMBINED NAME + ADDRESS SIMILARITY
        # ====================================================

        combined_similarity = (
            0.5 * name_token_set_similarity
            +
            0.5 * address_token_set_similarity
        )

        # ====================================================
        # STORE FEATURES
        # ====================================================

        features.append([

            row.source1_entity_id,
            row.candidate_entity_id,
            row.candidate_source,

            name_similarity,
            name_partial_similarity,
            name_token_sort_similarity,
            name_token_set_similarity,

            address_similarity,
            address_partial_similarity,
            address_token_sort_similarity,
            address_token_set_similarity,

            exact_name,
            exact_address,
            country_match,

            address_number_overlap,
            combined_similarity
        ])

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    return pd.DataFrame(

        features,

        columns=[

            "source1_entity_id",
            "candidate_entity_id",
            "source",

            "name_similarity",
            "name_partial_similarity",
            "name_token_sort_similarity",
            "name_token_set_similarity",

            "address_similarity",
            "address_partial_similarity",
            "address_token_sort_similarity",
            "address_token_set_similarity",

            "exact_name",
            "exact_address",
            "country_match",

            "address_number_overlap",
            "combined_similarity"
        ]
    )