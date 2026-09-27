import re
import pandas as pd
from unidecode import unidecode


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "pvt",
    "private",
}


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Transliterate where possible
    text = unidecode(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(text):
    text = normalize_text(text)

    if not text:
        return ""

    words = text.split()

    # Remove legal suffixes only from the end of the name
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()

    return " ".join(words)


COUNTRY_ALIASES = {
    "usa": "us",
    "u s a": "us",
    "united states": "us",
    "united states of america": "us",
    "uk": "gb",
    "united kingdom": "gb",
    "great britain": "gb",
    "fra": "france",
    "fr": "france",
    "ind": "india",
    "in": "india",
    "deu": "germany",
    "de": "germany",
}


def normalize_country(text):
    text = normalize_text(text)
    if not text:
        return ""
    return COUNTRY_ALIASES.get(text, text)


def preprocess_dataframe(df):
    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_business_name)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_text)
    )

    df["country_normalized"] = (
        df["country"].apply(normalize_country)
    )

    return df


def preprocess_all_sources(source1, source2, source3):
    source1 = preprocess_dataframe(source1)
    source2 = preprocess_dataframe(source2)
    source3 = preprocess_dataframe(source3)

    return source1, source2, source3
