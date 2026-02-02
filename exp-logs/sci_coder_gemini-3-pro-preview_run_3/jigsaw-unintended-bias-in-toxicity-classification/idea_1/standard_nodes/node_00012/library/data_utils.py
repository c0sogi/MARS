import os
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from library.config import Config


def clean_and_tokenize(text):
    """
    Basic cleaning. Tokenization is handled by the transformer tokenizer.
    """
    if pd.isna(text):
        return ""
    return str(text)


def build_or_load_vocabulary(load_cached_data=True):
    """
    Loads the transformer tokenizer.
    """
    print(f"Loading tokenizer for {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    return tokenizer


def identify_identity_indices(tokenizer):
    """
    Identifies the tokenizer indices corresponding to the identity terms.
    """
    identity_indices = set()
    target_terms = Config.get_identity_term_set()

    for term in target_terms:
        # Get token IDs for the term
        # We use add_special_tokens=False to get just the word's IDs
        ids = tokenizer.encode(term, add_special_tokens=False)
        for i in ids:
            identity_indices.add(i)

    return identity_indices
