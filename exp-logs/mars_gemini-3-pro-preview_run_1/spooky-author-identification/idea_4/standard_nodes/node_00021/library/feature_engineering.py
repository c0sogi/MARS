import os
import numpy as np
import pandas as pd
import spacy
import string
from scipy.stats import entropy
from library.config import Config
from library.utils import seed_everything


class SpacyPreprocessor:
    """
    A class to preprocess text using Spacy to extract Part-of-Speech (POS) tag sequences.
    This captures the syntactic structure of the text, independent of specific vocabulary.
    """

    def __init__(self, model_name=Config.SPACY_MODEL):
        self.model_name = model_name
        self.nlp = None

    def _load_model(self):
        """
        Loads the Spacy model if not already loaded.
        Disables unnecessary components (NER, Parser, Lemmatizer) for speed.
        """
        if self.nlp is None:
            # We only need the tagger and attribute ruler for POS tags
            try:
                self.nlp = spacy.load(
                    self.model_name, disable=["ner", "parser", "lemmatizer"]
                )
            except OSError:
                # Attempt to download if not present (though environment should have it)
                print(
                    f"Spacy model '{self.model_name}' not found. Attempting download..."
                )
                from spacy.cli import download

                download(self.model_name)
                self.nlp = spacy.load(
                    self.model_name, disable=["ner", "parser", "lemmatizer"]
                )

    def transform(self, texts, dataset_name, load_cached_data=True):
        """
        Converts a list of texts into POS tag sequences.
        Implements caching to avoid re-computation.

        Args:
            texts (list or pd.Series): The input text samples.
            dataset_name (str): Unique identifier for the dataset (e.g., 'train', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            list: A list of strings, where each string is a space-separated sequence of POS tags.
        """
        seed_everything()

        # Define cache path
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"pos_sequences_{dataset_name}.parquet")

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached POS sequences for {dataset_name} from {cache_path}..."
            )
            try:
                df = pd.read_parquet(cache_path)
                return df["pos_sequence"].tolist()
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Generating POS sequences for {dataset_name}...")
        self._load_model()

        # Ensure texts is a list of strings
        text_list = [str(t) for t in texts]

        pos_sequences = []
        # Use nlp.pipe for efficient batch processing
        # batch_size is set to a reasonable default; n_process=1 for safety
        for doc in self.nlp.pipe(text_list, batch_size=100, n_process=1):
            # Extract coarse-grained POS tags (e.g., DET, NOUN, VERB)
            tags = [token.pos_ for token in doc]
            pos_sequences.append(" ".join(tags))

        # 3. Save to cache
        try:
            df_out = pd.DataFrame({"pos_sequence": pos_sequences})
            df_out.to_parquet(cache_path, index=False)
            print(f"Saved POS sequences to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return pos_sequences


def extract_meta_features(texts, dataset_name, load_cached_data=True):
    """
    Extracts explicit meta-features from text: length, word count, and punctuation usage.

    Args:
        texts (list or pd.Series): The input text samples.
        dataset_name (str): Unique identifier for the dataset.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: A DataFrame containing the meta-features.
    """
    seed_everything()

    # Define cache path
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"meta_features_{dataset_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached meta-features for {dataset_name} from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing meta-features for {dataset_name}...")

    # Convert to pandas Series for efficient string operations
    s_texts = pd.Series(texts).astype(str).fillna("")

    features = pd.DataFrame()

    # Length features
    features["char_len"] = s_texts.str.len()
    features["word_count"] = s_texts.str.split().str.len()

    # Avoid division by zero
    features["avg_word_len"] = features["char_len"] / features["word_count"].replace(
        0, 1
    )

    # Punctuation counts
    # We look for specific punctuation marks that might indicate style
    punct_chars = [",", ";", ":", "?", "!", "-", '"', "'"]

    for p in punct_chars:
        # Escape special regex characters
        p_esc = "\\" + p if p in "?!." else p
        col_name = f"punct_{'quote' if p in ['"', "'"] else p}"
        # Handle duplicate names if both quotes map to same name, though here they are distinct chars
        if p == '"':
            col_name = "punct_double_quote"
        if p == "'":
            col_name = "punct_single_quote"

        features[col_name] = s_texts.str.count(p_esc)

    # Total punctuation count
    features["total_punct"] = s_texts.apply(
        lambda x: sum(1 for char in x if char in string.punctuation)
    )

    # Punctuation density
    features["punct_density"] = features["total_punct"] / features["char_len"].replace(
        0, 1
    )

    # 3. Save to cache
    try:
        features.to_parquet(cache_path, index=False)
        print(f"Saved meta-features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return features


def calculate_uncertainty_stats(probs):
    """
    Calculates uncertainty statistics from probability distributions.
    These features help the meta-learner decide how much to trust a base model.

    Args:
        probs (np.ndarray): Probability matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: A matrix of shape (n_samples, 3) containing:
            - Shannon Entropy
            - Standard Deviation
            - Max Probability (Confidence)
    """
    # Ensure input is numpy array
    probs = np.array(probs)

    # 1. Shannon Entropy (higher entropy = higher uncertainty)
    # base=None defaults to e, which is standard.
    # We add a small epsilon inside log implicitly handled by scipy or we can rely on it.
    # scipy.stats.entropy handles sum(p)=1 automatically.
    ent = entropy(probs, axis=1)

    # 2. Standard Deviation (spread of probabilities)
    # If one class is 1.0 and others 0.0, std is high (confident).
    # If all are 0.33, std is low (uncertain).
    std = np.std(probs, axis=1)

    # 3. Max Probability (Confidence)
    max_prob = np.max(probs, axis=1)

    # Stack features
    stats = np.column_stack([ent, std, max_prob])

    return stats
