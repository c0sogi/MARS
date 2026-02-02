import os
import re
import numpy as np
import pandas as pd
import nltk
from library.config import Config


class StructuralFeatureExtractor:
    """
    Extracts hand-crafted linguistic features including readability scores,
    sentence structure statistics, and lexical diversity metrics.
    """

    def __init__(self):
        # Regex for tokenization
        self.sentence_end_pattern = re.compile(r"[.!?]+")
        self.word_pattern = re.compile(
            r"\b[a-zA-Z]+\b"
        )  # Only alpha words for readability
        self.vowel_pattern = re.compile(r"[aeiouy]+", re.IGNORECASE)

        # Try to load NLTK words for spelling check
        self.english_vocab = set()
        self.has_vocab = False
        try:
            # Check if 'words' corpus is available
            nltk.data.find("corpora/words")
            from nltk.corpus import words

            self.english_vocab = set(words.words())
            self.has_vocab = True
        except (LookupError, ImportError):
            # Fallback: Dictionary features will be 0
            pass

    def count_syllables(self, word):
        """
        Heuristic syllable counter.
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Count vowel groups
        count = len(self.vowel_pattern.findall(word))

        # Subtract silent 'e' at the end
        if word.endswith("e") and not word.endswith("le"):
            count -= 1

        # Handle cases where we reduced to 0
        if count < 1:
            count = 1

        return count

    def process_text(self, text):
        """
        Computes a dictionary of features for a single text.
        """
        if pd.isna(text) or text == "":
            return self._get_empty_features()

        # Tokenization
        sentences = [
            s.strip() for s in self.sentence_end_pattern.split(text) if s.strip()
        ]
        words = self.word_pattern.findall(text)

        n_sentences = len(sentences)
        n_words = len(words)

        # Safety for division by zero
        if n_sentences == 0:
            n_sentences = 1
        if n_words == 0:
            n_words = 1

        # --- Basic Length Features ---
        avg_chars_per_word = np.mean([len(w) for w in words]) if words else 0

        # Sentence lengths (in words)
        sent_lengths = [len(self.word_pattern.findall(s)) for s in sentences]
        avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
        std_sent_len = np.std(sent_lengths) if sent_lengths else 0

        # --- Syllable & Complexity Features ---
        syllable_counts = [self.count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)

        # Complex words (3+ syllables)
        complex_words = sum(1 for c in syllable_counts if c >= 3)
        pct_complex_words = complex_words / n_words

        # --- Readability Scores ---
        # 1. Flesch Reading Ease
        # 206.835 - 1.015(total_words/total_sentences) - 84.6(total_syllables/total_words)
        flesch_re = (
            206.835
            - 1.015 * (n_words / n_sentences)
            - 84.6 * (total_syllables / n_words)
        )

        # 2. Flesch-Kincaid Grade Level
        # 0.39(total_words/total_sentences) + 11.8(total_syllables/total_words) - 15.59
        flesch_kincaid = (
            0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
        )

        # 3. Gunning Fog Index
        # 0.4 * ( (total_words/total_sentences) + 100 * (complex_words/total_words) )
        gunning_fog = 0.4 * ((n_words / n_sentences) + 100 * pct_complex_words)

        # --- Lexical Features ---
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / n_words  # Type-Token Ratio

        # Dictionary Errors (if vocab available)
        dict_errors = 0
        if self.has_vocab:
            # Simple check: word lower not in vocab
            # We filter very short words to avoid noise
            dict_errors = sum(
                1 for w in words if len(w) > 1 and w.lower() not in self.english_vocab
            )

        return {
            "word_count": n_words,
            "sentence_count": n_sentences,
            "avg_word_length": avg_chars_per_word,
            "avg_sentence_length": avg_sent_len,
            "std_sentence_length": std_sent_len,
            "total_syllables": total_syllables,
            "complex_word_count": complex_words,
            "flesch_reading_ease": flesch_re,
            "flesch_kincaid_grade": flesch_kincaid,
            "gunning_fog": gunning_fog,
            "type_token_ratio": ttr,
            "dictionary_errors": dict_errors,
        }

    def _get_empty_features(self):
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "std_sentence_length": 0,
            "total_syllables": 0,
            "complex_word_count": 0,
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "gunning_fog": 0,
            "type_token_ratio": 0,
            "dictionary_errors": 0,
        }


def extract_linguistic_features(
    df: pd.DataFrame, split: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main function to generate or load structural features for a given dataframe.

    Args:
        df: Input dataframe containing a 'full_text' column.
        split: Name of the data split (e.g., 'train', 'val', 'test') for caching.
        load_cached_data: If True, attempts to load from disk before computing.

    Returns:
        pd.DataFrame: A dataframe containing the computed numerical features.
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    cache_path = os.path.join(Config.cache_dir, f"structural_features_{split}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached structural features from {cache_path}")
        try:
            features_df = pd.read_parquet(cache_path)
            # Verify length matches
            if len(features_df) == len(df):
                return features_df
            else:
                print(
                    f"Cache length mismatch ({len(features_df)} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute features
    print(f"Computing structural features for {split} set...")
    extractor = StructuralFeatureExtractor()

    # Apply processing row by row
    # Using list comprehension is often faster than df.apply for text processing
    texts = df["full_text"].astype(str).tolist()
    features_list = [extractor.process_text(text) for text in texts]

    features_df = pd.DataFrame(features_list)

    # Optimize dtypes
    for col in features_df.columns:
        features_df[col] = features_df[col].astype(np.float32)

    # 3. Save to cache
    print(f"Saving structural features to {cache_path}")
    features_df.to_parquet(cache_path, index=False)

    return features_df
