import os
import re
import numpy as np
import pandas as pd
import spacy
import nltk
from library.config import Config


class MechanicsFeatureExtractor:
    """
    Implements the Mechanics Branch of the essay scoring architecture.
    Extracts explicit linguistic features including readability scores,
    length metrics, variance metrics, and error proxies.
    """

    def __init__(self):
        """
        Initialize NLP resources.
        Uses a blank Spacy model with a sentencizer for speed and robustness.
        Loads NLTK words for spelling error estimation.
        """
        # Initialize Spacy: Blank English model + Sentencizer is faster than en_core_web_sm
        # and sufficient for tokenization/sentence splitting.
        try:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except Exception as e:
            print(f"Warning: Failed to load Spacy model. Error: {e}")
            # Fallback to basic initialization if something goes wrong, though unlikely
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

        # Initialize NLTK Words for spelling check
        self._ensure_nltk_resources()
        try:
            from nltk.corpus import words

            self.valid_words = set(w.lower() for w in words.words())
        except Exception:
            # Fallback if corpus loading fails
            print(
                "Warning: NLTK words corpus not available. Spelling features will be 0."
            )
            self.valid_words = set()

    def _ensure_nltk_resources(self):
        """Ensure necessary NLTK data is downloaded."""
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            try:
                nltk.download("words", quiet=True)
            except Exception as e:
                print(f"Warning: Could not download nltk 'words': {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Basic text cleaning: normalize whitespace.
        """
        if pd.isna(text) or text == "":
            return ""
        # Replace various whitespace characters with a single space and trim
        return " ".join(str(text).split())

    def _count_syllables(self, word: str) -> int:
        """
        Heuristic syllable counter using regex.
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Count vowel groups
        count = len(re.findall(r"[aeiouy]+", word))

        # Adjust for silent 'e' at the end (unless it's 'le' like in 'apple')
        if word.endswith("e") and not word.endswith("le") and not word.endswith("les"):
            count -= 1

        # Heuristic: usually at least 1 syllable
        return max(1, count)

    def _compute_row_features(self, doc) -> dict:
        """
        Compute all mechanics features for a single Spacy Doc object.
        """
        # Basic extraction
        tokens = [t.text for t in doc]
        words = [t.text for t in doc if not t.is_punct and not t.is_space]
        sentences = list(doc.sents)

        # Counts
        char_count = len(doc.text)
        word_count = len(words)
        sentence_count = len(sentences)

        # Avoid division by zero
        safe_word_count = max(1, word_count)
        safe_sent_count = max(1, sentence_count)

        # 1. Length Metrics
        avg_sentence_length = word_count / safe_sent_count

        # 2. Variance Metrics
        # Calculate sentence lengths in words
        sent_lengths = [
            len([t for t in sent if not t.is_punct and not t.is_space])
            for sent in sentences
        ]
        if len(sent_lengths) > 1:
            sentence_length_var = np.var(sent_lengths)
        else:
            sentence_length_var = 0.0

        # 3. Vocabulary Richness (Type-Token Ratio)
        # Lowercase for fair comparison
        unique_words = set(w.lower() for w in words)
        vocab_richness = len(unique_words) / safe_word_count

        # 4. Readability Scores
        # Syllable calculations
        syllables_per_word = [self._count_syllables(w) for w in words]
        total_syllables = sum(syllables_per_word)

        # Complex words (>= 3 syllables)
        complex_word_count = sum(1 for s in syllables_per_word if s >= 3)

        # Flesch-Kincaid Grade Level
        # Formula: 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        fk_grade = (
            (0.39 * (word_count / safe_sent_count))
            + (11.8 * (total_syllables / safe_word_count))
            - 15.59
        )

        # Gunning Fog Index
        # Formula: 0.4 * ( (total_words / total_sentences) + 100 * (complex_words / total_words) )
        gunning_fog = 0.4 * (
            (word_count / safe_sent_count)
            + 100 * (complex_word_count / safe_word_count)
        )

        # 5. Error Proxies (Spelling)
        # Count alpha words not in dictionary
        if self.valid_words:
            spelling_errors = sum(
                1 for w in words if w.isalpha() and w.lower() not in self.valid_words
            )
        else:
            spelling_errors = 0

        return {
            "char_count": char_count,
            "word_count": word_count,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_var": sentence_length_var,
            "vocab_richness": vocab_richness,
            "flesch_kincaid_grade": fk_grade,
            "gunning_fog": gunning_fog,
            "spelling_error_count": spelling_errors,
        }

    def extract_features(
        self, df: pd.DataFrame, partition_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main method to extract features for a dataset.
        Implements caching logic using Parquet.

        Args:
            df (pd.DataFrame): Input dataframe containing 'full_text'.
            partition_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Dataframe containing only the computed features.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        cache_path = os.path.join(
            Config.CACHE_DIR, f"mechanics_features_{partition_name}.parquet"
        )

        # 1. Try to load cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached mechanics features from {cache_path}...")
            try:
                features_df = pd.read_parquet(cache_path)
                # Verify columns match config
                if all(col in features_df.columns for col in Config.MECHANICS_FEATURES):
                    return features_df[Config.MECHANICS_FEATURES]
                else:
                    print("Cached file schema mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Computing mechanics features for {partition_name}...")

        # Preprocess text
        texts = df["full_text"].apply(self.preprocess_text).tolist()

        # Use nlp.pipe for efficient batch processing
        # Disable unnecessary components if any (though we only added sentencizer)
        docs = self.nlp.pipe(texts, batch_size=64, n_process=1)

        feature_rows = []
        for doc in docs:
            feature_rows.append(self._compute_row_features(doc))

        features_df = pd.DataFrame(feature_rows)

        # Ensure column order and selection matches Config
        # Fill NaNs if any calculation resulted in NaN (though logic handles div/0)
        features_df = features_df.fillna(0.0)

        # Select only required columns
        final_df = features_df[Config.MECHANICS_FEATURES]

        # 3. Save to cache
        try:
            final_df.to_parquet(cache_path, index=False)
            print(f"Saved mechanics features to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return final_df
