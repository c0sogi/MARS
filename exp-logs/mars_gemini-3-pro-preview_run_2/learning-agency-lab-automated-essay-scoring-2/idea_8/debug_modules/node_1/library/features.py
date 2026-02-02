import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from library.config import CFG


class FeatureEngineer:
    """
    Extracts structural and linguistic features from essay text.
    Includes caching mechanisms and vocabulary management for OOV detection.
    """

    def __init__(self):
        self.output_dir = CFG.output_dir
        self.vocab_path = os.path.join(self.output_dir, "vocab.npy")
        self.vocab = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _count_syllables(self, word):
        """
        Heuristic syllable counter using regex.
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Remove silent 'e' at the end
        if word.endswith("e"):
            word = word[:-1]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Handle cases with no vowels found
        if count == 0:
            count = 1

        return count

    def _build_vocab(self, texts, min_freq=5):
        """
        Builds a vocabulary from the provided texts.
        Words must appear at least `min_freq` times to be included.
        """
        print("Building vocabulary from training data...")
        all_tokens = []
        for text in texts:
            tokens = re.findall(r"\b[a-z]{2,}\b", str(text).lower())
            all_tokens.extend(tokens)

        counts = Counter(all_tokens)
        # Keep words that appear at least min_freq times
        vocab_set = {word for word, count in counts.items() if count >= min_freq}

        self.vocab = vocab_set

        # Save to disk
        np.save(self.vocab_path, np.array(list(vocab_set)))
        print(f"Vocabulary built and saved. Size: {len(self.vocab)}")

    def _load_vocab(self):
        """
        Loads the vocabulary from disk.
        """
        if os.path.exists(self.vocab_path):
            vocab_list = np.load(self.vocab_path, allow_pickle=True)
            self.vocab = set(vocab_list)
            print(f"Vocabulary loaded. Size: {len(self.vocab)}")
        else:
            print("Warning: Vocabulary file not found. OOV features will be 0.")
            self.vocab = set()

    def _extract_text_features(self, text):
        """
        Computes features for a single text string.
        """
        text = str(text)
        if not text:
            return {
                "word_count": 0,
                "sent_count": 0,
                "char_count": 0,
                "avg_word_len": 0,
                "avg_sent_len": 0,
                "unique_ratio": 0,
                "flesch_kincaid": 0,
                "gunning_fog": 0,
                "oov_count": 0,
                "oov_ratio": 0,
            }

        # 1. Basic Tokenization
        # Simple word tokenization
        words = re.findall(r"\b\w+\b", text.lower())
        word_count = len(words)

        # Sentence segmentation (split by . ! ?)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = len(sentences)
        if sent_count == 0:
            sent_count = 1  # Avoid division by zero

        char_count = len(text)

        # 2. Length Metrics
        avg_word_len = char_count / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count

        # 3. Vocabulary Richness
        unique_words = set(words)
        unique_ratio = len(unique_words) / word_count if word_count > 0 else 0

        # 4. Complexity & Readability
        syllable_counts = [self._count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        complex_words = sum(1 for c in syllable_counts if c >= 3)

        # Flesch-Kincaid Grade Level
        # 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        fk_grade = 0
        if word_count > 0:
            fk_grade = (
                0.39 * (word_count / sent_count)
                + 11.8 * (total_syllables / word_count)
                - 15.59
            )

        # Gunning Fog Index
        # 0.4 * ( (total_words / total_sentences) + 100 * (complex_words / total_words) )
        gunning_fog = 0
        if word_count > 0:
            gunning_fog = 0.4 * (
                (word_count / sent_count) + 100 * (complex_words / word_count)
            )

        # 5. OOV / Error Proxy
        oov_count = 0
        if self.vocab is not None:
            # Check words that are purely alpha and length > 1
            check_words = [w for w in words if w.isalpha() and len(w) > 1]
            oov_count = sum(1 for w in check_words if w not in self.vocab)

        oov_ratio = oov_count / word_count if word_count > 0 else 0

        return {
            "word_count": word_count,
            "sent_count": sent_count,
            "char_count": char_count,
            "avg_word_len": avg_word_len,
            "avg_sent_len": avg_sent_len,
            "unique_ratio": unique_ratio,
            "flesch_kincaid": fk_grade,
            "gunning_fog": gunning_fog,
            "oov_count": oov_count,
            "oov_ratio": oov_ratio,
        }

    def extract_features(self, df, split_name="train", load_cached_data=True):
        """
        Main method to extract features for a dataframe.
        Handles caching and vocabulary management.

        Args:
            df (pd.DataFrame): Dataframe containing 'full_text'.
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Feature dataframe aligned with input df index.
        """
        cache_file = os.path.join(self.output_dir, f"features_{split_name}.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached features for {split_name} from {cache_file}")
            features_df = pd.read_parquet(cache_file)

            # Ensure index alignment just in case
            if len(features_df) == len(df):
                # If we are in val/test, we still need to ensure vocab is loaded for consistency
                # if we were to process more data later, but strictly for returning cached data, it's fine.
                # However, if we are in 'train' and loading cache, we assume vocab was built when cache was created.
                return features_df
            else:
                print(
                    f"Cached file length ({len(features_df)}) mismatch with df ({len(df)}). Recomputing."
                )

        # 2. Vocabulary Management
        if split_name == "train":
            # Always rebuild vocab on train split if not loading cache (or if cache invalid)
            self._build_vocab(df["full_text"].tolist())
        else:
            # For val/test, load the vocab built during training
            if self.vocab is None:
                self._load_vocab()

        # 3. Feature Extraction
        print(f"Extracting structural features for {split_name}...")

        # Use apply for simplicity; for 15k rows this is fast enough (~seconds)
        # compared to complex vectorization or multiprocessing overhead
        features_list = df["full_text"].apply(self._extract_text_features).tolist()
        features_df = pd.DataFrame(features_list, index=df.index)

        # 4. Save to Cache
        print(f"Saving features to {cache_file}")
        features_df.to_parquet(cache_file)

        return features_df
