import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


class MetaFeatureEngineer:
    """
    Engineers structural meta-features from the text data.
    Features include character counts, word counts, and length ratios.
    Applies log-normalization and Z-score standardization.
    """

    def __init__(self):
        self.text_cols = ["question_title", "question_body", "answer"]

    def _extract_raw_features(self, df):
        """
        Computes raw numerical features from the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing text columns.

        Returns:
            np.ndarray: Matrix of shape (n_samples, n_features).
        """
        # Ensure text columns are strings and handle NaNs
        for col in self.text_cols:
            if col not in df.columns:
                # Fallback for missing columns (though unlikely given schema)
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        features = pd.DataFrame()

        # 1. Character Lengths (Log-normalized)
        # We use log1p to handle the skewed distribution of text lengths
        features["q_title_len_char"] = np.log1p(df["question_title"].str.len())
        features["q_body_len_char"] = np.log1p(df["question_body"].str.len())
        features["answer_len_char"] = np.log1p(df["answer"].str.len())

        # 2. Word Counts (Log-normalized)
        # Simple whitespace splitting is sufficient for meta-features
        features["q_title_len_word"] = np.log1p(
            df["question_title"].apply(lambda x: len(x.split()))
        )
        features["q_body_len_word"] = np.log1p(
            df["question_body"].apply(lambda x: len(x.split()))
        )
        features["answer_len_word"] = np.log1p(
            df["answer"].apply(lambda x: len(x.split()))
        )

        # 3. Ratios (Answer vs Question Body)
        # We use the raw lengths (re-calculated or exponentiated) for ratios to capture physical proportion
        # Adding epsilon to denominator to prevent division by zero
        q_body_len_raw = df["question_body"].str.len()
        answer_len_raw = df["answer"].str.len()
        features["len_ratio_char"] = answer_len_raw / (q_body_len_raw + 1.0)

        q_body_word_raw = df["question_body"].apply(lambda x: len(x.split()))
        answer_word_raw = df["answer"].apply(lambda x: len(x.split()))
        features["len_ratio_word"] = answer_word_raw / (q_body_word_raw + 1.0)

        # 4. Explicit Interaction Features (Difference in lengths)
        features["diff_len_char"] = (
            features["answer_len_char"] - features["q_body_len_char"]
        )
        features["diff_len_word"] = (
            features["answer_len_word"] - features["q_body_len_word"]
        )

        return features.values.astype(np.float32)

    def process_splits(self, load_cached_data=True):
        """
        Generates or loads meta-features for Train, Validation, and Test sets.
        Standardizes features based on Training statistics.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (train_features, val_features, test_features) as numpy arrays.
        """
        # Paths for caching
        train_cache = Config.META_TRAIN_FEATS_PATH
        val_cache = Config.META_VAL_FEATS_PATH
        test_cache = Config.META_TEST_FEATS_PATH

        # Check if cache exists
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
            ):
                print("Loading cached meta-features...")
                train_feats = np.load(train_cache)
                val_feats = np.load(val_cache)
                test_feats = np.load(test_cache)
                return train_feats, val_feats, test_feats
            else:
                print("Cache not found. Computing meta-features from scratch...")

        # Load Data
        print("Loading metadata CSVs...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Extract Raw Features
        print("Extracting raw meta-features...")
        X_train_raw = self._extract_raw_features(train_df)
        X_val_raw = self._extract_raw_features(val_df)
        X_test_raw = self._extract_raw_features(test_df)

        # Standardize (Z-score)
        # Fit only on Train to prevent leakage
        print("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Save to cache
        print(f"Saving meta-features to {Config.WORKING_DIR}...")
        np.save(train_cache, X_train_scaled)
        np.save(val_cache, X_val_scaled)
        np.save(test_cache, X_test_scaled)

        return X_train_scaled, X_val_scaled, X_test_scaled
