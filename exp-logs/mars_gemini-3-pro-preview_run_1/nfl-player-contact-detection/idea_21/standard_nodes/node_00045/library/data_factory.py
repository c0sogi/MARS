import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    FEATURE_COLS,
    SEED,
)
from library.utils import get_logger, seed_everything
from library.feature_engineering import FeatureEngineer


class DataFactory:
    """
    Manages data loading, feature generation, and dataset construction for the
    Tri-Scout Diversity Mining pipeline.
    """

    def __init__(self):
        self.logger = get_logger("data_factory")
        self.fe = FeatureEngineer()
        seed_everything(SEED)

    def load_features(self, mode="train", load_cached_data=True):
        """
        Loads metadata and tracking data, then delegates to FeatureEngineer to
        generate or load cached features.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from parquet cache.

        Returns:
            pd.DataFrame: The processed feature dataframe with gating applied.
        """
        self.logger.info(f"Preparing features for mode: {mode}")

        if mode == "train":
            meta_path = TRAIN_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH
        elif mode == "val":
            meta_path = VAL_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH  # Val is a subset of Train data source
        elif mode == "test":
            meta_path = TEST_METADATA_PATH
            track_path = TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load Metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Generate Features (handles caching internally via FeatureEngineer)
        df_features = self.fe.generate_features(
            metadata_df=df_meta,
            tracking_path=track_path,
            mode=mode,
            load_cached_data=load_cached_data,
        )

        return df_features

    def get_scout_dataset(self, df_features, neg_ratio=1.0):
        """
        Constructs a balanced dataset for Scout training.

        Strategy:
        - Keep all Positives (Contact=1).
        - Sample Negatives (Contact=0) to match the count of positives * neg_ratio.

        Args:
            df_features (pd.DataFrame): The full gated training features.
            neg_ratio (float): Ratio of negatives to positives.

        Returns:
            (pd.DataFrame, pd.Series): X (features), y (labels)
        """
        self.logger.info("Constructing Scout Dataset (Balanced)...")

        pos_mask = df_features["contact"] == 1
        neg_mask = df_features["contact"] == 0

        df_pos = df_features[pos_mask]
        df_neg = df_features[neg_mask]

        n_pos = len(df_pos)
        n_neg = int(n_pos * neg_ratio)

        # Sample negatives
        if len(df_neg) > n_neg:
            df_neg_sampled = df_neg.sample(n=n_neg, random_state=SEED)
        else:
            df_neg_sampled = df_neg

        df_combined = (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

        X = df_combined[FEATURE_COLS]
        y = df_combined["contact"]

        self.logger.info(
            f"Scout Dataset: {len(X)} samples (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
        )
        return X, y

    def get_expert_dataset(self, df_features, hard_negative_indices, buffer_ratio=0.1):
        """
        Constructs the Expert Dataset using Hard Negative Mining.

        Strategy:
        - Keep all Positives.
        - Include all Hard Negatives (indices provided).
        - Include a random buffer of easy negatives (buffer_ratio * n_pos).

        Args:
            df_features (pd.DataFrame): The full gated training features.
            hard_negative_indices (array-like): Indices of hard negatives in df_features.
            buffer_ratio (float): Ratio of random negatives to add relative to positives.

        Returns:
            (pd.DataFrame, pd.Series): X (features), y (labels)
        """
        self.logger.info("Constructing Expert Dataset (Hard Negative Mining)...")

        # 1. Positives
        df_pos = df_features[df_features["contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid and correspond to negatives (sanity check)
        hard_neg_indices = np.array(hard_negative_indices)
        # Filter to ensure we are only taking from the dataframe provided
        valid_hard_indices = hard_neg_indices[hard_neg_indices < len(df_features)]
        df_hard_neg = df_features.iloc[valid_hard_indices]
        # Double check they are actually negatives (in case mining logic was loose)
        df_hard_neg = df_hard_neg[df_hard_neg["contact"] == 0]

        # 3. Random Buffer
        # Exclude hard negatives from the pool of remaining negatives
        hard_neg_idx_set = set(df_hard_neg.index)
        neg_mask = (df_features["contact"] == 0) & (
            ~df_features.index.isin(hard_neg_idx_set)
        )
        df_easy_neg_pool = df_features[neg_mask]

        n_buffer = int(len(df_pos) * buffer_ratio)
        if len(df_easy_neg_pool) > n_buffer:
            df_buffer = df_easy_neg_pool.sample(n=n_buffer, random_state=SEED)
        else:
            df_buffer = df_easy_neg_pool

        # Combine
        df_combined = pd.concat([df_pos, df_hard_neg, df_buffer], axis=0)
        df_combined = df_combined.sample(frac=1.0, random_state=SEED).reset_index(
            drop=True
        )

        X = df_combined[FEATURE_COLS]
        y = df_combined["contact"]

        self.logger.info(f"Expert Dataset: {len(X)} samples")
        self.logger.info(f"   Positives: {len(df_pos)}")
        self.logger.info(f"   Hard Negs: {len(df_hard_neg)}")
        self.logger.info(f"   Buffer:    {len(df_buffer)}")

        return X, y

    def get_validation_data(self, df_val):
        """
        Returns X, y for the validation set.
        """
        X = df_val[FEATURE_COLS]
        y = df_val["contact"]
        return X, y

    def get_test_data(self, df_test):
        """
        Returns X for the test set.
        """
        X = df_test[FEATURE_COLS]
        return X
