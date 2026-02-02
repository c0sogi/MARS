import pandas as pd
import numpy as np
import os
import logging
from library import config, utils, feature_engineering


class NFLDataLoader:
    """
    Data Loader for the Reference-Anchored Decoupled-Mining Ensemble (RAD-ME).
    Orchestrates metadata loading, feature generation via the library, and
    construction of specific datasets for Scout and Expert training phases.
    """

    def __init__(self):
        self.feature_gen = feature_engineering.ReferenceAnchoredFeatures()
        self.feature_cols = config.FEATURE_COLUMNS
        self.target_col = "contact"

    def load_metadata(self, split="train"):
        """
        Loads the metadata csv for the specified split.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe.
        """
        if split == "train":
            path = config.TRAIN_METADATA_PATH
        elif split == "val":
            path = config.VAL_METADATA_PATH
        elif split == "test":
            path = config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        logging.info(f"Loading {split} metadata from {path}")
        return pd.read_csv(path)

    def prepare_dataset(self, split="train", load_cached_data=True):
        """
        Orchestrates the loading, merging, gating, and feature engineering.
        Uses the ReferenceAnchoredFeatures library to handle the heavy lifting.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached parquet files.

        Returns:
            pd.DataFrame: The processed feature dataframe containing gated survivors.
        """
        # 1. Load Metadata
        df_meta = self.load_metadata(split)

        # 2. Generate Features (includes merging and gating)
        # The library handles caching internally based on split name
        df_features = self.feature_gen.generate_features(
            df_meta, split=split, load_cached_data=load_cached_data
        )

        # Ensure index is unique and reset for later index-based selection
        df_features = df_features.reset_index(drop=True)

        logging.info(f"Dataset prepared for {split}. Shape: {df_features.shape}")
        return df_features

    def get_scout_dataset(self, df):
        """
        Prepares the dataset for Scout training (Phase 1).
        Constructs a balanced dataset from the Relaxed Gated Survivors.

        Args:
            df (pd.DataFrame): The full gated training dataframe.

        Returns:
            tuple: (X, y)
        """
        logging.info("Preparing Scout Dataset (Balanced)...")

        if self.target_col not in df.columns:
            raise ValueError("Target column not found in dataframe")

        # Separate Positives and Negatives
        pos_mask = df[self.target_col] == 1
        neg_mask = df[self.target_col] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg = len(df_neg)

        # Balance: Downsample negatives to match positives (1:1 ratio)
        if n_neg > n_pos:
            df_neg_sampled = df_neg.sample(n=n_pos, random_state=config.SEED)
        else:
            df_neg_sampled = df_neg

        df_balanced = pd.concat([df_pos, df_neg_sampled], axis=0)
        df_balanced = df_balanced.sample(frac=1, random_state=config.SEED).reset_index(
            drop=True
        )

        X = df_balanced[self.feature_cols]
        y = df_balanced[self.target_col]

        logging.info(
            f"Scout Dataset: {len(df_balanced)} samples (Pos: {len(df_pos)}, Neg: {len(df_neg_sampled)})"
        )
        return X, y

    def get_expert_dataset(self, df, hard_negative_indices):
        """
        Prepares the dataset for Expert training (Phase 3).
        Consists of:
        1. All Positives
        2. Mined Hard Negatives (identified by Scouts)
        3. Random Easy Negatives (Anchors) at 1:1 ratio with Positives

        Args:
            df (pd.DataFrame): The full gated training dataframe.
            hard_negative_indices (array-like): Indices of hard negatives in df.

        Returns:
            tuple: (X, y)
        """
        logging.info("Preparing Expert Dataset (Anchored Mining)...")

        if self.target_col not in df.columns:
            raise ValueError("Target column not found in dataframe")

        # 1. All Positives
        pos_mask = df[self.target_col] == 1
        df_pos = df[pos_mask]
        n_pos = len(df_pos)

        # 2. Hard Negatives
        # Ensure indices are valid and present in the dataframe
        valid_hard_indices = [idx for idx in hard_negative_indices if idx in df.index]
        df_hard = df.loc[valid_hard_indices]

        # Strict check: Ensure they are actually negatives (exclude any mislabeled positives)
        df_hard = df_hard[df_hard[self.target_col] == 0]

        # 3. Random Anchors
        # We need negatives that are NOT in the hard negative set
        neg_mask = df[self.target_col] == 0
        is_hard = df.index.isin(df_hard.index)
        easy_neg_mask = neg_mask & (~is_hard)

        df_easy = df[easy_neg_mask]

        # Sample Anchors (1:1 Ratio with Positives defined in config)
        n_anchors = int(n_pos * config.ANCHOR_RATIO)
        if len(df_easy) > n_anchors:
            df_anchors = df_easy.sample(n=n_anchors, random_state=config.SEED)
        else:
            df_anchors = df_easy

        # Combine components
        df_expert = pd.concat([df_pos, df_hard, df_anchors], axis=0)
        df_expert = df_expert.sample(frac=1, random_state=config.SEED).reset_index(
            drop=True
        )

        X = df_expert[self.feature_cols]
        y = df_expert[self.target_col]

        logging.info(f"Expert Dataset: {len(df_expert)} samples")
        logging.info(f"  Positives: {len(df_pos)}")
        logging.info(f"  Hard Negatives: {len(df_hard)}")
        logging.info(f"  Anchors: {len(df_anchors)}")

        return X, y

    def merge_tracking_data(self, df_meta, df_tracking):
        """
        Merges tracking data onto metadata.
        Note: This functionality is encapsulated within ReferenceAnchoredFeatures.generate_features.
        This method is kept for interface compatibility if needed.
        """
        pass

    def apply_quadratic_gating(self, df):
        """
        Applies relaxed quadratic gating to filter candidates.
        Note: This functionality is encapsulated within ReferenceAnchoredFeatures.generate_features.
        This method is kept for interface compatibility if needed.
        """
        pass
