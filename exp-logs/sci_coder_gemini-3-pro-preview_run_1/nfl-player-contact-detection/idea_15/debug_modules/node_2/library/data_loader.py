import os
import pandas as pd
import numpy as np
from library import config, features, utils


class TrackingProcessor:
    """
    Handles loading and standardization of raw player tracking data.
    """

    @staticmethod
    def load_tracking(split: str) -> pd.DataFrame:
        """
        Loads the tracking data for the specified split.

        Args:
            split: 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: Raw tracking data.
        """
        if split in ["train", "val"]:
            path = config.TRAIN_TRACKING_PATH
        elif split == "test":
            path = config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found: {path}")

        return pd.read_csv(path)


class GeometricGate:
    """
    Implements the geometric gating logic to filter candidate pairs.
    """

    @staticmethod
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the distance-based gating threshold while preserving Ground contacts.
        Delegates to library.features logic to ensure consistency.

        Args:
            df: DataFrame containing 'distance' and 'nfl_player_id_2' columns.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return features.apply_geometric_gating(df)


class DatasetBuilder:
    """
    Manages the construction of datasets for the Scout and Expert training stages.
    Handles feature loading, caching, and sampling strategies.
    """

    def __init__(self):
        pass

    def load_data(self, split: str, load_cached: bool = True) -> pd.DataFrame:
        """
        Loads the feature-engineered dataset for a given split.
        Relies on library.features for generation and caching.

        Args:
            split: 'train', 'val', or 'test'.
            load_cached: Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed features.
        """
        # features.generate_features handles the caching logic internally
        # as per the requirement (check cache -> compute -> save).
        return features.generate_features(split=split, load_cached=load_cached)

    def build_scout_dataset(
        self, df: pd.DataFrame, negative_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        Constructs a balanced dataset for the Scout model (Phase 1).

        Strategy:
        - Keep all Positives (Contact = 1)
        - Sample Negatives (Contact = 0) to match the count of Positives * negative_ratio

        Args:
            df: The source dataframe (likely the gated training set).
            negative_ratio: Ratio of negatives to positives.

        Returns:
            pd.DataFrame: Balanced dataset.
        """
        # Separate classes
        pos_mask = df["contact"] == 1
        neg_mask = df["contact"] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        n_pos = len(df_pos)
        n_neg_target = int(n_pos * negative_ratio)

        # Sample negatives
        if n_neg_target > len(df_neg):
            n_neg_target = len(df_neg)

        df_neg_sampled = df_neg.sample(n=n_neg_target, random_state=config.RANDOM_STATE)

        # Combine and shuffle
        df_scout = pd.concat([df_pos, df_neg_sampled], axis=0)
        df_scout = df_scout.sample(
            frac=1, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

        return df_scout

    def build_expert_dataset(
        self,
        df: pd.DataFrame,
        hard_negative_indices: np.ndarray,
        random_negative_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Constructs the high-fidelity dataset for the Expert model (Phase 3).

        Strategy:
        - Keep all Positives.
        - Keep all identified Hard Negatives.
        - Add a buffer of Random Negatives (not in Hard Negatives).

        Args:
            df: The source dataframe (gated training set).
            hard_negative_indices: Array of indices in 'df' identified as hard negatives.
            random_negative_ratio: Ratio of random negatives to positives to add as buffer.

        Returns:
            pd.DataFrame: The expert dataset.
        """
        # Ensure indices are valid for this dataframe
        # We assume df has a RangeIndex or consistent index from loading
        valid_hard_indices = np.intersect1d(df.index.values, hard_negative_indices)

        # 1. Positives
        df_pos = df[df["contact"] == 1]

        # 2. Hard Negatives
        df_hard = df.loc[valid_hard_indices]

        # 3. Random Negatives (Buffer)
        # Filter: Contact is 0 AND Index is NOT in hard_negative_indices
        # Note: df_hard might contain some positives if the mining logic was loose,
        # but strictly hard negatives are ground truth 0.
        # Let's enforce ground truth 0 for the random pool.

        # Create a mask for hard negatives to exclude them from random pool
        is_hard = df.index.isin(valid_hard_indices)

        neg_pool_mask = (df["contact"] == 0) & (~is_hard)
        df_neg_pool = df[neg_pool_mask]

        n_pos = len(df_pos)
        n_buffer = int(n_pos * random_negative_ratio)

        if n_buffer > len(df_neg_pool):
            n_buffer = len(df_neg_pool)

        df_buffer = df_neg_pool.sample(n=n_buffer, random_state=config.RANDOM_STATE)

        # Combine all
        df_expert = pd.concat([df_pos, df_hard, df_buffer], axis=0)
        df_expert = df_expert.sample(
            frac=1, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

        return df_expert
