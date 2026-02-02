import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    ANCHOR_RATIO,
    SEED,
)
from library.features import (
    generate_train_features,
    generate_val_features,
    generate_test_features,
)


class DataLoader:
    """
    Handles data ingestion and dataset construction for the KAM-AE strategy.
    Orchestrates calls to the feature engineering library and implements
    specific sampling strategies for Dual-Scout and Expert training phases.
    """

    @staticmethod
    def load_train_data(load_cached_data=True):
        """
        Loads the full training dataset (Gated Survivors) with kinematically-aligned features.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: The processed training data.
        """
        print(f"[{DataLoader.__name__}] Requesting Training Data...")
        df = generate_train_features(
            metadata_path=TRAIN_METADATA_PATH,
            tracking_path=TRAIN_TRACKING_PATH,
            load_cached_data=load_cached_data,
        )
        return df

    @staticmethod
    def load_val_data(load_cached_data=True):
        """
        Loads the validation dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: The processed validation data.
        """
        print(f"[{DataLoader.__name__}] Requesting Validation Data...")
        df = generate_val_features(
            metadata_path=VAL_METADATA_PATH,
            tracking_path=TRAIN_TRACKING_PATH,
            load_cached_data=load_cached_data,
        )
        return df

    @staticmethod
    def load_test_data(load_cached_data=True):
        """
        Loads the test dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: The processed test data.
        """
        print(f"[{DataLoader.__name__}] Requesting Test Data...")
        df = generate_test_features(
            metadata_path=TEST_METADATA_PATH,
            tracking_path=TEST_TRACKING_PATH,
            load_cached_data=load_cached_data,
        )
        return df

    @staticmethod
    def load_raw_test_metadata():
        """
        Loads the raw test metadata CSV.
        Used to ensure the final submission contains all required contact_ids,
        including those filtered out by gating.
        """
        print(
            f"[{DataLoader.__name__}] Loading raw test metadata from {TEST_METADATA_PATH}..."
        )
        return pd.read_csv(TEST_METADATA_PATH)

    @staticmethod
    def sample_balanced_scout_data(df, random_state=SEED):
        """
        Creates a strictly balanced dataset for 'Scout' model training.
        Downsamples the majority class (contact=0) to match the minority class (contact=1).

        Args:
            df (pd.DataFrame): The full gated training dataframe.
            random_state (int): Seed for reproducibility.

        Returns:
            pd.DataFrame: A balanced subset of the input dataframe.
        """
        positives = df[df["contact"] == 1]
        negatives = df[df["contact"] == 0]

        n_pos = len(positives)
        # Downsample negatives
        if len(negatives) > n_pos:
            negatives = negatives.sample(n=n_pos, random_state=random_state)

        # Combine and shuffle
        balanced_df = (
            pd.concat([positives, negatives])
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )

        print(
            f"[{DataLoader.__name__}] Created Balanced Scout Dataset: {len(balanced_df)} rows (Pos: {len(positives)}, Neg: {len(negatives)})"
        )
        return balanced_df

    @staticmethod
    def prepare_expert_dataset(
        df, hard_negative_indices, anchor_ratio=ANCHOR_RATIO, random_state=SEED
    ):
        """
        Constructs the Expert Dataset for the final training phase.
        Composition:
        1. All Positives.
        2. Mined Hard Negatives (indices provided).
        3. Random Easy Negatives (Anchors) at a fixed ratio to (Positives + Hard Negatives).

        Args:
            df (pd.DataFrame): The full gated training dataframe.
            hard_negative_indices (array-like): Indices in 'df' identified as hard negatives.
            anchor_ratio (float): Ratio of Anchors to (Positives + Hard Negatives).
            random_state (int): Seed for reproducibility.

        Returns:
            pd.DataFrame: The constructed Expert dataset.
        """
        # 1. All Positives
        positives = df[df["contact"] == 1]

        # 2. Hard Negatives
        # Filter df by provided indices and ensure they are actually negatives (sanity check)
        # We use df.index.isin because hard_negative_indices are expected to be index labels
        hard_negatives = df.loc[
            df.index.isin(hard_negative_indices) & (df["contact"] == 0)
        ]

        # 3. Anchors (Random Easy Negatives)
        # Candidates are negatives that are NOT in the hard negative set
        # Note: We exclude hard_negative_indices to avoid duplication
        negative_mask = (df["contact"] == 0) & (~df.index.isin(hard_negative_indices))
        easy_negative_candidates = df[negative_mask]

        # Calculate required number of anchors
        count_core = len(positives) + len(hard_negatives)
        target_anchor_count = int(count_core * anchor_ratio)

        if len(easy_negative_candidates) > target_anchor_count:
            anchors = easy_negative_candidates.sample(
                n=target_anchor_count, random_state=random_state
            )
        else:
            anchors = easy_negative_candidates

        # Combine and shuffle
        expert_df = (
            pd.concat([positives, hard_negatives, anchors])
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )

        print(f"[{DataLoader.__name__}] Created Expert Dataset: {len(expert_df)} rows")
        print(f"   - Positives: {len(positives)}")
        print(f"   - Hard Negatives: {len(hard_negatives)}")
        print(f"   - Anchors: {len(anchors)}")

        return expert_df
