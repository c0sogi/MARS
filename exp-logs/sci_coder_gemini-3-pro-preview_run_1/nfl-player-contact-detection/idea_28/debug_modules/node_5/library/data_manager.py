import pandas as pd
import numpy as np
import logging
from library.config import Config
from library.features import generate_features
from library.utils import seed_everything


class DataManager:
    """
    Manages data ingestion, preprocessing orchestration, and dataset construction
    for the Scout and Expert models in the Dual-Basis VSAM Ensemble.
    """

    def __init__(self, debug=False):
        """
        Initialize the DataManager.

        Args:
            debug (bool): If True, pipelines will run on a subset of data for rapid iteration.
        """
        self.debug = debug
        self.logger = logging.getLogger("NFL_Contact_Detection")
        # Ensure reproducibility for sampling operations
        seed_everything(Config.SEED)

    def load_and_merge_data(self, split="train", load_cached_data=True):
        """
        Orchestrates the loading, merging, and feature engineering process by calling
        the features library.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from the parquet cache.

        Returns:
            pd.DataFrame: The processed dataframe containing metadata, targets, and
                          Dual-Basis features.
        """
        self.logger.info(f"DataManager: Loading {split} data (Debug={self.debug})...")
        # Delegate the heavy lifting to the features library which handles
        # merging, gating, vector projection, and caching.
        return generate_features(
            split=split, load_cached_data=load_cached_data, debug=self.debug
        )

    def get_scout_data(self, load_cached_data=True):
        """
        Retrieves the datasets required for training the Scout models.
        Scouts are trained on the 'Gated Survivors' - the subset of interactions
        that pass the Relaxed Quadratic Gating filter.

        Args:
            load_cached_data (bool): Whether to use cached data.

        Returns:
            tuple: (train_df, val_df)
        """
        train_df = self.load_and_merge_data("train", load_cached_data)
        val_df = self.load_and_merge_data("val", load_cached_data)
        return train_df, val_df

    def get_expert_data(self, train_df, hard_negative_indices):
        """
        Constructs the Anchored Dataset for the Expert model training.

        The Expert dataset consists of:
        1. All Positive samples (Contact = 1).
        2. Mined Hard Negatives (indices provided by Scouts).
        3. Random Easy Negatives (Anchors) to preserve global decision boundaries.

        Args:
            train_df (pd.DataFrame): The full training dataframe (Gated Survivors).
            hard_negative_indices (list or np.array): Indices of rows in train_df
                                                      identified as hard negatives.

        Returns:
            pd.DataFrame: A balanced and anchored dataset for the Expert.
        """
        self.logger.info("DataManager: Constructing Expert Dataset...")

        # 1. Extract Positives
        positives = train_df[train_df["contact"] == 1]

        # 2. Extract Hard Negatives
        # We assume hard_negative_indices correspond to the index of train_df.
        # Since features are saved/loaded with index=False, train_df has a RangeIndex.
        hard_neg_mask = train_df.index.isin(hard_negative_indices)
        hard_negatives = train_df[hard_neg_mask]

        # 3. Sample Anchors (Easy Negatives)
        # Candidates are negatives that were NOT flagged as hard negatives.
        anchor_candidates = train_df[(train_df["contact"] == 0) & (~hard_neg_mask)]

        # Calculate number of anchors based on the configured ratio relative to hard negatives
        n_hard = len(hard_negatives)
        n_anchors = int(n_hard * Config.ANCHOR_RATIO)

        # Sample anchors
        if len(anchor_candidates) > n_anchors:
            anchors = anchor_candidates.sample(n=n_anchors, random_state=Config.SEED)
        else:
            anchors = anchor_candidates

        self.logger.info(
            f"Expert Data Stats: Positives={len(positives)}, "
            f"HardNegs={len(hard_negatives)}, Anchors={len(anchors)}"
        )

        # Combine components
        expert_df = pd.concat([positives, hard_negatives, anchors], axis=0)

        # Shuffle the dataset to ensure random batch distribution
        expert_df = expert_df.sample(frac=1, random_state=Config.SEED).reset_index(
            drop=True
        )

        return expert_df

    def get_test_data(self, load_cached_data=True):
        """
        Retrieves the processed test dataset.
        Note: The test set is NOT gated (all rows must be predicted).

        Args:
            load_cached_data (bool): Whether to use cached data.

        Returns:
            pd.DataFrame: The processed test dataframe.
        """
        return self.load_and_merge_data("test", load_cached_data)

    def prepare_X_y(self, df, target_col="contact"):
        """
        Splits a dataframe into feature matrix X and target vector y.
        Filters columns to match the defined MODEL_FEATURES in Config.

        Args:
            df (pd.DataFrame): The dataset.
            target_col (str): The name of the target column.

        Returns:
            tuple: (X, y) where X is a DataFrame of features and y is a Series (or None).
        """
        # Select only the features defined in the configuration
        # This ensures we don't accidentally train on metadata or IDs
        feature_cols = [c for c in Config.MODEL_FEATURES if c in df.columns]

        X = df[feature_cols]

        if target_col in df.columns:
            y = df[target_col]
        else:
            y = None

        return X, y
