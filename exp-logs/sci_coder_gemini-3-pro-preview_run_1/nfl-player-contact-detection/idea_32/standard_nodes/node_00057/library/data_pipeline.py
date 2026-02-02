import logging
import pandas as pd
import numpy as np
from library.config import Config
from library.features import generate_features
from library.utils import seed_everything


class DataPipeline:
    """
    Orchestrates data loading and dataset construction for the
    Dual-Basis Time-Domain Anchored-Mining strategy.

    Delegates feature engineering to library.features and handles
    the logic for Scout (Balanced) and Expert (Mined + Anchored)
    dataset splits.
    """

    def __init__(self, config=Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self, mode="train", load_cached_data=True, debug=False):
        """
        Loads the feature-engineered dataset for the specified mode.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): Whether to use a small subset for debugging.

        Returns:
            pd.DataFrame: The processed dataframe with features and labels (if applicable).
        """
        self.logger.info(
            f"DataPipeline: Loading {mode} data (Cache={load_cached_data}, Debug={debug})..."
        )
        return generate_features(
            mode=mode, load_cached_data=load_cached_data, debug=debug
        )

    def construct_scout_dataset(self, df):
        """
        Constructs a balanced dataset for training Scout models.

        Strategy:
            - Keep all Positives (contact == 1).
            - Sample Negatives (contact == 0) to match the count of Positives (1:1 ratio).

        Args:
            df (pd.DataFrame): The full training dataframe (post-gating).

        Returns:
            pd.DataFrame: A shuffled, balanced dataframe.
        """
        seed_everything(self.config.SEED)

        # Separate classes
        positives = df[df["contact"] == 1]
        negatives = df[df["contact"] == 0]

        n_pos = len(positives)
        n_neg = len(negatives)

        self.logger.info(f"Constructing Scout Dataset from {len(df)} samples.")
        self.logger.info(f"Total Positives: {n_pos}, Total Negatives: {n_neg}")

        if n_pos == 0:
            self.logger.warning(
                "No positives found in dataset! Returning empty dataframe."
            )
            return pd.DataFrame(columns=df.columns)

        # Sample negatives to match positives
        # If fewer negatives than positives (unlikely in this domain), take all negatives
        n_sample = min(n_pos, n_neg)
        negatives_sampled = negatives.sample(n=n_sample, random_state=self.config.SEED)

        # Combine and shuffle
        df_scout = pd.concat([positives, negatives_sampled], axis=0)
        df_scout = df_scout.sample(frac=1.0, random_state=self.config.SEED).reset_index(
            drop=True
        )

        self.logger.info(
            f"Scout Dataset Created: {len(df_scout)} samples (Balanced 1:1)."
        )

        return df_scout

    def construct_expert_dataset(self, df, hard_negative_indices):
        """
        Constructs the Expert Dataset using Anchored Mining.

        Strategy:
            1. Include ALL Positives.
            2. Include ALL Mined Hard Negatives (identified by indices).
            3. Include Random Easy Negatives (Anchors) based on ANCHOR_RATIO relative to Hard Negatives.

        Args:
            df (pd.DataFrame): The full training dataframe.
            hard_negative_indices (np.ndarray or list): Indices of negatives classified as 'Hard' by Scouts.

        Returns:
            pd.DataFrame: The constructed expert training dataset.
        """
        seed_everything(self.config.SEED)

        self.logger.info("Constructing Expert Dataset with Anchored Mining...")

        # 1. Positives
        positives = df[df["contact"] == 1]

        # 2. Hard Negatives
        # Ensure we only select rows that are actually in the dataframe (indices might be from a full set)
        # and ensure they are actually negatives (sanity check)
        valid_hard_indices = np.intersect1d(df.index, hard_negative_indices)
        hard_negatives = df.loc[valid_hard_indices]

        # Filter out any potential mislabeled positives in the hard negative set (just in case)
        hard_negatives = hard_negatives[hard_negatives["contact"] == 0]

        # 3. Anchors (Easy Negatives)
        # Candidates are negatives that are NOT in the hard negative set
        # We use index exclusion
        is_negative = df["contact"] == 0
        is_hard = df.index.isin(valid_hard_indices)

        easy_negatives_mask = is_negative & (~is_hard)
        easy_negatives = df[easy_negatives_mask]

        # Calculate number of anchors to sample
        n_hard = len(hard_negatives)
        n_anchors = int(n_hard * self.config.ANCHOR_RATIO)

        # Limit anchors to available easy negatives
        n_anchors = min(n_anchors, len(easy_negatives))

        dfs_to_concat = [positives, hard_negatives]
        anchors_count = 0

        if n_anchors > 0:
            anchors = easy_negatives.sample(n=n_anchors, random_state=self.config.SEED)
            dfs_to_concat.append(anchors)
            anchors_count = len(anchors)

        # Combine
        df_expert = pd.concat(dfs_to_concat, axis=0)
        df_expert = df_expert.sample(
            frac=1.0, random_state=self.config.SEED
        ).reset_index(drop=True)

        self.logger.info(f"Expert Dataset Stats:")
        self.logger.info(f"  Positives: {len(positives)}")
        self.logger.info(f"  Hard Negatives: {len(hard_negatives)}")
        self.logger.info(
            f"  Anchors (Easy Negs): {anchors_count} (Ratio: {self.config.ANCHOR_RATIO})"
        )
        self.logger.info(f"  Total Samples: {len(df_expert)}")

        return df_expert
