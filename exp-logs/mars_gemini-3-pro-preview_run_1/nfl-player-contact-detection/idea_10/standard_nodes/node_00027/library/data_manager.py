import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, execute_with_cache, seed_everything
from library.feature_engineering import FeatureEngineer


class DataManager:
    """
    Orchestrates data loading, feature engineering, and dataset construction
    for the Ego-Centric Spatial Grid Mining Ensemble.

    Manages caching of the heavy feature matrices and implements the
    sampling logic for the Scout/Expert curriculum learning strategy.
    """

    def __init__(self):
        self.logger = setup_logger()
        self.fe = FeatureEngineer()
        seed_everything(Config.SEED)

    def _generate_full_train_data(self):
        """
        Internal function to generate the full gated training dataset using FeatureEngineer.
        This is passed to execute_with_cache.
        """
        self.logger.info("Generating Full Gated Training Dataset from scratch...")
        X, y, meta = self.fe.generate_dataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            tracking_path=Config.TRAIN_TRACKING_PATH,
            mode="train",
            load_cached_data=True,  # Allow FE to use its own tracking cache
        )

        # Combine into one dataframe for caching convenience
        # We prefix meta cols to avoid collision if necessary, but FE handles unique names
        # FE returns X (features), y (target), meta (ids)

        # Concatenate for storage
        df_full = pd.concat([meta, X], axis=1)
        if y is not None:
            df_full["target_contact"] = y

        return df_full

    def load_and_process_train_data(self, load_cached_data=True):
        """
        Loads the full training dataset, applying geometric gating and feature engineering.
        Uses caching to avoid re-computing grids.

        Returns:
            df_full (pd.DataFrame): The complete dataframe containing Meta, X, and y.
        """
        cache_filename = "train_features_gated_full.parquet"

        df_full = execute_with_cache(
            cache_filename,
            self._generate_full_train_data,
            load_cached_data=load_cached_data,
        )
        return df_full

    def get_scout_dataset(self, load_cached_data=True):
        """
        Constructs the 'Scout' dataset:
        - All Positives
        - Random sample of Negatives (controlled by Config.SCOUT_NEG_RATIO)

        Used to train the initial model that identifies hard negatives.
        """
        self.logger.info("Constructing Scout Dataset...")

        # Load full gated data
        df = self.load_and_process_train_data(load_cached_data=load_cached_data)

        # Separate Positives and Negatives
        pos_mask = df["target_contact"] == 1
        neg_mask = df["target_contact"] == 0

        df_pos = df[pos_mask]
        df_neg = df[neg_mask]

        # Sample Negatives
        n_pos = len(df_pos)
        n_neg = int(n_pos * Config.SCOUT_NEG_RATIO)

        # Ensure we don't sample more than available (though unlikely with gating)
        n_neg = min(n_neg, len(df_neg))

        self.logger.info(f"Scout Sampling: {n_pos} Positives, {n_neg} Negatives.")

        df_neg_sampled = df_neg.sample(n=n_neg, random_state=Config.SEED)

        # Combine and Shuffle
        df_scout = (
            pd.concat([df_pos, df_neg_sampled], axis=0)
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        # Split X, y
        y = df_scout["target_contact"]
        # Drop metadata and target to get X
        # We need to identify feature columns.
        # Strategy: Drop known metadata columns and target.
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "target_contact",
            "contact",
        ]
        exclude_cols += [c for c in df_scout.columns if "video_path" in c]

        X = df_scout.drop(columns=[c for c in exclude_cols if c in df_scout.columns])

        return X, y

    def get_mining_candidates(self, load_cached_data=True):
        """
        Returns the entire gated training set (X, y, meta).
        Used by the Scout model to predict on everything and find Hard Negatives.
        """
        self.logger.info("Loading Mining Candidates (Full Gated Set)...")
        df = self.load_and_process_train_data(load_cached_data=load_cached_data)

        y = df["target_contact"]

        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        meta = df[meta_cols].copy()

        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "target_contact",
            "contact",
        ]
        exclude_cols += [c for c in df.columns if "video_path" in c]

        X = df.drop(columns=[c for c in exclude_cols if c in df.columns])

        return X, y, meta

    def get_expert_dataset(self, hard_negative_indices, load_cached_data=True):
        """
        Constructs the 'Expert' dataset:
        - All Positives
        - All Mined Hard Negatives (provided by indices)
        - A random buffer of other negatives (to prevent forgetting easy cases)

        Args:
            hard_negative_indices (list or np.array): Indices of the full gated dataset
                                                      that were identified as hard negatives.
        """
        self.logger.info(
            f"Constructing Expert Dataset with {len(hard_negative_indices)} hard negatives..."
        )

        df = self.load_and_process_train_data(load_cached_data=load_cached_data)

        # 1. All Positives
        df_pos = df[df["target_contact"] == 1]

        # 2. Hard Negatives
        # Ensure indices are valid
        valid_indices = [i for i in hard_negative_indices if i < len(df)]
        df_hard_neg = df.iloc[valid_indices]

        # 3. Random Buffer (Easy Negatives)
        # We want some easy negatives so the model doesn't overfit to the boundary
        # Let's take a ratio relative to positives, e.g., 1:1
        n_buffer = len(df_pos)

        # Exclude already selected hard negatives and positives
        mask_excluded = df.index.isin(valid_indices) | (df["target_contact"] == 1)
        df_remaining = df[~mask_excluded]

        if not df_remaining.empty:
            n_buffer = min(n_buffer, len(df_remaining))
            df_buffer = df_remaining.sample(n=n_buffer, random_state=Config.SEED)
        else:
            # Cite debug_lesson_17: Preserve Dtypes When Concatenating Empty DataFrames
            df_buffer = df.iloc[:0].copy()

        self.logger.info(
            f"Expert Composition: {len(df_pos)} Pos, {len(df_hard_neg)} Hard Neg, {len(df_buffer)} Easy Neg"
        )

        # Combine
        df_expert = pd.concat([df_pos, df_hard_neg, df_buffer], axis=0)
        df_expert = df_expert.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

        y = df_expert["target_contact"]

        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "target_contact",
            "contact",
        ]
        exclude_cols += [c for c in df_expert.columns if "video_path" in c]

        X = df_expert.drop(columns=[c for c in exclude_cols if c in df_expert.columns])

        return X, y

    def get_val_dataset(self, load_cached_data=True):
        """
        Generates the validation dataset.
        Applies the same feature engineering and gating as training.
        """
        cache_filename = "val_features_gated.parquet"

        def _compute():
            self.logger.info("Generating Validation Dataset...")
            X, y, meta = self.fe.generate_dataset(
                metadata_path=Config.VAL_METADATA_PATH,
                tracking_path=Config.TRAIN_TRACKING_PATH,  # Val is subset of Train files
                mode="val",
                load_cached_data=True,
            )
            # Combine for caching
            df_full = pd.concat([meta, X], axis=1)
            if y is not None:
                df_full["target_contact"] = y
            return df_full

        df_val = execute_with_cache(
            cache_filename, _compute, load_cached_data=load_cached_data
        )

        y = df_val["target_contact"]
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "target_contact",
            "contact",
        ]
        exclude_cols += [c for c in df_val.columns if "video_path" in c]
        X = df_val.drop(columns=[c for c in exclude_cols if c in df_val.columns])
        meta = df_val[
            ["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
        ].copy()

        return X, y, meta

    def get_test_dataset(self, load_cached_data=True):
        """
        Generates the test dataset.
        Does NOT apply geometric gating (we must predict for all rows in submission).
        """
        cache_filename = "test_features_full.parquet"

        def _compute():
            self.logger.info("Generating Test Dataset...")
            X, y, meta = self.fe.generate_dataset(
                metadata_path=Config.TEST_METADATA_PATH,
                tracking_path=Config.TEST_TRACKING_PATH,
                mode="test",
                load_cached_data=True,
            )
            # Combine for caching
            df_full = pd.concat([meta, X], axis=1)
            # y is likely None or dummy 0s
            return df_full

        df_test = execute_with_cache(
            cache_filename, _compute, load_cached_data=load_cached_data
        )

        # Test usually doesn't have target, but generate_dataset might return None or dummy
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "target_contact",
            "contact",
        ]
        exclude_cols += [c for c in df_test.columns if "video_path" in c]

        X = df_test.drop(columns=[c for c in exclude_cols if c in df_test.columns])
        meta = df_test[
            ["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
        ].copy()

        return X, meta
