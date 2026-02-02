import pandas as pd
import numpy as np
import os
import gc
from library.config import KADM_CONFIG
from library.utils import setup_logger, process_with_cache
from library.feature_engineering import KinematicFeatureEngine

# Setup logger
logger = setup_logger(name="data_loader")


class DataLoader:
    """
    Handles data ingestion, merging, and gating for the KADM-AE pipeline.
    Acts as an orchestrator for KinematicFeatureEngine and prepares datasets for training/inference.
    """

    def __init__(self, config=KADM_CONFIG):
        self.config = config
        self.feature_engine = KinematicFeatureEngine(config)
        self.debug = config["settings"]["debug"]
        self.gating_threshold = config["feature_engineering"]["gating_threshold"]

    def load_metadata(self, split):
        """
        Loads the metadata CSV for the specified split.
        """
        if split == "train":
            path = self.config["paths"]["train_metadata"]
        elif split == "val":
            path = self.config["paths"]["val_metadata"]
        elif split == "test":
            path = self.config["paths"]["test_metadata"]
        else:
            raise ValueError(f"Unknown split: {split}")

        logger.info(f"Loading metadata from {path}")
        df = pd.read_csv(path)
        return df

    def load_tracking(self, split):
        """
        Loads the tracking data CSV for the specified split.
        """
        if split in ["train", "val"]:
            path = self.config["paths"]["train_tracking"]
        elif split == "test":
            path = self.config["paths"]["test_tracking"]
        else:
            raise ValueError(f"Unknown split: {split}")

        logger.info(f"Loading tracking data from {path}")
        df = pd.read_csv(path)
        return df

    def merge_tracking_data(self, metadata_df, tracking_df):
        """
        Aligns player positions with timestamps for the contact moment.
        Note: This performs a point-in-time merge. The KinematicFeatureEngine handles
        temporal window merging internally for feature generation.
        """
        logger.info("Merging tracking data with metadata...")

        # Ensure types match
        meta = metadata_df.copy()
        track = tracking_df.copy()

        # Merge Player 1
        merged = pd.merge(
            meta,
            track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )

        # Rename P1 columns if collision didn't occur or to standardise
        rename_map_p1 = {
            "x_position": "x_position_p1",
            "y_position": "y_position_p1",
            "speed": "speed_p1",
            "acceleration": "acceleration_p1",
            "direction": "direction_p1",
            "orientation": "orientation_p1",
        }
        merged.rename(columns=rename_map_p1, inplace=True)

        # Handle Ground for Player 2
        is_ground = merged["nfl_player_id_2"] == "G"

        # Prepare join column for P2
        merged["join_id_2"] = (
            pd.to_numeric(merged["nfl_player_id_2"], errors="coerce")
            .fillna(-999)
            .astype(int)
        )

        # Merge Player 2
        merged = pd.merge(
            merged,
            track,
            left_on=["game_play", "step", "join_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )

        # Rename P2 columns
        rename_map_p2 = {
            "x_position": "x_position_p2",
            "y_position": "y_position_p2",
            "speed": "speed_p2",
            "acceleration": "acceleration_p2",
            "direction": "direction_p2",
            "orientation": "orientation_p2",
        }
        merged.rename(columns=rename_map_p2, inplace=True)

        # Clean up
        drop_cols = [
            "nfl_player_id",
            "nfl_player_id_p1",
            "nfl_player_id_p2",
            "join_id_2",
        ]
        merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

        return merged

    def apply_relaxed_quadratic_gating(self, df):
        """
        Filters contact candidates based on the Relaxed Quadratic Gating logic.

        Args:
            df (pd.DataFrame): The dataframe containing features and the 'gating_pass' column.

        Returns:
            pd.DataFrame: The filtered dataframe containing only survivors.
        """
        logger.info("Applying Relaxed Quadratic Gating...")

        if "gating_pass" not in df.columns:
            logger.warning(
                "'gating_pass' column not found. Returning original dataframe."
            )
            return df

        initial_count = len(df)
        filtered_df = df[df["gating_pass"]].copy()
        final_count = len(filtered_df)

        logger.info(
            f"Gating reduced dataset from {initial_count} to {final_count} candidates "
            f"({(final_count/initial_count)*100:.2f}% survival rate)."
        )

        return filtered_df

    def load_dataset(self, split, apply_gating=True, load_cached_data=True):
        """
        Main entry point to load processed datasets for training or inference.

        Args:
            split (str): 'train', 'val', or 'test'.
            apply_gating (bool): Whether to filter based on the quadratic gating logic.
            load_cached_data (bool): Whether to use cached features.

        Returns:
            tuple: (X, y, meta_info)
                X (pd.DataFrame): Feature matrix.
                y (pd.Series or None): Target vector (None for test).
                meta_info (pd.DataFrame): Metadata columns (ids, game_play, etc.).
        """
        logger.info(f"Loading dataset for split: {split}")

        # 1. Generate/Load Features using the Engine
        # The engine handles caching internally
        features_df = self.feature_engine.generate_features(
            split, load_cached_data=load_cached_data
        )

        # 2. Apply Gating
        if apply_gating:
            features_df = self.apply_relaxed_quadratic_gating(features_df)

        # 3. Separate Components
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "gating_pass",
        ]
        target_col = "contact"

        # Identify feature columns (exclude meta and target)
        exclude_cols = meta_cols + [target_col]
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[feature_cols].copy()
        meta_info = features_df[meta_cols].copy()

        y = None
        if target_col in features_df.columns:
            y = features_df[target_col].copy()

        logger.info(f"Dataset loaded. X shape: {X.shape}")

        return X, y, meta_info
