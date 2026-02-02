import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import setup_logging, CacheManager


class DataLoader:
    """
    Handles loading, merging, and pre-filtering of metadata and tracking data.
    Implements the Relaxed Quadratic Gating strategy (KARP-AM Stage 0) to
    reduce the search space before expensive feature engineering.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.cache = CacheManager()

    def load_data(self, split: str, load_cached_data: bool = True):
        """
        Loads the filtered metadata and raw tracking data for a given split.

        Args:
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cached filtered metadata.

        Returns:
            metadata_df (pd.DataFrame): Filtered metadata (gated).
            tracking_df (pd.DataFrame): Raw tracking data.
        """
        # 1. Load Raw Tracking (Always needed for feature engineering later)
        tracking_df = self._load_raw_tracking(split)

        # 2. Try Loading Cached Filtered Metadata
        # We hash the gating parameters to ensure cache validity if params change
        params = {
            "split": split,
            "gating_dist": Config.GATING_DIST,
            "window_size": Config.WINDOW_SIZE,
            "strategy": "quadratic_gating_v1",
        }
        cache_filename = (
            self.cache.generate_key(f"filtered_meta_{split}", params) + ".parquet"
        )

        if load_cached_data and self.cache.exists(cache_filename):
            self.logger.info(f"Loading cached filtered metadata from {cache_filename}")
            metadata_df = self.cache.load(cache_filename)
        else:
            self.logger.info(f"Processing metadata for {split} (Cache miss)...")
            # Load Raw Metadata
            raw_meta = self._load_raw_metadata(split)

            # Apply Gating and Filtering
            metadata_df = self._process_metadata(raw_meta, tracking_df)

            # Save to Cache
            self.logger.info(f"Saving filtered metadata to {cache_filename}")
            self.cache.save(metadata_df, cache_filename)

        return metadata_df, tracking_df

    def _load_raw_metadata(self, split: str) -> pd.DataFrame:
        """Loads the raw metadata CSV based on the split."""
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        self.logger.info(f"Loading raw metadata from {path}")
        return pd.read_csv(path)

    def _load_raw_tracking(self, split: str) -> pd.DataFrame:
        """Loads the raw tracking CSV. Train and Val share the same source."""
        if split in ["train", "val"]:
            path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        self.logger.info(f"Loading raw tracking data from {path}")
        return pd.read_csv(path)

    def _process_metadata(
        self, meta_df: pd.DataFrame, track_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Orchestrates the filtering process:
        1. Separates Player-Ground (always kept).
        2. Merges Player-Player tracking for the specific contact step.
        3. Applies Quadratic Gating to filter physically impossible contacts.
        4. Recombines and returns.
        """
        # Separate Ground contacts
        mask_ground = meta_df["nfl_player_id_2"] == "G"
        meta_ground = meta_df[mask_ground].copy()
        meta_pp = meta_df[~mask_ground].copy()

        if meta_pp.empty:
            return meta_ground

        # Prepare Player-Player for Gating
        self.logger.info(
            f"Applying Quadratic Gating to {len(meta_pp)} Player-Player pairs..."
        )

        # We need to merge tracking data for the *current* step to estimate kinematics.
        # This is a lightweight merge compared to the full window expansion.

        # 1. Pre-calculate vectors for tracking data (Speed/Dir -> Vx/Vy)
        track_prep = track_df[
            [
                "game_play",
                "step",
                "nfl_player_id",
                "x_position",
                "y_position",
                "speed",
                "direction",
                "acceleration",
            ]
        ].copy()

        # Convert to radians (0 is North/Y, 90 is East/X)
        rad = np.radians(track_prep["direction"].fillna(0))
        track_prep["vx"] = track_prep["speed"] * np.sin(rad)
        track_prep["vy"] = track_prep["speed"] * np.cos(rad)
        track_prep["ax"] = track_prep["acceleration"] * np.sin(rad)
        track_prep["ay"] = track_prep["acceleration"] * np.cos(rad)

        cols_to_keep = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "vx",
            "vy",
            "ax",
            "ay",
        ]
        track_prep = track_prep[cols_to_keep]

        # Ensure types for merge
        meta_pp["nfl_player_id_2"] = meta_pp["nfl_player_id_2"].astype(int)

        # Merge P1
        merged = meta_pp.merge(
            track_prep,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="inner",
        ).rename(
            columns={
                c: f"{c}_p1"
                for c in ["x_position", "y_position", "vx", "vy", "ax", "ay"]
            }
        )

        # Merge P2
        merged = merged.drop(columns=["nfl_player_id"])  # Drop P1 id from track
        merged = merged.merge(
            track_prep,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="inner",
        ).rename(
            columns={
                c: f"{c}_p2"
                for c in ["x_position", "y_position", "vx", "vy", "ax", "ay"]
            }
        )

        merged = merged.drop(columns=["nfl_player_id"])  # Drop P2 id from track

        # Apply Gating
        survivors_mask = self.apply_quadratic_gating(merged)

        # Filter meta_pp based on surviving contact_ids
        surviving_ids = merged.loc[survivors_mask, "contact_id"].unique()
        meta_pp_filtered = meta_pp[meta_pp["contact_id"].isin(surviving_ids)].copy()

        self.logger.info(
            f"Gating complete. Kept {len(meta_pp_filtered)} / {len(meta_pp)} Player-Player pairs."
        )

        # Combine
        final_df = pd.concat([meta_pp_filtered, meta_ground], axis=0).reset_index(
            drop=True
        )
        return final_df

    def apply_quadratic_gating(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates min distance over the window [-1.0s, 1.0s] using Taylor expansion
        of the relative motion vectors.

        Returns:
            pd.Series: Boolean mask where True indicates the pair is kept.
        """
        # Relative State at t=0
        rx = df["x_position_p1"] - df["x_position_p2"]
        ry = df["y_position_p1"] - df["y_position_p2"]
        rvx = df["vx_p1"] - df["vx_p2"]
        rvy = df["vy_p1"] - df["vy_p2"]
        rax = df["ax_p1"] - df["ax_p2"]
        ray = df["ay_p1"] - df["ay_p2"]

        # Time steps to check (in seconds). Window is +/- 1.0s.
        # Checking every 0.2s provides sufficient resolution for the quadratic approximation.
        t_steps = np.linspace(-1.0, 1.0, 11)

        min_dists_sq = np.full(len(df), np.inf)

        for t in t_steps:
            # Quadratic projection: r(t) = r0 + v0*t + 0.5*a0*t^2
            rx_t = rx + rvx * t + 0.5 * rax * (t**2)
            ry_t = ry + rvy * t + 0.5 * ray * (t**2)

            dist_sq = rx_t**2 + ry_t**2
            min_dists_sq = np.minimum(min_dists_sq, dist_sq)

        # Threshold check
        mask = min_dists_sq < (Config.GATING_DIST**2)
        return mask
