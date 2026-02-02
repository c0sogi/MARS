import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    Handles data loading, global preprocessing, and caching.
    Specifically implements the Nearest-Neighbor Context generation for Stream B.
    """

    def __init__(self):
        self.logger = setup_logger(name="DataLoader")

    def load_metadata(self):
        """
        Loads the train, validation, and test metadata CSVs generated in the previous step.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        self.logger.info("Loading metadata...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        self.logger.info(
            f"Metadata loaded. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )
        return df_train, df_val, df_test

    def load_helmets(self, dataset_type="train"):
        """
        Loads the baseline helmet detection data.

        Args:
            dataset_type (str): 'train' or 'test'.

        Returns:
            pd.DataFrame: Helmet data.
        """
        path = (
            Config.TRAIN_HELMETS_PATH
            if dataset_type == "train"
            else Config.TEST_HELMETS_PATH
        )
        self.logger.info(f"Loading helmets from {path}...")
        df = pd.read_csv(path)
        return df

    def get_processed_tracking(self, dataset_type="train", load_cached_data=True):
        """
        Loads tracking data and computes Nearest-Neighbor Context features.
        Implements caching to avoid re-computing expensive spatial operations.

        Args:
            dataset_type (str): 'train' or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Processed tracking data with NN context features.
        """
        cache_filename = f"processed_tracking_{dataset_type}_context.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading processed tracking data from cache: {cache_path}"
            )
            try:
                df_tracking = pd.read_parquet(cache_path)
                return df_tracking
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        self.logger.info(f"Computing tracking data for {dataset_type}...")

        # Load raw data
        raw_path = (
            Config.TRAIN_TRACKING_PATH
            if dataset_type == "train"
            else Config.TEST_TRACKING_PATH
        )
        df_tracking = pd.read_csv(raw_path, usecols=Config.RAW_TRACKING_COLS)

        # Filter to relevant games if needed (optimization)
        # For this competition, we usually process the whole file or filter by metadata.
        # To ensure consistency, we process the whole tracking file provided.

        # Compute Nearest Neighbor Context
        df_processed = self._compute_nearest_neighbor_context(df_tracking)

        # 3. Save to cache
        self.logger.info(f"Saving processed tracking data to cache: {cache_path}")
        df_processed.to_parquet(cache_path, index=False)

        return df_processed

    def _compute_nearest_neighbor_context(self, df_tracking):
        """
        Computes explicit relational features between each player and their
        nearest opponent (Nearest Neighbor) at each timestep.

        This enables Stream B (Player-Ground) to infer context (e.g., "I am being tackled").

        Args:
            df_tracking (pd.DataFrame): Raw tracking data.

        Returns:
            pd.DataFrame: Tracking data enriched with 'nn_*' columns.
        """
        self.logger.info("Computing Nearest Neighbor Context features...")

        # Ensure datetime is standard (though we rely on 'step' for synchronization)
        # df_tracking['datetime'] = pd.to_datetime(df_tracking['datetime'])

        # Split into Home and Away
        # We need to find the nearest *Opponent*.
        # For a Home player, opponent is Away. For Away player, opponent is Home.

        # Standardize team names if necessary (usually 'home' and 'away')
        # Check unique values
        # teams = df_tracking['team'].unique()

        df_home = df_tracking[df_tracking["team"] == "home"].copy()
        df_away = df_tracking[df_tracking["team"] == "away"].copy()

        if df_home.empty or df_away.empty:
            self.logger.warning(
                "Tracking data missing one or both teams. Skipping NN computation."
            )
            # Return original with NaNs for NN features
            for col in Config.CONTEXT_FEATURES:
                df_tracking[col] = np.nan
            return df_tracking

        # Define columns to keep for the merge (coordinates and kinematics for physics)
        merge_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
        ]

        # --- Process Home Players (finding nearest Away) ---
        # Merge Home (Subject) with Away (Opponent) on game_play and step
        # This creates a cartesian product of players per frame (11x11 = 121 rows per frame)
        home_vs_away = pd.merge(
            df_home[merge_cols],
            df_away[merge_cols],
            on=["game_play", "step"],
            suffixes=("", "_opp"),
        )

        # Calculate Euclidean Distance
        home_vs_away["dist_temp"] = np.sqrt(
            (home_vs_away["x_position"] - home_vs_away["x_position_opp"]) ** 2
            + (home_vs_away["y_position"] - home_vs_away["y_position_opp"]) ** 2
        )

        # Find Nearest Opponent: Group by Subject (Home Player) and Step, take min distance
        # We sort by distance and drop duplicates to keep the nearest one
        home_vs_away = home_vs_away.sort_values(
            ["game_play", "step", "nfl_player_id", "dist_temp"]
        )
        nearest_away = home_vs_away.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"], keep="first"
        ).copy()

        # --- Process Away Players (finding nearest Home) ---
        away_vs_home = pd.merge(
            df_away[merge_cols],
            df_home[merge_cols],
            on=["game_play", "step"],
            suffixes=("", "_opp"),
        )

        away_vs_home["dist_temp"] = np.sqrt(
            (away_vs_home["x_position"] - away_vs_home["x_position_opp"]) ** 2
            + (away_vs_home["y_position"] - away_vs_home["y_position_opp"]) ** 2
        )

        away_vs_home = away_vs_home.sort_values(
            ["game_play", "step", "nfl_player_id", "dist_temp"]
        )
        nearest_home = away_vs_home.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"], keep="first"
        ).copy()

        # --- Combine Results ---
        # We only need the computed features and the join keys
        df_nn = pd.concat([nearest_away, nearest_home], ignore_index=True)

        # Calculate Relational Physics Features
        # 1. NN Distance
        df_nn["nn_dist"] = df_nn["dist_temp"]

        # 2. NN Relative Speed
        # Simple difference in scalar speed (can be refined to vector difference if needed)
        df_nn["nn_rel_speed"] = df_nn["speed"] - df_nn["speed_opp"]

        # 3. NN Relative Acceleration
        df_nn["nn_rel_accel"] = df_nn["acceleration"] - df_nn["acceleration_opp"]

        # 4. NN Relative Angle (Orientation/Direction difference)
        # Using direction (motion angle)
        # Calculate smallest difference between two angles
        diff = np.abs(df_nn["direction"] - df_nn["direction_opp"])
        df_nn["nn_rel_angle"] = np.minimum(diff, 360 - diff)

        # Select columns to merge back to main tracking df
        nn_features = ["game_play", "step", "nfl_player_id"] + Config.CONTEXT_FEATURES

        # Merge back to original tracking data
        # We use left join to preserve all original tracking rows (even if no opponent found, though rare)
        df_final = pd.merge(
            df_tracking,
            df_nn[nn_features],
            on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Fill NaNs for NN features (e.g., if only one team on field in tracking data) with defaults
        # Distance -> Large number, Speed/Accel -> 0
        df_final["nn_dist"] = df_final["nn_dist"].fillna(100.0)
        df_final["nn_rel_speed"] = df_final["nn_rel_speed"].fillna(0.0)
        df_final["nn_rel_accel"] = df_final["nn_rel_accel"].fillna(0.0)
        df_final["nn_rel_angle"] = df_final["nn_rel_angle"].fillna(180.0)

        self.logger.info(f"NN Context computed. Shape: {df_final.shape}")
        return df_final
