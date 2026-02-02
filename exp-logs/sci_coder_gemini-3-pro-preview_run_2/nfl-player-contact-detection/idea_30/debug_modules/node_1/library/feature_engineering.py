import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

    def calculate_shortest_arc(self, angle1, angle2):
        """
        Computes the shortest arc between two angles in degrees.
        Result is in [0, 180].
        """
        diff = np.abs(angle1 - angle2) % 360
        return np.minimum(diff, 360 - diff)

    def clamp_values(self, df):
        """
        Clamps columns based on Config.CLAMPING_RANGES.
        """
        for col, (min_val, max_val) in self.config.CLAMPING_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=min_val, upper=max_val)
        return df

    def select_best_view(self, helmets_df):
        """
        Max-Pooling Selection Strategy: Select the view with the largest helmet area.
        """
        # Calculate area if not present
        if "area" not in helmets_df.columns:
            helmets_df["area"] = helmets_df["width"] * helmets_df["height"]

        # Sort by game, step, player, and area (descending)
        helmets_df = helmets_df.sort_values(
            by=["game_play", "step", "nfl_player_id", "area"],
            ascending=[True, True, True, False],
        )

        # Drop duplicates keeping the first (largest area)
        best_view = helmets_df.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"], keep="first"
        )

        return best_view

    def process_tracking(self, tracking_path):
        """
        Loads and preprocesses tracking data with angular corrections and clamping.
        """
        # Load data
        df = pd.read_csv(tracking_path, usecols=self.config.TRACKING_COLS)

        # Fill NaNs in angles
        df["direction"] = df["direction"].fillna(0)
        df["orientation"] = df["orientation"].fillna(0)

        # Compute Trigonometric Features
        df["sin_direction"] = np.sin(np.radians(df["direction"]))
        df["cos_direction"] = np.cos(np.radians(df["direction"]))
        df["sin_orientation"] = np.sin(np.radians(df["orientation"]))
        df["cos_orientation"] = np.cos(np.radians(df["orientation"]))

        # Apply numerical stability clamping
        df = self.clamp_values(df)

        return df

    def process_helmets(self, helmets_path):
        """
        Loads and preprocesses helmet data, mapping frames to steps and selecting best view.
        """
        df = pd.read_csv(helmets_path)

        # Map Frame to Step (Snap is ~frame 300, 59.94Hz, 10Hz steps)
        # step = round((frame - 300) / 5.994)
        df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

        # Select Best View (Max-Pooling)
        df = self.select_best_view(df)

        # Ensure area exists for features
        if "area" not in df.columns:
            df["area"] = df["width"] * df["height"]

        # Select required columns
        keep_cols = ["game_play", "step", "nfl_player_id"] + self.config.VISUAL_FEATURES
        # Filter columns that exist (VISUAL_FEATURES should be in df)
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        # Apply numerical stability clamping
        df = self.clamp_values(df)

        return df

    def generate_features(
        self,
        metadata_path,
        tracking_path,
        helmets_path,
        output_name,
        load_cached_data=True,
    ):
        """
        Main pipeline to generate windowed features for training or inference.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, f"{output_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {output_name}...")

        # 1. Load Metadata
        meta_df = pd.read_csv(metadata_path)

        # 2. Load and Process Tracking & Helmets
        # Filter to relevant games to optimize memory
        relevant_games = meta_df["game_play"].unique()

        track_df = self.process_tracking(tracking_path)
        track_df = track_df[track_df["game_play"].isin(relevant_games)].copy()

        helmets_df = self.process_helmets(helmets_path)
        helmets_df = helmets_df[helmets_df["game_play"].isin(relevant_games)].copy()

        # 3. Prepare Metadata for Merging
        final_df = meta_df.copy()

        # Handle Player 2 ID (Ground is 'G')
        final_df["nfl_player_id_2_num"] = pd.to_numeric(
            final_df["nfl_player_id_2"], errors="coerce"
        )
        is_ground = final_df["nfl_player_id_2"] == "G"

        # Identify Kinematic Features to merge (exclude identifiers)
        exclude_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "datetime",
            "game_key",
            "play_id",
            "jersey_number",
            "position",
            "team",
        ]
        kin_feats = [c for c in track_df.columns if c not in exclude_cols]

        # Identify Visual Features
        vis_feats = self.config.VISUAL_FEATURES

        # 4. Time-Distributed Window Loop
        start_lag = -self.config.HALF_WINDOW
        end_lag = self.config.HALF_WINDOW

        for lag in range(start_lag, end_lag + 1):
            # Create temporary step column for merge
            final_df[f"step_lag_{lag}"] = final_df["step"] + lag

            # --- Merge Player 1 ---
            # Tracking
            p1_track = track_df.rename(
                columns={"step": f"step_lag_{lag}", "nfl_player_id": "nfl_player_id_1"}
            )
            rename_p1_kin = {c: f"{c}_p1_{lag}" for c in kin_feats}
            final_df = final_df.merge(
                p1_track[
                    ["game_play", f"step_lag_{lag}", "nfl_player_id_1"] + kin_feats
                ].rename(columns=rename_p1_kin),
                on=["game_play", f"step_lag_{lag}", "nfl_player_id_1"],
                how="left",
            )

            # Helmets
            p1_helmets = helmets_df.rename(
                columns={"step": f"step_lag_{lag}", "nfl_player_id": "nfl_player_id_1"}
            )
            rename_p1_vis = {c: f"{c}_p1_{lag}" for c in vis_feats}
            final_df = final_df.merge(
                p1_helmets[
                    ["game_play", f"step_lag_{lag}", "nfl_player_id_1"] + vis_feats
                ].rename(columns=rename_p1_vis),
                on=["game_play", f"step_lag_{lag}", "nfl_player_id_1"],
                how="left",
            )

            # --- Merge Player 2 ---
            # Tracking (Merge on numeric ID, Ground 'G' will be NaN and skipped)
            p2_track = track_df.rename(
                columns={
                    "step": f"step_lag_{lag}",
                    "nfl_player_id": "nfl_player_id_2_num",
                }
            )
            rename_p2_kin = {c: f"{c}_p2_{lag}" for c in kin_feats}
            final_df = final_df.merge(
                p2_track[
                    ["game_play", f"step_lag_{lag}", "nfl_player_id_2_num"] + kin_feats
                ].rename(columns=rename_p2_kin),
                on=["game_play", f"step_lag_{lag}", "nfl_player_id_2_num"],
                how="left",
            )

            # Helmets
            p2_helmets = helmets_df.rename(
                columns={
                    "step": f"step_lag_{lag}",
                    "nfl_player_id": "nfl_player_id_2_num",
                }
            )
            rename_p2_vis = {c: f"{c}_p2_{lag}" for c in vis_feats}
            final_df = final_df.merge(
                p2_helmets[
                    ["game_play", f"step_lag_{lag}", "nfl_player_id_2_num"] + vis_feats
                ].rename(columns=rename_p2_vis),
                on=["game_play", f"step_lag_{lag}", "nfl_player_id_2_num"],
                how="left",
            )

            # --- Ground Imputation ---
            # If P2 is Ground: Pos=P1, Vel=0, Vis=0

            # Coordinates
            final_df.loc[is_ground, f"x_position_p2_{lag}"] = final_df.loc[
                is_ground, f"x_position_p1_{lag}"
            ]
            final_df.loc[is_ground, f"y_position_p2_{lag}"] = final_df.loc[
                is_ground, f"y_position_p1_{lag}"
            ]

            # Zero Dynamics
            zero_cols = [
                "speed",
                "acceleration",
                "sa",
                "sin_direction",
                "cos_direction",
                "sin_orientation",
                "cos_orientation",
            ]
            for z in zero_cols:
                col_name = f"{z}_p2_{lag}"
                if col_name in final_df.columns:
                    final_df.loc[is_ground, col_name] = 0.0

            # Zero Visuals
            for v in vis_feats:
                col_name = f"{v}_p2_{lag}"
                if col_name in final_df.columns:
                    final_df.loc[is_ground, col_name] = 0.0

            # --- Relative Features ---
            # Relative Distance (rel_dist)
            dx = final_df[f"x_position_p1_{lag}"] - final_df[f"x_position_p2_{lag}"]
            dy = final_df[f"y_position_p1_{lag}"] - final_df[f"y_position_p2_{lag}"]
            dist = np.sqrt(dx**2 + dy**2)
            final_df[f"rel_dist_{lag}"] = dist

            # Relative Speed (scalar diff)
            final_df[f"rel_speed_{lag}"] = (
                final_df[f"speed_p1_{lag}"] - final_df[f"speed_p2_{lag}"]
            )

            # Relative Acceleration (scalar diff)
            final_df[f"rel_accel_{lag}"] = (
                final_df[f"acceleration_p1_{lag}"] - final_df[f"acceleration_p2_{lag}"]
            )

            # Closing Speed
            # Reconstruct velocity vectors
            vx1 = final_df[f"speed_p1_{lag}"] * final_df[f"sin_direction_p1_{lag}"]
            vy1 = final_df[f"speed_p1_{lag}"] * final_df[f"cos_direction_p1_{lag}"]
            vx2 = final_df[f"speed_p2_{lag}"] * final_df[f"sin_direction_p2_{lag}"]
            vy2 = final_df[f"speed_p2_{lag}"] * final_df[f"cos_direction_p2_{lag}"]

            rvx = vx1 - vx2
            rvy = vy1 - vy2
            dot = rvx * dx + rvy * dy

            # Avoid division by zero
            final_df[f"closing_speed_{lag}"] = -(dot) / (dist + 1e-6)

            # --- Clamp Derived Features ---
            derived_map = {
                f"rel_dist_{lag}": "rel_dist",
                f"rel_speed_{lag}": "rel_speed",
                f"rel_accel_{lag}": "rel_accel",
                f"closing_speed_{lag}": "closing_speed",
            }
            for col, key in derived_map.items():
                if key in self.config.CLAMPING_RANGES:
                    min_v, max_v = self.config.CLAMPING_RANGES[key]
                    final_df[col] = final_df[col].clip(min_v, max_v)

            # Cleanup
            final_df = final_df.drop(columns=[f"step_lag_{lag}"])

        # 5. Add Static Entity Embeddings (Position, Team)
        # Fetch unique player metadata from tracking
        player_meta = track_df[["nfl_player_id", "position", "team"]].drop_duplicates(
            subset=["nfl_player_id"]
        )

        # Merge P1
        final_df = final_df.merge(
            player_meta.rename(
                columns={
                    "nfl_player_id": "nfl_player_id_1",
                    "position": "position_1",
                    "team": "team_1",
                }
            ),
            on="nfl_player_id_1",
            how="left",
        )

        # Merge P2
        final_df = final_df.merge(
            player_meta.rename(
                columns={
                    "nfl_player_id": "nfl_player_id_2_num",
                    "position": "position_2",
                    "team": "team_2",
                }
            ),
            on="nfl_player_id_2_num",
            how="left",
        )

        # Impute Ground Categoricals
        final_df.loc[is_ground, "position_2"] = "G"
        final_df.loc[is_ground, "team_2"] = "G"

        # 6. Final Cleanup
        # Fill missing values (tracking might be missing for some players/frames)
        num_cols = final_df.select_dtypes(include=[np.number]).columns
        final_df[num_cols] = final_df[num_cols].fillna(0)

        cat_cols = ["position_1", "team_1", "position_2", "team_2"]
        final_df[cat_cols] = final_df[cat_cols].fillna("UNK")

        # Save to cache
        final_df.to_parquet(cache_path)
        print(f"Features saved to {cache_path}")

        # Clean up memory
        del track_df, helmets_df, player_meta
        gc.collect()

        return final_df
