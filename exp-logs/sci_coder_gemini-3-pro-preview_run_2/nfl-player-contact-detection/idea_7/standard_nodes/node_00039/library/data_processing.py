import os
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


class NFLContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Serves 1D tensors of shape (Features,).
    """

    def __init__(self, features, labels=None, contact_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.contact_ids = contact_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


class FeatureEngineer:
    """
    Handles data loading, vectorized lag creation, merging, and feature scaling.
    """

    def __init__(self, config: Config):
        self.config = config
        self.window_size = config.window_size
        self.half_window = self.window_size // 2

    def load_tracking(self, split_name, relevant_games):
        """
        Loads and filters tracking data.
        """
        filename = (
            "test_player_tracking.csv"
            if split_name == "test"
            else "train_player_tracking.csv"
        )
        path = os.path.join(self.config.input_dir, filename)

        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]

        df = pd.read_csv(path, usecols=cols)
        df = df[df["game_play"].isin(relevant_games)].copy()

        df["step"] = df["step"].astype(np.int32)
        float_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]
        for c in float_cols:
            df[c] = df[c].astype(np.float32)

        return df

    def create_wide_tracking(self, tracking_df):
        """
        Creates lagged features for tracking data using vectorized shifts.
        Cite: Lesson 00015 (Vectorized Lag-Shifting)
        """
        # Sort for correct shifting
        tracking_df = tracking_df.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).reset_index(drop=True)

        # Columns to lag
        lag_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]

        # GroupBy shift is safe for creating lags within player tracks
        grouped = tracking_df.groupby(["game_play", "nfl_player_id"])

        lagged_dfs = []

        # Lags: -half to +half
        offsets = range(-self.half_window, self.half_window + 1)

        for k in offsets:
            # Shift amount: if k=-5 (past), we need shift(5) to bring past value to current row.
            # if k=5 (future), we need shift(-5).
            shift_amount = -k

            # Select columns and shift
            shifted = grouped[lag_cols].shift(shift_amount)

            # Rename columns
            shifted.columns = [f"{c}_lag_{k}" for c in lag_cols]

            lagged_dfs.append(shifted)

        # Concatenate all lags horizontally
        wide_tracking = pd.concat(
            [tracking_df[["game_play", "step", "nfl_player_id"]]] + lagged_dfs, axis=1
        )

        return wide_tracking

    def process(self, metadata_df, tracking_df, split="train"):
        """
        Main processing pipeline: Wide Tracking -> Merge -> Impute -> Features -> Scale.
        """
        # 1. Create Wide Tracking Data
        print("  Creating vectorized lags...")
        wide_tracking = self.create_wide_tracking(tracking_df)

        # 2. Merge Player 1
        print("  Merging Player 1...")
        p1_cols = {
            c: f"{c}_1" for c in wide_tracking.columns if c not in ["game_play", "step"]
        }
        wide_p1 = wide_tracking.rename(columns=p1_cols)

        merged = metadata_df.merge(
            wide_p1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id_1_1"],
            how="left",
        )

        # 3. Merge Player 2
        print("  Merging Player 2...")
        merged["nfl_player_id_2_join"] = pd.to_numeric(
            merged["nfl_player_id_2"], errors="coerce"
        )

        p2_cols = {
            c: f"{c}_2" for c in wide_tracking.columns if c not in ["game_play", "step"]
        }
        wide_p2 = wide_tracking.rename(columns=p2_cols)

        merged = merged.merge(
            wide_p2,
            left_on=["game_play", "step", "nfl_player_id_2_join"],
            right_on=["game_play", "step", "nfl_player_id_2_2"],
            how="left",
        )

        # 4. Hybrid-Physics Ground Imputation
        # Cite: Lesson 00024 (Geometric Consistency) & Lesson 00034 (Kinematics Separation)
        print("  Imputing Ground physics...")
        is_ground = merged["nfl_player_id_2"] == "G"

        offsets = range(-self.half_window, self.half_window + 1)

        for k in offsets:
            # Position: Ground = Player 1
            merged.loc[is_ground, f"x_position_lag_{k}_2"] = merged.loc[
                is_ground, f"x_position_lag_{k}_1"
            ]
            merged.loc[is_ground, f"y_position_lag_{k}_2"] = merged.loc[
                is_ground, f"y_position_lag_{k}_1"
            ]

            # Kinematics: Ground = 0
            merged.loc[is_ground, f"speed_lag_{k}_2"] = 0.0
            merged.loc[is_ground, f"acceleration_lag_{k}_2"] = 0.0
            merged.loc[is_ground, f"orientation_lag_{k}_2"] = 0.0
            merged.loc[is_ground, f"direction_lag_{k}_2"] = 0.0

        # Fill NaNs (missing tracking)
        num_cols = [c for c in merged.columns if "_lag_" in c]
        merged[num_cols] = merged[num_cols].fillna(0.0)

        # 5. Feature Engineering (Derived Features)
        # Cite: Lesson 00005 (Non-Linear Transformations) & Lesson 00033 (Explicit Kinematics)
        print("  Calculating derived kinematic features...")

        feature_cols = []

        # Global features
        merged["is_ground"] = is_ground.astype(np.float32)
        feature_cols.append("is_ground")

        for k in offsets:
            suffix_1 = f"_lag_{k}_1"
            suffix_2 = f"_lag_{k}_2"

            # Distance
            dx = merged[f"x_position{suffix_1}"] - merged[f"x_position{suffix_2}"]
            dy = merged[f"y_position{suffix_1}"] - merged[f"y_position{suffix_2}"]
            dist = np.sqrt(dx**2 + dy**2)

            # Log Distance
            col_log_dist = f"log_dist_lag_{k}"
            merged[col_log_dist] = np.log1p(dist)
            feature_cols.append(col_log_dist)

            # Relative Speed
            dir_1 = np.radians(merged[f"direction{suffix_1}"])
            dir_2 = np.radians(merged[f"direction{suffix_2}"])

            vx_1 = merged[f"speed{suffix_1}"] * np.sin(dir_1)
            vy_1 = merged[f"speed{suffix_1}"] * np.cos(dir_1)
            vx_2 = merged[f"speed{suffix_2}"] * np.sin(dir_2)
            vy_2 = merged[f"speed{suffix_2}"] * np.cos(dir_2)

            dvx = vx_1 - vx_2
            dvy = vy_1 - vy_2
            rel_speed = np.sqrt(dvx**2 + dvy**2)

            col_rel_speed = f"rel_speed_lag_{k}"
            merged[col_rel_speed] = rel_speed
            feature_cols.append(col_rel_speed)

            # Closing Speed
            # Cite: Lesson 00007 (Numerical Stability)
            dot_prod = dvx * dx + dvy * dy
            clamped_dist = np.maximum(dist, 1e-6)
            closing_speed = -(dot_prod / clamped_dist)

            col_closing = f"closing_speed_lag_{k}"
            merged[col_closing] = closing_speed
            feature_cols.append(col_closing)

            # Add raw kinematics
            feature_cols.extend(
                [
                    f"speed{suffix_1}",
                    f"speed{suffix_2}",
                    f"acceleration{suffix_1}",
                    f"acceleration{suffix_2}",
                ]
            )

        # 6. Scaling
        # Cite: Lesson 00037 (Feature Scaling)
        print("  Scaling features...")
        X = merged[feature_cols].values.astype(np.float32)

        if split == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, self.config.scaler_path)
            print(f"  Scaler saved to {self.config.scaler_path}")
        else:
            if os.path.exists(self.config.scaler_path):
                scaler = joblib.load(self.config.scaler_path)
                X = scaler.transform(X)
            else:
                print(
                    "Warning: Scaler not found during validation/test. Using unscaled data."
                )

        return X


def load_and_process_data(
    split="train", debug=False, load_cached_data=True, config=None
):
    if config is None:
        config = Config()

    cache_prefix = f"{split}"
    if debug:
        cache_prefix += "_debug"

    path_X = os.path.join(config.artifact_dir, f"{cache_prefix}_X.npy")
    path_y = os.path.join(config.artifact_dir, f"{cache_prefix}_y.npy")
    path_ids = os.path.join(config.artifact_dir, f"{cache_prefix}_ids.npy")

    if load_cached_data and os.path.exists(path_X):
        print(f"Loading cached {split} data from {config.artifact_dir}...")
        X = np.load(path_X)

        # Cite debug_lesson_1: Verify Cache Consistency Against Runtime Configuration
        if X.ndim == 2:
            if split != "test":
                y = np.load(path_y)
            else:
                y = None

            meta_path = os.path.join(config.metadata_dir, f"{split}.csv")
            metadata_df = pd.read_csv(meta_path)
            if debug:
                metadata_df = metadata_df.iloc[: config.debug_sample_size].copy()

            return (
                NFLContactDataset(X, y, metadata_df["contact_id"].values),
                metadata_df,
            )

        print(
            f"  [Cache Invalid] Expected 2D data but found shape {X.shape}. Regenerating..."
        )

    print(f"Processing {split} data from scratch...")
    meta_path = os.path.join(config.metadata_dir, f"{split}.csv")
    metadata_df = pd.read_csv(meta_path)

    if debug:
        print(f"  Debug mode: Sampling {config.debug_sample_size} rows.")
        metadata_df = metadata_df.iloc[: config.debug_sample_size].copy()

    relevant_games = metadata_df["game_play"].unique()
    fe = FeatureEngineer(config)
    tracking_df = fe.load_tracking(split, relevant_games)

    X = fe.process(metadata_df, tracking_df, split=split)

    y = None
    if "contact" in metadata_df.columns and split != "test":
        y = metadata_df["contact"].values.astype(np.float32)

    print(f"Saving {split} data to cache...")
    np.save(path_X, X)
    if y is not None:
        np.save(path_y, y)
    np.save(path_ids, metadata_df["contact_id"].values)

    del tracking_df
    gc.collect()

    return NFLContactDataset(X, y, metadata_df["contact_id"].values), metadata_df
