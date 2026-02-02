import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import (
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    TRAIN_HELMETS_PATH,
    TEST_HELMETS_PATH,
    METADATA_TRAIN,
    METADATA_VAL,
    METADATA_TEST,
    WORKING_DIR,
    KINEMATIC_COLS,
    VISUAL_COLS,
    CAT_COLS,
    WINDOW_SIZE,
    CLAMP_MIN,
    CLAMP_MAX,
    SEED,
)
from library.utils import seed_everything


class DataProcessor:
    def __init__(self, debug=False):
        self.debug = debug
        self.scaler_path = os.path.join(WORKING_DIR, "scaler.joblib")
        self.encoders_path = os.path.join(WORKING_DIR, "encoders.joblib")
        seed_everything(SEED)

    def _load_metadata(self, split):
        if split == "train":
            return pd.read_csv(METADATA_TRAIN)
        elif split == "val":
            return pd.read_csv(METADATA_VAL)
        elif split == "test":
            return pd.read_csv(METADATA_TEST)
        else:
            raise ValueError(f"Unknown split: {split}")

    def preprocess_tracking(self, df_tracking, game_plays):
        """
        Generates windowed kinematic features with strict clamping.
        """
        # Filter relevant plays
        df = df_tracking[df_tracking["game_play"].isin(game_plays)].copy()

        # Sort for windowing
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Feature Engineering: Clamping
        # We clamp before windowing to ensure all inputs are stable
        for col in KINEMATIC_COLS:
            df[col] = df[col].clip(lower=CLAMP_MIN, upper=CLAMP_MAX)

        # Create Windows
        # We want [t-W, ..., t, ..., t+W]
        # We can use shifts.
        # Note: This assumes continuous steps. Tracking data is usually dense.
        # Ideally we reindex, but for efficiency we assume sort is sufficient and
        # we will merge on (game_play, nfl_player_id, step).

        feature_cols = []
        # We need to process each column for each lag
        # To do this efficiently without a loop over rows, we use shifts

        # Group by player to ensure shifts don't cross players
        grouped = df.groupby(["game_play", "nfl_player_id"])

        shifts = range(-WINDOW_SIZE, WINDOW_SIZE + 1)

        # We will construct a list of dataframes and concat
        dfs_shifted = []

        for s in shifts:
            # s < 0 means past (shift down), s > 0 means future (shift up)
            # pandas shift: positive shifts down (lag), negative shifts up (lead)
            # We want t-5 (past) to t+5 (future).
            # shift(5) gives value from t-5 at time t.
            # shift(-5) gives value from t+5 at time t.

            # Naming convention: col_lag_-5, col_lag_0, col_lag_5
            suffix = f"_lag_{s}"
            shifted = grouped[KINEMATIC_COLS].shift(s)  # shift(1) is t-1
            shifted.columns = [c + suffix for c in KINEMATIC_COLS]
            dfs_shifted.append(shifted)

        df_features = pd.concat(dfs_shifted, axis=1)

        # Add keys back
        df_features["game_play"] = df["game_play"]
        df_features["nfl_player_id"] = df["nfl_player_id"]
        df_features["step"] = df["step"]

        # Add categorical columns (no windowing needed, static per play/player usually, but tracking has them per step)
        # We take the value at time t (lag 0)
        for cat in CAT_COLS:
            df_features[cat] = df[cat]

        # Drop rows with NaNs created by shifting (edges of play)
        # Or fill them? Usually better to drop or fill.
        # Given the task, we might lose contact events at very start/end.
        # We will fill with 0 or nearest. Forward/Back fill is safer for tracking.
        # However, simple drop is standard if windows are small relative to play length.
        # Let's fill 0 for stability.
        df_features = df_features.fillna(0)

        return df_features

    def preprocess_helmets(self, df_helmets, game_plays):
        """
        Applies Max-Pooling Selection Strategy to visual features.
        """
        df = df_helmets[df_helmets["game_play"].isin(game_plays)].copy()

        # Map frame to step (Cite debug_lesson_17)
        # Snap is at frame 300 (5s * 59.94fps approx), Step is 10Hz
        df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

        # Calculate Area
        df["area"] = df["width"] * df["height"]

        # Max Pooling: Select row with max area per (game_play, step, nfl_player_id)
        # Sort by area descending
        df = df.sort_values("area", ascending=False)

        # Drop duplicates keeping first (max area)
        df_unique = df.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"], keep="first"
        )

        # Select features
        cols = ["game_play", "step", "nfl_player_id"] + VISUAL_COLS
        return df_unique[cols]

    def merge_and_format(self, df_meta, df_track, df_helm, is_train=True):
        """
        Merges labels with features and handles Ground imputation.
        """
        # Prepare Metadata
        # Ensure nfl_player_id_1 is numeric (it always is)
        # nfl_player_id_2 can be 'G'

        # 1. Merge Player 1 Tracking
        # Rename columns to indicate P1
        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        # Identify lag columns and cat columns

        df_merged = df_meta.merge(
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        # Rename P1 columns
        rename_dict_p1 = {c: c + "_p1" for c in track_cols}
        df_merged = df_merged.rename(columns=rename_dict_p1)

        # 2. Merge Player 2 Tracking
        # We need to handle 'G'. First merge on numeric ID.
        # Create a temp column for merge that is numeric, 'G' becomes NaN
        df_merged["p2_merge_id"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = df_merged.merge(
            df_track,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2_track"),
        ).drop(columns=["nfl_player_id", "p2_merge_id"])

        # Rename P2 columns
        rename_dict_p2 = {c: c + "_p2" for c in track_cols}
        df_merged = df_merged.rename(columns=rename_dict_p2)

        # 3. Ground Imputation (Kinematic)
        # If nfl_player_id_2 == 'G', set P2 pos = P1 pos, others = 0
        is_ground = df_merged["nfl_player_id_2"] == "G"

        if is_ground.any():
            # Identify columns
            # We have lag columns like x_position_lag_0_p1
            # We need to map p1 cols to p2 cols for imputation

            for col in KINEMATIC_COLS:
                for s in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                    col_name = f"{col}_lag_{s}"
                    p1_col = f"{col_name}_p1"
                    p2_col = f"{col_name}_p2"

                    if col in ["x_position", "y_position"]:
                        # Position: Copy P1 to P2
                        df_merged.loc[is_ground, p2_col] = df_merged.loc[
                            is_ground, p1_col
                        ]
                    else:
                        # Velocity/Accel/Angle: Set to 0
                        df_merged.loc[is_ground, p2_col] = 0.0

            # Handle Categorical for Ground
            for cat in CAT_COLS:
                p2_col = f"{cat}_p2"
                # We will assign a special token 'Ground' later, for now ensure string
                df_merged.loc[is_ground, p2_col] = "Ground"

        # 4. Merge Helmets P1
        df_merged = df_merged.merge(
            df_helm,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        rename_dict_vis_p1 = {c: c + "_p1" for c in VISUAL_COLS}
        df_merged = df_merged.rename(columns=rename_dict_vis_p1)

        # 5. Merge Helmets P2
        df_merged["p2_merge_id"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )
        df_merged = df_merged.merge(
            df_helm,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2_vis"),
        ).drop(columns=["nfl_player_id", "p2_merge_id"])

        rename_dict_vis_p2 = {c: c + "_p2" for c in VISUAL_COLS}
        df_merged = df_merged.rename(columns=rename_dict_vis_p2)

        # 6. Ground Imputation (Visual)
        # Set P2 visual features to 0 for Ground
        if is_ground.any():
            for c in VISUAL_COLS:
                p2_col = c + "_p2"
                df_merged.loc[is_ground, p2_col] = 0.0

        # Fill remaining NaNs (missing tracking/helmets) with 0
        # This is safe for neural nets with standardization
        df_merged = df_merged.fillna(0)

        return df_merged

    def process_split(self, split, load_cached_data=True):
        """
        Main processing function for a split.
        """
        cache_file = os.path.join(WORKING_DIR, f"features_{split}.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} data from cache...")
            return pd.read_parquet(cache_file)

        print(f"Processing {split} data from scratch...")

        # Load Metadata
        df_meta = self._load_metadata(split)
        if self.debug:
            df_meta = df_meta.sample(n=min(len(df_meta), 10000), random_state=SEED)

        unique_game_plays = df_meta["game_play"].unique()

        # Load Raw Data
        if split == "test":
            track_path = TEST_TRACKING_PATH
            helm_path = TEST_HELMETS_PATH
        else:
            track_path = TRAIN_TRACKING_PATH
            helm_path = TRAIN_HELMETS_PATH

        df_tracking = pd.read_csv(track_path)
        df_helmets = pd.read_csv(helm_path)

        # Preprocess
        df_track_proc = self.preprocess_tracking(df_tracking, unique_game_plays)
        df_helm_proc = self.preprocess_helmets(df_helmets, unique_game_plays)

        # Merge
        df_final = self.merge_and_format(
            df_meta, df_track_proc, df_helm_proc, is_train=(split != "test")
        )

        # Save Cache
        df_final.to_parquet(cache_file)

        return df_final

    def fit_transform_scalers(self, df_train):
        """
        Fits scalers on training data and transforms it.
        """
        # Identify feature columns
        # Kinematic: Windowed columns for P1 and P2
        # Visual: Visual columns for P1 and P2

        kin_cols = []
        for col in KINEMATIC_COLS:
            for s in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                kin_cols.append(f"{col}_lag_{s}_p1")
                kin_cols.append(f"{col}_lag_{s}_p2")

        vis_cols = []
        for col in VISUAL_COLS:
            vis_cols.append(f"{col}_p1")
            vis_cols.append(f"{col}_p2")

        # Fit StandardScaler
        scaler_kin = StandardScaler()
        scaler_vis = StandardScaler()

        # Fit on Train
        # Convert to float32 to save memory
        X_kin = df_train[kin_cols].astype(np.float32).values
        X_vis = df_train[vis_cols].astype(np.float32).values

        scaler_kin.fit(X_kin)
        scaler_vis.fit(X_vis)

        # Save
        joblib.dump({"kin": scaler_kin, "vis": scaler_vis}, self.scaler_path)

        # Encode Categoricals
        # We need to handle 'Ground' and unknown values
        # P1 and P2 share the same encoders
        encoders = {}
        for cat in CAT_COLS:
            le = LabelEncoder()
            # Gather all possible values from P1 and P2
            vals = (
                pd.concat([df_train[f"{cat}_p1"], df_train[f"{cat}_p2"]])
                .astype(str)
                .unique()
            )
            # Ensure 'Ground' is in there if not already
            if "Ground" not in vals:
                vals = np.append(vals, "Ground")

            le.fit(vals)
            encoders[cat] = le

        joblib.dump(encoders, self.encoders_path)

        return self._transform(
            df_train, scaler_kin, scaler_vis, encoders, kin_cols, vis_cols
        )

    def transform(self, df):
        """
        Transforms data using saved scalers.
        """
        scalers = joblib.load(self.scaler_path)
        encoders = joblib.load(self.encoders_path)

        kin_cols = []
        for col in KINEMATIC_COLS:
            for s in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                kin_cols.append(f"{col}_lag_{s}_p1")
                kin_cols.append(f"{col}_lag_{s}_p2")

        vis_cols = []
        for col in VISUAL_COLS:
            vis_cols.append(f"{col}_p1")
            vis_cols.append(f"{col}_p2")

        return self._transform(
            df, scalers["kin"], scalers["vis"], encoders, kin_cols, vis_cols
        )

    def _transform(self, df, scaler_kin, scaler_vis, encoders, kin_cols, vis_cols):
        X_kin = df[kin_cols].astype(np.float32).values
        X_vis = df[vis_cols].astype(np.float32).values

        X_kin = scaler_kin.transform(X_kin)
        X_vis = scaler_vis.transform(X_vis)

        # Categorical
        X_cat_list = []
        for cat in CAT_COLS:
            le = encoders[cat]
            # Handle unseen labels by mapping to a known one or mode?
            # Simple approach: Convert to string, map unknown to 'Ground' or first class
            # We'll use a safe transform helper

            def safe_transform(series, le):
                s = series.astype(str)
                # Find values not in classes
                mask = ~s.isin(le.classes_)
                if mask.any():
                    # Map to 'Ground' if available, else 0
                    fill_val = "Ground" if "Ground" in le.classes_ else le.classes_[0]
                    s[mask] = fill_val
                return le.transform(s)

            p1_cat = safe_transform(df[f"{cat}_p1"], le)
            p2_cat = safe_transform(df[f"{cat}_p2"], le)
            X_cat_list.append(p1_cat.reshape(-1, 1))
            X_cat_list.append(p2_cat.reshape(-1, 1))

        X_cat = np.hstack(
            X_cat_list
        )  # [N, 4] -> P1_Pos, P2_Pos, P1_Team, P2_Team order depends on loop

        # Extract Targets if available
        if "contact" in df.columns:
            y = df["contact"].values.astype(np.float32)
        else:
            y = np.zeros(len(df), dtype=np.float32)

        # Extract IDs for submission
        ids = df["contact_id"].values if "contact_id" in df.columns else None

        return {
            "X_kin": torch.tensor(X_kin, dtype=torch.float32),
            "X_vis": torch.tensor(X_vis, dtype=torch.float32),
            "X_cat": torch.tensor(X_cat, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
            "ids": ids,
        }

    def get_data(self, split="train", load_cached_data=True):
        """
        Public API to get processed tensors.
        """
        df = self.process_split(split, load_cached_data)

        if split == "train":
            return self.fit_transform_scalers(df)
        else:
            return self.transform(df)
