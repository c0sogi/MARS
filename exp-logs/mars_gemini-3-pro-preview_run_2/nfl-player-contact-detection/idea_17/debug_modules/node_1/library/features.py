import os
import gc
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything


class FeatureEngine:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

    def load_raw_data(self, split="train"):
        """
        Loads raw tracking, helmet, and metadata files.
        """
        print(f"Loading raw data for {split}...")
        if split == "train":
            meta_path = self.config.METADATA_TRAIN
            tracking_path = self.config.TRAIN_TRACKING_PATH
            helmets_path = self.config.TRAIN_HELMETS_PATH
            video_meta_path = self.config.TRAIN_VIDEO_META_PATH
        elif split == "validation":
            meta_path = self.config.METADATA_VAL
            # Validation uses train tracking/helmets but filtered by game_play
            tracking_path = self.config.TRAIN_TRACKING_PATH
            helmets_path = self.config.TRAIN_HELMETS_PATH
            video_meta_path = self.config.TRAIN_VIDEO_META_PATH
        else:  # test
            meta_path = self.config.METADATA_TEST
            tracking_path = self.config.TEST_TRACKING_PATH
            helmets_path = self.config.TEST_HELMETS_PATH
            video_meta_path = self.config.TEST_VIDEO_META_PATH

        # Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Debugging: Sample data if configured
        if not self.config.USE_ALL_DATA and split == "train":
            print(f"DEBUG: Sampling {self.config.DEBUG_SAMPLE_SIZE} rows...")
            game_plays = df_meta["game_play"].unique()
            sample_gps = np.random.choice(
                game_plays, min(len(game_plays), 20), replace=False
            )
            df_meta = df_meta[df_meta["game_play"].isin(sample_gps)].copy()

        # Load Tracking
        # Only load tracking for relevant game_plays
        req_gps = df_meta["game_play"].unique()
        df_tracking = pd.read_csv(tracking_path)
        df_tracking = df_tracking[df_tracking["game_play"].isin(req_gps)].copy()

        # Load Helmets
        df_helmets = pd.read_csv(helmets_path)
        df_helmets = df_helmets[df_helmets["game_play"].isin(req_gps)].copy()

        # Load Video Metadata
        df_vid_meta = pd.read_csv(video_meta_path)
        df_vid_meta = df_vid_meta[df_vid_meta["game_play"].isin(req_gps)].copy()

        return df_meta, df_tracking, df_helmets, df_vid_meta

    def preprocess_helmets(self, df_helmets):
        """
        Derives geometric features and aggregates duplicate views.
        """
        # Filter views - stick to reliable Sideline/Endzone
        df_helmets = df_helmets[df_helmets["view"].isin(["Sideline", "Endzone"])].copy()

        # Derived features
        df_helmets["helmet_area"] = df_helmets["width"] * df_helmets["height"]
        df_helmets["helmet_aspect_ratio"] = df_helmets["width"] / df_helmets["height"]
        df_helmets["helmet_centroid_x"] = df_helmets["left"] + df_helmets["width"] / 2
        df_helmets["helmet_centroid_y"] = df_helmets["top"] + df_helmets["height"] / 2

        # Aggregate duplicates (same player, same frame, different view)
        # We take the mean of features
        agg_cols = self.config.HELMET_FEATS + self.config.VISUAL_DERIVED_FEATS

        # Ensure nfl_player_id is string
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

        df_agg = (
            df_helmets.groupby(["game_play", "frame", "nfl_player_id"])[agg_cols]
            .mean()
            .reset_index()
        )
        return df_agg

    def align_visual_data(self, df_tracking, df_helmets, df_vid_meta):
        """
        Maps video frames to tracking steps using timestamps.
        """
        # Prepare Video Meta
        # We need start_time to align.
        # Note: Datetime formats can vary. Assuming ISO format or standard pd.to_datetime compatible.
        try:
            df_vid_meta["start_time"] = pd.to_datetime(
                df_vid_meta["start_time"], utc=True
            )
        except:
            # Fallback if parsing fails or format is simple string
            df_vid_meta["start_time"] = pd.to_datetime(df_vid_meta["start_time"])

        # Prepare Tracking
        # Ensure datetime is datetime object
        df_tracking["datetime"] = pd.to_datetime(df_tracking["datetime"], utc=True)

        # Merge start_time onto tracking
        # We use the 'Sideline' start time as reference (usually synced with Endzone)
        # If multiple views, drop duplicates for metadata
        meta_ref = df_vid_meta.drop_duplicates(subset=["game_play"])[
            ["game_play", "start_time"]
        ]

        df_merged = df_tracking.merge(meta_ref, on="game_play", how="left")

        # Calculate Frame
        # frame = (current_time - start_time) * 59.94
        # We add a small epsilon or round to nearest
        time_diff = (df_merged["datetime"] - df_merged["start_time"]).dt.total_seconds()
        df_merged["frame"] = (time_diff * 59.94).round().astype(int)

        # Handle cases where frame < 1 (before video start)
        df_merged["frame"] = df_merged["frame"].clip(lower=1)

        # Merge Helmet Features
        # Keys: game_play, frame, nfl_player_id
        df_merged["nfl_player_id"] = df_merged["nfl_player_id"].astype(str)

        # Perform Left Join
        df_final = df_merged.merge(
            df_helmets, on=["game_play", "frame", "nfl_player_id"], how="left"
        )

        # Impute missing helmet features with 0 or -1
        for col in self.config.HELMET_FEATS + self.config.VISUAL_DERIVED_FEATS:
            if col not in df_final.columns:
                df_final[col] = -1
            else:
                df_final[col] = df_final[col].fillna(-1)

        return df_final

    def engineer_features(self, df_meta, df_tracking_enriched):
        """
        Constructs pairwise interaction features and handles Ground imputation.
        """
        # Ensure IDs are strings
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_tracking_enriched["nfl_player_id"] = df_tracking_enriched[
            "nfl_player_id"
        ].astype(str)

        # We need to attach tracking data for P1 and P2 to the metadata rows
        # Metadata defines the pairs and steps.

        # Columns to keep from tracking
        track_cols = (
            ["game_play", "step", "nfl_player_id"]
            + self.config.TRACKING_FEATS
            + self.config.HELMET_FEATS
            + self.config.VISUAL_DERIVED_FEATS
        )

        # 1. Merge P1
        df_merged = df_meta.merge(
            df_tracking_enriched[track_cols],
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(
            columns=["nfl_player_id"]
        )  # Drop redundant join key

        # Rename P1 columns
        p1_suffix = "_1"
        rename_dict_p1 = {
            c: c + p1_suffix
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_dict_p1)

        # 2. Merge P2 (Players only first)
        # We split P2 into 'G' and Players to handle merge efficiently
        is_ground = df_merged["nfl_player_id_2"] == self.config.GROUND_ID

        df_players = df_merged[~is_ground].copy()
        df_ground = df_merged[is_ground].copy()

        # Merge P2 for players
        df_players = df_players.merge(
            df_tracking_enriched[track_cols],
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_2"),  # P1 cols already suffixed, this adds _2 to new cols
        ).drop(columns=["nfl_player_id"])

        # Rename P2 columns explicitly to ensure consistency
        # The merge suffixes might not catch everything if P1 cols were renamed manually
        # Let's verify: P1 cols are "speed_1". Incoming col is "speed". Merge makes it "speed".
        # We need to rename the new columns to "_2"
        new_cols = [
            c for c in track_cols if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_dict_p2 = {c: c + "_2" for c in new_cols}
        df_players = df_players.rename(columns=rename_dict_p2)

        # 3. Impute Ground P2
        # Ground P2 inherits P1 position, 0 velocity, and null visual features
        for col in new_cols:
            col_p1 = col + "_1"
            col_p2 = col + "_2"

            if "position" in col:
                # x_position_2 = x_position_1
                df_ground[col_p2] = df_ground[col_p1]
            elif col in ["speed", "acceleration", "sa"]:
                df_ground[col_p2] = 0.0
            elif col in ["direction", "orientation"]:
                df_ground[col_p2] = df_ground[col_p1]  # Relative angle 0
            else:
                # Visual features -> -1 or 0
                df_ground[col_p2] = -1.0

        # Concatenate back
        df_full = pd.concat([df_players, df_ground], axis=0).sort_index()

        # 4. Compute Interaction Features
        # Distance
        df_full["dx"] = df_full["x_position_1"] - df_full["x_position_2"]
        df_full["dy"] = df_full["y_position_1"] - df_full["y_position_2"]
        df_full["distance"] = np.sqrt(df_full["dx"] ** 2 + df_full["dy"] ** 2)
        df_full["log_distance"] = np.log1p(df_full["distance"])

        # Speed/Accel Diff
        df_full["speed_diff"] = df_full["speed_1"] - df_full["speed_2"]
        df_full["accel_diff"] = df_full["acceleration_1"] - df_full["acceleration_2"]

        # Visual IoU
        # Box: left, top, width, height
        # x1 = left, x2 = left + width
        # y1 = top, y2 = top + height

        def compute_iou(row):
            # If either helmet is missing (marked -1), IoU is 0
            if row["width_1"] <= 0 or row["width_2"] <= 0:
                return 0.0

            xA = max(row["left_1"], row["left_2"])
            yA = max(row["top_1"], row["top_2"])
            xB = min(row["left_1"] + row["width_1"], row["left_2"] + row["width_2"])
            yB = min(row["top_1"] + row["height_1"], row["top_2"] + row["height_2"])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = row["width_1"] * row["height_1"]
            boxBArea = row["width_2"] * row["height_2"]

            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou

        # Vectorized IoU
        # Create masks for missing helmets
        missing_mask = (df_full["width_1"] <= 0) | (df_full["width_2"] <= 0)

        xA = np.maximum(df_full["left_1"], df_full["left_2"])
        yA = np.maximum(df_full["top_1"], df_full["top_2"])
        xB = np.minimum(
            df_full["left_1"] + df_full["width_1"],
            df_full["left_2"] + df_full["width_2"],
        )
        yB = np.minimum(
            df_full["top_1"] + df_full["height_1"],
            df_full["top_2"] + df_full["height_2"],
        )

        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        boxAArea = df_full["width_1"] * df_full["height_1"]
        boxBArea = df_full["width_2"] * df_full["height_2"]

        df_full["visual_iou"] = interArea / (boxAArea + boxBArea - interArea + 1e-6)
        df_full.loc[missing_mask, "visual_iou"] = 0.0

        # Visual Pixel Distance
        df_full["visual_dist_pixel"] = np.sqrt(
            (df_full["helmet_centroid_x_1"] - df_full["helmet_centroid_x_2"]) ** 2
            + (df_full["helmet_centroid_y_1"] - df_full["helmet_centroid_y_2"]) ** 2
        )
        df_full.loc[missing_mask, "visual_dist_pixel"] = -1.0

        return df_full

    def create_wide_input(self, df):
        """
        Flattens temporal window into a single feature vector.
        """
        # Features to include in the window
        # P1 feats, P2 feats, Interaction feats
        base_feats = [c for c in df.columns if c.endswith("_1") or c.endswith("_2")]
        base_feats += self.config.INTERACTION_FEATS

        # Ensure we don't have duplicates or non-numeric
        # Exclude ID columns to prevent mixed-type errors (str vs int 0 from fillna)
        exclude_cols = ["nfl_player_id_1", "nfl_player_id_2"]
        base_feats = [
            c for c in base_feats if c in df.columns and c not in exclude_cols
        ]

        # Sort for shifting
        # Group by pair: game_play, p1, p2
        df = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Generate lags
        shifts = range(
            -self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1
        )  # e.g. -5 to +5

        shifted_dfs = []
        for s in shifts:
            # Shift features
            shifted = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])[
                base_feats
            ].shift(-s)

            # Rename columns
            suffix = f"_t{s:+d}" if s != 0 else "_t0"
            shifted.columns = [f"{col}{suffix}" for col in shifted.columns]
            shifted_dfs.append(shifted)

        # Concatenate all shifted features
        df_wide = pd.concat(shifted_dfs, axis=1)

        # Add metadata back
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        # Note: 'contact' might not exist in test
        available_meta = [c for c in meta_cols if c in df.columns]

        df_final = pd.concat([df[available_meta], df_wide], axis=1)

        # Drop rows with NaNs created by shifting (edges of play)
        # Or fill them? Usually fill with 0 or nearest.
        # Given "Wide Input" usually requires valid context.
        # However, for test set, we must predict every row.
        # We fill NaNs with 0 (padding).
        df_final = df_final.fillna(0)

        return df_final

    def process_split(self, split):
        """
        Orchestrates the pipeline for a specific split.
        """
        # Check cache
        cache_path = os.path.join(self.config.CACHE_DIR, f"{split}_features.parquet")

        # 1. Load Raw
        df_meta, df_tracking, df_helmets, df_vid_meta = self.load_raw_data(split)

        # 2. Preprocess Helmets
        print(f"[{split}] Preprocessing helmets...")
        df_helmets_proc = self.preprocess_helmets(df_helmets)

        # 3. Align Tracking & Visuals
        print(f"[{split}] Aligning visual data...")
        df_tracking_proc = self.align_visual_data(
            df_tracking, df_helmets_proc, df_vid_meta
        )

        # Free memory
        del df_tracking, df_helmets, df_vid_meta
        gc.collect()

        # 4. Engineer Features (Chunked by GamePlay to save RAM)
        print(f"[{split}] Engineering features...")
        game_plays = df_meta["game_play"].unique()
        chunk_size = 5  # Process 5 plays at a time

        processed_chunks = []

        for i in range(0, len(game_plays), chunk_size):
            gp_chunk = game_plays[i : i + chunk_size]

            meta_chunk = df_meta[df_meta["game_play"].isin(gp_chunk)].copy()
            track_chunk = df_tracking_proc[
                df_tracking_proc["game_play"].isin(gp_chunk)
            ].copy()

            # Compute interactions
            df_interact = self.engineer_features(meta_chunk, track_chunk)

            # Create wide input
            df_wide = self.create_wide_input(df_interact)

            processed_chunks.append(df_wide)

        # Concatenate
        df_final = pd.concat(processed_chunks, axis=0, ignore_index=True)

        # Save to cache
        print(f"[{split}] Saving to {cache_path}...")
        df_final.to_parquet(cache_path, index=False)

        return df_final

    def generate_features(self, load_cached_data=True):
        """
        Main entry point. Handles caching logic.
        """
        splits = ["train", "validation", "test"]
        results = {}

        for split in splits:
            cache_path = os.path.join(
                self.config.CACHE_DIR, f"{split}_features.parquet"
            )

            if load_cached_data and os.path.exists(cache_path):
                print(f"Loading cached {split} features from {cache_path}...")
                results[split] = pd.read_parquet(cache_path)
            else:
                print(f"Generating {split} features from scratch...")
                results[split] = self.process_split(split)

        return results["train"], results["validation"], results["test"]


def get_data(load_cached_data=True):
    """
    Wrapper function to be compatible with potential external calls.
    Returns X_train, y_train, X_val, y_val, X_test, test_ids
    """
    engine = FeatureEngine()
    train_df, val_df, test_df = engine.generate_features(
        load_cached_data=load_cached_data
    )

    # Separate Features and Targets
    # Identify feature columns (exclude metadata)
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
    ]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    print(f"Feature count: {len(feature_cols)}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["contact"].values.astype(np.float32)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["contact"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    test_ids = test_df["contact_id"].values

    return X_train, y_train, X_val, y_val, X_test, test_ids
