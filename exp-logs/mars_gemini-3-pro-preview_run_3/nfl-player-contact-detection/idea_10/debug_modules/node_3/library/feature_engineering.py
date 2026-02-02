import pandas as pd
import numpy as np
import os
import gc
from typing import List, Tuple, Optional

from library.config import Config
from library.utils import reduce_mem_usage

# =============================================================================
# Helper Functions
# =============================================================================


def compute_kinematics(df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Computes advanced kinematic features: Jerk and Orientation-Motion Alignment.
    Assumes df_tracking is sorted by game_play, nfl_player_id, step.
    """
    # Ensure sorted
    df_tracking = df_tracking.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Group by play and player to calculate diffs
    # Jerk = d(Acceleration) / dt. dt = 0.1s.
    # We can just take the diff of acceleration since dt is constant.
    # Using shift to avoid leakage across groups (though groupby apply is safer but slower)
    # Vectorized approach with groupby shift

    grp = df_tracking.groupby(["game_play", "nfl_player_id"])

    # Jerk
    df_tracking["jerk"] = grp["acceleration"].diff().fillna(0) / 0.1

    # Orientation-Motion Alignment
    # Orientation: 0-360 degrees. 0 is North (y-axis positive?), usually standard is 0=X or 0=Y.
    # Tracking data: 0 is usually Y-axis (short axis) or X-axis depending on standardization.
    # We assume standard unit circle or consistent reference frame.
    # Alignment = cos(orientation - direction)

    # Convert to radians
    # Note: If orientation/direction are NaN, alignment will be NaN. Fill with 0.
    ori_rad = np.radians(df_tracking["orientation"].fillna(0))
    dir_rad = np.radians(df_tracking["direction"].fillna(0))

    # Dot product logic: cos(theta) where theta is angle difference
    df_tracking["alignment"] = np.cos(ori_rad - dir_rad)

    return df_tracking


def create_lag_features(
    df: pd.DataFrame, feature_cols: List[str], lags: List[int]
) -> pd.DataFrame:
    """
    Creates flattened temporal window features using shifts.
    df must be sorted by game_play, nfl_player_id, step.
    """
    # We will pivot or simply join shifted versions.
    # Given the memory constraints and dataframe size, creating columns iteratively is efficient enough.

    out_df = df[["game_play", "nfl_player_id", "step"]].copy()

    grp = df.groupby(["game_play", "nfl_player_id"])

    for lag in lags:
        shifted = grp[feature_cols].shift(
            -lag
        )  # shift(-1) is t+1 (future), shift(1) is t-1 (past)
        # Wait, pandas shift(1) moves data down, so row t gets data from t-1.
        # We want at row t to have data from t-lag.
        # If lag is -4 (past), we want shift(4).
        # Let's standardize: lag k means t+k.
        # So lag -4 means t-4. We need shift(4).
        # Lag +4 means t+4. We need shift(-4).

        shift_amount = -lag
        suffix = f"_lag{lag}"

        shifted.columns = [c + suffix for c in feature_cols]
        out_df = pd.concat([out_df, shifted], axis=1)

    return out_df


def calculate_iou(box1_df, box2_df):
    """
    Vectorized IoU calculation.
    Inputs are DataFrames with columns [left, width, top, height]
    """
    # Box: x, y, w, h
    # x1, y1 is top-left. x2, y2 is bottom-right.

    b1_x1 = box1_df["left"]
    b1_y1 = box1_df["top"]
    b1_x2 = b1_x1 + box1_df["width"]
    b1_y2 = b1_y1 + box1_df["height"]

    b2_x1 = box2_df["left"]
    b2_y1 = box2_df["top"]
    b2_x2 = b2_x1 + box2_df["width"]
    b2_y2 = b2_y1 + box2_df["height"]

    # Intersection
    xi1 = np.maximum(b1_x1, b2_x1)
    yi1 = np.maximum(b1_y1, b2_y1)
    xi2 = np.minimum(b1_x2, b2_x2)
    yi2 = np.minimum(b1_y2, b2_y2)

    inter_width = np.maximum(0, xi2 - xi1)
    inter_height = np.maximum(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Union
    b1_area = box1_df["width"] * box1_df["height"]
    b2_area = box2_df["width"] * box2_df["height"]
    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    # Centroid Distance (Normalized by image diag or just pixels? Pixels is fine for tree models)
    c1_x = b1_x1 + box1_df["width"] / 2
    c1_y = b1_y1 + box1_df["height"] / 2
    c2_x = b2_x1 + box2_df["width"] / 2
    c2_y = b2_y1 + box2_df["height"] / 2

    dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    return iou, dist


# =============================================================================
# Main Feature Engineering Class
# =============================================================================


class FeatureEngineer:
    def __init__(self, config=Config):
        self.config = config
        self.tracking_cols = config.TRACKING_COLS + ["jerk", "alignment"]

        # Define lags: Micro (-4 to +4) and Macro (-15, +15)
        # We use a dense micro window and sparse macro bounds
        self.lags = list(range(-4, 5)) + [-15, 15]

    def load_and_prep_tracking(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = reduce_mem_usage(df)

        # Compute Kinematics
        df = compute_kinematics(df)

        return df

    def get_lagged_tracking(self, df_tracking: pd.DataFrame) -> pd.DataFrame:
        # Generate lag features
        # We need to preserve keys to merge back
        df_lags = create_lag_features(df_tracking, self.tracking_cols, self.lags)
        return df_lags

    def process_stream_a(
        self,
        meta_df: pd.DataFrame,
        df_tracking: pd.DataFrame,
        df_helmets: pd.DataFrame,
        cache_path: str,
        load_cached_data: bool,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Stream A features from {cache_path}")
            # Load parquet
            full_df = pd.read_parquet(cache_path)
            # Separate X and y
            # Assuming 'contact' is the target. If test set, it might be placeholder.
            y = full_df["contact"].values
            X = full_df.drop(
                columns=[
                    "contact",
                    "contact_id",
                    "game_play",
                    "step",
                    "nfl_player_id_1",
                    "nfl_player_id_2",
                    "datetime",
                    "video_path_sideline",
                    "video_path_endzone",
                    "video_path_all29",
                ],
                errors="ignore",
            )
            return X, y

        print("Generating Stream A features...")

        # Filter metadata for Stream A (Player vs Player)
        # nfl_player_id_2 is not 'G'
        stream_a_meta = meta_df[meta_df["nfl_player_id_2"] != "G"].copy()

        # Prepare Lagged Tracking
        df_lagged = self.get_lagged_tracking(df_tracking)

        # 1. Merge P1 Tracking
        # Rename columns to p1
        df_p1 = df_lagged.add_suffix("_p1")
        # Fix join keys
        df_p1 = df_p1.rename(
            columns={
                "game_play_p1": "game_play",
                "step_p1": "step",
                "nfl_player_id_p1": "nfl_player_id_1",
            }
        )

        # Ensure join keys are numeric to prevent merge errors
        stream_a_meta["nfl_player_id_1"] = pd.to_numeric(
            stream_a_meta["nfl_player_id_1"], errors="coerce"
        )
        df_p1["nfl_player_id_1"] = pd.to_numeric(
            df_p1["nfl_player_id_1"], errors="coerce"
        )

        merged = stream_a_meta.merge(
            df_p1, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # 2. Merge P2 Tracking
        df_p2 = df_lagged.add_suffix("_p2")
        df_p2 = df_p2.rename(
            columns={
                "game_play_p2": "game_play",
                "step_p2": "step",
                "nfl_player_id_p2": "nfl_player_id_2",
            }
        )

        # Ensure nfl_player_id_2 is same type
        merged["nfl_player_id_2"] = merged["nfl_player_id_2"].astype(str)
        df_p2["nfl_player_id_2"] = df_p2["nfl_player_id_2"].astype(str)

        merged = merged.merge(
            df_p2, on=["game_play", "step", "nfl_player_id_2"], how="left"
        )

        # 3. Interaction Features (Current Time t, i.e., lag0)
        # We use lag0 columns.
        # Columns are named like 'speed_lag0_p1'

        dx = merged["x_position_lag0_p1"] - merged["x_position_lag0_p2"]
        dy = merged["y_position_lag0_p1"] - merged["y_position_lag0_p2"]
        merged["dist_p1_p2"] = np.sqrt(dx**2 + dy**2)

        merged["speed_diff"] = np.abs(merged["speed_lag0_p1"] - merged["speed_lag0_p2"])

        # Closure Rate: -(v1 - v2) . (r1 - r2) / |r1 - r2|
        # Simple approx: derivative of distance (not implemented here) or just speed diff
        # Let's stick to simple robust features.

        # 4. Visual Features (Sideline & Endzone)
        # We need to merge helmet boxes for P1 and P2 at current step

        # Filter helmets to relevant plays to speed up
        relevant_plays = merged["game_play"].unique()
        df_helmets_filt = df_helmets[
            df_helmets["game_play"].isin(relevant_plays)
        ].copy()

        # Ensure IDs are strings
        df_helmets_filt["nfl_player_id"] = df_helmets_filt["nfl_player_id"].astype(str)

        for view in ["Sideline", "Endzone"]:
            view_helmets = df_helmets_filt[df_helmets_filt["view"] == view]

            # Merge P1 Helmets
            h_p1 = view_helmets.add_suffix(f"_{view}_p1")
            h_p1 = h_p1.rename(
                columns={
                    f"game_play_{view}_p1": "game_play",
                    f"frame_{view}_p1": "frame",  # Helmets are by frame, Tracking by step.
                    # Wait, Helmets file has 'frame'. Tracking has 'step'.
                    # Metadata has 'step'. We need to map step -> frame?
                    # The description says: "Sideline and Endzone video pairs are matched frame for frame in time... videos contain frame rate of 59.94 HZ. The moment of snap occurs 5 seconds into the video."
                    # Tracking step is 10Hz. Step 0 is snap.
                    # So Step 0 ~= Frame 300 (5s * 60fps).
                    # Step S (0.1s) -> Time = S * 0.1s. Frame = 300 + (S * 0.1 * 59.94).
                    f"nfl_player_id_{view}_p1": "nfl_player_id_1",
                }
            )

            # We need to map step to frame in merged df
            # Frame = 300 + step * 0.1 * 59.94
            # Round to nearest integer
            merged["est_frame"] = (
                (300 + merged["step"] * 0.1 * 59.94).round().astype(int)
            )

            # Now merge helmets on (game_play, est_frame, nfl_player_id)
            # Note: Helmet data might not have every frame or perfect sync.
            # We use left join.

            # P1
            merged = merged.merge(
                h_p1[
                    [
                        f"left_{view}_p1",
                        f"width_{view}_p1",
                        f"top_{view}_p1",
                        f"height_{view}_p1",
                        "game_play",
                        "frame",
                        "nfl_player_id_1",
                    ]
                ],
                left_on=["game_play", "est_frame", "nfl_player_id_1"],
                right_on=["game_play", "frame", "nfl_player_id_1"],
                how="left",
            ).drop(columns=["frame"])

            # P2
            h_p2 = view_helmets.add_suffix(f"_{view}_p2")
            h_p2 = h_p2.rename(
                columns={
                    f"game_play_{view}_p2": "game_play",
                    f"frame_{view}_p2": "frame",
                    f"nfl_player_id_{view}_p2": "nfl_player_id_2",
                }
            )

            merged = merged.merge(
                h_p2[
                    [
                        f"left_{view}_p2",
                        f"width_{view}_p2",
                        f"top_{view}_p2",
                        f"height_{view}_p2",
                        "game_play",
                        "frame",
                        "nfl_player_id_2",
                    ]
                ],
                left_on=["game_play", "est_frame", "nfl_player_id_2"],
                right_on=["game_play", "frame", "nfl_player_id_2"],
                how="left",
            ).drop(columns=["frame"])

            # Compute IoU and Dist
            # Create temp dfs for vectorized calc
            box1 = merged[
                [
                    f"left_{view}_p1",
                    f"width_{view}_p1",
                    f"top_{view}_p1",
                    f"height_{view}_p1",
                ]
            ].rename(columns=lambda x: x.replace(f"_{view}_p1", ""))
            box2 = merged[
                [
                    f"left_{view}_p2",
                    f"width_{view}_p2",
                    f"top_{view}_p2",
                    f"height_{view}_p2",
                ]
            ].rename(columns=lambda x: x.replace(f"_{view}_p2", ""))

            iou, dist = calculate_iou(box1, box2)

            merged[f"{view}_IoU"] = iou
            merged[f"{view}_Dist"] = dist

            # Impute missing visuals
            # If box is NaN, IoU is NaN. Fill with Sentinel.
            merged[f"{view}_IoU"] = merged[f"{view}_IoU"].fillna(
                self.config.VISUAL_SENTINEL_VALUE
            )
            merged[f"{view}_Dist"] = merged[f"{view}_Dist"].fillna(
                self.config.VISUAL_SENTINEL_VALUE
            )

            # Drop raw box cols to save memory
            drop_cols = [c for c in merged.columns if f"_{view}_p" in c]
            merged = merged.drop(columns=drop_cols)

        # Cleanup
        merged = merged.drop(columns=["est_frame"], errors="ignore")

        # Save to cache
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        merged.to_parquet(cache_path)

        y = merged["contact"].values
        X = merged.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
                "datetime",
                "video_path_sideline",
                "video_path_endzone",
                "video_path_all29",
            ],
            errors="ignore",
        )

        return X, y

    def process_stream_b(
        self,
        meta_df: pd.DataFrame,
        df_tracking: pd.DataFrame,
        cache_path: str,
        load_cached_data: bool,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Stream B features from {cache_path}")
            full_df = pd.read_parquet(cache_path)
            y = full_df["contact"].values
            X = full_df.drop(
                columns=[
                    "contact",
                    "contact_id",
                    "game_play",
                    "step",
                    "nfl_player_id_1",
                    "nfl_player_id_2",
                    "datetime",
                    "video_path_sideline",
                    "video_path_endzone",
                    "video_path_all29",
                ],
                errors="ignore",
            )
            return X, y

        print("Generating Stream B features...")

        # Filter metadata for Stream B (Player vs Ground)
        stream_b_meta = meta_df[meta_df["nfl_player_id_2"] == "G"].copy()

        # Prepare Lagged Tracking
        df_lagged = self.get_lagged_tracking(df_tracking)

        # 1. Merge P1 Tracking
        df_p1 = df_lagged.add_suffix("_p1")
        df_p1 = df_p1.rename(
            columns={
                "game_play_p1": "game_play",
                "step_p1": "step",
                "nfl_player_id_p1": "nfl_player_id_1",
            }
        )

        # Ensure join keys are numeric to prevent merge errors
        stream_b_meta["nfl_player_id_1"] = pd.to_numeric(
            stream_b_meta["nfl_player_id_1"], errors="coerce"
        )
        df_p1["nfl_player_id_1"] = pd.to_numeric(
            df_p1["nfl_player_id_1"], errors="coerce"
        )

        merged = stream_b_meta.merge(
            df_p1, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # Stream B is Uni-Modal Kinematic. No P2 features, No Visuals.

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        merged.to_parquet(cache_path)

        y = merged["contact"].values
        X = merged.drop(
            columns=[
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
                "datetime",
                "video_path_sideline",
                "video_path_endzone",
                "video_path_all29",
            ],
            errors="ignore",
        )

        return X, y

    def generate_train_features(self, load_cached_data: bool = True):
        """
        Generates features for training and validation sets.
        """
        # Load Metadata
        train_meta = pd.read_csv(self.config.TRAIN_META_PATH)
        val_meta = pd.read_csv(self.config.VAL_META_PATH)

        # Combine for processing, split later if needed, or process separately.
        # Processing separately is safer for memory.

        # Load Raw Data
        df_tracking = self.load_and_prep_tracking(self.config.TRAIN_TRACKING_PATH)
        df_helmets = pd.read_csv(self.config.TRAIN_HELMETS_PATH)
        df_helmets = reduce_mem_usage(df_helmets)

        # --- Stream A ---
        # Train
        X_train_A, y_train_A = self.process_stream_a(
            train_meta,
            df_tracking,
            df_helmets,
            os.path.join(self.config.WORKING_DIR, "train_streamA.parquet"),
            load_cached_data,
        )
        # Val
        X_val_A, y_val_A = self.process_stream_a(
            val_meta,
            df_tracking,
            df_helmets,
            os.path.join(self.config.WORKING_DIR, "val_streamA.parquet"),
            load_cached_data,
        )

        # --- Stream B ---
        # Train
        X_train_B, y_train_B = self.process_stream_b(
            train_meta,
            df_tracking,
            os.path.join(self.config.WORKING_DIR, "train_streamB.parquet"),
            load_cached_data,
        )
        # Val
        X_val_B, y_val_B = self.process_stream_b(
            val_meta,
            df_tracking,
            os.path.join(self.config.WORKING_DIR, "val_streamB.parquet"),
            load_cached_data,
        )

        return (
            (X_train_A, y_train_A),
            (X_val_A, y_val_A),
            (X_train_B, y_train_B),
            (X_val_B, y_val_B),
        )

    def generate_test_features(self, load_cached_data: bool = True):
        """
        Generates features for test set.
        """
        test_meta = pd.read_csv(self.config.TEST_META_PATH)

        df_tracking = self.load_and_prep_tracking(self.config.TEST_TRACKING_PATH)
        df_helmets = pd.read_csv(self.config.TEST_HELMETS_PATH)
        df_helmets = reduce_mem_usage(df_helmets)

        # --- Stream A ---
        X_test_A, _ = self.process_stream_a(
            test_meta,
            df_tracking,
            df_helmets,
            os.path.join(self.config.WORKING_DIR, "test_streamA.parquet"),
            load_cached_data,
        )

        # --- Stream B ---
        X_test_B, _ = self.process_stream_b(
            test_meta,
            df_tracking,
            os.path.join(self.config.WORKING_DIR, "test_streamB.parquet"),
            load_cached_data,
        )

        # We also need the IDs to map predictions back
        ids_A = test_meta[test_meta["nfl_player_id_2"] != "G"]["contact_id"].values
        ids_B = test_meta[test_meta["nfl_player_id_2"] == "G"]["contact_id"].values

        return X_test_A, ids_A, X_test_B, ids_B
