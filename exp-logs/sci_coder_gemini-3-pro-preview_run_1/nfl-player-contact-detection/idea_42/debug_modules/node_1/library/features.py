import os
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    WINDOW_SIZE,
    GATING_THRESHOLD,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
)
from library.utils import reduce_mem_usage, cache_result

# Suppress SettingWithCopyWarning for cleaner logs
pd.options.mode.chained_assignment = None


def compute_relative_vectors(df):
    """
    Computes relative position, velocity, and acceleration vectors between p1 and p2.
    """
    # 1. Position Difference
    df["r_x"] = df["x_position_p1"] - df["x_position_p2"]
    df["r_y"] = df["y_position_p1"] - df["y_position_p2"]

    # 2. Velocity Calculation (from Speed & Direction)
    # Convert direction (degrees) to radians.
    # Assuming standard tracking convention: 0 deg is Y-axis (North), 90 deg is X-axis (East).
    # v_x = speed * sin(theta), v_y = speed * cos(theta)
    for p in ["p1", "p2"]:
        rad = np.radians(df[f"direction_{p}"])
        df[f"v_x_{p}"] = df[f"speed_{p}"] * np.sin(rad)
        df[f"v_y_{p}"] = df[f"speed_{p}"] * np.cos(rad)

        # Acceleration Decomposition (Approximation using direction)
        # We use the same direction for acceleration as velocity for basis projection purposes
        df[f"a_x_{p}"] = df[f"acceleration_{p}"] * np.sin(rad)
        df[f"a_y_{p}"] = df[f"acceleration_{p}"] * np.cos(rad)

    # 3. Relative Vectors
    df["v_rel_x"] = df["v_x_p1"] - df["v_x_p2"]
    df["v_rel_y"] = df["v_y_p1"] - df["v_y_p2"]

    df["a_rel_x"] = df["a_x_p1"] - df["a_x_p2"]
    df["a_rel_y"] = df["a_y_p1"] - df["a_y_p2"]

    return df


def apply_dynamic_basis_alignment(df):
    """
    Projects relative vectors onto the dynamic basis defined by the Relative Velocity Vector.
    Resolves singularity issues by avoiding division by zero with epsilon.
    """
    # 1. Define Basis Vector u_t (Direction of Relative Velocity)
    v_rel_mag = np.sqrt(df["v_rel_x"] ** 2 + df["v_rel_y"] ** 2) + 1e-6

    u_x = df["v_rel_x"] / v_rel_mag
    u_y = df["v_rel_y"] / v_rel_mag

    # 2. Define Orthogonal Basis u_perp (-u_y, u_x)
    u_perp_x = -u_y
    u_perp_y = u_x

    # 3. Project Relative Position (r)
    df["r_long"] = df["r_x"] * u_x + df["r_y"] * u_y
    df["r_trans"] = df["r_x"] * u_perp_x + df["r_y"] * u_perp_y

    # 4. Project Relative Acceleration (a_rel)
    df["a_long"] = df["a_rel_x"] * u_x + df["a_rel_y"] * u_y
    df["a_trans"] = df["a_rel_x"] * u_perp_x + df["a_rel_y"] * u_perp_y

    # 5. Project Relative Velocity
    # v_long is effectively the signed magnitude of closing speed relative to the basis
    df["v_long"] = df["v_rel_x"] * u_x + df["v_rel_y"] * u_y
    df["v_trans"] = df["v_rel_x"] * u_perp_x + df["v_rel_y"] * u_perp_y

    return df


def add_interaction_primitives(df):
    """
    Computes explicit interaction primitives like Distance, TTC, and Jerk.
    """
    # Euclidean Distance
    df["dist"] = np.sqrt(df["r_x"] ** 2 + df["r_y"] ** 2)

    # Time-To-Collision (TTC)
    # TTC = Distance / Closing Speed.
    # Closing Speed = - (r . v_rel) / |r|
    r_dot_v = df["r_x"] * df["v_rel_x"] + df["r_y"] * df["v_rel_y"]

    # Avoid division by zero for distance
    safe_dist = df["dist"] + 1e-6
    closing_speed = -r_dot_v / safe_dist

    # We only define TTC if objects are closing (closing_speed > 0)
    # Cap TTC at 10.0 seconds for stability
    df["ttc"] = np.where(closing_speed > 0.1, df["dist"] / closing_speed, 10.0)

    return df


def flatten_temporal_window(df, feature_cols):
    """
    Pivots the temporal window [-WINDOW, +WINDOW] into a single wide row per contact_id.
    """
    # Pivot: Index=contact_id, Columns=offset, Values=features
    pivoted = df.pivot(index="contact_id", columns="offset", values=feature_cols)

    # Flatten MultiIndex columns: e.g., ('dist', -10) -> 'dist_-10'
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]

    return pivoted.reset_index()


def _process_chunk(chunk_meta, tracking_df):
    """
    Internal helper to process a chunk of plays.
    """
    # 1. Expand Metadata for Window
    # Create offsets [-WINDOW_SIZE, ..., +WINDOW_SIZE]
    offsets = np.arange(-WINDOW_SIZE, WINDOW_SIZE + 1)
    offset_df = pd.DataFrame({"offset": offsets})

    # Cross join metadata with offsets
    chunk_meta["_tmp"] = 1
    offset_df["_tmp"] = 1
    expanded = pd.merge(chunk_meta, offset_df, on="_tmp").drop("_tmp", axis=1)
    expanded["actual_step"] = expanded["step"] + expanded["offset"]

    # 2. Filter Tracking Data for this chunk
    relevant_plays = chunk_meta["game_play"].unique()
    chunk_tracking = tracking_df[tracking_df["game_play"].isin(relevant_plays)].copy()

    # 3. Merge Player 1 Tracking
    track_cols = [
        "game_play",
        "nfl_player_id",
        "step",
        "x_position",
        "y_position",
        "speed",
        "direction",
        "acceleration",
        "orientation",
    ]

    p1_track = (
        chunk_tracking[track_cols]
        .rename(
            columns={c: f"{c}_p1" for c in track_cols if c not in ["game_play", "step"]}
        )
        .rename(columns={"nfl_player_id": "nfl_player_id_1"})
    )

    merged = pd.merge(
        expanded,
        p1_track,
        left_on=["game_play", "nfl_player_id_1", "actual_step"],
        right_on=["game_play", "nfl_player_id_1", "step"],
        how="left",
    )
    if "step_y" in merged.columns:
        merged = merged.drop(columns=["step_y"])
    merged = merged.rename(columns={"step_x": "step"})

    # 4. Merge Player 2 Tracking
    # Handle 'G' (Ground) by creating a join key. If 'G', use dummy ID -999.
    merged["join_id_2"] = (
        pd.to_numeric(merged["nfl_player_id_2"], errors="coerce")
        .fillna(-999)
        .astype(int)
    )

    p2_track = (
        chunk_tracking[track_cols]
        .rename(
            columns={c: f"{c}_p2" for c in track_cols if c not in ["game_play", "step"]}
        )
        .rename(columns={"nfl_player_id": "join_id_2"})
    )

    merged = pd.merge(
        merged,
        p2_track,
        left_on=["game_play", "join_id_2", "actual_step"],
        right_on=["game_play", "join_id_2", "step"],
        how="left",
    )

    # 5. Handle Missing Data & Ground Logic
    # If P2 is Ground (or missing), fill P2 features with 0.
    # This effectively sets P2 to a stationary point at (0,0) with 0 velocity/accel.
    # Relative vectors will then reflect P1's absolute kinematics.
    p2_cols = [c for c in merged.columns if c.endswith("_p2")]
    merged[p2_cols] = merged[p2_cols].fillna(0)

    p1_cols = [c for c in merged.columns if c.endswith("_p1")]
    merged[p1_cols] = merged[p1_cols].fillna(0)

    # 6. Compute Features
    merged = compute_relative_vectors(merged)
    merged = apply_dynamic_basis_alignment(merged)
    merged = add_interaction_primitives(merged)

    # 7. Sentinel Value Strategy for Ground
    # Explicitly set distance to -1.0 for Ground interactions
    is_ground = merged["nfl_player_id_2"] == "G"
    merged.loc[is_ground, "dist"] = -1.0

    # 8. Flatten
    feats_to_keep = [
        "r_long",
        "r_trans",
        "v_long",
        "v_trans",
        "a_long",
        "a_trans",
        "speed_p1",
        "speed_p2",
        "dist",
        "ttc",
        "acceleration_p1",
        "acceleration_p2",
    ]
    flattened = flatten_temporal_window(merged, feats_to_keep)

    # 9. Relaxed Quadratic Gating
    # Filter based on minimum distance in the window
    # Ground contacts (-1.0) will always pass (< 3.0)
    dist_cols = [c for c in flattened.columns if c.startswith("dist_")]
    min_dists = flattened[dist_cols].min(axis=1)

    gated = flattened[min_dists < GATING_THRESHOLD].copy()

    # 10. Re-attach Metadata
    # We lost metadata during pivot, merge back using contact_id
    final_df = pd.merge(gated, chunk_meta, on="contact_id", how="inner")

    return final_df


def _generate_features_impl(metadata_path, tracking_path):
    """
    Driver function to load data and process in chunks.
    """
    print(f"Loading metadata from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    print(f"Loading tracking from {tracking_path}...")
    df_tracking = pd.read_csv(tracking_path)
    df_tracking = reduce_mem_usage(df_tracking)

    # Chunk by GamePlay
    unique_plays = df_meta["game_play"].unique()
    chunk_size = 50
    chunks = [
        unique_plays[i : i + chunk_size]
        for i in range(0, len(unique_plays), chunk_size)
    ]

    results = []
    print(f"Processing {len(unique_plays)} plays in {len(chunks)} chunks...")

    for i, play_chunk in enumerate(chunks):
        meta_chunk = df_meta[df_meta["game_play"].isin(play_chunk)].copy()
        processed_chunk = _process_chunk(meta_chunk, df_tracking)
        results.append(processed_chunk)

        if (i + 1) % 10 == 0:
            print(f"Processed chunk {i+1}/{len(chunks)}")

    if not results:
        return pd.DataFrame()

    full_df = pd.concat(results, axis=0)
    full_df = reduce_mem_usage(full_df)

    print(f"Feature generation complete. Shape: {full_df.shape}")
    return full_df


@cache_result(filename="features_train.parquet", file_format="parquet")
def generate_train_features(
    metadata_path=None, tracking_path=None, load_cached_data=False
):
    return _generate_features_impl(metadata_path, tracking_path)


@cache_result(filename="features_val.parquet", file_format="parquet")
def generate_val_features(
    metadata_path=None, tracking_path=None, load_cached_data=False
):
    return _generate_features_impl(metadata_path, tracking_path)


@cache_result(filename="features_test.parquet", file_format="parquet")
def generate_test_features(
    metadata_path=None, tracking_path=None, load_cached_data=False
):
    return _generate_features_impl(metadata_path, tracking_path)
