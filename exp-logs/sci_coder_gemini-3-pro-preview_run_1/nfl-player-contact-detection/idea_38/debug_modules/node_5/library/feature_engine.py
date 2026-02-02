import os
import gc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
from library.config import Config
from library.utils import reduce_mem_usage, save_parquet, load_parquet
from library.data_factory import DataFactory


class FeatureEngine:
    """
    Implements the Dynamic-Basis Context-Aware Anchored-Mining feature engineering pipeline.
    Responsible for Gating, Windowing, Dynamic Basis Projection, and Context Calculation.
    """

    @staticmethod
    def _quadratic_gating(df):
        """
        Applies Relaxed Quadratic Reachability Gating.
        Estimates min distance in the window [-1.0s, 1.0s] using Taylor expansion.
        Filters out pairs where the estimated minimum distance exceeds the threshold.
        """
        # Split PP and PG
        # We generally preserve Player-Ground (PG) interactions as their 'distance' is sentinel -1
        # and they are sparse/specific events.
        mask_ground = df["nfl_player_id_2"] == "G"
        df_pg = df[mask_ground].copy()
        df_pp = df[~mask_ground].copy()

        if not df_pp.empty:
            # Extract relative vectors at t=0
            rx = df_pp["x_position_p1"] - df_pp["x_position_p2"]
            ry = df_pp["y_position_p1"] - df_pp["y_position_p2"]

            # Convert speed/direction to velocity components
            # NFL Tracking: 0 deg is Y (North), 90 deg is X (East)
            # vx = speed * sin(dir), vy = speed * cos(dir)

            # Fill NaNs with 0 for safety
            dir_p1 = np.radians(df_pp["direction_p1"].fillna(0))
            dir_p2 = np.radians(df_pp["direction_p2"].fillna(0))

            vx = df_pp["speed_p1"] * np.sin(dir_p1) - df_pp["speed_p2"] * np.sin(dir_p2)
            vy = df_pp["speed_p1"] * np.cos(dir_p1) - df_pp["speed_p2"] * np.cos(dir_p2)

            # Approximate acceleration vector aligned with direction
            # (Tracking 'acceleration' is magnitude)
            ax = df_pp["acceleration_p1"] * np.sin(dir_p1) - df_pp[
                "acceleration_p2"
            ] * np.sin(dir_p2)
            ay = df_pp["acceleration_p1"] * np.cos(dir_p1) - df_pp[
                "acceleration_p2"
            ] * np.cos(dir_p2)

            # Check min distance at discrete points in the window [-1.0s, 1.0s]
            # t is in seconds
            min_dists = np.full(len(df_pp), np.inf)

            # Check 5 key points: -1.0, -0.5, 0.0, 0.5, 1.0
            for t in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                # r(t) = r0 + v0*t + 0.5*a0*t^2
                rt_x = rx + vx * t + 0.5 * ax * (t**2)
                rt_y = ry + vy * t + 0.5 * ay * (t**2)
                dist_t = np.sqrt(rt_x**2 + rt_y**2)
                min_dists = np.minimum(min_dists, dist_t)

            # Filter
            keep_mask = min_dists < Config.GATING_DISTANCE
            df_pp = df_pp[keep_mask]

        # Recombine
        return pd.concat([df_pp, df_pg], axis=0).sort_index()

    @staticmethod
    def _expand_windows(df_meta):
        """
        Expands the metadata to include +/- WINDOW_SIZE steps.
        Creates a row for every timestep in the window for every contact_id.
        """
        # Create offsets array
        offsets = np.arange(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        # Use index to map back to original rows
        df_meta = df_meta.reset_index(drop=True)

        # Vectorized expansion
        n_samples = len(df_meta)
        n_offsets = len(offsets)

        # Repeat indices and tile offsets
        expanded_indices = np.repeat(df_meta.index.values, n_offsets)
        tiled_offsets = np.tile(offsets, n_samples)

        # Create expanded dataframe
        df_expanded = df_meta.iloc[expanded_indices].copy()
        df_expanded["step_offset"] = tiled_offsets
        df_expanded["actual_step"] = df_expanded["step"] + df_expanded["step_offset"]

        return df_expanded

    @staticmethod
    def _compute_dynamic_features(df_window, df_tracking):
        """
        Merges tracking data for the expanded window, computes dynamic basis vectors,
        projects kinematics, and calculates Aligned Environmental Pressure.
        """
        # ---------------------------------------------------------
        # 1. Merge Tracking Data for P1 and P2
        # ---------------------------------------------------------
        track_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
        ]

        # Drop existing tracking columns to prevent duplicates (Cite debug_lesson_28)
        cols_to_drop = ["distance"]
        for c in track_cols:
            cols_to_drop.extend([f"{c}_p1", f"{c}_p2"])

        df_window = df_window.drop(
            columns=[c for c in cols_to_drop if c in df_window.columns]
        )

        # Merge P1
        df_merged = pd.merge(
            df_window,
            df_tracking,
            left_on=["game_play", "actual_step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_track_p1"),
        )
        rename_p1 = {c: f"{c}_p1" for c in track_cols}
        df_merged = df_merged.rename(columns=rename_p1)
        # Drop redundant columns from merge
        cols_to_drop = ["nfl_player_id", "step_track_p1"]
        df_merged = df_merged.drop(
            columns=[c for c in cols_to_drop if c in df_merged.columns]
        )

        # Handle P2 (Split logic for Ground)
        is_ground = df_merged["nfl_player_id_2"] == "G"

        # PP Subset
        df_pp = df_merged[~is_ground].copy()
        df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

        df_pp = pd.merge(
            df_pp,
            df_tracking,
            left_on=["game_play", "actual_step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_track_p2"),
        )
        rename_p2 = {c: f"{c}_p2" for c in track_cols}
        df_pp = df_pp.rename(columns=rename_p2)
        df_pp = df_pp.drop(
            columns=[c for c in cols_to_drop if c in df_pp.columns], errors="ignore"
        )

        # PG Subset
        df_pg = df_merged[is_ground].copy()
        for c in track_cols:
            df_pg[f"{c}_p2"] = 0.0

        df_full = pd.concat([df_pp, df_pg], axis=0).sort_index()

        # Ensure nfl_player_id_2 is consistently string for Parquet serialization
        df_full["nfl_player_id_2"] = df_full["nfl_player_id_2"].astype(str)

        # ---------------------------------------------------------
        # 2. Dynamic Basis Calculation
        # ---------------------------------------------------------
        # Vector P2->P1
        dx = df_full["x_position_p1"] - df_full["x_position_p2"]
        dy = df_full["y_position_p1"] - df_full["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # P1 Velocity Vector
        dir_rad_p1 = np.radians(df_full["direction_p1"].fillna(0))
        v1_x = df_full["speed_p1"] * np.sin(dir_rad_p1)
        v1_y = df_full["speed_p1"] * np.cos(dir_rad_p1)
        v1_mag = np.sqrt(v1_x**2 + v1_y**2)

        # Initialize Basis Vectors (ux, uy)
        ux = np.zeros_like(dx)
        uy = np.zeros_like(dy)

        # Recompute masks on df_full to ensure alignment (Cite debug_lesson_39)
        is_ground_full = df_full["nfl_player_id_2"] == "G"
        mask_pp = ~is_ground_full
        mask_pg = is_ground_full

        # PP Basis: Unit vector along P2->P1
        safe_dist = dist.copy()
        safe_dist[safe_dist < 1e-6] = 1.0
        ux[mask_pp] = dx[mask_pp] / safe_dist[mask_pp]
        uy[mask_pp] = dy[mask_pp] / safe_dist[mask_pp]

        # PG Basis: Unit vector along P1 Velocity
        safe_v = v1_mag.copy()
        safe_v[safe_v < 1e-6] = 1.0
        ux[mask_pg] = v1_x[mask_pg] / safe_v[mask_pg]
        uy[mask_pg] = v1_y[mask_pg] / safe_v[mask_pg]

        # Orthogonal Basis (Rotate -90 deg: x' = y, y' = -x? No, (x,y) -> (y, -x) is -90)
        # Let's use standard orthonormal: (-uy, ux)
        vx_basis = -uy
        vy_basis = ux

        # ---------------------------------------------------------
        # 3. Projections
        # ---------------------------------------------------------
        def project(feat_x, feat_y, bx, by):
            return feat_x * bx + feat_y * by

        # P1 Projections
        df_full["p1_v_long"] = project(v1_x, v1_y, ux, uy)
        df_full["p1_v_lat"] = project(v1_x, v1_y, vx_basis, vy_basis)

        a1_x = df_full["acceleration_p1"] * np.sin(dir_rad_p1)
        a1_y = df_full["acceleration_p1"] * np.cos(dir_rad_p1)
        df_full["p1_a_long"] = project(a1_x, a1_y, ux, uy)
        df_full["p1_a_lat"] = project(a1_x, a1_y, vx_basis, vy_basis)

        # P2 Projections
        dir_rad_p2 = np.radians(df_full["direction_p2"].fillna(0))
        v2_x = df_full["speed_p2"] * np.sin(dir_rad_p2)
        v2_y = df_full["speed_p2"] * np.cos(dir_rad_p2)
        a2_x = df_full["acceleration_p2"] * np.sin(dir_rad_p2)
        a2_y = df_full["acceleration_p2"] * np.cos(dir_rad_p2)

        df_full["p2_v_long"] = project(v2_x, v2_y, ux, uy)
        df_full["p2_v_lat"] = project(v2_x, v2_y, vx_basis, vy_basis)
        df_full["p2_a_long"] = project(a2_x, a2_y, ux, uy)
        df_full["p2_a_lat"] = project(a2_x, a2_y, vx_basis, vy_basis)

        # Relative Projections
        df_full["rel_v_long"] = df_full["p1_v_long"] - df_full["p2_v_long"]
        df_full["rel_v_lat"] = df_full["p1_v_lat"] - df_full["p2_v_lat"]

        # Update distance for PP (PG stays -1)
        df_full.loc[mask_pp, "distance"] = dist[mask_pp]

        # ---------------------------------------------------------
        # 4. Aligned Environmental Pressure (Context)
        # ---------------------------------------------------------
        # Efficient spatial search by grouping
        context_long = np.zeros(len(df_full))
        context_lat = np.zeros(len(df_full))

        df_full["temp_idx"] = np.arange(len(df_full))

        # Index tracking data for fast lookup
        df_track_indexed = df_tracking.set_index(["game_play", "step"]).sort_index()

        # Group targets by frame
        groups = df_full.groupby(["game_play", "actual_step"])

        for (gp, step), group in groups:
            try:
                track_frame = df_track_indexed.loc[(gp, step)]
            except KeyError:
                continue

            if track_frame.empty:
                continue

            # Prepare KDTree
            player_coords = track_frame[["x_position", "y_position"]].values

            # Prepare velocities for projection
            p_dirs = np.radians(track_frame["direction"].fillna(0).values)
            p_speeds = track_frame["speed"].values
            pv_x = p_speeds * np.sin(p_dirs)
            pv_y = p_speeds * np.cos(p_dirs)

            tree = cKDTree(player_coords)

            # Query targets
            target_coords = group[["x_position_p1", "y_position_p1"]].values
            # Query K+1 to account for self-match
            dists, idxs = tree.query(target_coords, k=Config.N_NEIGHBORS + 1)

            # Vectorized accumulation within the group is hard due to variable neighbor indices
            # Iterate targets in this frame
            row_indices = group["temp_idx"].values
            b_ux = ux[row_indices]
            b_uy = uy[row_indices]

            for i in range(len(group)):
                n_indices = idxs[i]
                n_dists = dists[i]

                pressure_l = 0.0
                pressure_t = 0.0
                count = 0

                for k, n_idx in enumerate(n_indices):
                    if n_idx >= len(player_coords):
                        continue

                    # Skip self (very small distance)
                    if n_dists[k] < 0.01:
                        continue

                    # Project neighbor velocity onto target basis
                    nv_x = pv_x[n_idx]
                    nv_y = pv_y[n_idx]

                    pl = nv_x * b_ux[i] + nv_y * b_uy[i]
                    pt = nv_x * -b_uy[i] + nv_y * b_ux[i]

                    pressure_l += pl
                    pressure_t += pt

                    count += 1
                    if count >= Config.N_NEIGHBORS:
                        break

                context_long[row_indices[i]] = pressure_l
                context_lat[row_indices[i]] = pressure_t

        df_full["context_pressure_long"] = context_long
        df_full["context_pressure_lat"] = context_lat

        return df_full

    @staticmethod
    def _flatten_features(df_processed):
        """
        Pivots the windowed data to create a single row per contact_id.
        """
        feature_cols = [
            "distance",
            "p1_v_long",
            "p1_v_lat",
            "p1_a_long",
            "p1_a_lat",
            "p2_v_long",
            "p2_v_lat",
            "p2_a_long",
            "p2_a_lat",
            "rel_v_long",
            "rel_v_lat",
            "context_pressure_long",
            "context_pressure_lat",
        ]

        # Pivot
        # We include 'contact' in index to preserve label
        index_cols = [
            "contact_id",
            "contact",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]

        pivot = df_processed.pivot_table(
            index=index_cols, columns="step_offset", values=feature_cols
        )

        # Flatten MultiIndex columns: feature_name + _t + offset
        pivot.columns = [f"{col}_t{offset}" for col, offset in pivot.columns]

        # Reset index to make index_cols normal columns again
        df_flat = pivot.reset_index()

        return df_flat

    @staticmethod
    def generate_features(split="train", load_cached_data=True, sample_size=None):
        """
        Main pipeline execution method.
        """
        # Determine cache filename
        suffix = f"_sample_{sample_size}" if sample_size else ""
        cache_file = f"features_{split}{suffix}.parquet"

        # 1. Try Load Cache
        if load_cached_data:
            df = load_parquet(cache_file)
            if df is not None:
                print(f"[{split}] Loaded features from cache: {cache_file}")
                return df

        print(f"[{split}] Generating features from scratch...")

        # 2. Load Base Data
        df_meta = DataFactory.prepare_base_data(
            split, load_cached_data=True, sample_size=sample_size
        )

        # 3. Apply Gating
        print(f"[{split}] Applying Relaxed Quadratic Gating...")
        len_before = len(df_meta)
        df_gated = FeatureEngine._quadratic_gating(df_meta)
        print(f"[{split}] Gating reduced pairs from {len_before} to {len(df_gated)}")

        if df_gated.empty:
            print("Warning: Gating removed all pairs.")
            return pd.DataFrame()

        # 4. Load Full Tracking Data (Required for context and window expansion)
        track_path = (
            Config.TRAIN_TRACKING_PATH if split != "test" else Config.TEST_TRACKING_PATH
        )
        print(f"[{split}] Loading full tracking data...")
        df_tracking = DataFactory.load_tracking(track_path)

        # 5. Expand Windows
        print(f"[{split}] Expanding windows (+/- {Config.WINDOW_SIZE})...")
        df_windows = FeatureEngine._expand_windows(df_gated)

        # 6. Compute Dynamic Features & Context
        print(f"[{split}] Computing Dynamic Basis and Context Features...")
        df_dynamic = FeatureEngine._compute_dynamic_features(df_windows, df_tracking)

        # 7. Flatten
        print(f"[{split}] Flattening feature vectors...")
        df_features = FeatureEngine._flatten_features(df_dynamic)

        # 8. Save to Cache
        save_parquet(df_features, cache_file)
        print(f"[{split}] Saved features to {cache_file}")

        # Cleanup
        del df_meta, df_gated, df_tracking, df_windows, df_dynamic
        gc.collect()

        return df_features
