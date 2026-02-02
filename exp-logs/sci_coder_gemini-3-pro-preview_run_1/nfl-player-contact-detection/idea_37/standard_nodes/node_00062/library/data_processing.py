import os
import gc
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.interpolate import interp1d
from library.config import Config
from library.utils import reduce_mem_usage, get_fingerprint, setup_logger

# Suppress warnings
warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self):
        self.logger = setup_logger("data_processing")
        self.window_size = Config.FEATURE_WINDOW_SIZE
        self.gating_threshold = Config.GATING_THRESHOLD
        self.cache_dir = Config.WORKING_DIR
        self.n_jobs = Config.N_JOBS

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, prefix, *args):
        """Generates a cache file path based on input arguments hash."""
        fingerprint = get_fingerprint(
            list(args) + [self.window_size, self.gating_threshold]
        )
        return os.path.join(self.cache_dir, f"{prefix}_{fingerprint}.parquet")

    def load_tracking_data(self, path):
        """Loads and optimizes tracking data."""
        self.logger.info(f"Loading tracking data from {path}...")
        df = pd.read_csv(path)
        df = reduce_mem_usage(df, verbose=False)

        # Ensure essential columns are correct types
        df["nfl_player_id"] = df["nfl_player_id"].astype(str)
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)

        return df

    def _process_single_play(self, game_play, play_meta, play_tracking):
        """
        Process a single play: constructs dense tensors and computes features
        for all interactions in this play.
        """
        # 1. Construct Dense Tensor for the Play
        # Get range of steps
        min_step = play_tracking["step"].min()
        max_step = play_tracking["step"].max()

        # Identify all unique players in tracking
        unique_players = play_tracking["nfl_player_id"].unique()
        player_map = {pid: i for i, pid in enumerate(unique_players)}
        n_players = len(unique_players)

        # Feature columns to extract
        # x, y, speed, acceleration, direction, orientation
        # We need x, y for distance/basis. speed, acc, dir, o for features.
        # Direction/Orientation are in degrees. Convert to radians for projection.

        # Pre-process angles
        play_tracking["dir_rad"] = np.deg2rad(play_tracking["direction"])
        play_tracking["o_rad"] = np.deg2rad(play_tracking["orientation"])

        # Calculate Vx, Vy, Ax, Ay from speed/acc/dir (if needed, or use raw)
        # The prompt suggests projecting v, a.
        # Vx = Speed * sin(dir), Vy = Speed * cos(dir) (NFL coordinates: 0 is Y axis usually, but let's stick to standard trig if x,y provided)
        # Actually, let's use the provided x, y for position.
        # For velocity vector, we can use (speed, direction).

        feat_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "dir_rad",
            "o_rad",
        ]
        n_feats = len(feat_cols)

        # Reindex to full step range to handle missing steps
        full_steps = np.arange(min_step, max_step + 1)

        # Create tensor: (Time, Players, Feats)
        # Initialize with NaNs
        tensor = np.full(
            (len(full_steps), n_players, n_feats), np.nan, dtype=np.float32
        )

        # Fill tensor
        # This part can be slow if iterated. Use pivot.
        # Pivot table: Index=step, Columns=player, Values=feat
        for f_idx, col in enumerate(feat_cols):
            pivot = play_tracking.pivot(
                index="step", columns="nfl_player_id", values=col
            )
            # Reindex to full range
            pivot = pivot.reindex(full_steps)
            # Interpolate missing values (linear) inside the play
            pivot = pivot.interpolate(method="linear", limit_direction="both")

            # Map columns to tensor indices
            # pivot columns are player_ids
            existing_players = pivot.columns
            indices = [player_map[p] for p in existing_players]

            # Assign
            tensor[:, indices, f_idx] = pivot.values

        # 2. Iterate over interactions (metadata rows)
        results = []

        # Filter metadata for this play
        # play_meta is already filtered by caller

        # Map steps to tensor indices
        step_to_idx = {s: i for i, s in enumerate(full_steps)}

        for row in play_meta.itertuples(index=False):
            # Row: contact_id, game_play, nfl_player_id_1, nfl_player_id_2, step, ...
            # Note: nfl_player_id_2 can be 'G'

            current_step = row.step
            p1_id = str(row.nfl_player_id_1)
            p2_id = str(row.nfl_player_id_2)

            # Check bounds
            if current_step not in step_to_idx:
                continue

            idx_center = step_to_idx[current_step]

            # Define Window Indices
            w = self.window_size
            idx_start = idx_center - w
            idx_end = idx_center + w + 1  # Inclusive of end

            # Check window bounds relative to tensor
            if idx_start < 0 or idx_end > tensor.shape[0]:
                # Pad or Skip?
                # For simplicity in this robust pipeline, we skip edge cases or pad.
                # Let's pad with edge values if possible, or just skip if too far off.
                # Given 10Hz and long plays, edge cases are rare.
                # We will clip indices and pad the result.
                pad_pre = 0
                pad_post = 0

                real_start = max(0, idx_start)
                real_end = min(tensor.shape[0], idx_end)

                pad_pre = real_start - idx_start
                pad_post = idx_end - real_end

                idx_start = real_start
                idx_end = real_end
            else:
                pad_pre = 0
                pad_post = 0

            # Extract P1 Data
            if p1_id not in player_map:
                continue  # Should not happen if tracking is complete
            p1_idx = player_map[p1_id]
            p1_data = tensor[idx_start:idx_end, p1_idx, :]  # (Window, Feats)

            # Extract P2 Data
            is_ground = p2_id == "G"
            if not is_ground:
                if p2_id not in player_map:
                    continue
                p2_idx = player_map[p2_id]
                p2_data = tensor[idx_start:idx_end, p2_idx, :]
            else:
                # Mock P2 data for Ground (Zeros)
                p2_data = np.zeros_like(p1_data)

            # Handle Padding
            if pad_pre > 0 or pad_post > 0:
                p1_data = np.pad(p1_data, ((pad_pre, pad_post), (0, 0)), mode="edge")
                p2_data = np.pad(p2_data, ((pad_pre, pad_post), (0, 0)), mode="edge")

            # --- Feature Engineering ---

            # Unpack P1
            # x, y, s, a, dir, o
            p1_pos = p1_data[:, 0:2]
            p1_s = p1_data[:, 2]
            p1_a = p1_data[:, 3]
            p1_dir = p1_data[:, 4]

            # Calculate P1 Velocity Vector
            # NFL: 0 degrees is Y axis? Usually 0 is North (Y). 90 is East (X).
            # V_x = S * sin(dir), V_y = S * cos(dir)
            p1_vel = np.stack([p1_s * np.sin(p1_dir), p1_s * np.cos(p1_dir)], axis=1)

            # Unpack P2
            if not is_ground:
                p2_pos = p2_data[:, 0:2]
                p2_s = p2_data[:, 2]
                p2_a = p2_data[:, 3]
                p2_dir = p2_data[:, 4]
                p2_vel = np.stack(
                    [p2_s * np.sin(p2_dir), p2_s * np.cos(p2_dir)], axis=1
                )

                # Relative Position Vector (P2 -> P1)
                r_vec = p1_pos - p2_pos
                dist = np.linalg.norm(r_vec, axis=1)

                # --- GATING ---
                # Relaxed Quadratic Gating: Min distance in window < Threshold
                if np.min(dist) > self.gating_threshold:
                    continue

                # Basis: Unit vector P2->P1
                # Handle zero distance
                with np.errstate(divide="ignore", invalid="ignore"):
                    u_rad = r_vec / dist[:, None]
                u_rad[np.isnan(u_rad)] = 0.0  # Collision point

            else:
                # Ground Interaction
                # Dist is sentinel
                dist = np.full(p1_pos.shape[0], Config.GROUND_DISTANCE_SENTINEL)

                # Basis: Unit vector of P1 Velocity
                # If speed is 0, use Orientation? Or just (0,1)
                p1_speed_safe = p1_s.copy()
                p1_speed_safe[p1_speed_safe < 1e-6] = 1.0  # Avoid div zero

                u_rad = p1_vel / p1_speed_safe[:, None]
                # If speed was 0, u_rad might be meaningless, but let's keep it zero or use orientation
                mask_zero = p1_s < 1e-6
                if np.any(mask_zero):
                    # Use orientation for static players
                    p1_o = p1_data[:, 5]
                    u_rad[mask_zero, 0] = np.sin(p1_o[mask_zero])
                    u_rad[mask_zero, 1] = np.cos(p1_o[mask_zero])

                # No gating for Ground based on distance (always reachable)
                # But we might want to gate based on 'event' likelihood?
                # For now, keep all ground samples provided in metadata.

                p2_vel = np.zeros_like(p1_vel)
                p2_a = np.zeros_like(p1_a)

            # Tangential Basis (Rotate 90 deg)
            # (x, y) -> (-y, x)
            u_tan = np.stack([-u_rad[:, 1], u_rad[:, 0]], axis=1)

            # --- Projections ---
            # P1 Projections
            p1_v_rad = np.sum(p1_vel * u_rad, axis=1)
            p1_v_tan = np.sum(p1_vel * u_tan, axis=1)
            # Accel is scalar in tracking? If so, we can't project perfectly without Accel Dir.
            # Assuming Accel is in direction of motion (Tangent to path) or we treat scalar 'a' as magnitude?
            # Tracking data has 'acceleration' (magnitude) and 'sa' (signed).
            # We only have magnitude 'acceleration' in our tensor.
            # Approximation: Project 'a' assuming it aligns with 'v' or just use scalar 'a' as feature.
            # Better: The prompt asks for "Projected Individual Motion Vectors".
            # Without Accel Vector, we can't project A.
            # However, we can differentiate V to get A vector.
            # V(t+1) - V(t-1) / 2dt.
            # Let's compute derived acceleration vector from velocity.
            # dt = 0.1s
            p1_acc_vec = np.gradient(p1_vel, axis=0) * 10.0
            p1_a_rad = np.sum(p1_acc_vec * u_rad, axis=1)
            p1_a_tan = np.sum(p1_acc_vec * u_tan, axis=1)

            if not is_ground:
                p2_acc_vec = np.gradient(p2_vel, axis=0) * 10.0
                p2_a_rad = np.sum(p2_acc_vec * u_rad, axis=1)
                p2_a_tan = np.sum(p2_acc_vec * u_tan, axis=1)

                p2_v_rad = np.sum(p2_vel * u_rad, axis=1)
                p2_v_tan = np.sum(p2_vel * u_tan, axis=1)
            else:
                p2_a_rad = np.zeros_like(p1_a_rad)
                p2_a_tan = np.zeros_like(p1_a_tan)
                p2_v_rad = np.zeros_like(p1_v_rad)
                p2_v_tan = np.zeros_like(p1_v_tan)

            # Basis Stability
            # Rotation: Dot product of u_rad(t) and u_rad(t-1)
            # cos(theta) = u(t) . u(t-1)
            u_rad_prev = np.roll(u_rad, 1, axis=0)
            u_rad_prev[0] = u_rad[0]
            dot_prod = np.sum(u_rad * u_rad_prev, axis=1)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            basis_rot = np.arccos(dot_prod)  # Radians per step

            # Jerk (Derivative of Radial Accel)
            jerk_rad = np.gradient(p1_a_rad) * 10.0

            # --- Flattening ---
            # We have arrays of shape (WindowSize,). We need to flatten to (WindowSize * Feats,).
            # Features per step:
            # dist, p1_v_rad, p1_v_tan, p1_a_rad, p1_a_tan, p2_v_rad, p2_v_tan, p2_a_rad, p2_a_tan, basis_rot, jerk_rad
            # Total 11 features per step. Window 21. Total 231 features.

            step_feats = np.stack(
                [
                    dist,
                    p1_v_rad,
                    p1_v_tan,
                    p1_a_rad,
                    p1_a_tan,
                    p2_v_rad,
                    p2_v_tan,
                    p2_a_rad,
                    p2_a_tan,
                    basis_rot,
                    jerk_rad,
                ],
                axis=1,
            )

            flat_feats = step_feats.flatten()

            # Create result dictionary
            res = {
                "contact_id": row.contact_id,
                "game_play": row.game_play,
                "step": row.step,
                "nfl_player_id_1": row.nfl_player_id_1,
                "nfl_player_id_2": row.nfl_player_id_2,
                "contact": getattr(
                    row, "contact", 0
                ),  # Handle test set where contact might be missing/0
            }

            # Add features with names
            # To save memory, we can just store the array and convert to DF later,
            # but for now let's make a dict or list.
            # List is faster.
            results.append(list(res.values()) + flat_feats.tolist())

        return results

    def process_dataset(
        self, metadata_path, tracking_path, dataset_name, load_cached=True, debug=False
    ):
        """
        Main driver to process a dataset (Train/Val/Test).
        """
        # Check Cache
        cache_path = self._get_cache_path(f"features_{dataset_name}", debug)
        if load_cached and os.path.exists(cache_path):
            self.logger.info(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        # Load Data
        df_meta = pd.read_csv(metadata_path)
        if debug:
            df_meta = df_meta.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).copy()

        df_tracking = self.load_tracking_data(tracking_path)

        # Filter tracking to relevant plays
        relevant_plays = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)]

        # Group by Play
        play_groups = df_meta.groupby("game_play")
        tracking_groups = dict(tuple(df_tracking.groupby("game_play")))

        tasks = []
        for game_play, group_meta in play_groups:
            if game_play in tracking_groups:
                tasks.append((game_play, group_meta, tracking_groups[game_play]))

        self.logger.info(f"Processing {len(tasks)} plays for {dataset_name}...")

        # Parallel Execution
        results_nested = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_single_play)(gp, gm, gt)
            for gp, gm, gt in tqdm(tasks, desc="Extracting Features")
        )

        # Flatten results
        # results_nested is list of lists
        flat_results = [item for sublist in results_nested for item in sublist]

        if not flat_results:
            self.logger.warning(
                f"No features generated for {dataset_name}. Check gating/data."
            )
            return pd.DataFrame()

        # Create DataFrame
        # Define Columns
        # Fixed cols + Feature cols
        fixed_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]

        # Generate feature names
        w = self.window_size
        feature_names = []
        # Features: dist, p1_v_rad, p1_v_tan, p1_a_rad, p1_a_tan, p2_v_rad, p2_v_tan, p2_a_rad, p2_a_tan, basis_rot, jerk_rad
        base_names = [
            "dist",
            "p1_v_rad",
            "p1_v_tan",
            "p1_a_rad",
            "p1_a_tan",
            "p2_v_rad",
            "p2_v_tan",
            "p2_a_rad",
            "p2_a_tan",
            "basis_rot",
            "jerk_rad",
        ]

        for t in range(-w, w + 1):
            for name in base_names:
                feature_names.append(f"{name}_t{t}")

        all_cols = fixed_cols + feature_names

        self.logger.info(f"Constructing DataFrame with {len(all_cols)} columns...")
        df_features = pd.DataFrame(flat_results, columns=all_cols)

        # Optimize memory
        df_features = reduce_mem_usage(df_features)

        # Save to Cache
        self.logger.info(f"Saving features to {cache_path}")
        df_features.to_parquet(cache_path)

        return df_features

    def get_train_data(self, load_cached=True, debug=False):
        return self.process_dataset(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_TRACKING_PATH,
            "train",
            load_cached,
            debug,
        )

    def get_val_data(self, load_cached=True, debug=False):
        return self.process_dataset(
            Config.VAL_METADATA_PATH,
            Config.TRAIN_TRACKING_PATH,  # Val uses train tracking file
            "val",
            load_cached,
            debug,
        )

    def get_test_data(self, load_cached=True, debug=False):
        return self.process_dataset(
            Config.TEST_METADATA_PATH,
            Config.TEST_TRACKING_PATH,
            "test",
            load_cached,
            debug,
        )
