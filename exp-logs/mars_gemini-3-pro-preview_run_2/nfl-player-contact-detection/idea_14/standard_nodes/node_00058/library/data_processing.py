import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.encoders = {}

        # Define cache paths
        self.cache_dir = self.config.WORKING_DIR
        self.scaler_path = os.path.join(self.cache_dir, "scaler.joblib")
        self.encoders_path = os.path.join(self.cache_dir, "encoders.joblib")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_wide_features(self, tracking_df):
        """
        Converts long-format tracking data into wide-format with temporal lags.
        """
        # Ensure sorting for correct shifting
        tracking_df = tracking_df.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).reset_index(drop=True)

        # Base columns to widen
        feature_cols = self.config.TRACKING_COLS

        # Result storage
        wide_dfs = []

        # Iterate over the window (e.g., -5 to +5)
        # Lag k means: value at step t comes from step t+k
        # If we want the window [t-5, ..., t+5], we shift by k where k in [-5, 5]
        # shift(k) shifts data DOWN by k. So row t gets data from t-k.
        # We want row t to have data from t-5. That is shift(5).
        # We want row t to have data from t+5. That is shift(-5).

        for k in range(-self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1):
            # We use shift(k) to bring past data (positive k) to current row
            # or future data (negative k) to current row.
            # However, standard convention: lag k usually means t-k.
            # Let's define: lag_k is the value at time t+k relative to current time t.
            # So lag_-5 is t-5. lag_5 is t+5.
            # To get value at t-5 into row t, we need to look at row t-5.
            # That corresponds to shift(5).
            # To get value at t+5 into row t, we need to look at row t+5.
            # That corresponds to shift(-5).

            shift_amount = -k
            suffix = f"_lag_{k}"

            shifted = tracking_df[feature_cols].shift(shift_amount)
            shifted.columns = [c + suffix for c in feature_cols]

            # Boundary check: Ensure we didn't shift data from a different play or player
            # We verify that the key columns align after shift
            gp_col = tracking_df["game_play"]
            pid_col = tracking_df["nfl_player_id"]

            gp_shifted = gp_col.shift(shift_amount)
            pid_shifted = pid_col.shift(shift_amount)

            mask = (gp_col == gp_shifted) & (pid_col == pid_shifted)

            # Apply mask to invalidate cross-play/cross-player shifts
            shifted = shifted.where(mask, np.nan)

            wide_dfs.append(shifted)

        # Concatenate all wide features horizontally
        # We also keep the keys
        keys = tracking_df[["game_play", "nfl_player_id", "step", "position", "team"]]
        wide_tracking = pd.concat([keys] + wide_dfs, axis=1)

        return wide_tracking

    def _impute_ground_and_physics(self, df):
        """
        Applies Hybrid Ground Imputation and calculates Explicit Relative Physics.
        """
        # Identify columns
        lags = range(-self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1)

        # 1. Hybrid Ground Imputation
        # If nfl_player_id_2 is 'G', we impute P2 features
        ground_mask = df["nfl_player_id_2"] == "G"

        if ground_mask.any():
            for k in lags:
                suffix = f"_lag_{k}"

                # Position: P2 = P1 (Distance -> 0)
                df.loc[ground_mask, f"x_position{suffix}_2"] = df.loc[
                    ground_mask, f"x_position{suffix}_1"
                ]
                df.loc[ground_mask, f"y_position{suffix}_2"] = df.loc[
                    ground_mask, f"y_position{suffix}_1"
                ]

                # Velocity/Accel/Orientation: P2 = 0
                for col in ["speed", "acceleration", "sa", "direction", "orientation"]:
                    if f"{col}{suffix}_2" in df.columns:
                        df.loc[ground_mask, f"{col}{suffix}_2"] = 0.0

        # 2. Explicit Relative Physics
        # Calculate for every lag step
        for k in lags:
            suffix = f"_lag_{k}"

            # Coordinates
            x1 = df[f"x_position{suffix}_1"]
            y1 = df[f"y_position{suffix}_1"]
            x2 = df[f"x_position{suffix}_2"]
            y2 = df[f"y_position{suffix}_2"]

            # Distance
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)

            # Log Distance (Feature 1)
            if self.config.USE_LOG_DISTANCE:
                df[f"log_dist{suffix}"] = np.log1p(dist)
            else:
                df[f"dist{suffix}"] = dist

            # Relative Speed (Feature 2)
            s1 = df[f"speed{suffix}_1"].fillna(0)
            s2 = df[f"speed{suffix}_2"].fillna(0)
            df[f"speed_diff{suffix}"] = s1 - s2

            # Note: We could add closing speed here, but given the complexity of
            # vectorized trig and potential for NaNs, distance and speed diff
            # are the most robust signals for the model.

        return df

    def _prepare_tensors(self, df, is_training=True):
        """
        Splits DataFrame into Continuous, Categorical, and Target tensors.
        Handles scaling and encoding.
        """
        # --- Define Feature Groups ---
        lags = range(-self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1)

        # Continuous Features
        cont_cols = []
        for k in lags:
            suffix = f"_lag_{k}"
            # P1 Kinematics
            cont_cols.extend([f"{c}{suffix}_1" for c in self.config.TRACKING_COLS])
            # P2 Kinematics
            cont_cols.extend([f"{c}{suffix}_2" for c in self.config.TRACKING_COLS])
            # Relative Physics
            if self.config.USE_LOG_DISTANCE:
                cont_cols.append(f"log_dist{suffix}")
            else:
                cont_cols.append(f"dist{suffix}")
            cont_cols.append(f"speed_diff{suffix}")

        # Categorical Features (Entity Embeddings)
        # We need P1 Position, P1 Team, P2 Position, P2 Team
        # Note: P2 Position/Team might be NaN for Ground. We treat Ground as a specific category.
        cat_cols = ["position_1", "team_1", "position_2", "team_2"]

        # --- Preprocessing ---

        # Handle NaNs in continuous features (e.g. missing tracking steps)
        # Fill with 0 is reasonable for standardized data where 0 is mean (approx)
        # or -1 for missing. Given scaling, 0 is safer.
        X_cont = df[cont_cols].fillna(0).values.astype(np.float32)

        # Handle Categoricals
        # Fill NaNs in categoricals with "Unknown"
        # For Ground, we can set Position='G', Team='G'
        df_cat = df[cat_cols].copy()

        # Explicitly label Ground
        ground_mask = df["nfl_player_id_2"] == "G"
        df_cat.loc[ground_mask, "position_2"] = "G"
        df_cat.loc[ground_mask, "team_2"] = "G"

        df_cat = df_cat.fillna("Unknown")

        X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)

        if is_training:
            # Fit Scaler
            self.scaler = StandardScaler()
            X_cont = self.scaler.fit_transform(X_cont)
            joblib.dump(self.scaler, self.scaler_path)

            # Fit Encoders
            for i, col in enumerate(cat_cols):
                le = LabelEncoder()
                # Ensure 'Unknown' and 'G' are in the vocabulary
                vals = df_cat[col].astype(str).unique().tolist()
                le.fit(vals + ["Unknown", "G"])
                self.encoders[col] = le
                X_cat[:, i] = le.transform(df_cat[col].astype(str))
            joblib.dump(self.encoders, self.encoders_path)

        else:
            # Load Scaler/Encoders if not in memory
            if self.scaler is None:
                self.scaler = joblib.load(self.scaler_path)
            if not self.encoders:
                self.encoders = joblib.load(self.encoders_path)

            # Transform
            X_cont = self.scaler.transform(X_cont)
            for i, col in enumerate(cat_cols):
                le = self.encoders[col]
                # Handle unseen labels by mapping to 'Unknown'
                vals = df_cat[col].astype(str).values
                # Efficiently map unseen to Unknown
                known_classes = set(le.classes_)
                vals = np.array([v if v in known_classes else "Unknown" for v in vals])
                X_cat[:, i] = le.transform(vals)

        # Target
        y = df["contact"].values.astype(np.float32) if "contact" in df.columns else None

        return X_cont, X_cat, y

    def _process_pipeline(self, meta_df, tracking_df, is_training=True):
        """
        Executes the full data processing pipeline for a given split.
        """
        print(f"Generating wide tracking features for {len(tracking_df)} rows...")
        wide_tracking = self._generate_wide_features(tracking_df)

        # Prepare for merge
        # Tracking nfl_player_id is float/int. Meta is string (due to 'G').
        # Convert tracking ID to string for consistent merging.
        wide_tracking["nfl_player_id"] = wide_tracking["nfl_player_id"].astype(str)

        # Ensure metadata IDs are strings to match tracking data types
        meta_df = meta_df.copy()
        meta_df["nfl_player_id_1"] = meta_df["nfl_player_id_1"].astype(str)
        meta_df["nfl_player_id_2"] = meta_df["nfl_player_id_2"].astype(str)

        # Rename columns for P1 and P2 merge
        # P1 Merge
        print("Merging Player 1 data...")
        p1_cols = {
            c: f"{c}_1"
            for c in wide_tracking.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        # We need to keep join keys
        p1_df = wide_tracking.rename(columns=p1_cols)

        merged = meta_df.merge(
            p1_df,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(
            columns=["nfl_player_id"]
        )  # Drop the join key from tracking

        # P2 Merge
        print("Merging Player 2 data...")
        p2_cols = {
            c: f"{c}_2"
            for c in wide_tracking.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        p2_df = wide_tracking.rename(columns=p2_cols)

        merged = merged.merge(
            p2_df,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        # Free memory
        del wide_tracking, p1_df, p2_df
        gc.collect()

        print("Imputing Ground and calculating physics...")
        merged = self._impute_ground_and_physics(merged)

        print("Preparing tensors (Scaling/Encoding)...")
        X_cont, X_cat, y = self._prepare_tensors(merged, is_training=is_training)

        return X_cont, X_cat, y

    def load_and_process_train_val(self, load_cached_data=True):
        """
        Loads training and validation data, processes it, and returns tensors.
        Uses caching to avoid re-processing.
        """
        cache_files = {
            "train": ["X_train.npy", "X_cat_train.npy", "y_train.npy"],
            "val": ["X_val.npy", "X_cat_val.npy", "y_val.npy"],
        }

        # Check cache
        if load_cached_data:
            all_exist = True
            for split in cache_files:
                for f in cache_files[split]:
                    if not os.path.exists(os.path.join(self.cache_dir, f)):
                        all_exist = False
                        break

            if all_exist:
                print("Loading cached training data...")
                data = {}
                for split in ["train", "val"]:
                    data[f"X_{split}"] = np.load(
                        os.path.join(self.cache_dir, f"X_{split}.npy")
                    )
                    data[f"X_cat_{split}"] = np.load(
                        os.path.join(self.cache_dir, f"X_cat_{split}.npy")
                    )
                    data[f"y_{split}"] = np.load(
                        os.path.join(self.cache_dir, f"y_{split}.npy")
                    )
                return (
                    data["X_train"],
                    data["X_cat_train"],
                    data["y_train"],
                    data["X_val"],
                    data["X_cat_val"],
                    data["y_val"],
                )

        # --- Process from Scratch ---
        print("Processing training data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(os.path.join(self.config.METADATA_DIR, "train.csv"))
        val_meta = pd.read_csv(os.path.join(self.config.METADATA_DIR, "validation.csv"))

        # Debug Sampling
        if self.config.DEBUG and self.config.SAMPLE_SIZE:
            print(f"Debug: Sampling {self.config.SAMPLE_SIZE} rows...")
            train_meta = train_meta.sample(
                n=min(len(train_meta), self.config.SAMPLE_SIZE),
                random_state=self.config.SEED,
            )
            val_meta = val_meta.sample(
                n=min(len(val_meta), int(self.config.SAMPLE_SIZE * 0.2)),
                random_state=self.config.SEED,
            )

        # Load Tracking
        # We only need tracking for the game_plays in our metadata
        relevant_gps = set(train_meta["game_play"].unique()) | set(
            val_meta["game_play"].unique()
        )
        tracking = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "train_player_tracking.csv")
        )
        tracking = tracking[tracking["game_play"].isin(relevant_gps)].copy()

        # Process Train
        # We process train first to fit scalers
        X_train, X_cat_train, y_train = self._process_pipeline(
            train_meta, tracking, is_training=True
        )

        # Process Val
        # Use the same tracking df (it contains val games too)
        X_val, X_cat_val, y_val = self._process_pipeline(
            val_meta, tracking, is_training=False
        )

        # Save to Cache
        print("Saving data to cache...")
        np.save(os.path.join(self.cache_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.cache_dir, "X_cat_train.npy"), X_cat_train)
        np.save(os.path.join(self.cache_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.cache_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.cache_dir, "X_cat_val.npy"), X_cat_val)
        np.save(os.path.join(self.cache_dir, "y_val.npy"), y_val)

        return X_train, X_cat_train, y_train, X_val, X_cat_val, y_val

    def load_and_process_test(self, load_cached_data=True):
        """
        Loads test data, processes it, and returns tensors.
        """
        cache_files = ["X_test.npy", "X_cat_test.npy", "test_ids.npy"]

        if load_cached_data:
            all_exist = True
            for f in cache_files:
                if not os.path.exists(os.path.join(self.cache_dir, f)):
                    all_exist = False
                    break
            if all_exist:
                print("Loading cached test data...")
                X_test = np.load(os.path.join(self.cache_dir, "X_test.npy"))
                X_cat_test = np.load(os.path.join(self.cache_dir, "X_cat_test.npy"))
                test_ids = np.load(
                    os.path.join(self.cache_dir, "test_ids.npy"), allow_pickle=True
                )
                return X_test, X_cat_test, test_ids

        print("Processing test data from scratch...")

        # Load Metadata
        test_meta = pd.read_csv(os.path.join(self.config.METADATA_DIR, "test.csv"))

        # Load Tracking
        tracking = pd.read_csv(
            os.path.join(self.config.INPUT_DIR, "test_player_tracking.csv")
        )

        # Process
        X_test, X_cat_test, _ = self._process_pipeline(
            test_meta, tracking, is_training=False
        )
        test_ids = test_meta["contact_id"].values

        # Cache
        np.save(os.path.join(self.cache_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.cache_dir, "X_cat_test.npy"), X_cat_test)
        np.save(os.path.join(self.cache_dir, "test_ids.npy"), test_ids)

        return X_test, X_cat_test, test_ids
