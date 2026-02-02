import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import generate_config_hash


class FeatureEngineer:
    """
    Handles feature engineering, scaling, and caching for the ventilator dataset.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.CACHE_DIR
        self.hash_str = generate_config_hash()

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self):
        """Returns a dictionary of paths for cached artifacts based on config hash."""
        return {
            "train_x": os.path.join(self.cache_dir, f"train_x_{self.hash_str}.npy"),
            "train_y": os.path.join(self.cache_dir, f"train_y_{self.hash_str}.npy"),
            "val_x": os.path.join(self.cache_dir, f"val_x_{self.hash_str}.npy"),
            "val_y": os.path.join(self.cache_dir, f"val_y_{self.hash_str}.npy"),
            "test_x": os.path.join(self.cache_dir, f"test_x_{self.hash_str}.npy"),
            "test_ids": os.path.join(self.cache_dir, f"test_ids_{self.hash_str}.npy"),
            "scaler_center": os.path.join(
                self.cache_dir, f"scaler_center_{self.hash_str}.npy"
            ),
            "scaler_scale": os.path.join(
                self.cache_dir, f"scaler_scale_{self.hash_str}.npy"
            ),
        }

    def _compute_physics_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes physics-fidelity and time-series features using vectorized operations.
        Reshapes data to (N_breaths, 80, N_features).
        """
        # Ensure data integrity
        n_rows = len(df)
        if n_rows % self.config.N_STEPS != 0:
            raise ValueError(
                f"Data length {n_rows} is not divisible by {self.config.N_STEPS}"
            )

        n_breaths = n_rows // self.config.N_STEPS

        # Extract base columns and reshape to (N_breaths, 80)
        # We assume data is sorted by breath_id and time_step (guaranteed by metadata script)
        u_in = df["u_in"].values.reshape(n_breaths, self.config.N_STEPS)
        u_out = df["u_out"].values.reshape(n_breaths, self.config.N_STEPS)
        R = df["R"].values.reshape(n_breaths, self.config.N_STEPS)
        C = df["C"].values.reshape(n_breaths, self.config.N_STEPS)
        time_step = df["time_step"].values.reshape(n_breaths, self.config.N_STEPS)

        # --- Physics Engineering ---

        # 1. Time Delta (dt)
        # dt[t] = time[t] - time[t-1]. First step is 0.
        dt = np.zeros_like(time_step)
        dt[:, 1:] = time_step[:, 1:] - time_step[:, :-1]

        # 2. Volume Integration (Integral of Flow)
        # volume = sum(u_in * dt)
        volume = np.cumsum(u_in * dt, axis=1)

        # 3. Dynamics (Lags)
        # Shift u_in to the right, padding with 0
        u_in_lag1 = np.hstack([np.zeros((n_breaths, 1)), u_in[:, :-1]])
        u_in_lag2 = np.hstack([np.zeros((n_breaths, 2)), u_in[:, :-2]])
        u_in_lag3 = np.hstack([np.zeros((n_breaths, 3)), u_in[:, :-3]])
        u_in_lag4 = np.hstack([np.zeros((n_breaths, 4)), u_in[:, :-4]])

        # 4. Dynamics (Differences)
        # diff1 = u_in[t] - u_in[t-1]
        u_in_diff1 = np.zeros_like(u_in)
        u_in_diff1[:, 1:] = u_in[:, 1:] - u_in[:, :-1]

        # diff2 = diff1[t] - diff1[t-1]
        u_in_diff2 = np.zeros_like(u_in)
        u_in_diff2[:, 1:] = u_in_diff1[:, 1:] - u_in_diff1[:, :-1]

        # 5. Physics Interactions (Equation of Motion Terms)
        R_u_in = R * u_in  # Resistive pressure component
        vol_C = volume / C  # Elastic pressure component

        # --- Feature Assembly ---

        # Map feature names to computed arrays
        feature_map = {
            "time_step": time_step,
            "u_in": u_in,
            "u_out": u_out,
            "R": R,
            "C": C,
            "u_in_lag1": u_in_lag1,
            "u_in_lag2": u_in_lag2,
            "u_in_lag3": u_in_lag3,
            "u_in_lag4": u_in_lag4,
            "u_in_diff1": u_in_diff1,
            "u_in_diff2": u_in_diff2,
            "volume": volume,
            "R_u_in": R_u_in,
            "vol_C": vol_C,
        }

        # Stack features along the last axis in the order defined by Config
        features_list = []
        for col in self.config.FEATURE_COLS:
            if col in feature_map:
                # Add new axis to stack: (N, 80) -> (N, 80, 1)
                features_list.append(feature_map[col][..., np.newaxis])
            else:
                raise KeyError(
                    f"Feature '{col}' defined in Config but not computed in FeatureEngineer."
                )

        # Concatenate to form (N_breaths, 80, N_features)
        X_3d = np.concatenate(features_list, axis=2)

        return X_3d

    def process_and_cache(self):
        """
        Loads raw CSVs, computes features, fits scaler, and saves processed data to cache.
        """
        paths = self._get_cache_paths()

        print("Loading raw metadata CSVs...")
        train_df = pd.read_csv(self.config.TRAIN_CSV)
        val_df = pd.read_csv(self.config.VAL_CSV)
        test_df = pd.read_csv(self.config.TEST_CSV)

        # Debug Sampling
        if self.config.DEBUG:
            print(
                f"DEBUG MODE: Sampling {self.config.DEBUG_SAMPLE_SIZE} breaths per split."
            )

            def sample_breaths(df):
                unique_breaths = df["breath_id"].unique()
                if len(unique_breaths) > self.config.DEBUG_SAMPLE_SIZE:
                    selected_breaths = unique_breaths[: self.config.DEBUG_SAMPLE_SIZE]
                    return df[df["breath_id"].isin(selected_breaths)].copy()
                return df

            train_df = sample_breaths(train_df)
            val_df = sample_breaths(val_df)
            test_df = sample_breaths(test_df)

        # Extract Targets and IDs (Reshape targets to match sequence structure)
        y_train = train_df[self.config.TARGET_COL].values.reshape(
            -1, self.config.N_STEPS
        )
        y_val = val_df[self.config.TARGET_COL].values.reshape(-1, self.config.N_STEPS)
        test_ids = test_df[
            self.config.ID_COL
        ].values  # Keep flat for submission mapping

        # Compute Features
        print("Computing physics features for Training set...")
        X_train = self._compute_physics_features(train_df)

        print("Computing physics features for Validation set...")
        X_val = self._compute_physics_features(val_df)

        print("Computing physics features for Test set...")
        X_test = self._compute_physics_features(test_df)

        # --- Scaling ---
        print("Fitting RobustScaler...")

        # Identify columns to scale (exclude binary 'u_out')
        cols_to_scale = [c for c in self.config.FEATURE_COLS if c != "u_out"]
        scale_indices = [self.config.FEATURE_COLS.index(c) for c in cols_to_scale]

        # Flatten the training subset for fitting the scaler
        X_train_subset = X_train[:, :, scale_indices].reshape(-1, len(scale_indices))

        scaler = RobustScaler()
        scaler.fit(X_train_subset)

        center = scaler.center_
        scale = scaler.scale_

        # Helper to apply scaling manually (avoiding pickle)
        def apply_scaling(X, indices, c, s):
            # X: (N, 80, F)
            X_scaled = X.copy()
            # Reshape stats for broadcasting: (1, 1, F_subset)
            c_reshaped = c.reshape(1, 1, -1)
            s_reshaped = s.reshape(1, 1, -1)

            # Apply (X - center) / scale
            # Note: RobustScaler handles small scales, but here we trust the fitted values
            subset = X_scaled[:, :, indices]
            subset = (subset - c_reshaped) / s_reshaped
            X_scaled[:, :, indices] = subset
            return X_scaled

        print("Applying scaling to datasets...")
        X_train = apply_scaling(X_train, scale_indices, center, scale)
        X_val = apply_scaling(X_val, scale_indices, center, scale)
        X_test = apply_scaling(X_test, scale_indices, center, scale)

        # --- Caching ---
        print(f"Saving processed datasets to {self.cache_dir}...")
        np.save(paths["train_x"], X_train)
        np.save(paths["train_y"], y_train)
        np.save(paths["val_x"], X_val)
        np.save(paths["val_y"], y_val)
        np.save(paths["test_x"], X_test)
        np.save(paths["test_ids"], test_ids)
        np.save(paths["scaler_center"], center)
        np.save(paths["scaler_scale"], scale)

        return {
            "train_x": X_train,
            "train_y": y_train,
            "val_x": X_val,
            "val_y": y_val,
            "test_x": X_test,
            "test_ids": test_ids,
        }

    def load_data(self, load_cached=True):
        """
        Main entry point. Loads from cache if available and requested, otherwise processes from scratch.
        """
        paths = self._get_cache_paths()
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached and all_exist:
            print(f"Loading cached features from {self.cache_dir}...")
            return {
                "train_x": np.load(paths["train_x"]),
                "train_y": np.load(paths["train_y"]),
                "val_x": np.load(paths["val_x"]),
                "val_y": np.load(paths["val_y"]),
                "test_x": np.load(paths["test_x"]),
                "test_ids": np.load(paths["test_ids"]),
            }
        else:
            if load_cached and not all_exist:
                print("Cached files not found.")
            return self.process_and_cache()


def get_datasets(load_cached=True):
    """
    Public API to retrieve processed datasets.
    Returns a dictionary containing:
        - train_x: (N_train, 80, F)
        - train_y: (N_train, 80)
        - val_x:   (N_val, 80, F)
        - val_y:   (N_val, 80)
        - test_x:  (N_test, 80, F)
        - test_ids: (N_test * 80,)
    """
    engineer = FeatureEngineer()
    return engineer.load_data(load_cached=load_cached)
