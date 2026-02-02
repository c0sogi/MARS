import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
from library.feature_engineering import FeatureEngineer


class NFLDataLoader:
    def __init__(self):
        self.config = Config
        self.fe = FeatureEngineer()

        # Artifact paths
        self.scaler_path = os.path.join(self.config.WORKING_DIR, "scaler.joblib")
        self.encoders_path = os.path.join(self.config.WORKING_DIR, "encoders.joblib")

        # Define the exact order of features per time step to ensure
        # the flattened vector can be correctly reshaped by the model.
        # Structure per lag: [P1_Kin, P1_Vis, P2_Kin, P2_Vis, Derived]
        self.feature_order_per_lag = []

        # 1. Player 1 Kinematic
        self.feature_order_per_lag.extend(
            [f"{c}_p1" for c in self.config.KINEMATIC_FEATURES]
        )
        # 2. Player 1 Visual
        self.feature_order_per_lag.extend(
            [f"{c}_p1" for c in self.config.VISUAL_FEATURES]
        )
        # 3. Player 2 Kinematic
        self.feature_order_per_lag.extend(
            [f"{c}_p2" for c in self.config.KINEMATIC_FEATURES]
        )
        # 4. Player 2 Visual
        self.feature_order_per_lag.extend(
            [f"{c}_p2" for c in self.config.VISUAL_FEATURES]
        )
        # 5. Derived Features (names as created in FeatureEngineer)
        # Note: In FE, these are named like 'rel_dist_{lag}', but here we define the base pattern
        self.derived_features = ["rel_dist", "rel_speed", "rel_accel", "closing_speed"]
        self.feature_order_per_lag.extend(self.derived_features)

    def get_feature_columns(self):
        """
        Constructs the full list of flattened feature column names across the time window.
        """
        cols = []
        start_lag = -self.config.HALF_WINDOW
        end_lag = self.config.HALF_WINDOW

        for lag in range(start_lag, end_lag + 1):
            # The FeatureEngineer appends _{lag} to all these columns
            # We reconstruct the exact column names expected in the dataframe

            # P1 Kinematic
            cols.extend([f"{c}_p1_{lag}" for c in self.config.KINEMATIC_FEATURES])
            # P1 Visual
            cols.extend([f"{c}_p1_{lag}" for c in self.config.VISUAL_FEATURES])
            # P2 Kinematic
            cols.extend([f"{c}_p2_{lag}" for c in self.config.KINEMATIC_FEATURES])
            # P2 Visual
            cols.extend([f"{c}_p2_{lag}" for c in self.config.VISUAL_FEATURES])
            # Derived
            cols.extend([f"{c}_{lag}" for c in self.derived_features])

        return cols

    def fit_scalers(self, df, num_cols):
        """
        Fits and saves StandardScaler on training data.
        """
        print("Fitting StandardScaler...")
        scaler = StandardScaler()
        scaler.fit(df[num_cols].astype(np.float32))
        joblib.dump(scaler, self.scaler_path)
        return scaler

    def fit_encoders(self, df, cat_cols):
        """
        Fits and saves LabelEncoders on training data.
        """
        print("Fitting LabelEncoders...")
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            # Ensure 'UNK' is in the classes if not present, though usually handled by FE filling
            unique_vals = df[col].astype(str).unique().tolist()
            if "UNK" not in unique_vals:
                unique_vals.append("UNK")
            le.fit(unique_vals)
            encoders[col] = le

        joblib.dump(encoders, self.encoders_path)
        return encoders

    def transform_data(self, df, num_cols, cat_cols, scaler, encoders):
        """
        Applies scaling and encoding to the dataframe.
        """
        # Numerical Scaling
        X_num = scaler.transform(df[num_cols].astype(np.float32))

        # Categorical Encoding
        X_cat_list = []
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels by mapping to 'UNK' or mode (here mapping to UNK index if exists)
            # We assume 'UNK' was fitted.

            # Efficient map with fallback
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            unk_val = mapping.get("UNK", 0)  # Default to 0 if UNK somehow missing

            encoded_col = (
                df[col].astype(str).map(mapping).fillna(unk_val).astype(int).values
            )
            X_cat_list.append(encoded_col.reshape(-1, 1))

        X_cat = np.hstack(X_cat_list)

        return X_num, X_cat

    def load_split(self, split_name, load_cached_data=True):
        """
        Main method to load, process, and return data for a specific split.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached parquet files from FE.

        Returns:
            dict: {
                'X_num': np.ndarray,
                'X_cat': np.ndarray,
                'y': np.ndarray (or None),
                'ids': np.ndarray (contact_ids),
                'meta': pd.DataFrame (metadata columns)
            }
        """
        print(f"[{split_name.upper()}] Loading and processing data...")

        # 1. Determine paths
        if split_name == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
            track_path = self.config.TRAIN_TRACKING_PATH
            helmets_path = self.config.TRAIN_HELMETS_PATH
            output_name = "train_features"
        elif split_name == "val":
            meta_path = self.config.VAL_METADATA_PATH
            track_path = self.config.TRAIN_TRACKING_PATH  # Val comes from train source
            helmets_path = self.config.TRAIN_HELMETS_PATH
            output_name = "val_features"
        elif split_name == "test":
            meta_path = self.config.TEST_METADATA_PATH
            track_path = self.config.TEST_TRACKING_PATH
            helmets_path = self.config.TEST_HELMETS_PATH
            output_name = "test_features"
        else:
            raise ValueError(f"Unknown split: {split_name}")

        # 2. Generate Features (handled by FeatureEngineer)
        df = self.fe.generate_features(
            metadata_path=meta_path,
            tracking_path=track_path,
            helmets_path=helmets_path,
            output_name=output_name,
            load_cached_data=load_cached_data,
        )

        # 3. Identify Columns
        num_cols = self.get_feature_columns()
        cat_cols = ["position_1", "team_1", "position_2", "team_2"]

        # Verify columns exist
        missing_cols = [c for c in num_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected feature columns: {missing_cols[:5]}...")

        # 4. Fit or Load Artifacts
        if split_name == "train":
            scaler = self.fit_scalers(df, num_cols)
            encoders = self.fit_encoders(df, cat_cols)
        else:
            if not os.path.exists(self.scaler_path) or not os.path.exists(
                self.encoders_path
            ):
                raise FileNotFoundError(
                    "Scaler/Encoders not found. Run 'train' split first."
                )
            scaler = joblib.load(self.scaler_path)
            encoders = joblib.load(self.encoders_path)

        # 5. Transform
        print(f"[{split_name.upper()}] Transforming features...")
        X_num, X_cat = self.transform_data(df, num_cols, cat_cols, scaler, encoders)

        # 6. Prepare Targets and Meta
        y = None
        if "contact" in df.columns:
            y = df["contact"].values.astype(int)

        ids = df["contact_id"].values

        # Keep relevant metadata for analysis/submission
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        meta_df = df[meta_cols].copy()

        print(f"[{split_name.upper()}] Data ready. Shape: {X_num.shape}")

        return {"X_num": X_num, "X_cat": X_cat, "y": y, "ids": ids, "meta": meta_df}
