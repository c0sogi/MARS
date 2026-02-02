import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


class FeatureEngineer:
    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.metadata_dir = Config.METADATA_DIR

        # Define file paths for cache
        self.cache_files = {
            "train_X": os.path.join(self.working_dir, "train_X.npy"),
            "train_y": os.path.join(self.working_dir, "train_y.npy"),
            "val_X": os.path.join(self.working_dir, "val_X.npy"),
            "val_y": os.path.join(self.working_dir, "val_y.npy"),
            "test_X": os.path.join(self.working_dir, "test_X.npy"),
            "test_ids": os.path.join(self.working_dir, "test_ids.npy"),
        }

    def _generate_features(self, df):
        """
        Applies physics-informed feature engineering to the dataframe.
        """
        # Ensure we work on a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # 1. Cyclical Augmentation (Keep raw Aspect as per strategy)
        # Convert degrees to radians for numpy functions
        aspect_rad = np.radians(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

        # 2. Geometric Magnitude
        # Euclidean distance to hydrology (hypotenuse)
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

        # 3. Directional Preservation
        # Absolute elevation of the water source
        df["Absolute_Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

        # 4. Global Context
        # Mean distance to key amenities
        df["Mean_Distance_To_Amenities"] = (
            df["Horizontal_Distance_To_Hydrology"]
            + df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Fire_Points"]
        ) / 3.0

        return df

    def process_data(self, load_cached_data=True):
        """
        Main pipeline to load, engineer, scale, and cache data.

        Args:
            load_cached_data (bool): If True, attempts to load from .npy files.

        Returns:
            tuple: (train_X, train_y, val_X, val_y, test_X, test_ids)
        """
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Check if cache exists
        all_cached = all(os.path.exists(path) for path in self.cache_files.values())

        if load_cached_data and all_cached:
            print(f"Loading cached data from {self.working_dir}...")
            train_X = np.load(self.cache_files["train_X"])
            train_y = np.load(self.cache_files["train_y"])
            val_X = np.load(self.cache_files["val_X"])
            val_y = np.load(self.cache_files["val_y"])
            test_X = np.load(self.cache_files["test_X"])
            test_ids = np.load(self.cache_files["test_ids"])
            return train_X, train_y, val_X, val_y, test_X, test_ids

        print("Processing data from scratch...")

        # 1. Load Data
        print(f"Loading parquet files from {self.metadata_dir}...")
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)
        df_test = pd.read_parquet(Config.TEST_PATH)

        # 2. Feature Engineering
        print("Generating physics-informed features...")
        df_train = self._generate_features(df_train)
        df_val = self._generate_features(df_val)
        df_test = self._generate_features(df_test)

        # 3. Define Column Groups
        # Continuous columns include original continuous + engineered ones
        # Config.CONTINUOUS_COLS already has the raw continuous cols.
        # We append the engineered ones.
        continuous_cols = Config.CONTINUOUS_COLS + Config.ENGINEERED_COLS
        binary_cols = Config.BINARY_COLS

        # 4. Scaling
        # Fit scaler ONLY on training data
        print("Fitting StandardScaler on training data...")
        scaler = StandardScaler()
        scaler.fit(df_train[continuous_cols])

        # Transform all sets
        train_cont = scaler.transform(df_train[continuous_cols])
        val_cont = scaler.transform(df_val[continuous_cols])
        test_cont = scaler.transform(df_test[continuous_cols])

        # 5. Assemble Final Arrays
        # Concatenate Scaled Continuous + Raw Binary
        print("Constructing final feature arrays...")

        # Helper to concat
        def get_X(df, cont_data):
            # Ensure binary cols are float32 for consistency with scaled data
            bin_data = df[binary_cols].values.astype(np.float32)
            return np.hstack([cont_data.astype(np.float32), bin_data])

        train_X = get_X(df_train, train_cont)
        val_X = get_X(df_val, val_cont)
        test_X = get_X(df_test, test_cont)

        # 6. Process Targets and IDs
        # Shift targets to 0-indexed (Class 1-7 -> 0-6)
        train_y = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
        val_y = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)
        test_ids = df_test[Config.ID_COL].values.astype(np.int64)

        # 7. Cache Data
        print(f"Saving processed data to {self.working_dir}...")
        np.save(self.cache_files["train_X"], train_X)
        np.save(self.cache_files["train_y"], train_y)
        np.save(self.cache_files["val_X"], val_X)
        np.save(self.cache_files["val_y"], val_y)
        np.save(self.cache_files["test_X"], test_X)
        np.save(self.cache_files["test_ids"], test_ids)

        return train_X, train_y, val_X, val_y, test_X, test_ids
