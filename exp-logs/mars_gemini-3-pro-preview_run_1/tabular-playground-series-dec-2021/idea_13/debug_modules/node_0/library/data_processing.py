import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from library.utils import get_cache_dir, seed_everything

# Constants
TARGET_COL = "Cover_Type"
ID_COL = "Id"


class DataProcessor:
    def __init__(self):
        self.cache_dir = get_cache_dir()
        self.metadata_dir = "./metadata"
        seed_everything(42)

    def load_raw_data(self):
        """
        Loads the stratified train/val splits and test set from metadata.
        """
        train_path = os.path.join(self.metadata_dir, "train.csv")
        val_path = os.path.join(self.metadata_dir, "val.csv")
        test_path = os.path.join(self.metadata_dir, "test.csv")

        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation was successful."
            )

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        return df_train, df_val, df_test

    def engineer_common_features(self, df):
        """
        Applies geometric and physics-informed feature engineering.
        """
        df = df.copy()

        # Euclidean Distance to Hydrology
        # sqrt(Horizontal^2 + Vertical^2)
        h_dist = df["Horizontal_Distance_To_Hydrology"]
        v_dist = df["Vertical_Distance_To_Hydrology"]
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

        # Relative Elevation (Elevation of Hydrology source)
        # Elevation - Vertical_Distance
        df["Relative_Elevation_Hydrology"] = df["Elevation"] - v_dist

        # Cyclic Aspect Encoding
        # Transform 0-360 degrees to continuous sin/cos components
        df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
        df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

        return df

    def _create_dense_index(self, df, prefix):
        """
        Converts a group of OHE columns (e.g., Soil_Type1...40) into a single dense integer index column.
        Returns the dense column and the list of OHE columns found.
        """
        # Identify columns starting with prefix
        cols = [c for c in df.columns if c.startswith(prefix)]

        # Sort numerically by the integer suffix
        def get_suffix(col_name):
            try:
                return int(col_name[len(prefix) :])
            except ValueError:
                return 0

        cols = sorted(cols, key=get_suffix)

        if not cols:
            return None, []

        # Create indices vector [1, 2, ..., N]
        # We use 1-based indexing so 0 can represent 'missing' or 'padding' if needed
        indices = np.arange(1, len(cols) + 1, dtype=np.int32)

        # Vectorized dot product to get the index where value is 1
        # shape: (n_samples, n_cols) @ (n_cols,) -> (n_samples,)
        dense_col = df[cols].values @ indices

        return dense_col.astype(np.int32), cols

    def get_xgb_data(self, load_cached_data=True):
        """
        Prepares data for XGBoost pipeline.
        - Retains OHE columns.
        - Adds dense integer indices.
        - Applies common engineering.
        - Returns (X_train, y_train, X_val, y_val, X_test, label_encoder)
        """
        # Define cache paths
        cache_files = {
            "X_train": os.path.join(self.cache_dir, "xgb_X_train.parquet"),
            "X_val": os.path.join(self.cache_dir, "xgb_X_val.parquet"),
            "X_test": os.path.join(self.cache_dir, "xgb_X_test.parquet"),
            "y_train": os.path.join(self.cache_dir, "xgb_y_train.npy"),
            "y_val": os.path.join(self.cache_dir, "xgb_y_val.npy"),
            "classes": os.path.join(self.cache_dir, "xgb_classes.npy"),
        }

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading cached XGBoost data...")
            X_train = pd.read_parquet(cache_files["X_train"])
            X_val = pd.read_parquet(cache_files["X_val"])
            X_test = pd.read_parquet(cache_files["X_test"])
            y_train = np.load(cache_files["y_train"])
            y_val = np.load(cache_files["y_val"])
            classes = np.load(cache_files["classes"])

            le = LabelEncoder()
            le.classes_ = classes
            return X_train, y_train, X_val, y_val, X_test, le

        print("Processing XGBoost data from scratch...")
        df_train, df_val, df_test = self.load_raw_data()

        # Process each split
        datasets = {"train": df_train, "val": df_val, "test": df_test}
        processed_dfs = {}

        for name, df in datasets.items():
            # 1. Common Engineering
            df = self.engineer_common_features(df)

            # 2. Densification (Keep OHE for XGB)
            soil_dense, _ = self._create_dense_index(df, "Soil_Type")
            df["Soil_Type_Index"] = soil_dense

            wild_dense, _ = self._create_dense_index(df, "Wilderness_Area")
            df["Wilderness_Area_Index"] = wild_dense

            # 3. Drop ID
            if ID_COL in df.columns:
                df = df.drop(columns=[ID_COL])

            processed_dfs[name] = df

        # Split X and y
        X_train = processed_dfs["train"].drop(columns=[TARGET_COL])
        y_train_raw = processed_dfs["train"][TARGET_COL].values

        X_val = processed_dfs["val"].drop(columns=[TARGET_COL])
        y_val_raw = processed_dfs["val"][TARGET_COL].values

        X_test = processed_dfs["test"]  # No target

        # Encode Targets
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val = le.transform(y_val_raw)

        # Save to cache
        print("Saving XGBoost data to cache...")
        X_train.to_parquet(cache_files["X_train"])
        X_val.to_parquet(cache_files["X_val"])
        X_test.to_parquet(cache_files["X_test"])
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["classes"], le.classes_)

        return X_train, y_train, X_val, y_val, X_test, le

    def get_nn_data(self, load_cached_data=True):
        """
        Prepares data for Neural Network pipeline.
        - Drops OHE columns (uses dense indices for embeddings).
        - Applies QuantileTransformer to continuous features.
        - Returns (X_train, y_train, X_val, y_val, X_test, label_encoder)
        """
        cache_files = {
            "X_train": os.path.join(self.cache_dir, "nn_X_train.parquet"),
            "X_val": os.path.join(self.cache_dir, "nn_X_val.parquet"),
            "X_test": os.path.join(self.cache_dir, "nn_X_test.parquet"),
            "y_train": os.path.join(self.cache_dir, "nn_y_train.npy"),
            "y_val": os.path.join(self.cache_dir, "nn_y_val.npy"),
            "classes": os.path.join(self.cache_dir, "nn_classes.npy"),
        }

        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading cached NN data...")
            X_train = pd.read_parquet(cache_files["X_train"])
            X_val = pd.read_parquet(cache_files["X_val"])
            X_test = pd.read_parquet(cache_files["X_test"])
            y_train = np.load(cache_files["y_train"])
            y_val = np.load(cache_files["y_val"])
            classes = np.load(cache_files["classes"])

            le = LabelEncoder()
            le.classes_ = classes
            return X_train, y_train, X_val, y_val, X_test, le

        print("Processing NN data from scratch...")
        df_train, df_val, df_test = self.load_raw_data()

        # 1. Common Engineering
        df_train = self.engineer_common_features(df_train)
        df_val = self.engineer_common_features(df_val)
        df_test = self.engineer_common_features(df_test)

        # 2. Densification and OHE Removal
        # For NN, we strictly want dense indices for embeddings and continuous vars for dense layers.
        cat_prefixes = ["Soil_Type", "Wilderness_Area"]
        dense_features = []

        for df in [df_train, df_val, df_test]:
            for prefix in cat_prefixes:
                dense, ohe_cols = self._create_dense_index(df, prefix)
                col_name = f"{prefix}_Index"
                df[col_name] = dense
                df.drop(columns=ohe_cols, inplace=True)
                if col_name not in dense_features:
                    dense_features.append(col_name)

        # 3. Identify Continuous Features
        # All columns that are not Target, ID, or Dense Indices
        exclude_cols = [TARGET_COL, ID_COL] + dense_features
        cont_features = [c for c in df_train.columns if c not in exclude_cols]

        # 4. Quantile Transformation (Normal)
        # Fit on Train, Transform All
        qt = QuantileTransformer(
            output_distribution="normal", random_state=42, subsample=100000
        )

        # Drop IDs before transform
        for df in [df_train, df_val, df_test]:
            if ID_COL in df.columns:
                df.drop(columns=[ID_COL], inplace=True)

        df_train[cont_features] = qt.fit_transform(df_train[cont_features])
        df_val[cont_features] = qt.transform(df_val[cont_features])
        df_test[cont_features] = qt.transform(df_test[cont_features])

        # Cast continuous to float32 to save memory
        for c in cont_features:
            df_train[c] = df_train[c].astype(np.float32)
            df_val[c] = df_val[c].astype(np.float32)
            df_test[c] = df_test[c].astype(np.float32)

        # Split X and y
        X_train = df_train.drop(columns=[TARGET_COL])
        y_train_raw = df_train[TARGET_COL].values

        X_val = df_val.drop(columns=[TARGET_COL])
        y_val_raw = df_val[TARGET_COL].values

        X_test = df_test

        # Encode Targets
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val = le.transform(y_val_raw)

        # Save to cache
        print("Saving NN data to cache...")
        X_train.to_parquet(cache_files["X_train"])
        X_val.to_parquet(cache_files["X_val"])
        X_test.to_parquet(cache_files["X_test"])
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["classes"], le.classes_)

        return X_train, y_train, X_val, y_val, X_test, le
