import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.features import generate_macro_features
from library.utils import set_seed


class LeafDataManager:
    """
    Manages data loading, merging, preprocessing, and caching for the leaf classification task.
    Implements the Multi-Resolution view strategy (Global, Macro, Combined).
    """

    def __init__(self, metadata_dir="./metadata", cache_dir="./working/idea_29"):
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self, load_cached_data=True):
        """
        Loads the dataset, performing feature extraction and preprocessing if not cached.

        Args:
            load_cached_data (bool): If True, attempts to load processed arrays from disk.

        Returns:
            dict: A dictionary containing training, validation, and test arrays for
                  Global, Macro, and Combined views, plus labels and IDs.
        """
        set_seed(42)

        # Define expected cache files
        cache_files = {
            "X_train_global": "X_train_global.npy",
            "X_train_macro": "X_train_macro.npy",
            "X_train_combined": "X_train_combined.npy",
            "y_train": "y_train.npy",
            "X_val_global": "X_val_global.npy",
            "X_val_macro": "X_val_macro.npy",
            "X_val_combined": "X_val_combined.npy",
            "y_val": "y_val.npy",
            "X_test_global": "X_test_global.npy",
            "X_test_macro": "X_test_macro.npy",
            "X_test_combined": "X_test_combined.npy",
            "test_ids": "test_ids.npy",
            "classes": "classes.npy",
        }

        # Check if cache exists
        cache_exists = all(
            os.path.exists(os.path.join(self.cache_dir, f))
            for f in cache_files.values()
        )

        if load_cached_data and cache_exists:
            print(f"Loading preprocessed data from cache: {self.cache_dir}")
            data = {}
            for key, filename in cache_files.items():
                data[key] = np.load(
                    os.path.join(self.cache_dir, filename), allow_pickle=True
                )
            return data

        print("Cache miss or reload requested. Processing data from scratch...")

        # 1. Load Metadata
        train_path = os.path.join(self.metadata_dir, "train.csv")
        val_path = os.path.join(self.metadata_dir, "val.csv")
        test_path = os.path.join(self.metadata_dir, "test.csv")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # 2. Generate/Load Macro Features (Morphometrics)
        # This calls the library function which handles its own caching of raw extraction
        macro_train_df = generate_macro_features(train_path, "train", load_cached_data)
        macro_val_df = generate_macro_features(val_path, "val", load_cached_data)
        macro_test_df = generate_macro_features(test_path, "test", load_cached_data)

        # 3. Extract and Merge Features
        # Identify Micro (Global) columns: margin_*, shape_*, texture_*
        micro_cols = [
            c for c in df_train.columns if c.startswith(("margin", "shape", "texture"))
        ]

        def process_split(df_meta, df_macro):
            # Extract Micro features directly from metadata
            X_micro = df_meta[micro_cols].values.astype(np.float64)

            # Merge Macro features on ID to ensure alignment
            # df_macro contains 'id' and feature columns
            df_merged = df_meta[["id"]].merge(df_macro, on="id", how="left")

            # Drop ID from macro features for the matrix
            macro_feat_cols = [c for c in df_macro.columns if c != "id"]
            X_macro = df_merged[macro_feat_cols].values.astype(np.float64)

            return X_micro, X_macro

        X_train_global_raw, X_train_macro_raw = process_split(df_train, macro_train_df)
        X_val_global_raw, X_val_macro_raw = process_split(df_val, macro_val_df)
        X_test_global_raw, X_test_macro_raw = process_split(df_test, macro_test_df)

        # 4. Process Targets
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["species"])
        y_val = le.transform(df_val["species"])
        classes = le.classes_
        test_ids = df_test["id"].values

        # 5. Preprocessing Pipeline
        # We apply PowerTransformer (Yeo-Johnson) to Gaussianize features.
        # CRITICAL: Fit on TRAIN only, Transform VAL and TEST.
        # We process Global and Macro views separately.

        # Pipeline for Global View
        pt_global = PowerTransformer(method="yeo-johnson")
        X_train_global = pt_global.fit_transform(X_train_global_raw).astype(np.float64)
        X_val_global = pt_global.transform(X_val_global_raw).astype(np.float64)
        X_test_global = pt_global.transform(X_test_global_raw).astype(np.float64)

        # Pipeline for Macro View
        pt_macro = PowerTransformer(method="yeo-johnson")
        X_train_macro = pt_macro.fit_transform(X_train_macro_raw).astype(np.float64)
        X_val_macro = pt_macro.transform(X_val_macro_raw).astype(np.float64)
        X_test_macro = pt_macro.transform(X_test_macro_raw).astype(np.float64)

        # Create Combined View (Concatenation of transformed views)
        # Since Yeo-Johnson is univariate, T(A+B) == T(A) + T(B) effectively.
        X_train_combined = np.hstack([X_train_global, X_train_macro])
        X_val_combined = np.hstack([X_val_global, X_val_macro])
        X_test_combined = np.hstack([X_test_global, X_test_macro])

        # 6. Save to Cache
        data = {
            "X_train_global": X_train_global,
            "X_train_macro": X_train_macro,
            "X_train_combined": X_train_combined,
            "y_train": y_train,
            "X_val_global": X_val_global,
            "X_val_macro": X_val_macro,
            "X_val_combined": X_val_combined,
            "y_val": y_val,
            "X_test_global": X_test_global,
            "X_test_macro": X_test_macro,
            "X_test_combined": X_test_combined,
            "test_ids": test_ids,
            "classes": classes,
        }

        for key, filename in cache_files.items():
            np.save(os.path.join(self.cache_dir, filename), data[key])

        print("Data processing complete and cached.")
        return data
