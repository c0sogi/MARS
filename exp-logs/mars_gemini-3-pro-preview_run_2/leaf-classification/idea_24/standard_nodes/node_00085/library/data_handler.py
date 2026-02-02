import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.image_processing import extract_morphological_features


class DataHandler:
    """
    Handles data loading, feature merging, view construction, and splitting for the RQLGE model.
    Manages caching of processed numpy arrays to ensure efficiency and reproducibility.
    """

    def __init__(self, debug=False):
        """
        Initialize the DataHandler.

        Args:
            debug (bool): If True, loads a small subset of data for debugging.
        """
        self.debug = debug
        self.config = Config(debug=debug)
        self.le = LabelEncoder()
        self.classes_ = None

    def get_data_splits(self, load_cached_data=True):
        """
        Loads training, validation, and test data, constructs feature views, and returns
        dictionaries containing the processed data.

        Args:
            load_cached_data (bool): If True, attempts to load processed splits from disk.

        Returns:
            tuple: (train_data, val_data, test_data)
                Each is a dictionary containing:
                - 'X_global': np.ndarray (N, 192)
                - 'X_morphological': np.ndarray (N, 11)
                - 'X_combined': np.ndarray (N, 203)
                - 'y': np.ndarray (N,) [only for train/val]
                - 'ids': np.ndarray (N,)
                - 'classes': np.ndarray [only for train]
        """
        # Define cache path for the consolidated splits
        cache_filename = "data_splits_debug.npz" if self.debug else "data_splits.npz"
        cache_path = os.path.join(self.config.CACHE_DIR, cache_filename)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading processed data splits from {cache_path}...")
            return self._load_splits_from_cache(cache_path)

        # 2. Load Metadata
        print("Loading metadata...")
        train_df = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(self.config.VAL_METADATA_PATH)
        test_df = pd.read_csv(self.config.TEST_METADATA_PATH)

        # Apply Debug Subsetting
        if self.debug:
            print("Debug mode: Subsetting data...")
            train_df = train_df.head(50)
            val_df = val_df.head(20)
            test_df = test_df.head(10)
            suffix = "_debug"
        else:
            suffix = ""

        # 3. Extract Morphological Features (Source B)
        # Note: The image_processing module handles its own caching logic.
        print("Extracting/Loading morphological features...")
        train_morph = extract_morphological_features(
            train_df, f"train{suffix}", load_cached_data
        )
        val_morph = extract_morphological_features(
            val_df, f"val{suffix}", load_cached_data
        )
        test_morph = extract_morphological_features(
            test_df, f"test{suffix}", load_cached_data
        )

        # 4. Process Splits and Construct Views
        print("Constructing feature views...")

        # Identify Global Feature Columns (Source A)
        # Exclude metadata columns
        exclude_cols = ["id", "species", "image_path"]
        global_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Identify Morphological Feature Columns
        # Exclude 'id'
        morph_cols = [c for c in train_morph.columns if c != "id"]

        # Process Train (Fit Encoder)
        train_data = self._process_single_split(
            train_df, train_morph, global_cols, morph_cols, is_train=True
        )

        # Process Val (Transform Encoder)
        val_data = self._process_single_split(
            val_df, val_morph, global_cols, morph_cols, is_train=False
        )

        # Process Test (No Labels)
        test_data = self._process_single_split(
            test_df, test_morph, global_cols, morph_cols, is_train=False, is_test=True
        )

        # 5. Save to Cache
        print(f"Saving processed splits to {cache_path}...")
        self._save_splits_to_cache(cache_path, train_data, val_data, test_data)

        return train_data, val_data, test_data

    def _process_single_split(
        self, meta_df, morph_df, global_cols, morph_cols, is_train=False, is_test=False
    ):
        """
        Helper to merge dataframes, extract arrays, and handle labels for a single split.
        """
        # Merge on ID
        merged_df = pd.merge(meta_df, morph_df, on="id", how="left")

        # Fill NaNs in morph features (if any image failed processing) with 0
        merged_df[morph_cols] = merged_df[morph_cols].fillna(0.0)

        # Extract IDs
        ids = merged_df["id"].values.astype(int)

        # Extract X Views (strictly float64)
        X_global = merged_df[global_cols].values.astype(np.float64)
        X_morph = merged_df[morph_cols].values.astype(np.float64)
        X_combined = np.hstack([X_global, X_morph]).astype(np.float64)

        data_dict = {
            "X_global": X_global,
            "X_morphological": X_morph,
            "X_combined": X_combined,
            "ids": ids,
        }

        # Handle Labels
        if not is_test:
            y_raw = merged_df["species"].values
            if is_train:
                self.le.fit(y_raw)
                self.classes_ = self.le.classes_

            y_enc = self.le.transform(y_raw)
            data_dict["y"] = y_enc

            if is_train:
                data_dict["classes"] = self.classes_

        return data_dict

    def _save_splits_to_cache(self, path, train_data, val_data, test_data):
        """
        Saves the dictionaries to a compressed npz file.
        Flattens the structure for storage.
        """
        save_dict = {}

        # Helper to add prefix
        def add_to_dict(prefix, data):
            for key, value in data.items():
                save_dict[f"{prefix}_{key}"] = value

        add_to_dict("train", train_data)
        add_to_dict("val", val_data)
        add_to_dict("test", test_data)

        np.savez_compressed(path, **save_dict)

    def _load_splits_from_cache(self, path):
        """
        Loads from npz and reconstructs the dictionaries.
        """
        loaded = np.load(path, allow_pickle=True)

        train_data = {}
        val_data = {}
        test_data = {}

        for key in loaded.files:
            if key.startswith("train_"):
                train_data[key.replace("train_", "")] = loaded[key]
            elif key.startswith("val_"):
                val_data[key.replace("val_", "")] = loaded[key]
            elif key.startswith("test_"):
                test_data[key.replace("test_", "")] = loaded[key]

        # Restore LabelEncoder classes if available
        if "classes" in train_data:
            self.classes_ = train_data["classes"]
            self.le.classes_ = self.classes_

        return train_data, val_data, test_data
