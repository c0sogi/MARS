import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import library.config as conf


class LeafDataManager:
    """
    Manages data loading, preprocessing, and caching for the Leaf Classification task.
    Implements view generation for the Dynamic Multi-View Ensemble strategy.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.cache_file = os.path.join(conf.WORKING_DIR, "processed_data.npz")

    def load_all_data(self, load_cached_data=True):
        """
        Loads data, processes it into views, and returns the datasets.
        Uses caching to speed up subsequent runs.

        Args:
            load_cached_data (bool): If True, attempts to load from disk cache.

        Returns:
            tuple: (X_train_views, y_train, X_val_views, y_val, X_test_views, test_ids, classes)
        """
        # Ensure working directory exists
        os.makedirs(conf.WORKING_DIR, exist_ok=True)

        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}...")
            try:
                return self._load_cache()
            except Exception as e:
                print(f"Failed to load cache ({e}). Reprocessing from scratch...")

        print("Processing data from scratch...")
        return self._process_from_scratch()

    def _process_from_scratch(self):
        # 1. Load Metadata CSVs
        if not os.path.exists(conf.TRAIN_PATH):
            raise FileNotFoundError(f"Train file not found at {conf.TRAIN_PATH}")

        df_train = pd.read_csv(conf.TRAIN_PATH)
        df_val = pd.read_csv(conf.VAL_PATH)
        df_test = pd.read_csv(conf.TEST_PATH)

        # 2. Process Targets
        # Fit encoder on training species only
        y_train = self.le.fit_transform(df_train[conf.TARGET_COL])
        # Transform validation species (assuming stratified split ensures all classes are present,
        # but handle potential unseen classes safely if needed - though metadata ensures stratification)
        y_val = self.le.transform(df_val[conf.TARGET_COL])
        classes = self.le.classes_

        # Extract Test IDs for submission
        test_ids = df_test[conf.ID_COL].values

        # 3. Process Features
        # Extract raw feature matrices
        X_train_raw = df_train[conf.ALL_FEATURE_COLS].values
        X_val_raw = df_val[conf.ALL_FEATURE_COLS].values
        X_test_raw = df_test[conf.ALL_FEATURE_COLS].values

        # Scale features (Fit on Train, Transform All)
        self.scaler.fit(X_train_raw)
        X_train_scaled = self.scaler.transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # 4. Create Views
        X_train_views = self._create_views(X_train_scaled)
        X_val_views = self._create_views(X_val_scaled)
        X_test_views = self._create_views(X_test_scaled)

        # 5. Cache Results
        self._save_cache(
            X_train_views, y_train, X_val_views, y_val, X_test_views, test_ids, classes
        )

        return (
            X_train_views,
            y_train,
            X_val_views,
            y_val,
            X_test_views,
            test_ids,
            classes,
        )

    def _create_views(self, X_scaled):
        """
        Slices the full scaled feature matrix into specific views defined in config.
        """
        # Map column names to integer indices based on the master list order
        col_to_idx = {name: i for i, name in enumerate(conf.ALL_FEATURE_COLS)}

        views = {}
        for view_name, col_names in conf.VIEWS.items():
            # Get indices for this view
            indices = [col_to_idx[c] for c in col_names]
            # Slice
            views[view_name] = X_scaled[:, indices]

        return views

    def _save_cache(
        self,
        X_train_views,
        y_train,
        X_val_views,
        y_val,
        X_test_views,
        test_ids,
        classes,
    ):
        """
        Saves data to a single .npz file. Dictionaries are flattened with prefixes.
        """
        data_dict = {
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
            "classes": classes,
        }

        # Flatten view dictionaries
        for view_name, data in X_train_views.items():
            data_dict[f"train_view_{view_name}"] = data

        for view_name, data in X_val_views.items():
            data_dict[f"val_view_{view_name}"] = data

        for view_name, data in X_test_views.items():
            data_dict[f"test_view_{view_name}"] = data

        np.savez(self.cache_file, **data_dict)
        print(f"Data successfully cached to {self.cache_file}")

    def _load_cache(self):
        """
        Loads data from .npz file and reconstructs the view dictionaries.
        """
        loaded = np.load(
            self.cache_file, allow_pickle=True
        )  # allow_pickle=True strictly for string arrays (classes) if needed, though standard types are fine

        y_train = loaded["y_train"]
        y_val = loaded["y_val"]
        test_ids = loaded["test_ids"]
        classes = loaded["classes"]

        X_train_views = {}
        X_val_views = {}
        X_test_views = {}

        # Reconstruct dictionaries based on key prefixes
        for key in loaded.files:
            if key.startswith("train_view_"):
                view_name = key.replace("train_view_", "")
                X_train_views[view_name] = loaded[key]
            elif key.startswith("val_view_"):
                view_name = key.replace("val_view_", "")
                X_val_views[view_name] = loaded[key]
            elif key.startswith("test_view_"):
                view_name = key.replace("test_view_", "")
                X_test_views[view_name] = loaded[key]

        return (
            X_train_views,
            y_train,
            X_val_views,
            y_val,
            X_test_views,
            test_ids,
            classes,
        )
