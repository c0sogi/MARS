import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    N_SPLITS,
    SEED,
)
from library.utils import save_npy, load_npy, seed_everything


class LeafDataManager:
    """
    Manages data loading, splitting, and topology transformations for the
    Manifold-Expanded Linear Discriminant Ensemble strategy.
    """

    def __init__(self, feature_extractor):
        """
        Args:
            feature_extractor: Instance of library.feature_extractor.FeatureExtractor
        """
        self.fe = feature_extractor
        self.le = LabelEncoder()

        # Data containers
        self.train_img = None
        self.train_tab = None
        self.train_y = None
        self.test_img = None
        self.test_tab = None
        self.test_ids = None

        # Cache directory for fold-specific data
        self.cache_dir = os.path.join(WORKING_DIR, "data_loader_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._is_setup = False

    def setup_data(self, load_cached_data=True):
        """
        Loads metadata, extracts base features, and encodes labels.
        Must be called before getting fold data.

        Args:
            load_cached_data (bool): Whether to attempt loading features from cache.
        """
        if self._is_setup:
            return

        print("Setting up data manager...")

        # 1. Load Metadata
        if not os.path.exists(TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Train metadata not found at {TRAIN_METADATA_PATH}"
            )
        if not os.path.exists(TEST_METADATA_PATH):
            raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_test = pd.read_csv(TEST_METADATA_PATH)

        # 2. Extract Features (Delegate to FeatureExtractor)
        # This handles the heavy lifting and its own caching
        print("Loading/Extracting training features...")
        self.train_img, self.train_tab, _, train_labels = self.fe.extract_features(
            df_train, "train", load_cached_data=load_cached_data
        )

        print("Loading/Extracting test features...")
        self.test_img, self.test_tab, self.test_ids, _ = self.fe.extract_features(
            df_test, "test", load_cached_data=load_cached_data
        )

        # 3. Encode Labels
        print("Encoding labels...")
        self.train_y = self.le.fit_transform(train_labels)

        self._is_setup = True
        print(f"Data setup complete. Classes: {len(self.le.classes_)}")

    def get_fold_data(self, fold_idx, load_cached_data=True):
        """
        Retrieves data for a specific fold with topology transformations.

        Training Data: Manifold Expansion (4 views -> 4 samples)
        Validation Data: Centroid Consolidation (4 views -> 1 mean sample)

        Args:
            fold_idx (int): Index of the fold (0 to N_SPLITS-1).
            load_cached_data (bool): Whether to use disk cache for the processed fold.

        Returns:
            tuple: (X_train, y_train, X_val, y_val)
                   X arrays are concatenated [ImageEmbeddings, TabularFeatures].
        """
        if not self._is_setup:
            self.setup_data(load_cached_data=load_cached_data)

        # Construct cache paths
        cache_prefix = os.path.join(self.cache_dir, f"fold_{fold_idx}")
        path_X_train = f"{cache_prefix}_X_train.npy"
        path_y_train = f"{cache_prefix}_y_train.npy"
        path_X_val = f"{cache_prefix}_X_val.npy"
        path_y_val = f"{cache_prefix}_y_val.npy"

        # 1. Try Cache
        if load_cached_data:
            if (
                os.path.exists(path_X_train)
                and os.path.exists(path_y_train)
                and os.path.exists(path_X_val)
                and os.path.exists(path_y_val)
            ):
                return (
                    load_npy(path_X_train),
                    load_npy(path_y_train),
                    load_npy(path_X_val),
                    load_npy(path_y_val),
                )

        # 2. Compute Data
        # Stratified Split
        # We use the same seed to ensure reproducibility
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        # We need to iterate to the specific fold
        splits = list(skf.split(self.train_tab, self.train_y))
        train_indices, val_indices = splits[fold_idx]

        # --- Process Training Data (Manifold Expansion) ---
        # Select subset
        tr_img = self.train_img[train_indices]  # (N_train, 4, D)
        tr_tab = self.train_tab[train_indices]  # (N_train, 192)
        tr_y = self.train_y[train_indices]  # (N_train,)

        # Expand
        # Flatten views: (N, 4, D) -> (N*4, D)
        # Reshape collapses the first two dimensions (N, 4) into one (N*4)
        # Order: Img0_V0, Img0_V1, Img0_V2, Img0_V3, Img1_V0...
        N_tr, V, D = tr_img.shape
        X_train_img = tr_img.reshape(N_tr * V, D)

        # Repeat tabular: (N, 192) -> (N*4, 192)
        # We repeat elements to match the view flattening order
        X_train_tab = np.repeat(tr_tab, V, axis=0)

        # Repeat labels
        y_train = np.repeat(tr_y, V, axis=0)

        # Concatenate: (N*4, D + 192)
        X_train = np.concatenate([X_train_img, X_train_tab], axis=1)

        # --- Process Validation Data (Centroid Consolidation) ---
        # Select subset
        val_img = self.train_img[val_indices]  # (N_val, 4, D)
        val_tab = self.train_tab[val_indices]  # (N_val, 192)
        y_val = self.train_y[val_indices]  # (N_val,)

        # Average views: (N, 4, D) -> (N, D)
        X_val_img = np.mean(val_img, axis=1)

        # Concatenate: (N, D + 192)
        X_val = np.concatenate([X_val_img, val_tab], axis=1)

        # 3. Save to Cache
        save_npy(X_train, path_X_train)
        save_npy(y_train, path_y_train)
        save_npy(X_val, path_X_val)
        save_npy(y_val, path_y_val)

        return X_train, y_train, X_val, y_val

    def get_test_data(self, load_cached_data=True):
        """
        Retrieves test data with Centroid Consolidation.

        Returns:
            tuple: (X_test, test_ids)
        """
        if not self._is_setup:
            self.setup_data(load_cached_data=load_cached_data)

        path_X_test = os.path.join(self.cache_dir, "test_X.npy")
        path_ids_test = os.path.join(self.cache_dir, "test_ids.npy")

        if load_cached_data:
            if os.path.exists(path_X_test) and os.path.exists(path_ids_test):
                return load_npy(path_X_test), load_npy(path_ids_test)

        # Process Test Data
        # Average views: (N, 4, D) -> (N, D)
        X_test_img = np.mean(self.test_img, axis=1)

        # Concatenate
        X_test = np.concatenate([X_test_img, self.test_tab], axis=1)

        # Save
        save_npy(X_test, path_X_test)
        save_npy(self.test_ids, path_ids_test)

        return X_test, self.test_ids

    @property
    def classes(self):
        """Returns the list of class names."""
        return self.le.classes_
