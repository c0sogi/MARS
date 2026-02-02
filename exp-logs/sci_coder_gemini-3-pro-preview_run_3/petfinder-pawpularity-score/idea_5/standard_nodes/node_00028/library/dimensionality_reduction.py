import os
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple, List

from library.config import Config
from library.utils import get_config_hash, seed_everything


class IndependentPCA:
    """
    Manages independent PCA compression for multiple feature backbones.
    Fits a separate PCA for each backbone to retain a specific variance,
    then concatenates the compressed features with metadata.
    """

    def __init__(
        self, variance_threshold: float = Config.PCA_VARIANCE, seed: int = Config.SEED
    ):
        self.variance_threshold = variance_threshold
        self.seed = seed
        self.backbones = Config.BACKBONES
        self.pcas: Dict[str, PCA] = {}
        self.fitted = False

    def fit(self, data_dict: Dict[str, np.ndarray]):
        """
        Fits separate PCA models for each backbone present in the data dictionary.

        Args:
            data_dict: Dictionary containing raw features (keys: 'features_{backbone_name}').
        """
        print(
            f"Fitting Independent PCA models (Variance Threshold: {self.variance_threshold})..."
        )

        for name in self.backbones:
            feature_key = f"features_{name}"
            if feature_key not in data_dict:
                raise KeyError(
                    f"Expected feature key '{feature_key}' not found in data dictionary."
                )

            features = data_dict[feature_key]

            # Initialize PCA
            pca = PCA(n_components=self.variance_threshold, random_state=self.seed)

            # Fit PCA
            pca.fit(features)
            self.pcas[name] = pca

            print(
                f"  - {name}: Reduced {features.shape[1]} -> {pca.n_components_} dimensions"
            )

        self.fitted = True

    def transform(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transforms features using the fitted PCA models and concatenates them
        with the metadata features.

        Args:
            data_dict: Dictionary containing raw features and metadata.

        Returns:
            np.ndarray: The concatenated dense feature matrix.
        """
        if not self.fitted:
            raise RuntimeError("PCA models must be fitted before calling transform.")

        component_list = []

        # 1. Transform and collect backbone features
        for name in self.backbones:
            feature_key = f"features_{name}"
            features = data_dict[feature_key]

            pca = self.pcas[name]
            transformed = pca.transform(features)
            component_list.append(transformed)

        # 2. Collect Metadata features
        # Metadata shape is (N, 12)
        if "metadata" not in data_dict:
            raise KeyError("Key 'metadata' not found in data dictionary.")

        # Cite solution_lesson_node_00025: Explicitly scaling binary auxiliary features
        metadata = data_dict["metadata"] * 10.0
        component_list.append(metadata)

        # 3. Concatenate all components horizontally
        # Result shape: (N, sum(pca_dims) + 12)
        final_features = np.hstack(component_list)

        return final_features


def get_dimensionality_reduced_features(
    train_data: Dict, val_data: Dict, test_data: Dict, load_cached_data: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrates the PCA fitting, transformation, and caching process.

    Args:
        train_data: Raw training data dictionary.
        val_data: Raw validation data dictionary.
        test_data: Raw test data dictionary.
        load_cached_data: Whether to attempt loading from disk.

    Returns:
        Tuple containing:
        (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    seed_everything(Config.SEED)

    # Generate hash for caching based on Config
    # We include 'pca' in the string to differentiate from raw features
    config_hash = get_config_hash(Config)

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "X_train": os.path.join(cache_dir, f"X_train_pca_{config_hash}.npy"),
        "y_train": os.path.join(cache_dir, f"y_train_pca_{config_hash}.npy"),
        "X_val": os.path.join(cache_dir, f"X_val_pca_{config_hash}.npy"),
        "y_val": os.path.join(cache_dir, f"y_val_pca_{config_hash}.npy"),
        "X_test": os.path.join(cache_dir, f"X_test_pca_{config_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_pca_{config_hash}.npy"),
    }

    # Check if all files exist
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and all_cached:
        print(
            f"Loading PCA-transformed features from {cache_dir} (Hash: {config_hash})..."
        )
        X_train = np.load(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"])
        X_val = np.load(cache_paths["X_val"])
        y_val = np.load(cache_paths["y_val"])
        X_test = np.load(cache_paths["X_test"])
        test_ids = np.load(
            cache_paths["test_ids"], allow_pickle=True
        )  # IDs are strings/objects

        return X_train, y_train, X_val, y_val, X_test, test_ids

    # If not cached, compute from scratch
    print(f"Computing PCA transformations (Hash: {config_hash})...")

    # Initialize and Fit PCA
    pca_processor = IndependentPCA(
        variance_threshold=Config.PCA_VARIANCE, seed=Config.SEED
    )
    pca_processor.fit(train_data)

    # Transform datasets
    print("Transforming Train set...")
    X_train = pca_processor.transform(train_data)
    y_train = train_data["targets"]

    print("Transforming Validation set...")
    X_val = pca_processor.transform(val_data)
    y_val = val_data["targets"]

    print("Transforming Test set...")
    X_test = pca_processor.transform(test_data)
    test_ids = test_data["ids"]

    # Save to cache
    print(f"Saving PCA-transformed features to {cache_dir}...")
    np.save(cache_paths["X_train"], X_train)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["X_val"], X_val)
    np.save(cache_paths["y_val"], y_val)
    np.save(cache_paths["X_test"], X_test)
    np.save(cache_paths["test_ids"], test_ids)

    print(f"Final Feature Dimensions: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
