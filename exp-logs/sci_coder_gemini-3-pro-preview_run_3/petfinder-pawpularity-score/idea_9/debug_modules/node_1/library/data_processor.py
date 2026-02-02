import os
import numpy as np
from sklearn.decomposition import PCA
from library.config import Config
from library.feature_engine import extract_and_cache_features
from library.utils import save_numpy, load_numpy, save_pickle, load_pickle


class MultiStreamPCA:
    """
    Manages PCA transformation for multiple feature streams.
    Fits independent PCA transformers for each backbone-view stream to retain
    specific variance (e.g., 95%).
    """

    def __init__(self, variance=0.95, random_state=42):
        self.variance = variance
        self.random_state = random_state
        # The specific keys expected from the feature engine
        self.stream_names = [
            "swin_global",
            "swin_zoomed",
            "effnet_global",
            "effnet_zoomed",
            "dino_global",
            "dino_zoomed",
            "clip_global",
            "clip_zoomed",
        ]
        self.pcas = {}

    def fit(self, data_dict):
        """
        Fits PCA on the provided data dictionary (typically training data).
        Args:
            data_dict (dict): Dictionary containing numpy arrays for each stream.
        """
        print("Fitting MultiStream PCA...")
        for name in self.stream_names:
            if name not in data_dict:
                print(f"Warning: Stream {name} not found in data.")
                continue

            X = data_dict[name]
            # PCA requires (N, D) input.
            pca = PCA(n_components=self.variance, random_state=self.random_state)
            pca.fit(X)
            self.pcas[name] = pca
            print(
                f"  Stream {name}: Retained {pca.n_components_} components (Original dim: {X.shape[1]})"
            )
        return self

    def transform(self, data_dict):
        """
        Transforms the data using fitted PCAs and concatenates them horizontally.
        Args:
            data_dict (dict): Dictionary containing numpy arrays for each stream.
        Returns:
            np.ndarray: Concatenated feature matrix.
        """
        transformed_arrays = []
        for name in self.stream_names:
            if name in self.pcas and name in data_dict:
                pca = self.pcas[name]
                X = data_dict[name]
                X_trans = pca.transform(X)
                transformed_arrays.append(X_trans)
            else:
                # This should not happen in a correct pipeline
                print(
                    f"Warning: Skipping transform for {name} (PCA not fitted or data missing)"
                )

        if not transformed_arrays:
            raise ValueError("No streams were transformed. Check data dictionary keys.")

        return np.hstack(transformed_arrays)


class MetadataScaler:
    """
    Handles scaling of metadata features.
    """

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def transform(self, metadata):
        return metadata * self.scale_factor


def process_and_cache_data(load_cached_data=True):
    """
    Orchestrates the data processing pipeline:
    1. Loads raw features (cached or computed).
    2. Fits MultiStreamPCA on training data.
    3. Transforms Train, Val, Test data.
    4. Merges with Metadata.
    5. Caches the final processed matrices.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """

    # Define file paths for processed data
    processed_files = {
        "X_train": os.path.join(Config.WORKING_DIR, "final_X_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "final_y_train.npy"),
        "X_val": os.path.join(Config.WORKING_DIR, "final_X_val.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "final_y_val.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, "final_X_test.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "final_test_ids.npy"),
        "pca_model": os.path.join(Config.WORKING_DIR, "multistream_pca.joblib"),
    }

    # Check if all files exist
    all_exist = all(os.path.exists(p) for p in processed_files.values())

    if load_cached_data and all_exist:
        print("Loading processed data from cache...")
        X_train = load_numpy(processed_files["X_train"])
        y_train = load_numpy(processed_files["y_train"])
        X_val = load_numpy(processed_files["X_val"])
        y_val = load_numpy(processed_files["y_val"])
        X_test = load_numpy(processed_files["X_test"])
        test_ids = load_numpy(processed_files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # 1. Load Raw Features (this function handles its own caching of raw extraction)
    # We pass load_cached_data to it so it can reuse raw extractions if available
    train_raw = extract_and_cache_features("train", load_cached_data=load_cached_data)
    val_raw = extract_and_cache_features("val", load_cached_data=load_cached_data)
    test_raw = extract_and_cache_features("test", load_cached_data=load_cached_data)

    # 2. Fit PCA on Train
    pca_processor = MultiStreamPCA(
        variance=Config.PCA_VARIANCE, random_state=Config.SEED
    )
    pca_processor.fit(train_raw)

    # Save PCA model for reproducibility
    save_pickle(pca_processor, processed_files["pca_model"])

    # 3. Transform All
    print("Transforming features with PCA...")
    X_train_pca = pca_processor.transform(train_raw)
    X_val_pca = pca_processor.transform(val_raw)
    X_test_pca = pca_processor.transform(test_raw)

    # 4. Handle Metadata
    # Note: PetDataset (in data_handling.py) already scales metadata by Config.METADATA_SCALE.
    # Therefore, we use MetadataScaler with factor 1.0 to avoid double scaling.
    scaler = MetadataScaler(scale_factor=1.0)

    meta_train = scaler.transform(train_raw["metadata"])
    meta_val = scaler.transform(val_raw["metadata"])
    meta_test = scaler.transform(test_raw["metadata"])

    # Concatenate PCA features with Metadata
    print("Concatenating PCA features with metadata...")
    X_train = np.hstack([X_train_pca, meta_train])
    X_val = np.hstack([X_val_pca, meta_val])
    X_test = np.hstack([X_test_pca, meta_test])

    # Extract Targets and IDs
    y_train = train_raw["targets"]
    y_val = val_raw["targets"]
    test_ids = test_raw["ids"]

    # 5. Save to Cache
    print("Saving processed data...")
    save_numpy(X_train, processed_files["X_train"])
    save_numpy(y_train, processed_files["y_train"])
    save_numpy(X_val, processed_files["X_val"])
    save_numpy(y_val, processed_files["y_val"])
    save_numpy(X_test, processed_files["X_test"])
    save_numpy(test_ids, processed_files["test_ids"])

    return X_train, y_train, X_val, y_val, X_test, test_ids
