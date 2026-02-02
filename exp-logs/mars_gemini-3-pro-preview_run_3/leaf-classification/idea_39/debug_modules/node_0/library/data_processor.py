import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import (
    METADATA_DIR,
    CACHE_DIR,
    CENTROID_INDICES,
    TABULAR_PREFIXES,
    TOTAL_TABULAR_FEATURES,
)
from library.utils import setup_logger, save_array, load_array
from library.feature_extraction import run_extraction

# Initialize logger
logger = setup_logger("data_processor.log")


def make_orthogonal_centroids(features: np.ndarray) -> np.ndarray:
    """
    Aggregates 12-view features into 3 orthogonal centroids (A, B, C) per image.

    Args:
        features (np.ndarray): Input features of shape (N_images, 12, Feature_Dim).

    Returns:
        np.ndarray: Densified features of shape (N_images * 3, Feature_Dim).
                    Order: [Img1_A, Img1_B, Img1_C, Img2_A, ...].
    """
    N, V, D = features.shape

    # Containers for the three centroids
    centroids_A = []
    centroids_B = []
    centroids_C = []

    # Indices for aggregation
    idx_A = CENTROID_INDICES["A"]
    idx_B = CENTROID_INDICES["B"]
    idx_C = CENTROID_INDICES["C"]

    # Compute means for each centroid type across the batch
    # features[:, idx_A, :] shape becomes (N, 4, D) -> mean -> (N, D)
    c_A = np.mean(features[:, idx_A, :], axis=1)
    c_B = np.mean(features[:, idx_B, :], axis=1)
    c_C = np.mean(features[:, idx_C, :], axis=1)

    # Stack them: (N, 3, D)
    # axis=1 ensures [Img1_A, Img1_B, Img1_C] structure
    densified = np.stack([c_A, c_B, c_C], axis=1)

    # Flatten to (N * 3, D)
    densified = densified.reshape(N * 3, D)

    return densified


def process_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts tabular features from the dataframe.

    Args:
        df (pd.DataFrame): Metadata dataframe containing feature columns.

    Returns:
        np.ndarray: Array of shape (N, 192).
    """
    # Identify columns
    cols = []
    for prefix in TABULAR_PREFIXES:
        # Assuming 1-based indexing as per description (margin_1 ... margin_64)
        cols.extend([f"{prefix}_{i}" for i in range(1, 65)])

    # Extract and ensure float32
    return df[cols].values.astype(np.float32)


def prepare_split(
    split_name: str, visual_data: dict, meta_df: pd.DataFrame, is_test: bool = False
):
    """
    Merges visual and tabular data, applying densification.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        visual_data (dict): Dictionary containing 'dino', 'conv', 'ids'.
        meta_df (pd.DataFrame): Metadata dataframe.
        is_test (bool): Whether this is the test set (no labels).

    Returns:
        dict: Contains 'X', 'y' (if not test), 'ids', 'original_ids'.
    """
    logger.info(f"Processing {split_name} split...")

    # 1. Visual Features (Densification)
    # Input: (N, 12, D) -> Output: (N*3, D)
    dino_dense = make_orthogonal_centroids(visual_data["dino"])
    conv_dense = make_orthogonal_centroids(visual_data["conv"])

    # 2. Tabular Features
    # Input: (N, 192)
    tab_features = process_tabular_features(meta_df)

    # Replicate tabular features 3 times to match densified visual features
    # np.repeat with repeats=3, axis=0 produces [Row1, Row1, Row1, Row2, ...]
    tab_dense = np.repeat(tab_features, 3, axis=0)

    # 3. Concatenate all features
    # Structure: [DINO (1024) | Conv (1536) | Tabular (192)]
    X = np.concatenate([dino_dense, conv_dense, tab_dense], axis=1)

    # 4. Handle IDs
    # Replicate IDs 3 times to track origin
    original_ids = visual_data["ids"]
    ids_dense = np.repeat(original_ids, 3)

    result = {
        "X": X,
        "ids": ids_dense,
        "original_ids": original_ids,  # Kept for reference/aggregation
    }

    # 5. Handle Labels (if available)
    if not is_test:
        if "species" not in meta_df.columns:
            raise ValueError(f"Species column missing in {split_name} metadata")

        labels = meta_df["species"].values
        # Replicate labels 3 times
        labels_dense = np.repeat(labels, 3)
        result["y"] = labels_dense

    return result


def load_dataset(load_cached_data: bool = True, debug_sample_size: int = None):
    """
    Main function to load, process, and structure the dataset.

    Args:
        load_cached_data (bool): If True, attempts to load processed arrays from disk.
        debug_sample_size (int, optional): Limit dataset size for debugging.

    Returns:
        dict: Contains processed train, val, test data and feature indices.
    """
    # Define cache filenames for the FINAL processed data
    cache_files = {
        "train_X": "densified_train_X.npy",
        "train_y": "densified_train_y.npy",
        "train_ids": "densified_train_ids.npy",
        "val_X": "densified_val_X.npy",
        "val_y": "densified_val_y.npy",
        "val_ids": "densified_val_ids.npy",
        "test_X": "densified_test_X.npy",
        "test_ids": "densified_test_ids.npy",
        "classes": "classes.npy",
        "feature_indices": "feature_indices.npy",  # Stores start/end indices for modalities
    }

    # Check if all cache files exist
    all_cached = True
    if load_cached_data:
        for fname in cache_files.values():
            if load_array(fname) is None:
                all_cached = False
                break
    else:
        all_cached = False

    if all_cached and load_cached_data:
        logger.info("Loading processed dataset from cache...")
        data = {
            "train": {
                "X": load_array(cache_files["train_X"]),
                "y": load_array(cache_files["train_y"]),
                "ids": load_array(cache_files["train_ids"]),
            },
            "val": {
                "X": load_array(cache_files["val_X"]),
                "y": load_array(cache_files["val_y"]),
                "ids": load_array(cache_files["val_ids"]),
            },
            "test": {
                "X": load_array(cache_files["test_X"]),
                "ids": load_array(cache_files["test_ids"]),
            },
            "classes": load_array(cache_files["classes"]),
            "feature_indices": load_array(cache_files["feature_indices"]),
        }
        return data

    # --- Process from Scratch ---
    logger.info("Processing dataset from scratch...")

    # 1. Get Raw Features (Visual)
    # This handles its own caching of the raw 12-view extraction
    raw_train, raw_val, raw_test = run_extraction(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # 2. Load Metadata
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    if debug_sample_size:
        meta_train = meta_train.head(debug_sample_size)
        meta_val = meta_val.head(debug_sample_size)
        # We don't limit test usually, but for consistency in debug mode if needed:
        # meta_test = meta_test.head(debug_sample_size)
        # (Usually better to keep full test to avoid submission errors, but respecting param)

    # 3. Process Splits (Densification + Merging)
    train_data = prepare_split("train", raw_train, meta_train, is_test=False)
    val_data = prepare_split("val", raw_val, meta_val, is_test=False)
    test_data = prepare_split("test", raw_test, meta_test, is_test=True)

    # 4. Encode Labels
    # We fit on Train + Val to ensure we capture all classes (though stratification should handle this)
    # However, strictly we should fit on Train. But given the fixed class list in sample_submission,
    # we should ensure alignment.
    # Let's fit on the union of unique species in Train and Val.
    unique_classes = np.unique(np.concatenate([train_data["y"], val_data["y"]]))
    unique_classes.sort()  # Ensure deterministic order

    le = LabelEncoder()
    le.fit(unique_classes)

    train_data["y_enc"] = le.transform(train_data["y"])
    val_data["y_enc"] = le.transform(val_data["y"])

    # 5. Define Feature Indices
    # X structure: [DINO | Conv | Tabular]
    dino_dim = raw_train["dino"].shape[2]
    conv_dim = raw_train["conv"].shape[2]
    tab_dim = TOTAL_TABULAR_FEATURES

    indices = np.array(
        [
            0,  # Start DINO
            dino_dim,  # End DINO / Start Conv
            dino_dim + conv_dim,  # End Conv / Start Tabular
            dino_dim + conv_dim + tab_dim,  # End Tabular
        ]
    )

    # 6. Save to Cache
    save_array(train_data["X"], cache_files["train_X"])
    save_array(train_data["y_enc"], cache_files["train_y"])
    save_array(train_data["ids"], cache_files["train_ids"])

    save_array(val_data["X"], cache_files["val_X"])
    save_array(val_data["y_enc"], cache_files["val_y"])
    save_array(val_data["ids"], cache_files["val_ids"])

    save_array(test_data["X"], cache_files["test_X"])
    save_array(test_data["ids"], cache_files["test_ids"])

    save_array(le.classes_, cache_files["classes"])
    save_array(indices, cache_files["feature_indices"])

    logger.info("Dataset processing complete and cached.")

    return {
        "train": {
            "X": train_data["X"],
            "y": train_data["y_enc"],
            "ids": train_data["ids"],
        },
        "val": {"X": val_data["X"], "y": val_data["y_enc"], "ids": val_data["ids"]},
        "test": {"X": test_data["X"], "ids": test_data["ids"]},
        "classes": le.classes_,
        "feature_indices": indices,
    }
