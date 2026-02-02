import numpy as np
import os
from library.config import Config
from library.utils import seed_everything, get_logger
from library.feature_extraction import DeepFeatureExtractor
from library.densification import ManifoldDensifier
from library.modeling import run_cross_validation


def merge_datasets(data_list):
    """
    Merges a list of raw feature dictionaries by concatenating arrays along axis 0.
    Assumes all dictionaries have the same keys.

    Args:
        data_list (list): List of dictionaries containing numpy arrays.

    Returns:
        dict: A single dictionary with concatenated arrays.
    """
    if not data_list:
        return {}

    merged = {}
    keys = data_list[0].keys()

    for key in keys:
        # Collect arrays for this key from all dictionaries
        arrays = [d[key] for d in data_list]
        # Concatenate along the first dimension (samples)
        merged[key] = np.concatenate(arrays, axis=0)

    return merged


def run_workflow(load_cached_data=True):
    """
    Orchestrates the Selective-Topology Orthogonal Manifold-Densified LDA Ensemble pipeline.

    Steps:
    1. Extract Deep (DINOv2, ConvNeXt) and Tabular features for Train, Val, and Test.
    2. Merge Train and Val sets to utilize all labeled data for Cross-Validation.
    3. Apply Manifold Densification (Orthogonal View-Set Averaging) to generate
       3 centroids per image, structurally increasing sample size.
    4. Execute Stratified K-Fold Cross-Validation to train the ensemble and
       generate aggregated predictions for the test set.

    Args:
        load_cached_data (bool): If True, attempts to load intermediate feature arrays
                                 from the cache directory to save time.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("workflow")
    logger.info(
        f"Initializing workflow (Debug={Config.DEBUG}, Device={Config.DEVICE})..."
    )

    # 2. Feature Extraction
    # We extract features for all splits.
    extractor = DeepFeatureExtractor()

    logger.info("Step 1/4: Extracting features...")

    # Extract Train features
    train_raw = extractor.extract_features("train", load_cached_data=load_cached_data)

    # Extract Validation features
    val_raw = extractor.extract_features("val", load_cached_data=load_cached_data)

    # Extract Test features
    test_raw = extractor.extract_features("test", load_cached_data=load_cached_data)

    # 3. Merge Train and Val
    # We combine the training and validation metadata splits into a single dataset
    # for the Stratified K-Fold Cross-Validation process.
    logger.info("Step 2/4: Merging Train and Validation sets for full CV...")
    full_train_raw = merge_datasets([train_raw, val_raw])
    logger.info(f"Merged dataset size: {len(full_train_raw['ids'])} samples.")

    # 4. Manifold Densification
    # Transforms (N, 12, D) -> (3N, D) using orthogonal centroids.
    densifier = ManifoldDensifier()

    logger.info("Step 3/4: Densifying manifolds...")

    # Densify Training Data (Merged)
    # We use a custom split name 'train_full' to ensure the cache file is unique
    # and does not conflict with partial 'train' or 'val' caches.
    densified_train = densifier.prepare_densified_dataset(
        full_train_raw, split="train_full", load_cached_data=load_cached_data
    )

    # Densify Test Data
    densified_test = densifier.prepare_densified_dataset(
        test_raw, split="test", load_cached_data=load_cached_data
    )

    # 5. Modeling (CV + Inference)
    # This function handles the K-Fold loop, pipeline training, validation scoring,
    # and final test submission generation.
    logger.info("Step 4/4: Running Cross-Validation and Inference...")
    run_cross_validation(densified_train, densified_test)

    logger.info("Workflow execution completed.")
