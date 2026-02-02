import os
from library.config import TRAIN_META_PATH, VAL_META_PATH, TEST_META_PATH, SEED
from library.utils import seed_everything
from library.image_processing import generate_dataset_features


def run_feature_generation(load_cached_data=True):
    """
    Orchestrates the feature generation pipeline for Train, Validation, and Test sets.

    This function utilizes the logic in library.image_processing to extract
    spatially-stratified EfficientNet features and lung volumetrics.

    It ensures the Training set is processed first. This is critical because the
    PCA model (used for dimensionality reduction) is fitted on the training data
    and then applied to the validation and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load features from the cache directory.
                                 If False or if cache is missing, regenerates features.

    Returns:
        tuple: A tuple containing three dictionaries (train_features, val_features, test_features).
               Each dictionary maps Patient_ID (str) to Feature_Vector (numpy array).
    """
    # Set seed for deterministic behavior
    seed_everything(SEED)

    # 1. Process Training Data
    # The 'train' subset triggers PCA fitting in generate_dataset_features
    print("[Feature Generation] Processing Training Data...")
    train_features = generate_dataset_features(
        metadata_path=TRAIN_META_PATH,
        subset_name="train",
        load_cached_data=load_cached_data,
    )

    # 2. Process Validation Data
    # Uses the PCA model fitted in step 1
    print("[Feature Generation] Processing Validation Data...")
    val_features = generate_dataset_features(
        metadata_path=VAL_META_PATH,
        subset_name="val",
        load_cached_data=load_cached_data,
    )

    # 3. Process Test Data
    # Uses the PCA model fitted in step 1
    print("[Feature Generation] Processing Test Data...")
    test_features = generate_dataset_features(
        metadata_path=TEST_META_PATH,
        subset_name="test",
        load_cached_data=load_cached_data,
    )

    return train_features, val_features, test_features
