import os
import sys
import pandas as pd
import numpy as np
import torch

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.feature_extractor import FeatureEngine
from library.dimensionality_reduction import DimensionalityReducer
from library.ensemble_model import EnsembleTrainer


def create_debug_metadata(sample_size):
    """
    Creates subset metadata files to ensure consistency across
    FeatureEngine (which uses PetDataset) and DimensionalityReducer
    (which reads CSVs directly).
    """
    print(f"Creating debug metadata with {sample_size} samples...")

    # Define new paths in working directory
    debug_train_path = os.path.join(Config.WORKING_DIR, "debug_train_meta.csv")
    debug_val_path = os.path.join(Config.WORKING_DIR, "debug_val_meta.csv")
    debug_test_path = os.path.join(Config.WORKING_DIR, "debug_test_meta.csv")

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Slice and save
    df_train.head(sample_size).to_csv(debug_train_path, index=False)
    df_val.head(sample_size).to_csv(debug_val_path, index=False)
    df_test.head(sample_size).to_csv(debug_test_path, index=False)

    return debug_train_path, debug_val_path, debug_test_path


def main():
    # 1. Setup
    seed_everything(42)
    logger = get_logger("DemoScript")
    logger.info("Starting End-to-End Demo...")

    # 2. Override Configuration for Speed
    # We use a very small sample size and lightweight settings
    DEBUG_SIZE = 10

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = DEBUG_SIZE
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Use only one backbone to save time
    Config.BACKBONES = {"swin": "swin_large_patch4_window7_224"}

    # Reduce Ensemble complexity
    Config.N_FOLDS = 2
    Config.LGBM_PARAMS["n_estimators"] = 5
    Config.EXTRATREES_PARAMS["n_estimators"] = 5
    Config.LGBM_PARAMS["verbose"] = -1
    Config.LGBM_PARAMS["verbosity"] = -1

    # PCA Variance: High enough to keep dims for small N
    Config.PCA_VARIANCE = 0.99

    # 3. Create and Link Debug Metadata
    # This is crucial because DimensionalityReducer reads the CSV file directly.
    # If we don't point it to a small CSV, it will try to merge 10 image features
    # with 7000 metadata rows, causing a shape mismatch.
    train_path, val_path, test_path = create_debug_metadata(DEBUG_SIZE)
    Config.TRAIN_META_PATH = train_path
    Config.VAL_META_PATH = val_path
    Config.TEST_META_PATH = test_path

    # Ensure working directory exists (Config.setup() created it, but we might have changed it)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 4. Feature Extraction
    logger.info("Step 1: Feature Extraction")
    feature_engine = FeatureEngine()
    # Force run (load_cached_data=False) to demonstrate execution
    feature_engine.run(load_cached_data=False)

    # Verify intermediate files
    expected_feat_file = os.path.join(Config.WORKING_DIR, "swin_train_features.npy")
    if not os.path.exists(expected_feat_file):
        raise FileNotFoundError(
            f"Feature extraction failed. {expected_feat_file} not found."
        )

    feats = np.load(expected_feat_file)
    logger.info(f"Verified extracted features shape: {feats.shape}")
    if feats.shape[0] != DEBUG_SIZE:
        raise AssertionError(f"Expected {DEBUG_SIZE} features, got {feats.shape[0]}")

    # 5. Dimensionality Reduction
    logger.info("Step 2: Dimensionality Reduction & Fusion")
    dim_reducer = DimensionalityReducer()
    X_train, y_train, X_val, y_val, X_test, ids_test = dim_reducer.run(
        load_cached_data=False
    )

    logger.info(f"Fused Train Shape: {X_train.shape}")
    logger.info(f"Fused Test Shape: {X_test.shape}")

    # Verify shapes
    if X_train.shape[0] != DEBUG_SIZE:
        raise AssertionError(f"X_train rows {X_train.shape[0]} != {DEBUG_SIZE}")
    if X_test.shape[0] != DEBUG_SIZE:
        raise AssertionError(f"X_test rows {X_test.shape[0]} != {DEBUG_SIZE}")

    # 6. Ensemble Modeling
    logger.info("Step 3: Ensemble Training & Prediction")
    trainer = EnsembleTrainer()
    submission_df = trainer.run(
        X_train, y_train, X_val, y_val, X_test, ids_test, load_cached_data=False
    )

    # 7. Final Verification
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    logger.info(f"Submission loaded. Shape: {df_sub.shape}")

    if len(df_sub) != DEBUG_SIZE:
        raise AssertionError(
            f"Submission length {len(df_sub)} does not match test size {DEBUG_SIZE}"
        )

    if df_sub.isnull().any().any():
        raise AssertionError("Submission contains NaN values.")

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
