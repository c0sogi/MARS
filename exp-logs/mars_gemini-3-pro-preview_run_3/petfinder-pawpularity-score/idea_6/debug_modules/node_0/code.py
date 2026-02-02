import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import PawpularityDataset, load_metadata_splits, get_transforms
from library.feature_extractor import FeatureEngine
from library.preprocessing import StreamProcessor
from library.models import Level1Estimators, StackingMetaLearner
from library.train_eval import CrossValidator


def setup_demo_environment():
    """
    Configures the environment for a fast demonstration run.
    Overrides Config values to use a tiny subset of data and fewer folds.
    """
    print("Setting up demonstration environment...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample size for speed

    # Reduce CV folds for speed
    Config.N_FOLDS = 2

    # Set a specific working directory for the demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set seed
    set_seed(Config.SEED)
    print("Configuration updated for demo mode.")


def verify_data_loader():
    """
    Demonstrates and verifies the PawpularityDataset and Data Loading logic.
    """
    print("\n=== Verifying Data Loader ===")

    # Load metadata splits (will be subsampled due to DEBUG=True)
    train_df, val_df, test_df = load_metadata_splits()

    print(f"Train DF shape: {train_df.shape}")
    print(f"Val DF shape: {val_df.shape}")

    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"

    # Initialize Dataset
    transform = get_transforms(Config.IMAGE_SIZE)
    dataset = PawpularityDataset(
        train_df, Config.INPUT_DIR, transform=transform, use_tta=False
    )

    # Fetch one sample
    image, meta, target, sample_id = dataset[0]

    # Verify shapes
    # Image: (3, 224, 224)
    assert image.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Unexpected image shape: {image.shape}"
    # Metadata: (12,)
    assert meta.shape == (
        len(Config.METADATA_COLS),
    ), f"Unexpected metadata shape: {meta.shape}"
    # Target: Scalar
    assert isinstance(target.item(), float), "Target is not a float"

    print("Standard Dataset item verified.")

    # Verify TTA (Test Time Augmentation)
    dataset_tta = PawpularityDataset(
        train_df, Config.INPUT_DIR, transform=transform, use_tta=True
    )
    image_tta, _, _, _ = dataset_tta[0]

    # TTA Image: (2, 3, 224, 224) -> Stack of [Original, Flipped]
    assert image_tta.shape == (
        2,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Unexpected TTA image shape: {image_tta.shape}"

    print("TTA Dataset item verified.")


def verify_feature_extraction():
    """
    Demonstrates feature extraction.
    Note: This loads models and runs inference, which is the heaviest part.
    We rely on the small DEBUG_SAMPLE_SIZE to keep this fast.
    """
    print("\n=== Verifying Feature Extraction ===")

    engine = FeatureEngine()

    # Extract features (force re-computation by setting load_cached_data=False for demo purposes)
    # In a real run, we would use True.
    # We will only check the output structure.
    features_dict, meta_dict, target_dict, ids_dict = engine.extract_features(
        load_cached_data=False
    )

    # Check structure
    splits = ["train", "val", "test"]
    for split in splits:
        assert split in features_dict, f"Missing split {split} in features"
        assert split in meta_dict, f"Missing split {split} in metadata"

        # Check one backbone
        backbone_name = Config.BACKBONES[0]["name"]
        feat_shape = features_dict[split][backbone_name].shape

        print(f"Split '{split}' - Backbone '{backbone_name}' shape: {feat_shape}")

        # Verify sample count matches debug size
        assert (
            feat_shape[0] == Config.DEBUG_SAMPLE_SIZE
        ), f"Feature count mismatch for {split}"

    return features_dict, meta_dict, target_dict


def verify_preprocessing(features_dict, meta_dict):
    """
    Demonstrates the StreamProcessor (PCA + Concatenation).
    """
    print("\n=== Verifying Preprocessing ===")

    processor = StreamProcessor()

    # Process features
    X_train, X_val, X_test = processor.process_features(
        features_dict, meta_dict, load_cached_data=False
    )

    print(f"Processed X_train shape: {X_train.shape}")
    print(f"Processed X_val shape: {X_val.shape}")

    # Verify rows
    assert X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE

    # Verify columns:
    # 4 backbones * PCA components + 12 metadata features
    # Exact number depends on PCA variance retention, but should be > 12
    assert (
        X_train.shape[1] > 12
    ), "Feature dimension too small, PCA/Concat might have failed"

    return X_train, X_val, X_test


def verify_modeling(X_train, y_train):
    """
    Demonstrates Level 1 Estimators and Level 2 Stacking.
    """
    print("\n=== Verifying Modeling ===")

    # 1. Level 1 Estimators
    l1_estimators = Level1Estimators()

    # Get OOF preds
    # Note: We are passing X_train which has 20 samples.
    # N_FOLDS is set to 2.
    oof_preds = l1_estimators.get_oof_predictions(X_train, y_train)

    print(f"OOF Predictions shape: {oof_preds.shape}")

    # Shape should be (n_samples, n_models)
    # We have 3 base models (SVR, Ridge, LGBM)
    assert oof_preds.shape == (Config.DEBUG_SAMPLE_SIZE, 3), "OOF shape incorrect"

    # Fit all on full data
    l1_estimators.fit_all(X_train, y_train)

    # 2. Level 2 Meta Learner
    meta_learner = StackingMetaLearner()
    meta_learner.fit(oof_preds, y_train)

    # Test prediction flow
    # Predict using L1 models
    base_preds = l1_estimators.predict(X_train)
    # Predict using L2 model
    final_preds = meta_learner.predict(base_preds)

    assert len(final_preds) == Config.DEBUG_SAMPLE_SIZE
    print("Modeling components verified.")


def run_full_pipeline_check():
    """
    Runs the CrossValidator to ensure the entire pipeline logic holds together.
    """
    print("\n=== Running Full Pipeline via CrossValidator ===")

    validator = CrossValidator()

    # Run pipeline
    # We use load_cached_data=True here because we just computed and cached
    # features in the previous verification steps (FeatureEngine saves to disk).
    validator.train_and_predict(load_cached_data=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file generated at {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df.shape}")
        assert len(df) == Config.DEBUG_SAMPLE_SIZE, "Submission row count mismatch"
        assert "Id" in df.columns and "Pawpularity" in df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Verify Data Loader
    verify_data_loader()

    # 3. Verify Feature Extraction
    # Returns raw features needed for next steps
    feats, metas, targets = verify_feature_extraction()

    # 4. Verify Preprocessing
    # Returns processed numpy arrays
    X_train, X_val, X_test = verify_preprocessing(feats, metas)

    # 5. Verify Modeling
    # Use train targets
    y_train = targets["train"]
    verify_modeling(X_train, y_train)

    # 6. Verify Full Pipeline Wrapper
    run_full_pipeline_check()

    print("\nAll demonstrations completed successfully.")
