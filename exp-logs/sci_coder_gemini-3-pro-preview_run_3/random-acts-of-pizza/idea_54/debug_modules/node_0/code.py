import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set global seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. CONFIGURATION & PATCHING
# =============================================================================
# We import config and patch hyperparameters to make the demo run fast.
import library.config as config

print("Patching configuration for rapid demonstration...")

# Define demo-specific constants
DEMO_N_FOLDS = 2
DEMO_N_ESTIMATORS = 10
DEMO_EARLY_STOPPING = 2

# Patch dictionary parameters in config in-place
params_list = [
    config.LEXICAL_BAGGER_PARAMS,
    config.COMMUNITY_BAGGER_PARAMS,
    config.SEMANTIC_BAGGER_PARAMS,
    config.SEMANTIC_BOOSTER_PARAMS,
    config.SEMANTIC_GRADIENT_PARAMS,
    config.TEMPORAL_BOOSTER_PARAMS,
]

for params in params_list:
    if "n_estimators" in params:
        params["n_estimators"] = DEMO_N_ESTIMATORS
    if "early_stopping_rounds" in params:
        params["early_stopping_rounds"] = DEMO_EARLY_STOPPING
    # Ensure silent mode
    params["verbose"] = -1
    params["verbosity"] = 0

# Patch N_FOLDS in inference engine since it imports the integer directly
import library.inference_engine

library.inference_engine.N_FOLDS = DEMO_N_FOLDS

# Import remaining modules after patching
from library.data_loader import load_raw_dataset, clean_dataset, get_stratified_folds
from library.feature_engineering import FeatureFactory
from library.training_engine import HybridTrainer
from library.inference_engine import HybridPredictor
from library.config import CACHE_DIR, TARGET_COL

if __name__ == "__main__":
    print("\n=== Starting Self-Contained Demo ===\n")

    # Clean working directory for a fresh run
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # 2. DATA LOADING & PREPARATION
    # =========================================================================
    print("Loading and preparing data...")

    # Load full datasets
    df_train_full = load_raw_dataset("full_train")
    df_test_full = load_raw_dataset("test")

    # Subsample for demonstration speed (50 train, 20 test)
    df_train = df_train_full.iloc[:50].copy().reset_index(drop=True)
    df_test = df_test_full.iloc[:20].copy().reset_index(drop=True)

    # Clean datasets
    df_train = clean_dataset(df_train, is_test=False)
    df_test = clean_dataset(df_test, is_test=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # Verify target exists in train
    assert TARGET_COL in df_train.columns, "Target column missing from training data"

    # =========================================================================
    # 3. FEATURE ENGINEERING
    # =========================================================================
    print("\nInitializing FeatureFactory...")
    feature_factory = FeatureFactory()

    # Fit on training data
    feature_factory.fit(df_train)

    # Transform Train (Force no cache to test generation logic)
    print("Generating training features...")
    train_features = feature_factory.transform(df_train, "demo_train", load_cache=False)

    # Transform Test
    print("Generating test features...")
    test_features = feature_factory.transform(df_test, "demo_test", load_cache=False)

    # Verification
    expected_keys = ["lexical", "behavioral", "semantic", "metadata"]
    for key in expected_keys:
        assert key in train_features, f"Missing key '{key}' in train features"
        assert key in test_features, f"Missing key '{key}' in test features"
        # Check rows match
        assert train_features[key].shape[0] == len(
            df_train
        ), f"Row mismatch in train {key}"
        assert test_features[key].shape[0] == len(
            df_test
        ), f"Row mismatch in test {key}"

    print("Feature generation verified.")

    # =========================================================================
    # 4. TRAINING PIPELINE
    # =========================================================================
    print("\nInitializing HybridTrainer...")
    trainer = HybridTrainer(feature_factory)

    # Generate Folds
    folds = get_stratified_folds(df_train, n_folds=DEMO_N_FOLDS)

    # Level 1 Training
    print("Running Level 1 Training...")
    oof_preds = trainer.train_level_1(df_train, folds)

    # Verify OOF predictions
    assert len(oof_preds) == len(df_train), "OOF predictions length mismatch"
    assert not oof_preds.isnull().values.any(), "NaNs found in OOF predictions"

    # Level 2 Training
    print("Running Level 2 Training...")
    y_true = df_train[TARGET_COL].values
    trainer.train_level_2(oof_preds, y_true)

    # Verify models saved
    models_dir = os.path.join(CACHE_DIR, "models")
    assert os.path.exists(
        os.path.join(models_dir, "meta_learner.joblib")
    ), "Meta learner not saved"
    # Check for a stable model (e.g., lexical_bagger_full)
    assert os.path.exists(
        os.path.join(models_dir, "lexical_bagger_full.joblib")
    ), "Stable model not saved"
    # Check for a volatile model fold (e.g., semantic_booster_fold_0)
    assert os.path.exists(
        os.path.join(models_dir, "semantic_booster_fold_0.joblib")
    ), "Volatile model fold not saved"

    # =========================================================================
    # 5. INFERENCE PIPELINE
    # =========================================================================
    print("\nInitializing HybridPredictor...")
    predictor = HybridPredictor(feature_factory)

    # Generate Submission
    print("Generating submission...")
    predictor.generate_submission(df_test)

    # Verify Submission
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found"

    submission_df = pd.read_csv(submission_path)
    assert len(submission_df) == len(df_test), "Submission row count mismatch"
    assert "request_id" in submission_df.columns, "request_id column missing"
    assert (
        "requester_received_pizza" in submission_df.columns
    ), "prediction column missing"

    # Check probabilities range
    probs = submission_df["requester_received_pizza"]
    assert (
        probs.min() >= 0 and probs.max() <= 1
    ), "Predictions out of probability range [0, 1]"

    print("\n=== Demo Completed Successfully ===")
    print(f"Submission generated at: {submission_path}")
    print("Head of submission:")
    print(submission_df.head())
