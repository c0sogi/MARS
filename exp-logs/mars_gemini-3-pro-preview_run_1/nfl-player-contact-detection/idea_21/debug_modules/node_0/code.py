import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
import logging

# Import the provided library modules
import library.config
import library.utils
import library.feature_engineering
import library.data_factory
import library.model_factory
import library.trainer

from library.utils import seed_everything, get_logger

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================
DEMO_DIR = "./working/demo_execution"
DATA_CACHE_DIR = os.path.join(DEMO_DIR, "data_cache")
MODELS_DIR = os.path.join(DEMO_DIR, "models")

# Define paths for mini-datasets
MINI_TRAIN_META = os.path.join(DEMO_DIR, "mini_train_metadata.csv")
MINI_VAL_META = os.path.join(DEMO_DIR, "mini_val_metadata.csv")
MINI_TEST_META = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
MINI_TRAIN_TRACK = os.path.join(DEMO_DIR, "mini_train_tracking.csv")
MINI_TEST_TRACK = os.path.join(DEMO_DIR, "mini_test_tracking.csv")

# Define paths for artifacts
MINI_TRAIN_FEATS = os.path.join(DATA_CACHE_DIR, "train_features_gated_full.parquet")
MINI_VAL_FEATS = os.path.join(DATA_CACHE_DIR, "val_features_gated.parquet")
MINI_TEST_FEATS = os.path.join(DATA_CACHE_DIR, "test_features_full.parquet")
MINI_HARD_NEGS = os.path.join(DATA_CACHE_DIR, "hard_negative_indices.npy")


def setup_demo_data():
    """
    Creates mini versions of the datasets by sampling a few plays.
    This ensures the code runs quickly for demonstration purposes.
    """
    print("Setting up mini-datasets for demo...")
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load original metadata
    print("Loading original metadata...")
    df_train_meta = pd.read_csv(library.config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(library.config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(library.config.TEST_METADATA_PATH)

    # 2. Sample Plays
    # Train: 3 plays, Val: 1 play, Test: 1 play
    train_plays = df_train_meta["game_play"].unique()[:3]
    val_plays = df_val_meta["game_play"].unique()[:1]
    test_plays = df_test_meta["game_play"].unique()[:1]

    # 3. Filter Metadata
    mini_train_meta = df_train_meta[df_train_meta["game_play"].isin(train_plays)].copy()
    mini_val_meta = df_val_meta[df_val_meta["game_play"].isin(val_plays)].copy()
    mini_test_meta = df_test_meta[df_test_meta["game_play"].isin(test_plays)].copy()

    # Save Metadata
    mini_train_meta.to_csv(MINI_TRAIN_META, index=False)
    mini_val_meta.to_csv(MINI_VAL_META, index=False)
    mini_test_meta.to_csv(MINI_TEST_META, index=False)

    print(f"Mini Train Meta: {len(mini_train_meta)} rows")
    print(f"Mini Val Meta: {len(mini_val_meta)} rows")
    print(f"Mini Test Meta: {len(mini_test_meta)} rows")

    # 4. Filter Tracking Data
    # We need tracking data for the selected plays
    print("Filtering tracking data...")

    # Train + Val Tracking (sourced from train_player_tracking.csv)
    needed_train_plays = np.concatenate([train_plays, val_plays])

    # Read in chunks or full? 1.2GB is fine to read full for filtering
    df_track_train = pd.read_csv(library.config.TRAIN_TRACKING_PATH)
    mini_track_train = df_track_train[
        df_track_train["game_play"].isin(needed_train_plays)
    ].copy()
    mini_track_train.to_csv(MINI_TRAIN_TRACK, index=False)

    # Test Tracking
    df_track_test = pd.read_csv(library.config.TEST_TRACKING_PATH)
    # Note: Test metadata contact_id parsing in script might not match exactly if game_play format differs,
    # but we rely on the provided metadata structure.
    mini_track_test = df_track_test[df_track_test["game_play"].isin(test_plays)].copy()
    mini_track_test.to_csv(MINI_TEST_TRACK, index=False)

    print(f"Mini Train/Val Tracking: {len(mini_track_train)} rows")
    print(f"Mini Test Tracking: {len(mini_track_test)} rows")


def patch_libraries():
    """
    Monkey-patches the imported library modules to use the mini-datasets
    and reduced hyperparameters. This is necessary because the library files
    cannot be modified directly.
    """
    print("Patching library configuration for demo...")

    # Define Reduced Hyperparameters
    # Very low estimators for speed
    DEMO_LGBM = library.config.LGBM_PARAMS.copy()
    DEMO_LGBM.update({"n_estimators": 10, "num_leaves": 8})

    DEMO_XGB = library.config.XGB_PARAMS.copy()
    DEMO_XGB.update({"n_estimators": 10, "max_depth": 3})

    DEMO_CAT = library.config.CAT_PARAMS.copy()
    DEMO_CAT.update({"iterations": 10, "depth": 3})

    # List of modules that import constants from config
    modules_to_patch = [
        library.config,
        library.data_factory,
        library.feature_engineering,
        library.trainer,
        library.model_factory,
    ]

    # Patch Paths
    for mod in modules_to_patch:
        if hasattr(mod, "TRAIN_METADATA_PATH"):
            mod.TRAIN_METADATA_PATH = MINI_TRAIN_META
        if hasattr(mod, "VAL_METADATA_PATH"):
            mod.VAL_METADATA_PATH = MINI_VAL_META
        if hasattr(mod, "TEST_METADATA_PATH"):
            mod.TEST_METADATA_PATH = MINI_TEST_META

        if hasattr(mod, "TRAIN_TRACKING_PATH"):
            mod.TRAIN_TRACKING_PATH = MINI_TRAIN_TRACK
        if hasattr(mod, "TEST_TRACKING_PATH"):
            mod.TEST_TRACKING_PATH = MINI_TEST_TRACK

        if hasattr(mod, "TRAIN_FEATURES_PATH"):
            mod.TRAIN_FEATURES_PATH = MINI_TRAIN_FEATS
        if hasattr(mod, "VAL_FEATURES_PATH"):
            mod.VAL_FEATURES_PATH = MINI_VAL_FEATS
        if hasattr(mod, "TEST_FEATURES_PATH"):
            mod.TEST_FEATURES_PATH = MINI_TEST_FEATS

        if hasattr(mod, "HARD_NEGATIVE_INDICES_PATH"):
            mod.HARD_NEGATIVE_INDICES_PATH = MINI_HARD_NEGS
        if hasattr(mod, "MODEL_DIR"):
            mod.MODEL_DIR = MODELS_DIR
        if hasattr(mod, "WORKING_DIR"):
            mod.WORKING_DIR = DEMO_DIR

    # Patch Model Params in model_factory and config
    library.config.LGBM_PARAMS = DEMO_LGBM
    library.config.XGB_PARAMS = DEMO_XGB
    library.config.CAT_PARAMS = DEMO_CAT

    library.model_factory.LGBM_PARAMS = DEMO_LGBM
    library.model_factory.XGB_PARAMS = DEMO_XGB
    library.model_factory.CAT_PARAMS = DEMO_CAT


def verify_artifacts():
    """
    Checks if the expected output files were created.
    """
    print("Verifying artifacts...")

    # Check Features
    if not os.path.exists(MINI_TRAIN_FEATS):
        raise FileNotFoundError("Train features parquet not created.")

    # Check Models
    expected_models = [
        "scout_lgbm.joblib",
        "scout_xgb.joblib",
        "scout_cat.joblib",
        "lgbm_expert.joblib",
        "xgb_expert.joblib",
        "cat_expert.joblib",
    ]
    for m in expected_models:
        path = os.path.join(MODELS_DIR, m)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {m} not found at {path}")

    # Check Threshold
    if not os.path.exists(os.path.join(DEMO_DIR, "best_threshold.npy")):
        raise FileNotFoundError("Threshold file not found.")

    print("Artifact verification passed.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    setup_demo_data()
    patch_libraries()

    # 2. Instantiate Trainer
    print("\nInitializing Trainer...")
    trainer = library.trainer.Trainer()

    # 3. Run Pipeline
    # Disable loading cached data to force execution of the logic
    print("\nRunning Training Pipeline...")
    ensemble, best_threshold = trainer.run_pipeline(
        load_cached_features=False, load_cached_mining=False
    )

    # 4. Verify Pipeline Output
    print(f"\nPipeline Complete. Best Threshold: {best_threshold}")
    verify_artifacts()

    # 5. Inference Demonstration
    print("\nRunning Inference on Mini Test Set...")

    # Load test features (generated via DataFactory to ensure consistency)
    df_factory = library.data_factory.DataFactory()

    # Note: load_features for 'test' will generate features if not cached
    # We patch the path, so it uses MINI_TEST_META and MINI_TEST_TRACK
    df_test_features = df_factory.load_features(mode="test", load_cached_data=False)

    # Get X
    X_test = df_factory.get_test_data(df_test_features)

    # Predict
    y_probs = ensemble.predict(X_test)
    y_preds = (y_probs >= best_threshold).astype(int)

    # 6. Create Submission
    submission = pd.DataFrame(
        {"contact_id": df_test_features["contact_id"], "contact": y_preds}
    )

    submission_path = os.path.join(DEMO_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission.head())

    # Final Assertion
    assert len(submission) == len(df_test_features), "Submission length mismatch"
    assert submission["contact"].isin([0, 1]).all(), "Invalid predictions found"

    print("\nDemo completed successfully.")
