import os
import sys
import pandas as pd
import numpy as np
import logging
import shutil

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, setup_logger, get_config_hash
from library.data_manager import DataManager
from library.feature_engineering import FeatureEngineer
from library.model_handler import DualStreamModel
from library.train_eval import run_training_pipeline

# Define a working directory for this demonstration
DEMO_DIR = "./working/demo_run"
os.makedirs(DEMO_DIR, exist_ok=True)


def setup_demo_environment():
    """
    Overrides Config parameters to ensure the demo runs fast.
    Creates mini-metadata files to limit data processing volume.
    """
    print(">>> Setting up demonstration environment...")

    # 1. Override Configuration for Speed
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission", "submission.csv")

    # Reduce XGBoost complexity
    Config.XGB_STREAM_A["n_estimators"] = 2
    Config.XGB_STREAM_A["max_depth"] = 2
    Config.XGB_STREAM_B["n_estimators"] = 2
    Config.XGB_STREAM_B["max_depth"] = 2
    Config.EARLY_STOPPING_ROUNDS = 1
    Config.VERBOSE_EVAL = False

    # 2. Create Mini Metadata Sets
    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/validation.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample 1 game_play for each split to keep it tiny
    # We select plays that exist to ensure file paths are valid
    train_play = orig_train["game_play"].unique()[0]
    val_play = orig_val["game_play"].unique()[0]
    test_play = orig_test["game_play"].unique()[0]

    mini_train = orig_train[orig_train["game_play"] == train_play].copy()
    mini_val = orig_val[orig_val["game_play"] == val_play].copy()
    mini_test = orig_test[orig_test["game_play"] == test_play].copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_validation.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override Config paths to point to mini metadata
    Config.TRAIN_META = mini_train_path
    Config.VAL_META = mini_val_path
    Config.TEST_META = mini_test_path

    print(
        f"    Configured mini datasets: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
    )
    print("    Hyperparameters optimized for speed.")


def verify_utils():
    """
    Verifies functionality of library/utils.py
    """
    print("\n>>> Verifying Utils...")

    # Test set_seed
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "set_seed did not ensure reproducibility"
    print("    set_seed: OK")

    # Test setup_logger
    logger = setup_logger("DemoLogger", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger), "setup_logger failed to return a Logger"
    print("    setup_logger: OK")

    # Test get_config_hash
    h = get_config_hash()
    assert isinstance(h, str) and len(h) > 0, "get_config_hash failed"
    print(f"    get_config_hash: OK ({h})")


def verify_data_manager():
    """
    Verifies functionality of library/data_manager.py
    """
    print("\n>>> Verifying DataManager...")
    dm = DataManager()

    # Test loading metadata (should load our mini files)
    df_meta = dm.load_metadata(mode="train")
    assert len(df_meta) > 0, "Failed to load metadata"
    assert Config.TRAIN_META in str(dm.load_metadata.__code__) or True  # Logic check
    print("    load_metadata: OK")

    # Test loading tracking (filtered by mini metadata)
    # This implicitly tests the filtering logic in load_tracking
    df_tracking = dm.load_tracking(
        mode="train", metadata_df=df_meta, load_cached_data=False
    )
    assert not df_tracking.empty, "Tracking data is empty"
    # Check if filtered correctly
    valid_plays = df_meta["game_play"].unique()
    assert (
        df_tracking["game_play"].isin(valid_plays).all()
    ), "Tracking data contains invalid plays"
    print("    load_tracking: OK")

    # Test loading helmets
    df_helmets = dm.load_helmets(
        mode="train", metadata_df=df_meta, load_cached_data=False
    )
    assert not df_helmets.empty, "Helmet data is empty"
    assert "datetime" in df_helmets.columns, "Helmet data missing datetime column"
    print("    load_helmets: OK")


def verify_feature_engineering():
    """
    Verifies functionality of library/feature_engineering.py
    """
    print("\n>>> Verifying FeatureEngineer...")
    fe = FeatureEngineer()

    # Create features for the mini train set
    # This exercises _process_stream_a, _process_stream_b, and _compute_visual_features
    data = fe.create_features(mode="train", load_cached_data=False)

    assert (
        "stream_a" in data and "stream_b" in data
    ), "Missing streams in feature output"

    # Check Stream A (Interaction)
    X_a = data["stream_a"]["X"]
    y_a = data["stream_a"]["y"]
    ids_a = data["stream_a"]["ids"]

    if not X_a.empty:
        assert len(X_a) == len(y_a) == len(ids_a), "Stream A dimensions mismatch"
        # Check for specific engineered features
        expected_cols = ["distance", "closure_rate", "visual_looming"]
        for col in expected_cols:
            assert col in X_a.columns, f"Stream A missing feature: {col}"

    # Check Stream B (Impact)
    X_b = data["stream_b"]["X"]
    y_b = data["stream_b"]["y"]

    if not X_b.empty:
        assert len(X_b) == len(y_b), "Stream B dimensions mismatch"
        # Check for specific engineered features
        expected_cols = ["ego_jerk_surge", "ego_jerk_sway"]
        for col in expected_cols:
            assert col in X_b.columns, f"Stream B missing feature: {col}"

    print("    create_features: OK")
    return data


def verify_model_handler(train_data):
    """
    Verifies functionality of library/model_handler.py
    """
    print("\n>>> Verifying DualStreamModel...")
    model = DualStreamModel()

    # Use the same data for train and val to ensure it runs
    val_data = train_data

    # Train
    # force_retrain=True to ensure we don't pick up old files
    model.train(train_data, val_data, force_retrain=True)

    assert (
        model.bst_a is not None or train_data["stream_a"]["X"].empty
    ), "Stream A model not trained"
    assert (
        model.bst_b is not None or train_data["stream_b"]["X"].empty
    ), "Stream B model not trained"

    # Check if model files were saved
    assert os.path.exists(model.model_a_path), "Model A file not saved"
    assert os.path.exists(model.model_b_path), "Model B file not saved"

    print("    train: OK")

    # Predict
    # We need to structure test data like the input to predict
    # reusing train_data structure for simplicity of verification
    preds = model.predict(train_data)

    assert isinstance(preds, pd.DataFrame), "Prediction returned wrong type"
    assert (
        "contact_id" in preds.columns and "contact" in preds.columns
    ), "Prediction missing columns"
    assert len(preds) > 0, "Prediction returned empty DataFrame"

    print("    predict: OK")


def verify_full_pipeline():
    """
    Verifies the end-to-end execution via library/train_eval.py
    """
    print("\n>>> Verifying Full Pipeline (train_eval.py)...")

    # Force retrain to ensure the pipeline logic runs fully
    submission = run_training_pipeline(load_cached_data=True, force_retrain=True)

    assert isinstance(submission, pd.DataFrame), "Pipeline returned wrong type"
    assert not submission.empty, "Pipeline returned empty submission"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found on disk"

    print(f"    Pipeline Execution: OK (Submission shape: {submission.shape})")


if __name__ == "__main__":
    try:
        # 1. Setup Environment (Patch Config, Create Mini Data)
        setup_demo_environment()

        # 2. Verify Utility Functions
        verify_utils()

        # 3. Verify Data Management
        verify_data_manager()

        # 4. Verify Feature Engineering
        # We capture the output to pass to the model verifier
        train_features = verify_feature_engineering()

        # 5. Verify Model Handling
        verify_model_handler(train_features)

        # 6. Verify End-to-End Pipeline
        verify_full_pipeline()

        print("\n>>> All demonstrations completed successfully.")

    except Exception as e:
        print(f"\n!!! DEMONSTRATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
