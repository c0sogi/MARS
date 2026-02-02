import os
import pandas as pd
import numpy as np
import sys
import shutil

# Import library components
from library.config import Config
from library.data_manager import DataManager
from library.model_factory import ContactGBDT
from library.optimization import optimize_thresholds, generate_submission
from library.utils import seed_everything


def create_mini_metadata():
    """
    Creates a subset of the training and validation metadata to speed up the demo.
    We select a small number of unique games/plays.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Sample 2 unique plays for training, 1 for validation
    train_plays = train_meta["game_play"].unique()[:2]
    val_plays = val_meta["game_play"].unique()[:1]

    mini_train = train_meta[train_meta["game_play"].isin(train_plays)].copy()
    mini_val = val_meta[val_meta["game_play"].isin(val_plays)].copy()

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_validation.csv")

    # Save to working directory
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)

    print(f"Mini Train shape: {mini_train.shape}")
    print(f"Mini Val shape: {mini_val.shape}")

    return mini_train_path, mini_val_path


def configure_demo(mini_train_path, mini_val_path):
    """
    Patches the Config class to use mini datasets and faster model parameters.
    """
    print("Patching Configuration for demo...")

    # Override Metadata Paths
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path

    # Override XGBoost parameters for speed (fewer trees)
    # Stream A
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_A["early_stopping_rounds"] = 5

    # Stream B
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_B["early_stopping_rounds"] = 5

    # Disable caching for the demo to ensure feature engineering code runs
    # In a real run, you would keep this True.
    return False


def run_pipeline():
    seed_everything(Config.SEED)

    # 1. Setup Data
    mini_train_path, mini_val_path = create_mini_metadata()
    use_cache = configure_demo(mini_train_path, mini_val_path)

    # 2. Initialize Data Manager
    print("\n=== Initializing Data Manager ===")
    dm = DataManager(config=Config)

    # 3. Load/Generate Training Data
    # This triggers FeatureEngineer to process the mini datasets
    print("\n=== Generating/Loading Training Data ===")
    data_splits = dm.get_train_data(load_cached_data=use_cache)

    # Extract splits
    X_train_A, y_train_A = data_splits["stream_A"]["train"]
    X_val_A, y_val_A = data_splits["stream_A"]["val"]

    X_train_B, y_train_B = data_splits["stream_B"]["train"]
    X_val_B, y_val_B = data_splits["stream_B"]["val"]

    # Validation: Ensure we have data
    assert len(X_train_A) > 0, "Stream A training data is empty"
    assert len(X_train_B) > 0, "Stream B training data is empty"
    print(f"Stream A Train Samples: {len(X_train_A)}")
    print(f"Stream B Train Samples: {len(X_train_B)}")

    # 4. Train Models
    print("\n=== Training Stream A (Interaction Model) ===")
    model_A = ContactGBDT(Config.XGB_PARAMS_STREAM_A)
    model_A.train(X_train_A, y_train_A, X_val_A, y_val_A, verbose=True)

    print("\n=== Training Stream B (Impact Model) ===")
    model_B = ContactGBDT(Config.XGB_PARAMS_STREAM_B)
    model_B.train(X_train_B, y_train_B, X_val_B, y_val_B, verbose=True)

    # Save models
    model_A_path = os.path.join(Config.WORKING_DIR, "model_A.json")
    model_B_path = os.path.join(Config.WORKING_DIR, "model_B.json")
    model_A.save(model_A_path)
    model_B.save(model_B_path)

    assert os.path.exists(model_A_path), "Model A failed to save"

    # 5. Optimize Thresholds
    print("\n=== Optimizing Thresholds ===")
    # Get probabilities on validation set
    probs_val_A = model_A.predict_proba(X_val_A)
    probs_val_B = model_B.predict_proba(X_val_B)

    optimization_results = optimize_thresholds(
        y_val_A, probs_val_A, y_val_B, probs_val_B
    )

    thresh_A = optimization_results["thresh_A"]
    thresh_B = optimization_results["thresh_B"]

    # 6. Inference on Test Set
    print("\n=== Running Inference on Test Set ===")
    # Note: We run on full test set as it's required for submission
    X_test_A, ids_A, X_test_B, ids_B = dm.get_test_data(load_cached_data=use_cache)

    print(f"Test Stream A size: {len(X_test_A)}")
    print(f"Test Stream B size: {len(X_test_B)}")

    probs_test_A = model_A.predict_proba(X_test_A)
    probs_test_B = model_B.predict_proba(X_test_B)

    # 7. Generate Submission
    print("\n=== Generating Submission ===")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    generate_submission(
        ids_A,
        probs_test_A,
        ids_B,
        probs_test_B,
        thresh_A,
        thresh_B,
        output_path=submission_path,
    )

    # Final Validation
    assert os.path.exists(submission_path), "Submission file was not created"
    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Verify against sample submission length if available
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        assert len(df_sub) == len(
            sample
        ), f"Submission length mismatch. Expected {len(sample)}, got {len(df_sub)}"

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    run_pipeline()
