import os
import pandas as pd
import numpy as np
import sys

# Import library components
from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import DatasetManager
from library.model_xgb import XGBWrapper
from library.metrics import calculate_mcc


def main():
    # 1. Setup
    set_seed(42)
    logger = get_logger("demo_script")
    logger.info("Starting demo script...")

    # Define a custom working directory for this demo to avoid cache conflicts
    demo_working_dir = "./working/demo_run"
    os.makedirs(demo_working_dir, exist_ok=True)

    # Update Config to use this directory for caching and model saving
    Config.WORKING_DIR = demo_working_dir
    Config.MODEL_PATH = os.path.join(demo_working_dir, "xgb_model.json")

    # 2. Create Data Subsets for Speed
    # We need to preserve temporal structure, so we sample by game_play.
    # Random row sampling would break lag feature generation.
    logger.info("Creating data subsets...")

    # --- Train Subset ---
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(
            f"Original train metadata not found at {Config.TRAIN_META_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    unique_games_train = df_train["game_play"].unique()
    # Select first 2 games for training to keep it very fast
    subset_games_train = unique_games_train[:2]
    df_train_subset = df_train[df_train["game_play"].isin(subset_games_train)].copy()

    train_subset_path = os.path.join(demo_working_dir, "train_subset.csv")
    df_train_subset.to_csv(train_subset_path, index=False)
    # Point Config to the new subset
    Config.TRAIN_META_PATH = train_subset_path

    # --- Validation Subset ---
    df_val = pd.read_csv(Config.VAL_META_PATH)
    unique_games_val = df_val["game_play"].unique()
    # Select first 1 game for validation
    subset_games_val = unique_games_val[:1]
    df_val_subset = df_val[df_val["game_play"].isin(subset_games_val)].copy()

    val_subset_path = os.path.join(demo_working_dir, "val_subset.csv")
    df_val_subset.to_csv(val_subset_path, index=False)
    # Point Config to the new subset
    Config.VAL_META_PATH = val_subset_path

    # --- Test Subset ---
    df_test = pd.read_csv(Config.TEST_META_PATH)
    unique_games_test = df_test["game_play"].unique()
    # Select first 1 game for testing
    subset_games_test = unique_games_test[:1]
    df_test_subset = df_test[df_test["game_play"].isin(subset_games_test)].copy()

    test_subset_path = os.path.join(demo_working_dir, "test_subset.csv")
    df_test_subset.to_csv(test_subset_path, index=False)
    # Point Config to the new subset
    Config.TEST_META_PATH = test_subset_path

    logger.info(
        f"Train subset games: {len(subset_games_train)} | Rows: {len(df_train_subset)}"
    )
    logger.info(
        f"Val subset games: {len(subset_games_val)} | Rows: {len(df_val_subset)}"
    )

    # 3. Configure Model for Speed
    # Drastically reduce estimators for demonstration purposes
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    # Use 'hist' tree method to ensure compatibility even if GPU allocation is tricky with tiny data
    Config.XGB_PARAMS["tree_method"] = "hist"

    # 4. Data Loading & Feature Engineering
    logger.info("Initializing DatasetManager...")
    # Force regeneration (load_cached_data=False) to ensure we use our new subsets
    # instead of any potentially existing cache in the working dir.
    dm = DatasetManager(load_cached_data=False)

    logger.info("Loading Train Data...")
    X_train, y_train, ids_train = dm.get_train_data()

    # Logic Verification
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_train) == len(y_train), "Mismatch in X and y lengths (Train)"
    assert not X_train.isnull().values.any(), "NaNs found in training features"

    logger.info("Loading Validation Data...")
    X_val, y_val, ids_val = dm.get_validation_data()
    assert len(X_val) > 0, "Validation data is empty"

    # 5. Model Training
    logger.info("Initializing and Training Model...")
    model_wrapper = XGBWrapper()
    model_wrapper.train(X_train, y_train, X_val, y_val)

    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file was not saved at {Config.MODEL_PATH}"

    # 6. Evaluation
    logger.info("Evaluating on Validation Set...")
    val_probs = model_wrapper.predict(X_val)
    assert len(val_probs) == len(y_val), "Prediction length mismatch"

    # Optimize threshold
    best_thresh = model_wrapper.optimize_threshold(y_val, val_probs)

    # Calculate final MCC
    val_preds = (val_probs >= best_thresh).astype(int)
    mcc = calculate_mcc(y_val, val_preds)
    logger.info(f"Validation MCC: {mcc:.4f}")
    assert -1.0 <= mcc <= 1.0, "MCC score out of valid range [-1, 1]"

    # 7. Inference on Test
    logger.info("Running Inference on Test Set...")
    X_test, _, ids_test = dm.get_test_data()

    test_probs = model_wrapper.predict(X_test)
    test_preds = (test_probs >= best_thresh).astype(int)

    # 8. Create Submission
    logger.info("Creating Submission File...")
    submission_df = pd.DataFrame({"contact_id": ids_test, "contact": test_preds})

    # Ensure submission dir exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    # Validate submission
    assert os.path.exists(submission_path), "Submission file not created"

    saved_df = pd.read_csv(submission_path)
    assert list(saved_df.columns) == [
        "contact_id",
        "contact",
    ], "Incorrect submission columns"
    assert len(saved_df) == len(ids_test), "Submission row count mismatch"
    assert (
        saved_df["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values"

    logger.info(f"Submission saved to {submission_path}")
    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    main()
