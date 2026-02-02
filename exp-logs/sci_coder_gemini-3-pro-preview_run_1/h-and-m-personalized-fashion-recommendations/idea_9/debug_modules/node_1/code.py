import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import warnings

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_manager import DataManager
from library.model_dwsc import DWSCRecommender
from library.evaluation import calculate_map12

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting DWSC Recommender Demonstration...")
    set_seed(Config.SEED)

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    print("\n[Step 1] Configuring for fast demonstration...")
    # Enable DEBUG mode to load only a small tail of the transaction logs
    Config.DEBUG = True
    Config.DEBUG_ROWS = 50000  # Process only 50k rows for this demo

    # Reduce computational complexity for the demo
    Config.CF_NEIGHBORS = 10  # Keep top-10 neighbors instead of 100
    Config.BATCH_SIZE = 1000  # Smaller batch size
    Config.TOP_K = 12

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Loading & Preprocessing (Validation Split)
    # =========================================================================
    print("\n[Step 2] Loading and preparing validation data...")
    dm = DataManager()

    # We force load_cached_data=False to demonstrate the processing logic
    train_df, target_df, test_users = dm.get_validation_data(load_cached_data=False)

    # --- Verification ---
    print(f"  Train shape: {train_df.shape}")
    print(f"  Target shape: {target_df.shape}")
    print(f"  Test users count: {len(test_users)}")

    assert not train_df.empty, "Training dataframe should not be empty."
    assert not target_df.empty, "Target dataframe should not be empty."
    assert len(test_users) > 0, "Test users list should not be empty."
    assert (
        "days_elapsed" in train_df.columns
    ), "days_elapsed column missing in train_df."
    assert train_df["days_elapsed"].min() >= 0, "days_elapsed should be non-negative."

    # =========================================================================
    # 3. Model Training (Fitting)
    # =========================================================================
    print("\n[Step 3] Fitting DWSC Model...")
    model = DWSCRecommender()

    # Get dimensions from the encoder
    n_users = len(dm.encoder.user_to_idx)
    n_items = len(dm.encoder.item_to_idx)
    print(f"  Encoder Dimensions: Users={n_users}, Items={n_items}")

    model.fit(train_df, n_users, n_items, load_cached_data=False)

    # --- Verification ---
    assert model.S is not None, "Similarity matrix S was not created."
    assert sp.issparse(model.S), "Similarity matrix S should be sparse."
    assert model.trend_scores is not None, "Trend scores were not computed."
    assert len(model.trend_scores) == n_items, "Trend scores dimension mismatch."
    print("  Model fit verification passed.")

    # =========================================================================
    # 4. Inference (Validation)
    # =========================================================================
    print("\n[Step 4] Generating predictions for validation set...")

    # Predict for the users in the validation target set
    val_preds_df = model.predict(
        test_users, train_df, dm.encoder, load_cached_data=False
    )

    # --- Verification ---
    print("  Predictions generated.")
    print(val_preds_df.head(2))

    assert "customer_id" in val_preds_df.columns
    assert "prediction" in val_preds_df.columns
    assert len(val_preds_df) == len(test_users), "Prediction count mismatch."

    # Check format of prediction string (should be space-separated IDs)
    sample_pred = val_preds_df.iloc[0]["prediction"]
    assert isinstance(sample_pred, str), "Prediction should be a string."
    assert len(sample_pred.split()) <= Config.TOP_K, "Too many items predicted."

    # =========================================================================
    # 5. Evaluation
    # =========================================================================
    print("\n[Step 5] Evaluating MAP@12...")

    map_score = calculate_map12(target_df, val_preds_df)
    print(f"  Validation MAP@12 Score: {map_score:.6f}")

    # Sanity check: Score should be a float between 0 and 1
    assert 0.0 <= map_score <= 1.0, "MAP@12 score out of range."

    # =========================================================================
    # 6. Submission Workflow (Dry Run)
    # =========================================================================
    print("\n[Step 6] Running Submission Workflow (Dry Run)...")

    # Load submission data (Train = last 10 weeks relative to max date, Test = sample_submission users)
    sub_train_df, sub_test_users = dm.get_submission_data(load_cached_data=False)

    print(f"  Submission Train shape: {sub_train_df.shape}")
    print(f"  Submission Test users: {len(sub_test_users)}")

    # Re-fit model on submission training data
    # Note: In a real run, we would fit on the full dataset.
    # Here we fit on the debug subset returned by get_submission_data.
    model.fit(sub_train_df, n_users, n_items, load_cached_data=False)

    # Predict
    final_submission_df = model.predict(
        sub_test_users, sub_train_df, dm.encoder, load_cached_data=False
    )

    # Final check
    assert len(final_submission_df) == len(sub_test_users)
    print(f"  Final submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
