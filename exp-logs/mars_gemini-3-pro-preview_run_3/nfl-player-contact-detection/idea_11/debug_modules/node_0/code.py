import os
import sys
import pandas as pd
import numpy as np
import shutil

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_manager import DataManager
from library.feature_builder import FeatureBuilder
from library.model_factory import DualStreamModel
from library.optimizer import ThresholdOptimizer


def run_demo_pipeline():
    print("=== Starting NFL Contact Detection Demo Pipeline ===")

    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # Define temporary paths for the demo subset
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_validation.csv")

    # 2. Create a Mini Dataset for Speed
    # We will load the full metadata, select 2 unique game_plays, and save as a new metadata file.
    # We then patch Config to point to these new files.
    print("\n[Step 1] Creating Mini Dataset...")
    dm = DataManager()

    # Load original metadata
    full_train_meta = dm.load_metadata("train")

    # Select 2 unique plays
    unique_plays = full_train_meta["game_play"].unique()
    selected_plays = unique_plays[:2]
    print(f"Selected plays for demo: {selected_plays}")

    # Filter metadata
    mini_df = full_train_meta[full_train_meta["game_play"].isin(selected_plays)].copy()

    # Split into mini-train and mini-val (50-50 split of the 2 plays for demo purposes)
    # Play 1 -> Train, Play 2 -> Val
    mini_train_df = mini_df[mini_df["game_play"] == selected_plays[0]].copy()
    mini_val_df = mini_df[mini_df["game_play"] == selected_plays[1]].copy()

    # Save to working directory
    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)

    # Patch Config to use these mini files
    # We must patch the class attributes directly
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path

    # Patch XGBoost params for speed (reduce estimators)
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 2

    print(f"Mini Train Shape: {mini_train_df.shape}")
    print(f"Mini Val Shape: {mini_val_df.shape}")

    # 3. Feature Engineering
    print("\n[Step 2] Building Features...")
    fb = FeatureBuilder()

    # Force reload by setting load_cached_data=False or ensuring cache doesn't exist for these specific paths.
    # Since the split names are still 'train' and 'val', we should clear specific cache files if they exist
    # to avoid loading full dataset features.
    for split in ["train", "val"]:
        for stream in ["streamA", "streamB"]:
            cache_file = Config.get_feature_cache_path(stream, split)
            if os.path.exists(cache_file):
                os.remove(cache_file)
            npy_ids = cache_file.replace(".parquet", "_ids.npy")
            if os.path.exists(npy_ids):
                os.remove(npy_ids)
            npy_y = cache_file.replace(".parquet", "_y.npy")
            if os.path.exists(npy_y):
                os.remove(npy_y)

    # --- Stream A (Player-Player) ---
    print("Building Stream A (Interaction)...")
    X_train_A, ids_train_A, y_train_A = fb.build_stream_a_features(
        "train", load_cached_data=False
    )
    X_val_A, ids_val_A, y_val_A = fb.build_stream_a_features(
        "val", load_cached_data=False
    )

    # Verify Stream A
    assert X_train_A.shape[0] == len(
        y_train_A
    ), "Stream A Train features/labels mismatch"
    assert X_val_A.shape[0] == len(y_val_A), "Stream A Val features/labels mismatch"
    assert (
        "distance" in X_train_A.columns
    ), "Stream A missing derived feature 'distance'"
    assert (
        "Sideline_IoU" in X_train_A.columns
    ), "Stream A missing visual feature 'Sideline_IoU'"
    print(f"Stream A Train Features: {X_train_A.shape}")

    # --- Stream B (Player-Ground) ---
    print("Building Stream B (Impact)...")
    X_train_B, ids_train_B, y_train_B = fb.build_stream_b_features(
        "train", load_cached_data=False
    )
    X_val_B, ids_val_B, y_val_B = fb.build_stream_b_features(
        "val", load_cached_data=False
    )

    # Verify Stream B
    assert X_train_B.shape[0] == len(
        y_train_B
    ), "Stream B Train features/labels mismatch"
    assert "jerk" in X_train_B.columns, "Stream B missing physics feature 'jerk'"
    assert (
        "Sideline_IoU" not in X_train_B.columns
    ), "Stream B should not have visual features"
    print(f"Stream B Train Features: {X_train_B.shape}")

    # 4. Model Training
    print("\n[Step 3] Training Dual Stream Model...")
    model = DualStreamModel()

    data_bundle = {
        "X_train_A": X_train_A,
        "y_train_A": y_train_A,
        "X_val_A": X_val_A,
        "y_val_A": y_val_A,
        "X_train_B": X_train_B,
        "y_train_B": y_train_B,
        "X_val_B": X_val_B,
        "y_val_B": y_val_B,
    }

    model.train(data_bundle)

    # Verify models exist
    assert model.model_a is not None, "Model A failed to train"
    assert model.model_b is not None, "Model B failed to train"
    print("Training complete.")

    # 5. Threshold Optimization
    print("\n[Step 4] Optimizing Thresholds...")
    # Using the optimizer logic integrated in DualStreamModel
    model.optimize_thresholds(X_val_A, y_val_A, X_val_B, y_val_B)

    print(f"Optimized Threshold A: {model.threshold_a}")
    print(f"Optimized Threshold B: {model.threshold_b}")

    # Verify thresholds are within search range
    assert (
        Config.THRESHOLD_SEARCH_START <= model.threshold_a < Config.THRESHOLD_SEARCH_END
    )
    assert (
        Config.THRESHOLD_SEARCH_START <= model.threshold_b < Config.THRESHOLD_SEARCH_END
    )

    # 6. Inference / Prediction
    print("\n[Step 5] Running Inference (Simulation)...")
    # We will use the validation set as a proxy for the test set to demonstrate prediction

    submission = model.predict(X_val_A, ids_val_A, X_val_B, ids_val_B)

    print(f"Submission Shape: {submission.shape}")
    print("Sample Predictions:")
    print(submission.head())

    # Verify Submission Format
    assert "contact_id" in submission.columns, "Submission missing 'contact_id'"
    assert "contact" in submission.columns, "Submission missing 'contact'"
    assert (
        submission["contact"].isin([0, 1]).all()
    ), "Predictions must be binary (0 or 1)"

    # Check that we have predictions for both streams combined
    expected_rows = len(ids_val_A) + len(ids_val_B)
    assert (
        len(submission) == expected_rows
    ), f"Expected {expected_rows} predictions, got {len(submission)}"

    print("\n=== Demo Pipeline Completed Successfully ===")


if __name__ == "__main__":
    # Ensure errors are raised explicitly
    try:
        run_demo_pipeline()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
