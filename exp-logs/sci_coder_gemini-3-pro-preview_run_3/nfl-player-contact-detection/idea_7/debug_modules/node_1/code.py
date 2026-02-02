import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, Timer
from library.data_manager import DataManager
from library.model_factory import StreamModel
from library.ensemble import EnsembleOptimizer


def create_mini_metadata():
    """
    Creates smaller versions of metadata files to speed up the demonstration.
    Selects a few plays from the original metadata.
    """
    print("Creating mini metadata files for rapid execution...")

    # Define source and target paths
    meta_files = {
        "train": (
            Config.TRAIN_META_PATH,
            os.path.join(Config.WORKING_DIR, "train_mini.csv"),
        ),
        "validation": (
            Config.VAL_META_PATH,
            os.path.join(Config.WORKING_DIR, "val_mini.csv"),
        ),
        "test": (
            Config.TEST_META_PATH,
            os.path.join(Config.WORKING_DIR, "test_mini.csv"),
        ),
    }

    for split, (src, dst) in meta_files.items():
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Select top 2 unique plays to keep data volume very low
            unique_plays = df["game_play"].unique()[:2]
            df_mini = df[df["game_play"].isin(unique_plays)].copy()
            df_mini.to_csv(dst, index=False)
            print(
                f"  Created {dst} with {len(df_mini)} rows (Plays: {len(unique_plays)})"
            )
        else:
            print(f"  Warning: Source {src} not found.")

    return meta_files


def override_config(mini_meta_paths):
    """
    Overrides default configuration for speed and demonstration purposes.
    """
    print("Overriding configuration for demo...")

    # Point to mini metadata
    Config.TRAIN_META_PATH = mini_meta_paths["train"][1]
    Config.VAL_META_PATH = mini_meta_paths["validation"][1]
    Config.TEST_META_PATH = mini_meta_paths["test"][1]

    # Reduce XGBoost complexity
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    Config.XGB_PARAMS["max_depth"] = 3

    # Reduce Ensemble search space
    Config.BLEND_WEIGHTS = np.linspace(0, 1, 5)  # 5 steps instead of 21
    Config.THRESHOLDS = np.linspace(0.1, 0.9, 5)  # 5 steps instead of 81

    # Reduce Window sizes for faster feature gen
    Config.WINDOW_MICRO = 2
    Config.WINDOW_MACRO = 5


def main():
    # 1. Setup
    set_seed(42)

    # Create mini datasets and override config
    mini_meta_paths = create_mini_metadata()
    override_config(mini_meta_paths)

    # Initialize DataManager
    dm = DataManager()

    # =========================================================================
    # 2. Data Loading & Feature Generation (Stream A: Tracking)
    # =========================================================================
    with Timer("Stream A (Tracking) Data Loading"):
        # Load Train
        X_train_a, y_train_a, meta_train_a = dm.load_stream_data(
            stream="tracking", split="train", load_cached_data=False
        )
        # Load Validation
        X_val_a, y_val_a, meta_val_a = dm.load_stream_data(
            stream="tracking", split="validation", load_cached_data=False
        )

    # Validation
    assert not X_train_a.empty, "Stream A Training features should not be empty"
    assert len(X_train_a) == len(y_train_a), "Stream A X and y lengths mismatch"
    print(f"Stream A Train Shape: {X_train_a.shape}")

    # =========================================================================
    # 3. Data Loading & Feature Generation (Stream B: Helmets)
    # =========================================================================
    with Timer("Stream B (Helmets) Data Loading"):
        # Load Train
        X_train_b, y_train_b, meta_train_b = dm.load_stream_data(
            stream="helmets", split="train", load_cached_data=False
        )
        # Load Validation
        X_val_b, y_val_b, meta_val_b = dm.load_stream_data(
            stream="helmets", split="validation", load_cached_data=False
        )

    # Validation
    assert not X_train_b.empty, "Stream B Training features should not be empty"
    # Ensure alignment between streams (since we used deterministic mini-metadata)
    # Note: If DataManager._subsample was used inside load_stream_data with debug_sample < 1.0,
    # alignment isn't guaranteed unless seeded carefully. Here we loaded full mini-sets (default debug_sample=1.0).
    assert len(X_train_a) == len(X_train_b), "Stream A and B training set sizes differ"
    print(f"Stream B Train Shape: {X_train_b.shape}")

    # =========================================================================
    # 4. Model Training
    # =========================================================================
    # --- Train Stream A ---
    model_a = StreamModel(stream_name="tracking")
    with Timer("Train Stream A"):
        model_a.train(X_train_a, y_train_a, meta_train_a, X_val_a, y_val_a, meta_val_a)

    assert model_a.model_pp is not None, "Stream A PP model failed to train"
    assert model_a.model_pg is not None, "Stream A PG model failed to train"

    # --- Train Stream B ---
    model_b = StreamModel(stream_name="helmets")
    with Timer("Train Stream B"):
        model_b.train(X_train_b, y_train_b, meta_train_b, X_val_b, y_val_b, meta_val_b)

    assert model_b.model_pp is not None, "Stream B PP model failed to train"

    # =========================================================================
    # 5. Ensemble Optimization
    # =========================================================================
    print("\nGenerating Validation Predictions for Ensemble...")
    preds_a = model_a.predict_proba(X_val_a, meta_val_a)
    preds_b = model_b.predict_proba(X_val_b, meta_val_b)

    optimizer = EnsembleOptimizer()
    best_weight, best_thresh, best_mcc = optimizer.optimize(y_val_a, preds_a, preds_b)

    assert 0.0 <= best_weight <= 1.0, "Optimal weight out of bounds"
    assert 0.0 < best_thresh < 1.0, "Optimal threshold out of bounds"

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\nRunning Inference on Test Set...")

    # Load Test Data
    X_test_a, _, meta_test_a = dm.load_stream_data(
        "tracking", "test", load_cached_data=False
    )
    X_test_b, _, meta_test_b = dm.load_stream_data(
        "helmets", "test", load_cached_data=False
    )

    # Predict
    test_preds_a = model_a.predict_proba(X_test_a, meta_test_a)
    test_preds_b = model_b.predict_proba(X_test_b, meta_test_b)

    # Blend
    final_probs = optimizer.blend_predictions(test_preds_a, test_preds_b, best_weight)

    # Create Submission
    submission_df = dm.prepare_submission(
        meta_test_a, final_probs, threshold=best_thresh
    )

    # Save
    optimizer.save_submission(submission_df)

    # Verify
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
    print(f"Submission Head:\n{submission_df.head()}")


if __name__ == "__main__":
    main()
