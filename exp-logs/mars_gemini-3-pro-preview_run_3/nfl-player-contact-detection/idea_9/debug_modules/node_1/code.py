import os
import pandas as pd
import numpy as np
import sys
import shutil

# Import provided library modules
import library.config as config
import library.features as features_module
import library.pipeline as pipeline
from library.utils import seed_everything
from library.model_wrapper import DualStreamModel


def main():
    print("Initializing Demonstration...")

    # 1. Setup & Configuration
    # Set seed for reproducibility
    seed_everything(config.SEED)

    # Override XGBoost params for speed (Demo purposes only)
    config.XGB_PARAMS.update(
        {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": 4,
            "tree_method": "hist",  # Fast for small data
        }
    )

    # Clean working directory to ensure fresh run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Define Mock Data Loader for Speed
    # This prevents loading the full 4GB+ datasets
    def mock_load_dataset(mode="train", load_cached_data=False):
        print(f"  [Mock] Loading small subset for mode='{mode}'...")

        # Determine paths based on mode
        if mode in ["train", "validation"]:
            tracking_path = config.TRAIN_TRACKING_PATH
            helmets_path = config.TRAIN_HELMETS_PATH
            meta_path = (
                config.TRAIN_META_PATH if mode == "train" else config.VAL_META_PATH
            )
        else:
            tracking_path = config.TEST_TRACKING_PATH
            helmets_path = config.TEST_HELMETS_PATH
            meta_path = config.TEST_META_PATH

        # Load small chunks of tracking and helmets (e.g., 10k rows)
        # This ensures we have enough data for a few plays
        tracking_df = pd.read_csv(tracking_path, nrows=10000)
        helmets_df = pd.read_csv(helmets_path, nrows=10000)

        # Get unique game_plays from tracking to filter metadata
        valid_game_plays = tracking_df["game_play"].unique()

        # Load metadata and filter to match tracking data
        # This ensures referential integrity for merges
        labels_df = pd.read_csv(meta_path)
        labels_df = labels_df[labels_df["game_play"].isin(valid_game_plays)].copy()

        # If labels are empty (e.g. validation set split might not align with first 10k rows of tracking),
        # we fallback to taking the first few rows of labels and faking the tracking data match
        # or just proceeding (FeatureEngineer handles missing tracking with NaNs).
        # For this demo, let's ensure we have some data.
        if len(labels_df) == 0:
            print(
                "  [Mock] Warning: No matching game_plays in metadata. Taking head of metadata."
            )
            labels_df = pd.read_csv(meta_path, nrows=100)
            # We won't have tracking for these, but the pipeline should handle NaNs/zeros.

        print(
            f"  [Mock] Loaded: Labels={len(labels_df)}, Tracking={len(tracking_df)}, Helmets={len(helmets_df)}"
        )
        return labels_df, tracking_df, helmets_df

    # 3. Monkey-Patch the FeatureEngineer's data loader
    # We patch the module where load_dataset is imported
    features_module.load_dataset = mock_load_dataset

    # 4. Run Training Pipeline
    print("\n--- Step 1: Running Training Pipeline ---")
    # We use debug=True to further slice the generated features before training
    model = pipeline.run_training(
        load_cached_data=False,  # Force re-compute with our mock data
        debug=True,
        n_estimators=10,
    )

    # Validation: Check if model files were created
    print("\n--- Validating Training Artifacts ---")
    model_a_path = os.path.join(config.WORKING_DIR, "model_A.json")
    model_b_path = os.path.join(config.WORKING_DIR, "model_B.json")
    thresholds_path = os.path.join(config.WORKING_DIR, "thresholds.json")

    assert os.path.exists(model_a_path), "Model A artifact missing!"
    assert os.path.exists(model_b_path), "Model B artifact missing!"
    assert os.path.exists(thresholds_path), "Thresholds artifact missing!"
    print("  [Check] Model artifacts verified.")

    # Validation: Check Model Object
    assert isinstance(model, DualStreamModel), "Returned object is not DualStreamModel"
    assert "A" in model.models, "Model A not in memory"
    assert "B" in model.models, "Model B not in memory"
    print("  [Check] In-memory model verified.")

    # 5. Run Inference Pipeline
    print("\n--- Step 2: Running Inference Pipeline ---")
    # We pass the trained model directly to avoid reloading from disk (though reloading is also supported)
    submission_df = pipeline.run_inference(model=model, load_cached_data=False)

    # Validation: Check Submission
    print("\n--- Validating Submission ---")
    assert isinstance(
        submission_df, pd.DataFrame
    ), "Inference did not return a DataFrame"
    assert "contact_id" in submission_df.columns, "contact_id column missing"
    assert "contact" in submission_df.columns, "contact column missing"

    # Check if submission file exists on disk
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found on disk"

    # Check content
    saved_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"  Submission Shape: {saved_df.shape}")
    print("  Sample Predictions:")
    print(saved_df.head())

    assert len(saved_df) > 0, "Submission file is empty"
    assert (
        saved_df["contact"].isin([0, 1]).all()
    ), "Predictions contain non-binary values"
    print("  [Check] Submission file verified.")

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
