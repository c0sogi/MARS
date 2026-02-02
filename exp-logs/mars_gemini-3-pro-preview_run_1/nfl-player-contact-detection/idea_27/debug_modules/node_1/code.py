import os
import shutil
import pandas as pd
import numpy as np
import warnings
import joblib

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.training_pipeline import TrainingPipeline
from library.inference import InferencePipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by:
    1. Defining a working directory.
    2. Creating mini-datasets to speed up execution.
    3. Monkey-patching the Config class to use these mini-datasets and reduced hyperparameters.
    """
    print(">>> Setting up Demo Environment...")

    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # --- 1. Create Mini Datasets ---
    # We select a few plays to keep data consistent between metadata and tracking
    print("Creating mini-datasets...")

    # Load full metadata to sample plays
    full_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select plays that have at least one contact event to ensure positive samples
    plays_with_contact = full_train_meta[full_train_meta["contact"] == 1][
        "game_play"
    ].unique()
    # Sample 2 plays with contact
    selected_plays = np.random.choice(
        plays_with_contact, size=min(2, len(plays_with_contact)), replace=False
    )

    # Filter Metadata
    mini_train_meta = full_train_meta[
        full_train_meta["game_play"].isin(selected_plays)
    ].copy()

    # Create a mini val set (using different plays if possible, or split the sample)
    # For demo speed, we'll just take a small slice of the remaining data for validation
    remaining_meta = full_train_meta[~full_train_meta["game_play"].isin(selected_plays)]
    mini_val_meta = remaining_meta.head(500).copy()  # Arbitrary small number

    # Create mini test set from sample_submission logic
    # We need to construct a valid test metadata file. The provided test_metadata.csv exists in metadata dir.
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_meta = full_test_meta.head(500).copy()

    # Save Mini Metadata
    mini_train_meta_path = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_test_meta_path = os.path.join(demo_dir, "mini_test_metadata.csv")

    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    mini_test_meta.to_csv(mini_test_meta_path, index=False)

    print(f"Mini Train Rows: {len(mini_train_meta)}")
    print(f"Mini Val Rows: {len(mini_val_meta)}")

    # --- 2. Create Mini Tracking Data ---
    # We must load the huge tracking file once to filter it
    print("Filtering tracking data (this may take a moment)...")
    full_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)

    # Filter for the selected plays (Train + Val)
    relevant_plays = set(mini_train_meta["game_play"]).union(
        set(mini_val_meta["game_play"])
    )
    mini_train_tracking = full_tracking[
        full_tracking["game_play"].isin(relevant_plays)
    ].copy()

    # For test tracking, we read the test tracking file
    full_test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)
    test_plays = set(mini_test_meta["game_play"])
    mini_test_tracking = full_test_tracking[
        full_test_tracking["game_play"].isin(test_plays)
    ].copy()

    # Save Mini Tracking
    mini_train_tracking_path = os.path.join(demo_dir, "mini_train_tracking.csv")
    mini_test_tracking_path = os.path.join(demo_dir, "mini_test_tracking.csv")

    mini_train_tracking.to_csv(mini_train_tracking_path, index=False)
    mini_test_tracking.to_csv(mini_test_tracking_path, index=False)

    # --- 3. Monkey Patch Config ---
    print("Patching Config with demo settings...")

    # Paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = mini_train_meta_path
    Config.VAL_METADATA_PATH = mini_val_meta_path
    Config.TEST_METADATA_PATH = mini_test_meta_path
    Config.TRAIN_TRACKING_PATH = mini_train_tracking_path
    Config.TEST_TRACKING_PATH = mini_test_tracking_path
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Hyperparameters (Speed optimization)
    Config.SCOUT_EPOCHS = 10
    Config.EXPERT_EPOCHS = 10
    Config.EARLY_STOPPING_ROUNDS = 5

    # Reduce Model Complexity
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    # Ensure N_JOBS is reasonable
    Config.N_JOBS = 4

    return demo_dir


def run_training_pipeline_demo(demo_dir):
    print("\n>>> Starting Training Pipeline Demo...")

    # Instantiate Pipeline
    pipeline = TrainingPipeline()

    # Run Pipeline
    # This handles Data Loading -> Feature Engineering -> Scout Training -> Mining -> Expert Training -> Optimization
    experts, best_threshold = pipeline.run()

    # --- Verification ---
    print("Verifying Training Artifacts...")

    # 1. Check Feature Cache
    assert os.path.exists(
        os.path.join(demo_dir, "features_train_full.parquet")
    ), "Train features not cached."
    assert os.path.exists(
        os.path.join(demo_dir, "features_val_full.parquet")
    ), "Val features not cached."

    # 2. Check Scout Models
    assert os.path.exists(
        os.path.join(demo_dir, "scout_lgbm.joblib")
    ), "Scout LGBM not saved."
    assert os.path.exists(
        os.path.join(demo_dir, "scout_xgb.joblib")
    ), "Scout XGB not saved."

    # 3. Check Hard Negative Indices
    assert os.path.exists(
        os.path.join(demo_dir, "hard_negative_indices.npy")
    ), "Hard negatives not saved."

    # 4. Check Expert Models
    assert len(experts) == 2, f"Expected 2 expert models, got {len(experts)}"
    assert os.path.exists(
        os.path.join(demo_dir, "expert_lgbm.joblib")
    ), "Expert LGBM not saved."

    # 5. Check Threshold
    assert os.path.exists(
        os.path.join(demo_dir, "best_threshold.npy")
    ), "Best threshold not saved."
    print(f"Training Verified. Best Threshold: {best_threshold}")


def run_inference_pipeline_demo(demo_dir):
    print("\n>>> Starting Inference Pipeline Demo...")

    # Instantiate Pipeline
    inf_pipeline = InferencePipeline()

    # Run Inference
    # Note: We use the threshold optimized during training
    inf_pipeline.predict_test_set(use_optimized_threshold=True)

    # --- Verification ---
    print("Verifying Inference Artifacts...")

    # 1. Check Submission File
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file not created."

    # 2. Validate Content
    df_sub = pd.read_csv(sub_path)
    print(f"Submission Shape: {df_sub.shape}")

    # Check columns
    assert "contact_id" in df_sub.columns, "contact_id column missing."
    assert "contact" in df_sub.columns, "contact column missing."

    # Check values
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Predictions contain non-binary values."

    print("Inference Verified.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Setup Environment & Data
        demo_dir = setup_demo_environment()

        # 2. Run Training
        run_training_pipeline_demo(demo_dir)

        # 3. Run Inference
        run_inference_pipeline_demo(demo_dir)

        print("\n>>> DEMO COMPLETED SUCCESSFULLY.")

    except AssertionError as e:
        print(f"\n!!! ASSERTION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n!!! UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
