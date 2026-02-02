import pandas as pd
import numpy as np
import os
import shutil
from library.config import Config
from library.utils import set_seed
from library.tessc_model import TESSCRecommender


def setup_demo_data():
    """
    Creates a small subset of the metadata files for demonstration purposes.
    We use the 'tail' of the datasets to ensure we capture the most recent
    transactions, which is critical for the time-decay and validation split logic.
    """
    print("--- Setting up Demo Data ---")
    demo_input_dir = "./working/demo_input"
    os.makedirs(demo_input_dir, exist_ok=True)

    # Define paths for demo files
    demo_train_path = os.path.join(demo_input_dir, "train.csv")
    demo_val_path = os.path.join(demo_input_dir, "val.csv")
    demo_test_path = os.path.join(demo_input_dir, "test.csv")

    # 1. Sample Train Data (Last 50k rows to get recent history)
    print(f"Sampling {Config.TRAIN_PATH}...")
    # We read the full file then take tail because we can't easily seek to tail in CSV
    # without reading. For 25M rows, this might take a moment, but it's safe.
    # Optimization: Read a chunk from the end if possible, but standard pandas read is safest.
    # To save time in this specific demo environment, we assume the file is large and
    # we just want a functional subset. We'll read a chunk.

    # Reading the last 50,000 rows efficiently
    # We'll use a larger sample to ensure good overlap for similarity
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_train_sample = df_train.tail(50000).copy()
    df_train_sample.to_csv(demo_train_path, index=False)

    # 2. Sample Validation Data (Last 10k rows)
    print(f"Sampling {Config.VAL_PATH}...")
    df_val = pd.read_csv(Config.VAL_PATH)
    df_val_sample = df_val.tail(10000).copy()
    df_val_sample.to_csv(demo_val_path, index=False)

    # 3. Sample Test Data (First 1000 customers from submission)
    print(f"Sampling {Config.TEST_PATH}...")
    df_test = pd.read_csv(Config.TEST_PATH)
    df_test_sample = df_test.head(1000).copy()
    df_test_sample.to_csv(demo_test_path, index=False)

    print("Demo data created successfully.")
    return demo_train_path, demo_val_path, demo_test_path


def run_demo():
    # 1. Environment Setup
    set_seed(42)

    # 2. Prepare Data
    demo_train, demo_val, demo_test = setup_demo_data()

    # 3. Override Configuration
    # We patch the Config class directly. Since it's a singleton class object,
    # these changes propagate to the library modules.
    Config.TRAIN_PATH = demo_train
    Config.VAL_PATH = demo_val
    Config.TEST_PATH = demo_test

    # Use a separate working directory for this demo run
    Config.WORKING_DIR = "./working/demo_run/cache"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create these new directories
    Config.setup()

    print(f"\nConfiguration updated. Working Dir: {Config.WORKING_DIR}")

    # 4. Instantiate Model
    recommender = TESSCRecommender()

    # ==========================================
    # Phase A: Validation Mode
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE A: Validation Mode")
    print("=" * 40)

    # Fit the model using the validation split logic.
    # load_cached_data=False ensures we process the new demo data from scratch.
    recommender.fit(use_validation=True, load_cached_data=False)

    # Verify internal state
    assert recommender.X is not None, "Interaction matrix X should be initialized"
    assert recommender.S is not None, "Similarity matrix S should be initialized"
    assert (
        recommender.val_df is not None
    ), "Validation DataFrame should be populated in validation mode"

    # Evaluate
    # This calculates MAP@12 on the hold-out set (last 7 days of the provided data)
    map_score = recommender.evaluate()

    # Check score validity
    print(f"Computed MAP@12: {map_score}")
    assert isinstance(map_score, float), "MAP score must be a float"
    assert 0.0 <= map_score <= 1.0, "MAP score must be between 0 and 1"

    # ==========================================
    # Phase B: Submission Mode
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE B: Submission Mode")
    print("=" * 40)

    # Refit the model on ALL available data (Train + Val combined)
    # This maximizes information for the final prediction
    recommender.fit(use_validation=False, load_cached_data=False)

    # Verify internal state changes
    assert (
        recommender.val_df is None
    ), "Validation DataFrame should be None in submission mode"

    # Generate Submission
    recommender.generate_submission()

    # ==========================================
    # Phase C: Verification
    # ==========================================
    print("\n" + "=" * 40)
    print("PHASE C: Output Verification")
    print("=" * 40)

    # Check file existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Load submission to check format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions (should match the sampled test set size)
    expected_rows = 1000
    if len(sub_df) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in submission, found {len(sub_df)}"
        )

    # Check columns
    required_cols = ["customer_id", "prediction"]
    if not all(col in sub_df.columns for col in required_cols):
        raise AssertionError(f"Submission missing required columns: {required_cols}")

    # Check prediction format (space-separated strings)
    sample_pred = sub_df.iloc[0]["prediction"]
    if not isinstance(sample_pred, str):
        raise AssertionError(
            f"Prediction column must contain strings. Found: {type(sample_pred)}"
        )

    # Check number of items predicted
    pred_items = sample_pred.split()
    if len(pred_items) > 12:
        raise AssertionError(f"Predicted more than 12 items: {len(pred_items)}")

    print("Verification Successful!")
    print(f"Sample Prediction for {sub_df.iloc[0]['customer_id']}: {sample_pred}")


if __name__ == "__main__":
    run_demo()
