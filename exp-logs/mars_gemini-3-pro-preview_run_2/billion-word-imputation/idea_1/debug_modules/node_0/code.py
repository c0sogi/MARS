import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings
import sys

# Import provided library modules
from library.config import Config
from library.data_factory import get_dataloaders
from library.model_factory import load_model_and_tokenizer
from library.train_engine import run_training
from library.inference_engine import predict_submission


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demonstration by:
    1. Creating a specific working directory.
    2. Creating small subsets of the training, validation, and test data.
    3. Overriding Config parameters to point to these subsets and reduce runtime.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to keep demo artifacts isolated
    Config.WORKING_DIR = demo_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model")
    Config.TOKENIZER_SAVE_PATH = os.path.join(demo_dir, "tokenizer")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    # -------------------------------------------------------------------------
    # Create Data Subsets
    # -------------------------------------------------------------------------
    # We read a tiny fraction of the actual metadata to create valid parquet files
    # This ensures the schema matches exactly what the library expects.

    # Load original metadata (just first 100 rows)
    print("Creating data subsets...")
    df_train_orig = pd.read_parquet("./metadata/train.parquet").head(100)
    df_val_orig = pd.read_parquet("./metadata/val.parquet").head(50)
    df_test_orig = pd.read_parquet("./metadata/test.parquet").head(20)

    # Save to demo directory
    demo_train_path = os.path.join(demo_dir, "train_subset.parquet")
    demo_val_path = os.path.join(demo_dir, "val_subset.parquet")
    demo_test_path = os.path.join(demo_dir, "test_subset.parquet")

    df_train_orig.to_parquet(demo_train_path, index=False)
    df_val_orig.to_parquet(demo_val_path, index=False)
    df_test_orig.to_parquet(demo_test_path, index=False)

    # -------------------------------------------------------------------------
    # Override Config Parameters
    # -------------------------------------------------------------------------
    print("Overriding configuration for speed...")

    # Point to subsets
    Config.TRAIN_DATA_PATH = demo_train_path
    Config.VAL_DATA_PATH = demo_val_path
    Config.TEST_DATA_PATH = demo_test_path

    # Reduce Compute Requirements
    Config.TRAIN_BATCH_SIZE = 8
    Config.VAL_BATCH_SIZE = 8
    Config.INFERENCE_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce overhead for small data

    # Reduce Training Duration
    Config.EPOCHS = 1
    Config.MAX_STEPS = 10  # Only run 10 steps
    Config.LOGGING_STEPS = 2
    Config.EVAL_STEPS = 5
    Config.SAVE_STEPS = 5
    Config.WARMUP_RATIO = 0.0

    # Ensure Model Name is valid (using the one in Config or a smaller one if needed)
    # distilroberta-base is small enough for a quick demo on A100.

    print("Environment setup complete.")


def test_data_loading():
    """
    Demonstrates and validates the data loading pipeline.
    """
    print("\n=== Testing Data Loading ===")

    # Load tokenizer
    _, tokenizer = load_model_and_tokenizer()

    # Force re-computation of cache by setting load_cached_data=False initially
    # or just relying on the fact that our demo_dir is fresh.
    train_loader, val_loader = get_dataloaders(tokenizer, load_cached_data=False)

    # Verification
    print("Verifying DataLoader outputs...")

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "labels" in batch, "Batch missing labels (Collator should add this)"

    # Check shapes
    expected_shape = (Config.TRAIN_BATCH_SIZE, Config.MAX_SEQ_LEN)
    assert (
        batch["input_ids"].shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {batch['input_ids'].shape}"

    # Check masking logic (labels should not be all -100)
    # In MLM, -100 is ignored, other values are targets.
    labels = batch["labels"]
    assert (labels != -100).sum() > 0, "No tokens were masked for prediction!"

    print("Data loading verification successful.")


def test_training_loop():
    """
    Executes the training engine with the reduced configuration.
    """
    print("\n=== Testing Training Loop ===")

    # Run training
    # This will use the Config overrides (MAX_STEPS=10)
    run_training(load_cached_data=True)

    # Verify artifacts
    print("Verifying training artifacts...")

    # Check if model was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not saved to {Config.MODEL_SAVE_PATH}")

    # Check for config.json and pytorch_model.bin (or safetensors)
    files = os.listdir(Config.MODEL_SAVE_PATH)
    has_model_file = any(
        f.endswith(".bin") or f.endswith(".safetensors") for f in files
    )
    assert has_model_file, "No model weights file found in save directory."

    print("Training loop verification successful.")


def test_inference_engine():
    """
    Executes the inference engine to generate a submission.
    """
    print("\n=== Testing Inference Engine ===")

    # Run prediction
    # subset_size=-1 means use all (which is just 20 rows in our demo subset)
    predict_submission(subset_size=-1)

    # Verify submission file
    print("Verifying submission file...")

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    # Check content format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "sentence" in df_sub.columns, "Submission missing 'sentence' column"

    # Check row count (should match our demo test subset size)
    expected_count = 20
    assert (
        len(df_sub) == expected_count
    ), f"Submission row count mismatch. Expected {expected_count}, got {len(df_sub)}"

    # Check that sentences are strings and not empty
    assert (
        df_sub["sentence"].apply(lambda x: isinstance(x, str) and len(x) > 0).all()
    ), "Some submitted sentences are invalid."

    print("Inference verification successful.")
    print(f"Submission saved to: {Config.SUBMISSION_FILE}")


def main():
    # 1. Set Seed for Reproducibility
    # Note: Config.SEED is 42 by default, but we ensure it's applied globally.
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        # 2. Setup Environment
        setup_demo_environment()

        # 3. Test Data Loading
        test_data_loading()

        # 4. Test Training
        test_training_loop()

        # 5. Test Inference
        test_inference_engine()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAILURE] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILURE] An error occurred: {e}")
        # Print stack trace for debugging if needed, but keeping it simple per requirements
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
