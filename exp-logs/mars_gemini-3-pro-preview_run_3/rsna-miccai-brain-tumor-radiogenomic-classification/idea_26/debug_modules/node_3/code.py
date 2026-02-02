import os
import shutil
import pandas as pd
import numpy as np
import torch
import sys

# Import provided library modules
import library.config
import library.data
import library.model
import library.train
import library.predict
import library.utils


def run_demo():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Setup Demo Environment & Monkey Patching
    # ==========================================
    # We use a separate directory for this demo to avoid conflicts with full runs
    # and to demonstrate the pipeline on a small subset of data for speed.
    DEMO_DIR = "./working/demo_execution"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    print(f"Created demo working directory: {DEMO_DIR}")

    # Monkey-patch the configuration to point to our demo directory.
    # This ensures caches, models, and submissions are saved here.
    library.config.WORKING_DIR = DEMO_DIR
    library.data.WORKING_DIR = DEMO_DIR

    # Update model and submission paths in relevant modules
    demo_model_path = os.path.join(DEMO_DIR, "best_model.pth")
    demo_submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")

    library.config.MODEL_PATH = demo_model_path
    library.train.MODEL_PATH = demo_model_path
    library.predict.MODEL_PATH = demo_model_path
    library.predict.SUBMISSION_PATH = demo_submission_path

    # ==========================================
    # 2. Create Subset Metadata (Fast Execution)
    # ==========================================
    print("Creating subset metadata for fast execution...")

    # Load original metadata
    orig_train = pd.read_parquet(library.config.TRAIN_META_PATH)
    orig_val = pd.read_parquet(library.config.VAL_META_PATH)
    orig_test = pd.read_parquet(library.config.TEST_META_PATH)

    # Create tiny subsets (e.g., 4 train, 2 val, 2 test)
    # This ensures the data processing step finishes in seconds rather than minutes.
    demo_train = orig_train.head(4).copy()
    demo_val = orig_val.head(2).copy()
    demo_test = orig_test.head(2).copy()

    # Save to demo metadata directory
    demo_train_path = os.path.join(DEMO_META_DIR, "train.parquet")
    demo_val_path = os.path.join(DEMO_META_DIR, "val.parquet")
    demo_test_path = os.path.join(DEMO_META_DIR, "test.parquet")

    demo_train.to_parquet(demo_train_path)
    demo_val.to_parquet(demo_val_path)
    demo_test.to_parquet(demo_test_path)

    print(
        f"Subset metadata saved. Train: {len(demo_train)}, Val: {len(demo_val)}, Test: {len(demo_test)}"
    )

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\nVerifying Model Architecture...")
    library.utils.seed_everything(42)

    model = library.model.SSBHDNetwork()
    model.eval()

    # Create a dummy input tensor: (Batch=2, Channels=128, Height=224, Width=224)
    # Channels = 32 slices * 4 modalities = 128
    dummy_input = torch.randn(2, 128, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("Model architecture verification passed.")

    # ==========================================
    # 4. Run Training Pipeline
    # ==========================================
    print("\nStarting Training Pipeline...")

    # We run for 2 epochs with a small batch size to ensure convergence logic runs
    # without consuming significant time.
    library.train.run_training(
        train_meta_path=demo_train_path,
        val_meta_path=demo_val_path,
        model_save_path=demo_model_path,
        epochs=2,
        batch_size=2,
        lr=1e-4,
        load_cached_data=False,  # Force processing of our new subset
        patience=2,
    )

    # Verify artifacts
    assert os.path.exists(demo_model_path), "Model checkpoint was not saved."
    assert os.path.exists(
        os.path.join(DEMO_DIR, "cached_train_X.npy")
    ), "Training cache X not created."
    assert os.path.exists(
        os.path.join(DEMO_DIR, "cached_train_y.npy")
    ), "Training cache y not created."
    print("Training pipeline completed successfully.")

    # ==========================================
    # 5. Run Inference Pipeline
    # ==========================================
    print("\nStarting Inference Pipeline...")

    library.predict.generate_submission(
        test_meta_path=demo_test_path,
        model_path=demo_model_path,
        submission_output_path=demo_submission_path,
        batch_size=2,
        load_cached_data=False,
    )

    # Verify submission
    assert os.path.exists(demo_submission_path), "Submission file not found."

    df_sub = pd.read_csv(demo_submission_path)
    print("Generated Submission:")
    print(df_sub)

    # assertions on submission format
    assert list(df_sub.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Incorrect submission columns."
    assert len(df_sub) == 2, f"Expected 2 predictions, got {len(df_sub)}"
    assert (
        df_sub["MGMT_value"].min() >= 0.0 and df_sub["MGMT_value"].max() <= 1.0
    ), "Probabilities out of range."
    assert not df_sub.isnull().values.any(), "Submission contains null values."

    print("Inference pipeline completed successfully.")
    print("\nAll demonstration steps passed.")


if __name__ == "__main__":
    run_demo()
