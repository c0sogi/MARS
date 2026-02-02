import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library
from library.config import Config
from library.trainer import Runner


def main():
    print("Initializing demonstration...")

    # 1. Configuration Setup for Speed and Isolation
    # We create a separate working directory for this demo to avoid conflicts with existing caches
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Update Config paths to point to the new working directory.
    # Since these are class attributes defined at import time, we must update them manually.
    Config.WORKING_DIR = demo_working_dir
    Config.PROCESSED_TRAIN_PATH = os.path.join(demo_working_dir, "train_data.npz")
    Config.PROCESSED_VAL_PATH = os.path.join(demo_working_dir, "val_data.npz")
    Config.PROCESSED_TEST_PATH = os.path.join(demo_working_dir, "test_data.npz")
    Config.SCALERS_PATH = os.path.join(demo_working_dir, "scalers.npz")
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "best_model.pt")

    # Update submission path
    demo_submission_dir = "./working/demo_submission"
    os.makedirs(demo_submission_dir, exist_ok=True)
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Reduce hyperparameters for a quick demonstration
    Config.NUM_EPOCHS = 2  # Run only 2 epochs to verify the loop works
    Config.BATCH_SIZE = 32  # Standard batch size
    Config.ATOM_HIDDEN_DIM = 64  # Reduce model size for speed
    Config.GLOBAL_HIDDEN_DIM = 32
    Config.FUSION_HIDDEN_DIM = 32

    print("Configuration updated for demonstration:")
    Config.print_config()

    # 2. Instantiate Runner
    # This initializes the model, optimizer, loss function, and data processor.
    # The model architecture will use the updated Config dimensions.
    print("\nInstantiating Runner...")
    runner = Runner()

    # 3. Execute Training
    # We set load_cached_data=False to force the DataProcessor to read the metadata CSVs,
    # parse the XYZ geometry files, compute features, and create the cache files.
    print("\nStarting Training Loop...")
    try:
        runner.train(load_cached_data=False)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify model checkpoint was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not generated at {Config.MODEL_SAVE_PATH}"
        )
    print(f"Model checkpoint verified at {Config.MODEL_SAVE_PATH}")

    # 4. Execute Prediction
    # We use load_cached_data=True here because the training step has already processed
    # and cached the test data. This saves time.
    print("\nStarting Prediction Loop...")
    try:
        runner.predict(load_cached_data=True)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        raise e

    # 5. Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    print(f"Submission file verified at {Config.SUBMISSION_PATH}")

    # Validate submission content format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df.shape}")

    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
        )

    # Check if we have the expected number of test samples (240 based on metadata info)
    expected_count = 240
    if len(df) != expected_count:
        raise ValueError(
            f"Submission row count mismatch. Expected {expected_count}, got {len(df)}"
        )

    # Check for NaNs
    if df.isnull().any().any():
        raise ValueError("Submission contains NaN values.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
