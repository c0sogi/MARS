import os
import pandas as pd
import torch
import numpy as np
from unittest.mock import patch

# Import library components
import library.config as config
import library.data_processing as dp
import library.dataset as ds
import library.trainer as tr
import library.model as md
from library.utils import seed_everything, get_device


def run_demo():
    # --- 1. Configuration Overrides for Speed/Demo ---
    # We create a separate working directory for this demo run
    DEMO_DIR = "./working/demo_run_v1"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update WORKING_DIR in all relevant modules to ensure files are saved/loaded from demo dir
    config.WORKING_DIR = DEMO_DIR
    dp.WORKING_DIR = DEMO_DIR
    ds.WORKING_DIR = DEMO_DIR
    tr.WORKING_DIR = DEMO_DIR

    # Update Training Parameters for a quick run
    tr.EPOCHS = 1  # Train for only 1 epoch
    ds.BATCH_SIZE = 512  # Use a reasonable batch size

    # Update Submission Path to be inside the demo directory
    SUB_PATH = os.path.join(DEMO_DIR, "submission.csv")
    config.SUBMISSION_PATH = SUB_PATH
    tr.SUBMISSION_PATH = SUB_PATH

    print(f"Configuration: Epochs={tr.EPOCHS}, Working Dir={DEMO_DIR}")

    # --- 2. Monkey-Patching for Data Subset ---
    # We wrap pd.read_csv to limit the rows read from metadata files.
    # This forces the pipeline to process only a small subset of data, ensuring speed.

    original_read_csv = pd.read_csv

    def limited_read_csv(*args, **kwargs):
        # args[0] is usually the filepath
        path = args[0] if len(args) > 0 else kwargs.get("filepath_or_buffer")

        # Check if we are reading one of the metadata CSVs
        if isinstance(path, str) and "metadata" in path and path.endswith(".csv"):
            # Limit to 2000 rows. This provides enough data to form batches
            # but is small enough to process in seconds.
            kwargs["nrows"] = 2000

        return original_read_csv(*args, **kwargs)

    # Apply the patch to pandas.read_csv
    with patch("pandas.read_csv", side_effect=limited_read_csv):

        # --- 3. Data Preparation ---
        print("\n=== Step 1: Loading and Processing Data (Subset) ===")
        # load_cached_data=False forces the pipeline to run 'process_data' using our patched read_csv
        train_loader, val_loader, test_loader, input_dims = ds.get_dataloaders(
            load_cached_data=False
        )

        # Verify DataLoaders are not empty
        batch = next(iter(train_loader))
        data, target = batch
        print(f"Data Loaded. Batch Kinematic Shape: {data['kinematic'].shape}")

        if data["kinematic"].shape[0] == 0:
            raise ValueError("DataLoader is empty!")

        # --- 4. Model Initialization ---
        print("\n=== Step 2: Initializing EARVN Model ===")
        model = md.EARVN(input_dims=input_dims)
        device = get_device()
        model.to(device)
        print("Model initialized and moved to device.")

        # --- 5. Training and Inference ---
        print("\n=== Step 3: Training and Generating Submission ===")
        # train_model handles the training loop, validation, threshold optimization, and submission generation
        tr.train_model(model, train_loader, val_loader, test_loader)

    # --- 6. Final Verification ---
    print("\n=== Step 4: Verifying Outputs ===")

    # Check if submission file exists
    if not os.path.exists(SUB_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUB_PATH}")

    df_sub = pd.read_csv(SUB_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Verify columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise ValueError(f"Submission missing columns. Found: {df_sub.columns}")

    # Verify binary predictions
    if not df_sub["contact"].isin([0, 1]).all():
        raise ValueError("Submission contains non-binary values in 'contact' column.")

    # Verify model weights were saved
    model_path = os.path.join(DEMO_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Best model weights not saved.")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    run_demo()
