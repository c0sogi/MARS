import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library components
from library.config import Config
from library.preprocessor import get_data
from library.dataset import get_dataloaders, SequenceDataset
from library.model import BiGRUModel, generate_submission
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting End-to-End Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Setup Temporary Environment and Override Config
    # ---------------------------------------------------------
    TEMP_DIR = "./temp_demo_env"
    WORKING_DIR = os.path.join(TEMP_DIR, "working")
    SUBMISSION_DIR = os.path.join(TEMP_DIR, "submission")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(WORKING_DIR)
    os.makedirs(SUBMISSION_DIR)

    print(f"Created temporary directory: {TEMP_DIR}")

    # Load original metadata to create subsets
    print("Loading original metadata...")
    train_meta_orig = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta_orig = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta_orig = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select 1 trip for each split to ensure speed
    train_trip = train_meta_orig["tripId"].unique()[0]
    val_trip = val_meta_orig["tripId"].unique()[0]
    test_trip = test_meta_orig["tripId"].unique()[0]

    print(f"Selected Train Trip: {train_trip}")
    print(f"Selected Val Trip:   {val_trip}")
    print(f"Selected Test Trip:  {test_trip}")

    # Save subset metadata
    temp_train_meta_path = os.path.join(TEMP_DIR, "train_meta_subset.csv")
    temp_val_meta_path = os.path.join(TEMP_DIR, "val_meta_subset.csv")
    temp_test_meta_path = os.path.join(TEMP_DIR, "test_meta_subset.csv")

    train_meta_orig[train_meta_orig["tripId"] == train_trip].to_csv(
        temp_train_meta_path, index=False
    )
    val_meta_orig[val_meta_orig["tripId"] == val_trip].to_csv(
        temp_val_meta_path, index=False
    )
    test_meta_orig[test_meta_orig["tripId"] == test_trip].to_csv(
        temp_test_meta_path, index=False
    )

    # Override Config attributes
    print("Overriding configuration for demonstration...")
    Config.TRAIN_METADATA_PATH = temp_train_meta_path
    Config.VAL_METADATA_PATH = temp_val_meta_path
    Config.TEST_METADATA_PATH = temp_test_meta_path

    Config.WORKING_DIR = WORKING_DIR
    Config.SUBMISSION_DIR = SUBMISSION_DIR

    # Redirect cache files to temp dir
    Config.CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    Config.CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    Config.CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    Config.CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    Config.CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    Config.CACHE_TEST_META = os.path.join(WORKING_DIR, "test_meta.parquet")
    Config.CACHE_SCALER = os.path.join(WORKING_DIR, "scaler.json")
    Config.MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Reduce compute load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # ---------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # ---------------------------------------------------------
    print("\n=== Step 2: Data Loading & Preprocessing ===")
    # load_cached_data=False forces processing from the new subset metadata
    train_loader, val_loader, test_loader, test_meta_df = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify batch structure
    sample_X, sample_y = next(iter(train_loader))
    print(f"Sample Batch X shape: {sample_X.shape}")  # (Batch, Window, Features)
    print(f"Sample Batch y shape: {sample_y.shape}")  # (Batch, 2)

    # Assertions to ensure data correctness
    assert (
        sample_X.shape[1] == Config.WINDOW_SIZE
    ), f"Expected window size {Config.WINDOW_SIZE}, got {sample_X.shape[1]}"
    assert (
        sample_X.shape[2] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {sample_X.shape[2]}"
    assert (
        sample_y.shape[1] == Config.OUTPUT_DIM
    ), f"Expected output dim {Config.OUTPUT_DIM}, got {sample_y.shape[1]}"

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n=== Step 3: Model Initialization ===")
    device = torch.device(Config.DEVICE)
    model = BiGRUModel().to(device)
    print(f"Model initialized on {device}")

    # Verify forward pass
    dummy_input = sample_X.to(device)
    dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        sample_X.shape[0],
        Config.OUTPUT_DIM,
    ), "Model output shape mismatch"

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("\n=== Step 4: Training Loop ===")
    trainer = Trainer(model)
    # Fit the model (runs for 1 epoch as configured)
    trained_model = trainer.fit(train_loader, val_loader)

    # Verify model checkpoint exists
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model successfully saved to {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved.")

    # ---------------------------------------------------------
    # 5. Inference and Submission
    # ---------------------------------------------------------
    print("\n=== Step 5: Inference & Submission ===")
    # Generate submission using the trained model and test data
    generate_submission(trained_model, test_loader, test_meta_df)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(sub_df.head())

        # Check columns
        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            col in sub_df.columns for col in expected_cols
        ), "Submission missing required columns"

        # Check length matches test metadata
        assert len(sub_df) == len(
            test_meta_df
        ), f"Submission length {len(sub_df)} != Test Metadata length {len(test_meta_df)}"
        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    # ---------------------------------------------------------
    # 6. Cleanup
    # ---------------------------------------------------------
    print("\n=== Cleanup ===")
    shutil.rmtree(TEMP_DIR)
    print(f"Removed temporary directory: {TEMP_DIR}")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
