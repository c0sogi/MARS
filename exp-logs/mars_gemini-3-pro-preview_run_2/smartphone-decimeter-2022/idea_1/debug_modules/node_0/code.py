import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import ecef_to_lla, haversine_distance
from library.data_loader import get_data, GNSSDataset
from library.model import WindowedMLP, train_model, generate_submission
from library.trainer import run_experiment


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Demonstration Script...")
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring for fast demonstration...")
    # Override Config values to run quickly on a small subset
    Config.DEBUG_TRIP_COUNT = 3  # Process only 3 trips
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 32  # Small batch size
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller model
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directories are clean/ready (Config creates them on import,
    # but we double check or clean if needed. Here we just rely on overwrite).
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Submission Directory: {Config.SUBMISSION_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loading and Processing...")

    # Load Training Data (force processing from raw files)
    print("  Loading Training Data...")
    X_train, y_train, meta_train = get_data(split="train", load_cached_data=False)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Validation: Check shapes match
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Mismatch in training samples and targets"
    assert (
        X_train.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {X_train.shape[1]}"
    assert (
        y_train.shape[1] == Config.OUTPUT_DIM
    ), f"Expected output dim {Config.OUTPUT_DIM}, got {y_train.shape[1]}"

    # Load Validation Data
    print("  Loading Validation Data...")
    X_val, y_val, meta_val = get_data(split="val", load_cached_data=False)
    print(f"  X_val shape: {X_val.shape}")

    # Create DataLoaders
    train_dataset = GNSSDataset(X_train, y_train, mode="train")
    val_dataset = GNSSDataset(X_val, y_val, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify DataLoader yields correct shapes
    batch_X, batch_y = next(iter(train_loader))
    print(f"  Batch X shape: {batch_X.shape}")
    print(f"  Batch y shape: {batch_y.shape}")
    assert batch_X.shape == (Config.BATCH_SIZE, Config.INPUT_DIM)

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Instantiation...")

    model = WindowedMLP(
        input_dim=Config.INPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(Config.DEVICE)
    print("  Model initialized successfully.")
    print(model)

    # Test Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(10, Config.INPUT_DIM).to(Config.DEVICE)
        output = model(dummy_input)

    assert output.shape == (10, Config.OUTPUT_DIM), "Model output shape incorrect"
    print("  Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training Loop...")

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.EARLY_STOPPING_PATIENCE,
        device=Config.DEVICE,
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
    )

    # Verify checkpoint creation
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise AssertionError(
            f"Model checkpoint not created at {Config.MODEL_CHECKPOINT_PATH}"
        )
    print("  Training complete. Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Testing Inference and Submission Generation...")

    # Load Test Data
    # For test, get_data returns (X, meta_list, df_original)
    X_test, meta_test, df_test_original = get_data(split="test", load_cached_data=False)
    print(f"  X_test shape: {X_test.shape}")

    test_dataset = GNSSDataset(X_test, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Generate Submission
    df_submission = generate_submission(
        model=trained_model,
        test_loader=test_loader,
        meta_list=meta_test,
        df_test_original=df_test_original,
        submission_path=Config.SUBMISSION_FILE_PATH,
        device=Config.DEVICE,
    )

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_FILE_PATH):
        raise AssertionError(
            f"Submission file not created at {Config.SUBMISSION_FILE_PATH}"
        )

    # Verify Submission Content
    expected_cols = [
        Config.COL_TRIP_ID,
        Config.COL_UNIX_TIME,
        Config.COL_LATITUDE,
        Config.COL_LONGITUDE,
    ]
    if not all(col in df_submission.columns for col in expected_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {df_submission.columns}"
        )

    # Since we used DEBUG_TRIP_COUNT on test data loading as well, the submission might be smaller
    # than the full sample_submission.csv. However, the generate_submission function loads the
    # template to ensure alignment. Rows not predicted (due to debug filtering) will be NaNs or
    # missing if inner join logic was strict.
    # In the provided generate_submission implementation:
    # df_final = pd.merge(df_template, df_merged, on=required_cols, how="left")
    # This ensures full length, but unpredicted rows will have NaNs.

    # Let's check if we have predictions for the processed trips
    processed_trip_ids = set([m[0] for m in meta_test])

    # Check a row corresponding to a processed trip
    sample_processed_trip = list(processed_trip_ids)[0]
    sample_row = df_submission[
        df_submission[Config.COL_TRIP_ID] == sample_processed_trip
    ].iloc[0]

    print(f"  Sample prediction for {sample_processed_trip}:")
    print(sample_row)

    if pd.isna(sample_row[Config.COL_LATITUDE]) or pd.isna(
        sample_row[Config.COL_LONGITUDE]
    ):
        raise AssertionError("Prediction resulted in NaN for a processed trip.")

    print("\n[6] Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
