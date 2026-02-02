import os
import pandas as pd
import torch
import numpy as np
import warnings

# Import library modules
from library.config import Config
from library.data_preprocessing import get_data
from library.dataset import GNSSSequenceDataset, collate_padded_sequences
from library.model import TransUNet1D
from library.trainer import Trainer
from library.inference import generate_predictions
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting GNSS Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("1. Configuring Environment...")

    # Set up specific working directories for this demo run
    demo_work_dir = "./working/demo_run"
    demo_sub_dir = "./working/demo_submission"

    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_sub_dir, exist_ok=True)

    # Patch the Config class to use these directories and speed up training
    Config.WORKING_DIR = demo_work_dir
    Config.SUBMISSION_DIR = demo_sub_dir
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.ENCODER_CHANNELS = [16, 32]  # Smaller model for speed
    Config.DECODER_CHANNELS = [16]
    Config.TRANSFORMER_D_MODEL = 32
    Config.TRANSFORMER_NHEAD = 2
    Config.TRANSFORMER_NUM_LAYERS = 1
    Config.TRANSFORMER_DIM_FEEDFORWARD = 64

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Submission Directory: {Config.SUBMISSION_DIR}")
    print("   Configuration updated for fast demonstration.\n")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing (Train/Val)
    # -------------------------------------------------------------------------
    print("2. Loading and Preprocessing Data...")

    # Load full metadata
    full_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Subset metadata to just one drive each to save time
    # We pick the first available drive in each set
    train_drive_id = full_train_meta["drive_id"].unique()[0]
    val_drive_id = full_val_meta["drive_id"].unique()[0]

    mini_train_meta = full_train_meta[
        full_train_meta["drive_id"] == train_drive_id
    ].copy()
    mini_val_meta = full_val_meta[full_val_meta["drive_id"] == val_drive_id].copy()

    # Save these mini metadata files for reference (optional)
    mini_train_meta.to_csv(
        os.path.join(demo_work_dir, "mini_train_meta.csv"), index=False
    )
    mini_val_meta.to_csv(os.path.join(demo_work_dir, "mini_val_meta.csv"), index=False)

    print(f"   Selected Train Drive: {train_drive_id} ({len(mini_train_meta)} samples)")
    print(f"   Selected Val Drive:   {val_drive_id} ({len(mini_val_meta)} samples)")

    # Process data (Load GNSS features and Target residuals)
    # We set load_cached_data=False to ensure we demonstrate the processing logic
    print("   Processing Train Data...")
    df_train = get_data(mini_train_meta, load_cached_data=False)

    print("   Processing Val Data...")
    df_val = get_data(mini_val_meta, load_cached_data=False)

    # Verification
    assert not df_train.empty, "Training dataframe is empty!"
    assert not df_val.empty, "Validation dataframe is empty!"
    assert all(
        col in df_train.columns for col in Config.INPUT_FEATURES
    ), "Missing input features in train data"
    assert all(
        col in df_train.columns for col in Config.TARGET_COLS
    ), "Missing target columns in train data"
    print("   Data loaded and verified successfully.\n")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("3. Creating Datasets and DataLoaders...")

    # Create Datasets
    train_dataset = GNSSSequenceDataset(
        df_train,
        feature_cols=Config.INPUT_FEATURES,
        target_cols=Config.TARGET_COLS,
        mode="train",
        scaler_dir=Config.WORKING_DIR,
    )

    val_dataset = GNSSSequenceDataset(
        df_val,
        feature_cols=Config.INPUT_FEATURES,
        target_cols=Config.TARGET_COLS,
        mode="val",
        scaler_dir=Config.WORKING_DIR,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_padded_sequences,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )

    # Verify batch structure
    features, targets, mask, meta = next(iter(train_loader))
    print(f"   Batch Features Shape: {features.shape} (Batch, Channels, Length)")
    print(f"   Batch Targets Shape:  {targets.shape} (Batch, Length, OutputDim)")
    print(f"   Batch Mask Shape:     {mask.shape} (Batch, Length)")

    assert features.shape[1] == len(
        Config.INPUT_FEATURES
    ), "Incorrect feature dimension"
    assert targets.shape[2] == len(Config.TARGET_COLS), "Incorrect target dimension"
    print("   DataLoaders operational.\n")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("4. Initializing Model...")

    model = TransUNet1D()

    # Move to device
    device = Config.DEVICE
    model = model.to(device)

    # Test forward pass
    with torch.no_grad():
        dummy_input = features.to(device)
        dummy_mask = mask.to(device)
        output = model(dummy_input, dummy_mask)

    print(f"   Model Output Shape: {output.shape}")

    # Output should be (Batch, OutputDim, Length)
    assert output.shape == (
        features.shape[0],
        len(Config.TARGET_COLS),
        features.shape[2],
    )
    print("   Model initialized and forward pass successful.\n")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("5. Running Training Loop...")

    trainer = Trainer(model, train_loader, val_loader, device=device)
    trained_model = trainer.fit()

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint not found!"
    print("   Training finished successfully.\n")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline
    # -------------------------------------------------------------------------
    print("6. Running Inference Pipeline...")

    # Prepare a mini test metadata file
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    # Pick one trip
    test_trip_id = full_test_meta["tripId"].unique()[0]
    mini_test_meta = full_test_meta[full_test_meta["tripId"] == test_trip_id].copy()

    # Save mini test metadata
    mini_test_path = os.path.join(demo_work_dir, "mini_test_meta.csv")
    mini_test_meta.to_csv(mini_test_path, index=False)

    # Patch Config to point to this mini test file
    Config.TEST_METADATA_PATH = mini_test_path

    # Run inference
    # This will load the best model from Config.WORKING_DIR/best_model.pth
    # and save submission to Config.SUBMISSION_DIR/submission.csv
    generate_predictions(
        model_path=best_model_path, load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated!"

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission generated with {len(df_sub)} rows.")
    print("   First few rows:")
    print(df_sub.head())

    assert len(df_sub) == len(mini_test_meta), "Submission row count mismatch!"
    print("   Inference pipeline verified.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
