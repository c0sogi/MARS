import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import SmartphoneLocationDataset, collate_fn
from library.model import HR1DResNet
from library.trainer import Trainer
from library.inference import run_inference
from library.loss import MultiResolutionMAELoss


def setup_demo_environment():
    """
    Sets up a demo environment by creating mini metadata files
    referencing a small subset of the data to ensure speed.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_work_dir = "./working/demo_run"
    os.makedirs(demo_work_dir, exist_ok=True)

    mini_meta_dir = os.path.join(demo_work_dir, "metadata")
    os.makedirs(mini_meta_dir, exist_ok=True)

    mini_cache_dir = os.path.join(demo_work_dir, "cache")
    os.makedirs(mini_cache_dir, exist_ok=True)

    # 1. Create Mini Train Metadata
    # Read original metadata and take only the first drive
    orig_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    first_train_drive = orig_train_meta["drive_id"].unique()[0]
    mini_train_df = orig_train_meta[
        orig_train_meta["drive_id"] == first_train_drive
    ].copy()
    mini_train_path = os.path.join(mini_meta_dir, "train_metadata.csv")
    mini_train_df.to_csv(mini_train_path, index=False)
    print(
        f"Created mini train metadata with {len(mini_train_df)} rows (Drive: {first_train_drive})"
    )

    # 2. Create Mini Val Metadata
    orig_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    first_val_drive = orig_val_meta["drive_id"].unique()[0]
    mini_val_df = orig_val_meta[orig_val_meta["drive_id"] == first_val_drive].copy()
    mini_val_path = os.path.join(mini_meta_dir, "val_metadata.csv")
    mini_val_df.to_csv(mini_val_path, index=False)
    print(
        f"Created mini val metadata with {len(mini_val_df)} rows (Drive: {first_val_drive})"
    )

    # 3. Create Mini Test Metadata
    orig_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    first_test_drive = orig_test_meta["drive_id"].unique()[0]
    mini_test_df = orig_test_meta[orig_test_meta["drive_id"] == first_test_drive].copy()
    mini_test_path = os.path.join(mini_meta_dir, "test_metadata.csv")
    mini_test_df.to_csv(mini_test_path, index=False)
    print(
        f"Created mini test metadata with {len(mini_test_df)} rows (Drive: {first_test_drive})"
    )

    # 4. Patch Config to use demo paths and settings
    Config.WORKING_DIR = demo_work_dir
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    Config.CACHE_TRAIN = os.path.join(mini_cache_dir, "train_processed.parquet")
    Config.CACHE_VAL = os.path.join(mini_cache_dir, "val_processed.parquet")
    Config.CACHE_TEST = os.path.join(mini_cache_dir, "test_processed.parquet")

    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimize for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1

    return demo_work_dir


def test_dataset_loading():
    print("\n--- Testing Dataset Loading ---")

    # Load Train Dataset
    # load_cached=False ensures we process the mini metadata we just created
    train_ds = SmartphoneLocationDataset(split="train", load_cached=False)
    print(f"Train Dataset loaded. Sequences: {len(train_ds)}")

    if len(train_ds) > 0:
        item = train_ds[0]
        print("Sample Item Keys:", item.keys())
        print("Features Shape:", item["features"].shape)
        print("Targets Shape:", item["targets"].shape)

        # Verify shapes
        assert (
            item["features"].shape[0] == Config.IN_CHANNELS
        ), f"Expected {Config.IN_CHANNELS} channels"
        assert item["targets"].shape[0] == 2, "Expected 2 target channels (North, East)"
        assert (
            item["features"].shape[1] == item["targets"].shape[1]
        ), "Time dimension mismatch"

    # Load Val Dataset
    val_ds = SmartphoneLocationDataset(split="val", load_cached=False)
    print(f"Val Dataset loaded. Sequences: {len(val_ds)}")

    return train_ds, val_ds


def test_model_forward():
    print("\n--- Testing Model Architecture ---")

    model = HR1DResNet()

    # Create dummy input: [Batch=2, Channels=28, Time=100]
    dummy_input = torch.randn(2, Config.IN_CHANNELS, 100)

    # Forward pass
    outputs = model(dummy_input)

    print(f"Model Output Type: {type(outputs)}")
    print(f"Number of Output Heads: {len(outputs)}")

    # Check High-Res Output
    high_res_out = outputs[0]
    print(f"High-Res Output Shape: {high_res_out.shape}")

    # Verify output shape matches input length (due to cropping in model)
    assert high_res_out.shape == (
        2,
        2,
        100,
    ), f"Expected (2, 2, 100), got {high_res_out.shape}"

    return model


def test_loss_function():
    print("\n--- Testing Loss Function ---")

    criterion = MultiResolutionMAELoss()

    # Dummy data
    T = 100
    # Model outputs a list of tensors for deep supervision
    # Stream 0: Full res, Stream 1: 1/4 res, Stream 2: 1/16 res
    # Note: Model internal padding might change sizes slightly, but here we simulate ideal case
    preds = [
        torch.randn(2, 2, T),  # High Res
        torch.randn(2, 2, T // 4),  # Med Res
        torch.randn(2, 2, T // 16),  # Low Res
    ]

    targets = torch.randn(2, 2, T)
    mask = torch.ones(2, T, dtype=torch.bool)

    loss = criterion(preds, targets, mask)
    print(f"Computed Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss > 0, "Loss should be positive"


def test_training_loop(train_ds, val_ds, model):
    print("\n--- Testing Training Loop ---")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    trainer = Trainer(model)

    # Run fit (Config.EPOCHS is set to 1)
    trained_model = trainer.fit(train_loader, val_loader, epochs=1)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint saved successfully at {checkpoint_path}")
    else:
        raise FileNotFoundError("Model checkpoint not found after training.")

    return checkpoint_path


def test_inference(checkpoint_path):
    print("\n--- Testing Inference ---")

    # run_inference handles dataset loading (test split), model loading, and submission generation
    # load_cached=False to ensure we process the mini test metadata
    run_inference(
        checkpoint_path=checkpoint_path,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        load_cached=False,
    )

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated with {len(df_sub)} rows.")
        print(df_sub.head())

        # Verify columns
        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            c in df_sub.columns for c in expected_cols
        ), "Missing columns in submission"

        # Verify no NaNs in coordinates (unless input was empty, but we have data)
        if len(df_sub) > 0:
            assert not df_sub["LatitudeDegrees"].isnull().any(), "NaNs in Latitude"
            assert not df_sub["LongitudeDegrees"].isnull().any(), "NaNs in Longitude"
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Dataset
        train_ds, val_ds = test_dataset_loading()

        # 3. Model
        model = test_model_forward()

        # 4. Loss
        test_loss_function()

        # 5. Training
        checkpoint_path = test_training_loop(train_ds, val_ds, model)

        # 6. Inference
        test_inference(checkpoint_path)

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
