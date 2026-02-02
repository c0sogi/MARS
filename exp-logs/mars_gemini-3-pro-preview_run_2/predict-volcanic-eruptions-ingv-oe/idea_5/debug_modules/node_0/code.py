import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import VolcanoDataset
from library.model import HybridResNet34
from library.engine import fit, generate_submission


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print("--- Starting Volcano Prediction Demo ---")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_WORK_DIR = os.path.join(DEMO_DIR, "working")
    DEMO_SUB_DIR = os.path.join(DEMO_DIR, "submission")

    os.makedirs(DEMO_META_DIR, exist_ok=True)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)
    os.makedirs(DEMO_SUB_DIR, exist_ok=True)

    # Override Config paths to use the demo directories
    Config.METADATA_DIR = DEMO_META_DIR
    Config.WORKING_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_DIR = DEMO_SUB_DIR

    Config.TRAIN_METADATA_PATH = os.path.join(DEMO_META_DIR, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(DEMO_META_DIR, "val.csv")
    Config.TEST_METADATA_PATH = os.path.join(DEMO_META_DIR, "test.csv")

    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_WORK_DIR, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_WORK_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_WORK_DIR, "test_features.parquet")

    Config.STATS_SCALER_MEAN_PATH = os.path.join(DEMO_WORK_DIR, "stats_scaler_mean.npy")
    Config.STATS_SCALER_SCALE_PATH = os.path.join(
        DEMO_WORK_DIR, "stats_scaler_scale.npy"
    )
    Config.TARGET_MEAN_PATH = os.path.join(DEMO_WORK_DIR, "target_mean.npy")
    Config.TARGET_STD_PATH = os.path.join(DEMO_WORK_DIR, "target_std.npy")

    Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORK_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUB_DIR, "submission.csv")

    # Override Hyperparameters for Speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = False  # We will manually subset metadata instead

    # -------------------------------------------------------------------------
    # 2. Create Mini-Datasets (Subsetting Metadata)
    # -------------------------------------------------------------------------
    print("Creating mini-datasets for rapid execution...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample small subsets (e.g., 20 samples for train, 10 for val/test)
    mini_train = orig_train.head(20)
    mini_val = orig_val.head(10)
    mini_test = orig_test.head(10)

    # Save to demo metadata location
    mini_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    mini_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    mini_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(f"Mini-Train size: {len(mini_train)}")
    print(f"Mini-Val size: {len(mini_val)}")
    print(f"Mini-Test size: {len(mini_test)}")

    # -------------------------------------------------------------------------
    # 3. Instantiate Datasets & Loaders
    # -------------------------------------------------------------------------
    print("\nInitializing Datasets (Feature Extraction & Scaling)...")

    # Instantiate datasets
    # This triggers process_and_cache_features internally using the paths in Config
    train_dataset = VolcanoDataset(mode="train", load_cached_data=False)
    val_dataset = VolcanoDataset(mode="val", load_cached_data=False)
    test_dataset = VolcanoDataset(mode="test", load_cached_data=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Verification: Check batch shapes
    print("Verifying DataLoader shapes...")
    dummy_spec, dummy_feat, dummy_target = next(iter(train_loader))

    # Expected Shapes:
    # Spectrogram: (Batch, 20, 128, Time) -> Time depends on signal length/hop ~ 235
    # Features: (Batch, Num_Stats_Features) -> 14 stats * 10 sensors = 140
    # Target: (Batch)

    assert dummy_spec.ndim == 4, f"Spectrogram dim mismatch. Got {dummy_spec.ndim}"
    assert (
        dummy_spec.shape[1] == 20
    ), f"Spectrogram channels mismatch. Got {dummy_spec.shape[1]}"
    assert dummy_feat.ndim == 2, f"Features dim mismatch. Got {dummy_feat.ndim}"
    assert (
        dummy_feat.shape[1] == 140
    ), f"Feature count mismatch. Got {dummy_feat.shape[1]}"
    assert dummy_target.ndim == 1, f"Target dim mismatch. Got {dummy_target.ndim}"

    print("Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Initialize Model
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = HybridResNet34(num_stats_features=140)
    model = model.to(device)

    # Define Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # -------------------------------------------------------------------------
    # 5. Train Model
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop...")

    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        config=Config,
    )

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 6. Generate Submission
    # -------------------------------------------------------------------------
    print("\nGenerating Submission...")

    generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # -------------------------------------------------------------------------
    # 7. Final Verification
    # -------------------------------------------------------------------------
    print("\nVerifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    assert len(df_sub) == len(
        mini_test
    ), f"Submission row count mismatch. Expected {len(mini_test)}, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns mismatch."

    # Check for NaNs
    assert not df_sub.isna().any().any(), "Submission contains NaNs."

    # Check values are numeric
    assert pd.api.types.is_numeric_dtype(
        df_sub["time_to_eruption"]
    ), "Prediction column is not numeric."

    print("Verification passed.")
    print(f"Demo completed successfully. Output stored in {DEMO_DIR}")


if __name__ == "__main__":
    run_demo()
