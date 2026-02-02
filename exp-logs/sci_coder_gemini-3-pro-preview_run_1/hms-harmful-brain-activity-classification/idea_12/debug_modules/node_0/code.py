import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import HMSDataset
from library.model import AsymmetricCoordinateNet
from library.engine import train_model, inference


# --- Setup & Configuration ---
def setup_demo_config():
    """
    Overrides default configuration for a quick demonstration.
    """
    # Enable Debug mode to use a small subset of data (100 samples)
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Ensure directories exist (Config.setup() does this, but being explicit)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set seed
    seed_everything(Config.SEED)

    print(f"Configuration Configured:")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")


def verify_dataset_shapes(dataset, name="Dataset"):
    """
    Verifies that the dataset returns items with expected shapes.
    """
    print(f"\nVerifying {name} shapes...")
    eeg, spec, target = dataset[0]

    # Check EEG Shape: (Channels, Seq_Len) -> (20, 5000)
    # Note: The dataset returns (20, 5000) because it's transposed in __getitem__
    # before returning if it was (Seq_Len, Channels), but let's check the actual output.
    # Looking at dataset.py:
    # eeg_tensor = torch.tensor(eeg_data) -> eeg_data is (Seq_Len, Channels)
    # It does NOT permute EEG in __getitem__.
    # However, the Model's EEGEncoder expects (B, C, L) and does the permute internally.
    # Let's check what __getitem__ actually returns.

    print(f"  EEG Shape: {eeg.shape}")
    print(f"  Spec Shape: {spec.shape}")
    print(f"  Target Shape: {target.shape}")

    # Assertions based on Config
    assert eeg.shape == (
        Config.EEG_SEQ_LEN,
        Config.EEG_CHANNELS,
    ), f"Expected EEG shape ({Config.EEG_SEQ_LEN}, {Config.EEG_CHANNELS}), got {eeg.shape}"

    # Spec shape: (5, 512, 512) -> 4 regions + 1 coord map
    assert spec.shape == (
        Config.SPEC_CHANNELS,
        Config.SPEC_SIZE[0],
        Config.SPEC_SIZE[1],
    ), f"Expected Spec shape ({Config.SPEC_CHANNELS}, {Config.SPEC_SIZE[0]}, {Config.SPEC_SIZE[1]}), got {spec.shape}"

    # Target shape: (6,)
    if name != "Test Dataset":
        assert target.shape == (
            Config.NUM_CLASSES,
        ), f"Expected Target shape ({Config.NUM_CLASSES},), got {target.shape}"

    print(f"  {name} verification passed.")


def main():
    setup_demo_config()

    # -------------------------------------------------------------------------
    # 1. Load Metadata
    # -------------------------------------------------------------------------
    print("\n--- Loading Metadata ---")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Apply Debug Slicing
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SIZE]
        # Test set is small enough, but let's slice it too for speed consistency
        test_df = test_df.iloc[: Config.DEBUG_SIZE]

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # -------------------------------------------------------------------------
    # 2. Prepare Datasets and Loaders
    # -------------------------------------------------------------------------
    print("\n--- Initializing Datasets ---")
    train_dataset = HMSDataset(train_df, config=Config, mode="train", augment=True)
    val_dataset = HMSDataset(val_df, config=Config, mode="val", augment=False)
    test_dataset = HMSDataset(test_df, config=Config, mode="test", augment=False)

    # Verify shapes
    verify_dataset_shapes(train_dataset, "Train Dataset")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to avoid batch norm issues with size 1
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Initialize Model
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    device = torch.device(Config.DEVICE)
    model = AsymmetricCoordinateNet(config=Config)
    model.to(device)

    # Dummy Forward Pass Check
    dummy_eeg, dummy_spec, _ = next(iter(train_loader))
    dummy_eeg = dummy_eeg.to(device)
    dummy_spec = dummy_spec.to(device)

    with torch.no_grad():
        # Use AMP context just like in training
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=Config.USE_AMP,
        ):
            dummy_out = model(dummy_eeg, dummy_spec)

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch!"

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=Config,
    )

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n--- Running Inference ---")

    # Load Best Model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    predictions = inference(
        model=model, dataloader=test_loader, device=device, config=Config
    )

    print(f"Predictions Shape: {predictions.shape}")
    assert predictions.shape == (
        len(test_df),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch!"

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    submission = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission.insert(0, "eeg_id", test_df["eeg_id"].values)

    # Verify Probability Sum Constraint
    row_sums = submission[Config.CLASS_NAMES].sum(axis=1)
    # Allow small float error
    is_valid_sum = np.allclose(row_sums, 1.0, atol=1e-4)

    if not is_valid_sum:
        print("Warning: Probabilities do not sum to 1. Normalizing...")
        submission[Config.CLASS_NAMES] = submission[Config.CLASS_NAMES].div(
            row_sums, axis=0
        )

    # Final assertion
    assert np.allclose(
        submission[Config.CLASS_NAMES].sum(axis=1), 1.0, atol=1e-4
    ), "Final submission probabilities do not sum to 1.0"

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
