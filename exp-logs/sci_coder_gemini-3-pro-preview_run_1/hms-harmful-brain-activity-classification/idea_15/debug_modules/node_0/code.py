import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
import tqdm.auto

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# 1. Patch tqdm to disable progress bars (Requirement: Silent Execution)
# -------------------------------------------------------------------------
class SilentTqdm:
    """Mock class to replace tqdm and suppress output."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


# Apply patch before importing library modules that use tqdm
tqdm.auto.tqdm = SilentTqdm

# -------------------------------------------------------------------------
# 2. Import Library Components
# -------------------------------------------------------------------------
from library.config import Config
from library.utils import seed_everything
from library.dataset import HMSDataset
from library.models import DeepSupervisedModel
from library.engine import Trainer, predict_and_submit


def run_demo():
    print("Initializing Demo Script...")

    # ---------------------------------------------------------------------
    # 3. Configure for Speed (Demo Mode)
    # ---------------------------------------------------------------------
    # We modify Config class attributes directly to create a lightweight run
    Config.EPOCHS = 2
    Config.TRAIN_SAMPLE_SIZE = 64  # Small subset for fast demonstration
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Use modest parallelism

    # Isolate demo outputs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create necessary directories
    Config.setup()

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ---------------------------------------------------------------------
    # 4. Data Loading & Verification
    # ---------------------------------------------------------------------
    print("\n--- Setting up DataLoaders ---")

    # Initialize Datasets
    # Note: HMSDataset reads Config.TRAIN_SAMPLE_SIZE to limit data loading
    train_dataset = HMSDataset(mode="train", use_cache=True)
    val_dataset = HMSDataset(mode="val", use_cache=True)

    # Limit validation set size for demo speed
    val_dataset.df = val_dataset.df.iloc[:32].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Verify Data Shapes
    print("\n--- Verifying Data Shapes ---")
    try:
        batch = next(iter(train_loader))
        X_eeg, X_spec, y = batch

        # Expected Shapes based on Config
        # EEG: (Batch, 20, 5000) -> 50s @ 100Hz
        # Spec: (Batch, 5, 512, 512) -> 4 regions + 1 coord map
        # Target: (Batch, 6)

        assert X_eeg.shape == (
            Config.BATCH_SIZE,
            20,
            5000,
        ), f"EEG shape mismatch: {X_eeg.shape}"
        assert X_spec.shape == (
            Config.BATCH_SIZE,
            5,
            512,
            512,
        ), f"Spec shape mismatch: {X_spec.shape}"
        assert y.shape == (Config.BATCH_SIZE, 6), f"Target shape mismatch: {y.shape}"

        print("Data shapes verified successfully.")

    except Exception as e:
        print(f"Data verification failed: {e}")
        raise e

    # ---------------------------------------------------------------------
    # 5. Model Initialization
    # ---------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    device = Config.DEVICE
    print(f"Device: {device}")

    model = DeepSupervisedModel(
        num_classes=Config.NUM_CLASSES,
        eeg_channels=Config.EEG_CHANNELS,
        spec_channels=Config.SPEC_CHANNELS,
    )
    model.to(device)

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    with torch.no_grad():
        X_eeg_dev = X_eeg.to(device)
        X_spec_dev = X_spec.to(device)

        # Model returns 3 logits: Joint, EEG-only, Spec-only
        logits = model(X_eeg_dev, X_spec_dev)

        assert len(logits) == 3, "Model should return 3 outputs (Deep Supervision)"
        assert logits[0].shape == (Config.BATCH_SIZE, 6), "Joint logit shape mismatch"

    print("Forward pass verified.")

    # ---------------------------------------------------------------------
    # 6. Training Loop
    # ---------------------------------------------------------------------
    print("\n--- Starting Training Loop ---")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Run Fit
    trainer.fit()

    # ---------------------------------------------------------------------
    # 7. Inference & Submission
    # ---------------------------------------------------------------------
    print("\n--- Running Inference ---")

    test_dataset = HMSDataset(mode="test", use_cache=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Generates submission.csv in Config.SUBMISSION_PATH
    predict_and_submit(test_loader, model, device)

    # ---------------------------------------------------------------------
    # 8. Verify Submission File
    # ---------------------------------------------------------------------
    print("\n--- Verifying Submission ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check row count matches test set
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    expected_rows = len(test_meta_df)
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["eeg_id"] + Config.OUTPUT_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check probability constraints (Sum to 1.0)
    prob_sums = sub_df[Config.OUTPUT_COLS].sum(axis=1)

    # Allow small floating point tolerance
    assert np.allclose(prob_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    print("Submission verified successfully.")
    print("Demo execution complete.")


if __name__ == "__main__":
    run_demo()
