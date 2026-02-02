import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import library components
import library.config as config
from library.dataset import SpeechCommandsDataset
from library.model import DilatedEfficientNet
from library.engine import Trainer
from library.utils import set_seed

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("===========================================================")
    print("       Speech Command Recognition - Demo Execution         ")
    print("===========================================================")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(42)

    print("\n[1] Configuring environment for rapid demonstration...")

    # Override training parameters for speed
    # We set epochs to 2: Epoch 0 for standard training, Epoch 1 for SWA
    config.TRAINING_PARAMS["epochs"] = 2
    config.TRAINING_PARAMS["swa_start_epoch"] = 2  # Starts at epoch index 1 (2nd epoch)
    config.TRAINING_PARAMS["batch_size"] = 8
    config.TRAINING_PARAMS["num_workers"] = 2

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"    Epochs: {config.TRAINING_PARAMS['epochs']}")
    print(f"    Batch Size: {config.TRAINING_PARAMS['batch_size']}")
    print(f"    SWA Start Epoch: {config.TRAINING_PARAMS['swa_start_epoch']}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subsampling)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing subsampled datasets...")

    # Load full metadata
    df_train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(config.VAL_METADATA_PATH)
    df_test_full = pd.read_csv(config.TEST_METADATA_PATH)

    # Subsample for demo (50 samples each to ensure speed)
    # We ensure we have at least some variety in labels
    df_train_small = df_train_full.sample(n=50, random_state=42).reset_index(drop=True)
    df_val_small = df_val_full.sample(n=20, random_state=42).reset_index(drop=True)
    df_test_small = df_test_full.sample(n=20, random_state=42).reset_index(drop=True)

    print(f"    Train samples: {len(df_train_small)}")
    print(f"    Val samples:   {len(df_val_small)}")
    print(f"    Test samples:  {len(df_test_small)}")

    # Instantiate Datasets
    # Note: We skip the complex balancing logic in get_dataloaders for this demo
    # and instantiate the Dataset class directly.
    train_dataset = SpeechCommandsDataset(df_train_small, phase="train")
    val_dataset = SpeechCommandsDataset(df_val_small, phase="val")
    test_dataset = SpeechCommandsDataset(df_test_small, phase="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=True,
        num_workers=config.TRAINING_PARAMS["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=False,
        num_workers=config.TRAINING_PARAMS["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=False,
        num_workers=config.TRAINING_PARAMS["num_workers"],
    )

    # Verify Data Loading
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"    Batch Shape: {sample_batch.shape}")  # Expected: [B, 1, 128, T]

    # Assertions
    assert sample_batch.dim() == 4, "Batch should be 4D [B, C, F, T]"
    assert sample_batch.size(1) == 1, "Input should have 1 channel (spectrogram)"
    assert (
        sample_batch.size(2) == config.AUDIO_PARAMS["n_mels"]
    ), f"Freq dim should be {config.AUDIO_PARAMS['n_mels']}"

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = DilatedEfficientNet(config=config.MODEL_PARAMS)
    model.eval()

    # Dummy Forward Pass
    with torch.no_grad():
        output = model(sample_batch)

    print(f"    Output Shape: {output.shape}")

    # Assertions
    expected_classes = len(config.FINE_GRAINED_LABELS)
    assert output.shape == (
        sample_batch.size(0),
        expected_classes,
    ), f"Output shape mismatch. Expected ({sample_batch.size(0)}, {expected_classes}), got {output.shape}"

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Trainer and starting training loop...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run training
    # This will run 1 epoch of standard training and 1 epoch of SWA update
    trainer.fit()

    # Verify artifacts
    assert os.path.exists(
        trainer.best_model_path
    ), "Best model checkpoint (Phase 1) not found."
    assert os.path.exists(trainer.swa_model_path), "SWA model checkpoint not found."
    print("    Training completed successfully. Checkpoints saved.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    trainer.predict_and_submit()

    # Verify Submission File
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission file loaded. Rows: {len(df_sub)}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Assertions
    assert len(df_sub) == len(
        df_test_small
    ), f"Submission row count mismatch. Expected {len(df_test_small)}, got {len(df_sub)}"
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission missing required columns 'fname' or 'label'"

    # Check if labels are valid competition labels
    valid_labels = config.TARGET_LABELS_SET.union({"silence", "unknown"})
    invalid_preds = df_sub[~df_sub["label"].isin(valid_labels)]

    if not invalid_preds.empty:
        print(
            f"    WARNING: Found invalid labels in submission: {invalid_preds['label'].unique()}"
        )
        raise AssertionError(
            "Submission contains labels outside the allowed competition set."
        )
    else:
        print("    All predicted labels are valid competition labels.")

    print("\n===========================================================")
    print("       Demo Completed Successfully                         ")
    print("===========================================================")


if __name__ == "__main__":
    run_demo()
