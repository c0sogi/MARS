import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import get_datasets, AudioDataset
from library.model import ConvNeXtAudio
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for fast demonstration...")

    # Modify Config for speed and isolation
    Config.debug = True  # Uses only 100 samples per split
    Config.experiment_name = "demo_run"
    Config.working_dir = os.path.join("./working", Config.experiment_name)
    Config.checkpoint_path = os.path.join(Config.working_dir, "best_model.pth")

    # Reduce compute load for demonstration
    Config.epochs = 2
    Config.batch_size = 8
    Config.num_workers = 2
    Config.duration = 5.0  # Reduce audio duration to 5s for faster processing
    Config.target_length = int(Config.sample_rate * Config.duration)

    # Ensure clean working directory
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.seed)

    print(f"    Debug Mode: {Config.debug}")
    print(f"    Epochs: {Config.epochs}")
    print(f"    Working Directory: {Config.working_dir}")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying LWLRAP Metric Logic...")

    # Case 1: Perfect predictions
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_score = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    score = calculate_lwlrap(y_true, y_score)
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    # Case 2: Inverse predictions (worst case for these labels)
    # Sample 0 (True: 0): Preds [0.1, 0.9, 0.0] -> Rank 2 (Prec 1/2=0.5)
    # Sample 1 (True: 1): Preds [0.8, 0.1, 0.1] -> Rank 2 (Prec 1/2=0.5)
    # Avg = 0.5
    y_score_bad = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    score_bad = calculate_lwlrap(y_true, y_score_bad)
    assert np.isclose(score_bad, 0.5), f"Expected 0.5, got {score_bad}"

    print("    Metric verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset Preparation
    # -------------------------------------------------------------------------
    print("\n>>> Preparing Datasets (Spectrogram Generation)...")

    # This will trigger spectrogram computation for the 100 debug samples
    # load_cached_data=False ensures we demonstrate the processing logic
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False)

    # Assertions to verify dataset integrity
    assert len(train_ds) <= 100, "Train dataset size should be <= 100 in debug mode"
    assert len(val_ds) <= 100, "Val dataset size should be <= 100 in debug mode"

    # Check item shape
    sample_spec, sample_target = train_ds[0]
    # Expected shape: (1, n_mels, time_steps)
    # time_steps = (duration * sr) / hop_length
    expected_time_steps = int(Config.target_length / Config.hop_length) + 1

    assert sample_spec.ndim == 3, f"Expected 3 dims, got {sample_spec.ndim}"
    assert sample_spec.shape[0] == 1, f"Expected 1 channel, got {sample_spec.shape[0]}"
    assert (
        sample_spec.shape[1] == Config.n_mels
    ), f"Expected {Config.n_mels} mels, got {sample_spec.shape[1]}"
    # Allow small off-by-one in time steps due to padding/striding logic
    assert (
        abs(sample_spec.shape[2] - expected_time_steps) < 5
    ), f"Expected ~{expected_time_steps} time steps, got {sample_spec.shape[2]}"

    print(f"    Train size: {len(train_ds)}")
    print(f"    Val size: {len(val_ds)}")
    print(f"    Sample shape: {sample_spec.shape}")
    print("    Dataset preparation successful.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")

    model = ConvNeXtAudio(config=Config)
    model = model.to(Config.device)

    # Test Forward Pass with dummy batch
    dummy_input = torch.randn(2, 1, Config.n_mels, sample_spec.shape[2]).to(
        Config.device
    )
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        Config.num_classes,
    ), f"Expected output shape (2, {Config.num_classes}), got {dummy_output.shape}"

    print("    Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Starting Training Loop...")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=Config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler)

    # Run Fit
    trainer.fit()

    # Verify Checkpoint
    assert os.path.exists(Config.checkpoint_path), "Checkpoint file was not created!"
    print(f"    Training complete. Checkpoint saved at {Config.checkpoint_path}")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Running Inference on Test Set...")

    # Load Best Model
    checkpoint = torch.load(Config.checkpoint_path, map_location=Config.device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    assert len(all_preds) == len(test_ds), "Prediction count mismatch"
    assert all_preds.shape[1] == Config.num_classes, "Class count mismatch"

    print(f"    Inference complete. Predictions shape: {all_preds.shape}")

    # -------------------------------------------------------------------------
    # 7. Submission File Generation (Mock)
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission File...")

    # Get column names from sample submission
    sample_sub = pd.read_csv(Config.sample_submission)
    columns = sample_sub.columns.tolist()
    id_col = columns[0]
    label_cols = columns[1:]

    # Create DataFrame
    # Note: test_ds.fnames matches the order of the loader because shuffle=False
    submission_df = pd.DataFrame(all_preds, columns=label_cols)
    submission_df.insert(0, id_col, test_ds.fnames)

    # Save
    submission_df.to_csv(Config.submission_path, index=False)

    print(f"    Submission saved to {Config.submission_path}")
    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    run_demo()
