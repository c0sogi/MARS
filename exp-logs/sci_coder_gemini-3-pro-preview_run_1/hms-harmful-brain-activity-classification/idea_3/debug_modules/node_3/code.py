import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Import from provided library
from library.config import Config
from library.utils import set_seed, KLDivLossWithLogits
from library.dataset import get_dataloaders
from library.model import HybridModel
from library.train import train_one_epoch, run_training
from library.inference import run_inference


def demonstrate_components():
    print("\n=== 1. Configuration Setup ===")
    # Override Config defaults for a fast demonstration
    # We use a temporary working directory for this demo
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    # Instantiate Config with overrides
    config = Config(
        debug=True,
        debug_subset_size=16,  # Small subset for speed
        batch_size=4,
        epochs=1,
        working_dir=demo_working_dir,
        device="cpu",  # Force CPU for simple logic verification if GPU not needed, but environment has GPU.
        # Let's use default device logic from Config (cuda if available)
    )

    # Ensure reproducibility
    set_seed(config.seed)
    print(f"Config initialized. Device: {config.device}")
    print(f"Working directory: {config.working_dir}")

    print("\n=== 2. Data Loading & Verification ===")
    # Load DataLoaders
    loaders = get_dataloaders(config, load_cached_data=False)

    # Verify keys
    assert "train" in loaders, "Train loader missing"
    assert "val" in loaders, "Validation loader missing"
    assert "test" in loaders, "Test loader missing"

    train_loader = loaders["train"]
    print(f"Train loader batches: {len(train_loader)}")

    # Fetch one batch
    eeg_batch, spec_batch, target_batch = next(iter(train_loader))

    # Verify Shapes
    # EEG: (Batch, Channels, Time) -> (4, 20, 5000)
    expected_eeg_shape = (config.batch_size, config.eeg_channels, config.eeg_seq_len)
    assert (
        eeg_batch.shape == expected_eeg_shape
    ), f"EEG shape mismatch: {eeg_batch.shape} != {expected_eeg_shape}"

    # Spectrogram: (Batch, 1, Height, Width) -> (4, 1, 512, 512)
    expected_spec_shape = (
        config.batch_size,
        1,
        config.spec_size[0],
        config.spec_size[1],
    )
    assert (
        spec_batch.shape == expected_spec_shape
    ), f"Spec shape mismatch: {spec_batch.shape} != {expected_spec_shape}"

    # Targets: (Batch, Num_Classes) -> (4, 6)
    expected_target_shape = (config.batch_size, config.num_classes)
    assert (
        target_batch.shape == expected_target_shape
    ), f"Target shape mismatch: {target_batch.shape} != {expected_target_shape}"

    print("Data shapes verified successfully.")
    print(
        f"EEG Batch: {eeg_batch.shape}, Spec Batch: {spec_batch.shape}, Targets: {target_batch.shape}"
    )

    print("\n=== 3. Model Initialization & Forward Pass ===")
    device = torch.device(config.device)
    model = HybridModel(config).to(device)

    # Move batch to device
    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(eeg_batch, spec_batch)

    assert (
        logits.shape == expected_target_shape
    ), f"Output shape mismatch: {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Forward pass successful. Logits shape:", logits.shape)

    print("\n=== 4. Training Step Verification ===")
    # Setup training components
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = KLDivLossWithLogits()
    scaler = GradScaler(
        enabled=False
    )  # Disable AMP for simple logic check or set enabled=config.use_amp

    # Mock scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=100
    )

    # Run one epoch manually (using the provided train_one_epoch function)
    # We use the train_loader which has only a few batches due to debug=True
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, device, scaler
    )

    assert avg_loss > 0, "Training loss should be positive"
    print(f"Manual training step successful. Average Loss: {avg_loss:.4f}")

    print("\n=== 5. Full Pipeline Execution (Training & Inference) ===")
    # To demonstrate run_training, we must patch the Config class attributes
    # because run_training instantiates Config() internally without arguments.

    # Patch Config class defaults for the demo
    original_debug = Config.debug
    original_epochs = Config.epochs
    original_working_dir = Config.working_dir
    original_batch_size = Config.batch_size

    Config.debug = True
    Config.epochs = 1
    Config.working_dir = demo_working_dir
    Config.debug_subset_size = 20  # Ensure enough data for train/val split in debug
    Config.batch_size = 4

    print("Running library.train.run_training()...")
    run_training(load_cached_data=False)

    # Verify outputs exist
    model_path = os.path.join(demo_working_dir, "best_model.pth")
    submission_path = os.path.join(Config.submission_dir, "submission.csv")

    assert os.path.exists(model_path), "Model checkpoint was not saved."
    assert os.path.exists(submission_path), "Submission file was not created."
    print("Training pipeline completed. Artifacts verified.")

    print("Running library.inference.run_inference()...")
    # run_inference accepts kwargs to override Config
    run_inference(
        load_cached_data=False,
        debug=True,
        working_dir=demo_working_dir,
        model_save_path=model_path,
    )

    # Check submission file content
    df_sub = pd.read_csv(submission_path)
    print(f"Inference completed. Submission shape: {df_sub.shape}")

    # Verify probabilities sum to 1
    prob_cols = [c for c in df_sub.columns if c != "eeg_id"]
    sums = df_sub[prob_cols].sum(axis=1)
    assert np.allclose(
        sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1.0"
    print("Submission integrity check passed.")

    # Restore Config defaults (good practice, though script ends here)
    Config.debug = original_debug
    Config.epochs = original_epochs
    Config.working_dir = original_working_dir
    Config.batch_size = original_batch_size

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_components()
