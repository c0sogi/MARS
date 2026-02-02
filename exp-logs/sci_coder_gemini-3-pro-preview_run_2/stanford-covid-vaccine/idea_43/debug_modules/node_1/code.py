import os
import shutil
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MetricTracker
from library.data import get_loaders
from library.model import TISRNModel
from library.loss import MCRMSELoss
from library.train import train_one_epoch, validate


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup Configuration
    # Enable debug mode to use a small subset of data (batch_size * 2 rows)
    config = Config(debug=True)

    # Redirect cache to a separate demo directory in ./working to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    config.cache_dir = demo_dir
    config.train_cache = os.path.join(demo_dir, "mini_train.npz")
    config.val_cache = os.path.join(demo_dir, "mini_val.npz")
    config.test_cache = os.path.join(demo_dir, "mini_test.npz")

    # Set seeds for reproducibility
    seed_everything(config.seed)

    print(f"Configuration initialized. Device: {config.device}")
    print(f"Cache directory: {config.cache_dir}")

    # 2. Data Loading
    print("\n--- Testing Data Loading ---")
    # This will process the first few rows of the metadata files and save .npz caches
    train_loader, val_loader, test_loader = get_loaders(config, load_cached_data=False)

    # Verify dataset size (Debug mode should load batch_size * 2 = 8 samples)
    expected_samples = config.batch_size * 2
    print(f"Train dataset size: {len(train_loader.dataset)}")
    assert (
        len(train_loader.dataset) == expected_samples
    ), f"Expected {expected_samples} samples in debug mode, got {len(train_loader.dataset)}"

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(config.device)
    partner_indices = batch["partner_indices"].to(config.device)
    targets = batch["targets"].to(config.device)
    mask = batch["mask"].to(config.device)

    # Input shape: (Batch, Seq_Len, Channels=18)
    # Channels: 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner) = 18
    print(f"Input batch shape: {inputs.shape}")
    assert inputs.shape == (
        config.batch_size,
        config.seq_len,
        18,
    ), "Incorrect input shape"
    assert targets.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), "Incorrect target shape"
    assert mask.shape == (config.batch_size, config.seq_len), "Incorrect mask shape"

    # 3. Model Initialization & Forward Pass
    print("\n--- Testing Model Architecture ---")
    model = TISRNModel(config).to(config.device)

    # The model returns a tuple: (Predictions_Pass1, Predictions_Pass2)
    preds_p1, preds_p2 = model(inputs, partner_indices, mask)

    print(f"Prediction Pass 1 shape: {preds_p1.shape}")
    print(f"Prediction Pass 2 shape: {preds_p2.shape}")

    assert preds_p1.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), "Pass 1 output shape mismatch"
    assert preds_p2.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), "Pass 2 output shape mismatch"

    # Verify that gradients are tracked
    assert preds_p2.requires_grad, "Model output does not require grad (broken graph)"

    # 4. Loss Function
    print("\n--- Testing Loss Function ---")
    criterion = MCRMSELoss()

    # Calculate loss
    loss = criterion(preds_p2, targets, mask)
    print(f"Calculated Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Training Loop Component
    print("\n--- Testing Training Step ---")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Run one epoch (on the tiny debug dataset)
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, config.device, config
    )
    print(f"Train One Epoch Loss: {train_loss:.6f}")

    # Run validation
    val_loss, val_mcrmse = validate(model, val_loader, criterion, config.device)
    print(f"Validation Loss: {val_loss:.6f}, MCRMSE: {val_mcrmse:.6f}")

    # 6. Metric Tracker Verification
    print("\n--- Testing Metric Tracker Logic ---")
    tracker = MetricTracker()

    # Create dummy data
    # Scored indices in MetricTracker are [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    # Let's make a batch of 1 sample, length 1
    # Target: [1.0, 1.0, 0.0, 1.0, 0.0] -> Scored: [1.0, 1.0, 1.0]
    # Pred:   [2.0, 1.0, 0.0, 3.0, 0.0] -> Scored: [2.0, 1.0, 3.0]
    # Errors: [1.0, 0.0, 2.0]
    # Squared: [1.0, 0.0, 4.0]
    # MSE per col: [1.0, 0.0, 4.0]
    # RMSE per col: [1.0, 0.0, 2.0]
    # MCRMSE = (1 + 0 + 2) / 3 = 1.0

    dummy_targets = torch.tensor([[[1.0, 1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    dummy_preds = torch.tensor([[[2.0, 1.0, 0.0, 3.0, 0.0]]], dtype=torch.float32)

    tracker.update(dummy_preds, dummy_targets)
    computed_mcrmse = tracker.compute()

    print(f"Computed MCRMSE on dummy data: {computed_mcrmse}")
    assert np.isclose(
        computed_mcrmse, 1.0
    ), f"Metric Tracker logic failed. Expected 1.0, got {computed_mcrmse}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
