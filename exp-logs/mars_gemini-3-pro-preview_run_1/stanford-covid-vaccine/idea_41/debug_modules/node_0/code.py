import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import load_data
from library.model import CapacityStabilizedBiGRU
from library.engine import train_one_epoch, validate

if __name__ == "__main__":
    # 1. Setup and Reproducibility
    print("Initializing demonstration...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Verify Metric Logic (MCRMSE)
    print("Verifying MCRMSE metric logic...")
    # Create dummy ground truth and predictions
    # Shape: (2 samples, 3 columns) for simplicity in manual calc
    y_true_dummy = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])  # (2, 1, 3)
    y_pred_dummy = np.array([[[1.1, 1.9, 3.0]], [[4.2, 4.8, 6.0]]])  # (2, 1, 3)

    # Col 1 (idx 0): (1.0-1.1)^2 = 0.01, (4.0-4.2)^2 = 0.04 -> Mean MSE = 0.025 -> RMSE = sqrt(0.025) ~= 0.1581
    # Col 2 (idx 1): (2.0-1.9)^2 = 0.01, (5.0-4.8)^2 = 0.04 -> Mean MSE = 0.025 -> RMSE = sqrt(0.025) ~= 0.1581
    # Col 3 (idx 2): (3.0-3.0)^2 = 0.0,  (6.0-6.0)^2 = 0.0  -> Mean MSE = 0.0   -> RMSE = 0.0
    # MCRMSE = (0.1581 + 0.1581 + 0.0) / 3 ~= 0.1054

    calc_mcrmse = mcrmse_loss(y_true_dummy, y_pred_dummy)
    expected_mcrmse = (np.sqrt(0.025) * 2) / 3

    assert np.isclose(
        calc_mcrmse, expected_mcrmse, atol=1e-4
    ), f"MCRMSE calculation failed. Got {calc_mcrmse}, expected {expected_mcrmse}"
    print("MCRMSE metric verified.")

    # 3. Data Loading
    print("Loading data via library.data...")
    # We force reload to demonstrate processing logic, though caching is supported
    # Note: Config.cache_dir is used internally.
    train_ds, val_ds, test_ds = load_data(load_cached_data=True)

    # Verify dataset lengths
    print(f"Original Train size: {len(train_ds)}")
    print(f"Original Val size: {len(val_ds)}")
    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."

    # Create Subsets for Speed (Fast Demo)
    subset_indices = range(32)  # Use only 32 samples
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)

    # Create DataLoaders
    # Using a small batch size
    demo_batch_size = 8
    train_loader = DataLoader(train_subset, batch_size=demo_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=demo_batch_size, shuffle=False)

    # 4. Model Instantiation
    print("Instantiating CapacityStabilizedBiGRU model...")
    model = CapacityStabilizedBiGRU(Config).to(device)

    # 5. Forward Pass Verification
    print("Verifying forward pass shapes...")
    # Fetch a single batch
    sample_batch = next(iter(train_loader))
    seq = sample_batch["seq"].to(device)
    loop = sample_batch["loop"].to(device)
    dist = sample_batch["dist"].to(device)

    # Forward
    output = model(seq, loop, dist)

    # Check Output Shape: (Batch, Seq_Len, 3)
    # 3 outputs correspond to the 3 targets: reactivity, deg_Mg_pH10, deg_Mg_50C
    expected_shape = (demo_batch_size, Config.seq_len, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"
    print(f"Forward pass successful. Output shape: {output.shape}")

    # 6. Training Loop Demonstration
    print("Running training loop (1 epoch on subset)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert train_loss >= 0, "Training loss should be non-negative."
    print(f"Train Loss (MSE): {train_loss:.6f}")

    # 7. Validation Demonstration
    print("Running validation loop...")
    val_score = validate(model, val_loader, device)

    assert not np.isnan(val_score), "Validation score returned NaN."
    assert val_score >= 0, "Validation score should be non-negative."
    print(f"Validation MCRMSE: {val_score:.6f}")

    # 8. Inference Demonstration (Test Set)
    print("Running inference on test subset...")
    test_subset = Subset(test_ds, range(8))
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        preds = model(seq, loop, dist)

        # Verify predictions are finite
        assert torch.isfinite(preds).all(), "Model produced non-finite predictions."

    print("Inference successful.")
    print("\nDemonstration completed successfully.")
