import os
import sys
import torch
import torch.optim as optim
import numpy as np

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_metric
from library.data import get_dataloaders
from library.model import DeepBiasRefinedBiGRU
from library.loss import MCRMSELoss


def main():
    print("==== RNA Degradation Prediction Pipeline Demo ====")

    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for speed and demonstration purposes
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_run"  # Separate dir for demo artifacts

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)
    print(f"    Device: {device}")

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("    Seed set.")

    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data (Subset)...")

    # Load a small subset (e.g., 16 samples) to verify pipeline quickly
    subset_size = 16
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing from metadata
        subset_size=subset_size,
    )

    print(f"    Train Loader batches: {len(train_loader)}")
    print(f"    Val Loader batches:   {len(val_loader)}")
    print(f"    Test Loader batches:  {len(test_loader)}")

    # Validation: Check Batch Structure
    sample_batch = next(iter(train_loader))
    features = sample_batch["features"]
    pair_indices = sample_batch["pair_indices"]
    targets = sample_batch["targets"]

    print(
        f"    Feature shape: {features.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.IN_CHANNELS}])"
    )
    print(
        f"    Indices shape: {pair_indices.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}])"
    )
    print(
        f"    Targets shape: {targets.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, 5])"
    )

    # Assertions
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.IN_CHANNELS,
    ), "Feature shape mismatch"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair indices shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Targets shape mismatch"

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")

    model = DeepBiasRefinedBiGRU().to(device)

    # Validation: Check Model Forward Pass
    features = features.to(device)
    pair_indices = pair_indices.to(device)

    # Forward pass
    outputs = model(features, pair_indices)

    print(
        f"    Output shape: {outputs.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, 5])"
    )
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Model output shape mismatch"

    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[4] Simulating Training Loop...")

    criterion = MCRMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    model.train()
    total_loss = 0.0

    # Run for a few batches
    for i, batch in enumerate(train_loader):
        # Move data to device
        b_features = batch["features"].to(device)
        b_indices = batch["pair_indices"].to(device)
        b_targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward
        preds = model(b_features, b_indices)

        # Loss
        loss = criterion(preds, b_targets)

        # Backward
        loss.backward()

        # Clip gradients (as per Config)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        print(f"    Batch {i+1}/{len(train_loader)} - Loss: {loss_val:.4f}")

        # Validation: Loss should be scalar and non-negative
        assert not np.isnan(loss_val), "Loss is NaN"
        assert loss_val >= 0, "Loss is negative"

    avg_loss = total_loss / len(train_loader)
    print(f"    Average Train Loss: {avg_loss:.4f}")

    # 5. Evaluation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[5] Evaluating on Validation Set...")

    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            b_features = batch["features"].to(device)
            b_indices = batch["pair_indices"].to(device)
            b_targets = batch["targets"].to(device)

            preds = model(b_features, b_indices)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(b_targets.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    print(f"    Validation Preds Shape: {val_preds.shape}")

    # Calculate Metric
    # Note: calculate_metric handles slicing to Config.PRED_LEN (68) internally
    metric_score = calculate_metric(val_targets, val_preds)
    print(f"    Validation MCRMSE Score: {metric_score:.4f}")

    assert isinstance(metric_score, float), "Metric should be a float"

    # 6. Inference on Test Set
    # -------------------------------------------------------------------------
    print("\n[6] Inference on Test Set...")

    test_batch = next(iter(test_loader))
    t_features = test_batch["features"].to(device)
    t_indices = test_batch["pair_indices"].to(device)

    with torch.no_grad():
        t_preds = model(t_features, t_indices)

    print(f"    Test Input Shape: {t_features.shape}")
    print(f"    Test Prediction Shape: {t_preds.shape}")

    # Verify predictions are not all zero (random initialization should produce non-zero outputs)
    assert torch.abs(t_preds).sum() > 0, "Model predictions are all zeros"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
