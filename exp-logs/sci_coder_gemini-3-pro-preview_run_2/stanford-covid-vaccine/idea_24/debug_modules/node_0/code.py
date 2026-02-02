import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import library components
from library.config import Config
from library.dataset import RNADataset
from library.model import RNA_Model
from library.loss import MCRMSELoss
from library.train_eval import train_epoch, validate


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Setup and Configuration Overrides for Speed
    print("1. Configuring environment for rapid demonstration...")
    Config.set_seed(42)

    # Override Config defaults to run on a tiny subset
    Config.SUBSET_SIZE = 50  # Only use 50 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 2  # Few epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory for cache exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"   Subset Size: {Config.SUBSET_SIZE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Device: {Config.DEVICE}")

    # 2. Data Loading and Verification
    print("\n2. Verifying Dataset and Data Loading...")

    # Initialize Train Dataset
    train_dataset = RNADataset(mode="train", load_cached_data=False)
    print(f"   Train Dataset Length: {len(train_dataset)}")

    # Fetch a single sample to verify structure
    sample = train_dataset[0]
    inputs = sample["inputs"]
    partner_indices = sample["partner_indices"]
    targets = sample["targets"]
    sample_id = sample["ids"]

    print(f"   Sample ID: {sample_id}")
    print(f"   Input Shape: {inputs.shape} (Expected: 107, 18)")
    print(f"   Partner Indices Shape: {partner_indices.shape} (Expected: 107,)")
    print(f"   Targets Shape: {targets.shape} (Expected: 68, 5)")

    # Assertions for data integrity
    assert inputs.shape == (107, 18), "Incorrect input shape"
    assert partner_indices.shape == (107,), "Incorrect partner indices shape"
    assert targets.shape == (68, 5), "Incorrect targets shape"
    assert inputs.dtype == torch.float32, "Input dtype should be float32"
    assert partner_indices.dtype == torch.long, "Partner indices dtype should be long"

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    print("   Train DataLoader initialized successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n3. Verifying Model Architecture...")
    model = RNA_Model().to(Config.DEVICE)

    # Create a batch from the single sample for testing
    batch_inputs = inputs.unsqueeze(0).to(Config.DEVICE)  # (1, 107, 18)
    batch_partners = partner_indices.unsqueeze(0).to(Config.DEVICE)  # (1, 107)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch_inputs, batch_partners)

    print(f"   Model Output Shape: {outputs.shape} (Expected: 1, 107, 5)")

    # Assertions for model output
    assert outputs.shape == (1, 107, 5), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"
    print("   Forward pass successful.")

    # 4. Loss Function Logic Verification
    print("\n4. Verifying MCRMSE Loss Logic...")
    criterion = MCRMSELoss().to(Config.DEVICE)

    # Create synthetic data to verify calculation
    # Scored targets are indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # We create targets = 0 and predictions = 1 for these columns.
    # The loss should be sqrt(1^2) = 1.0

    # Batch of size 2
    synth_targets = torch.zeros((2, 68, 5), device=Config.DEVICE)
    synth_preds = torch.zeros((2, 107, 5), device=Config.DEVICE)

    # Set predictions for scored columns to 1.0
    # Indices 0, 1, 3 are scored
    scored_indices = [0, 1, 3]
    for idx in scored_indices:
        synth_preds[:, :68, idx] = 1.0

    # Set predictions for unscored columns to 100.0 (should be ignored)
    unscored_indices = [2, 4]
    for idx in unscored_indices:
        synth_preds[:, :68, idx] = 100.0

    loss_val = criterion(synth_preds, synth_targets)
    print(f"   Calculated Loss: {loss_val.item():.4f}")

    # Assertions for loss
    # Expected: RMSE for scored columns is 1.0. Mean of RMSEs is 1.0.
    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Loss verification failed. Expected 1.0, got {loss_val.item()}"
    print(
        "   Loss function logic verified (correctly ignores unscored columns and length mismatch)."
    )

    # 5. Training Loop Execution
    print("\n5. Executing Training Loop (Demo)...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Load Validation Data
    val_dataset = RNADataset(mode="val", load_cached_data=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"   Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_score = validate(model, val_loader, Config.DEVICE)
        print(
            f"   Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        # Basic sanity check that loss is not exploding or zero
        assert train_loss > 0, "Train loss should be positive"
        assert val_score > 0, "Validation score should be positive"

    print("   Training loop completed successfully.")

    # 6. Inference on Test Set
    print("\n6. Verifying Inference on Test Set...")
    test_dataset = RNADataset(mode="test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    sample_preds = []
    sample_ids = []

    print(f"   Processing {len(test_dataset)} test samples...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(Config.DEVICE)
            partner_indices = batch["partner_indices"].to(Config.DEVICE)
            ids = batch["ids"]

            outputs = model(inputs, partner_indices)

            # Move to CPU for processing
            outputs = outputs.cpu().numpy()

            sample_preds.append(outputs)
            sample_ids.extend(ids)

    # Concatenate predictions
    all_preds = np.concatenate(sample_preds, axis=0)

    print(
        f"   Total Predictions Shape: {all_preds.shape} (Expected: {len(test_dataset)}, 107, 5)"
    )
    assert all_preds.shape == (
        len(test_dataset),
        107,
        5,
    ), "Test prediction shape mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
