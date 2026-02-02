import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import RNADataset
from library.model import DDFRN
from library.loss import MaskedMCRMSELoss
from library.train import Trainer


def demo_data_loading():
    print("\n=== 1. Demonstrating Data Loading ===")

    # Use a small subset for demonstration speed
    subset_size = 16
    print(f"Loading train dataset (subset={subset_size})...")

    # Initialize Dataset
    # We force reload to ensure we aren't just reading old cache files for this demo
    ds = RNADataset(mode="train", load_cached_data=False, debug_subset=subset_size)

    print(f"Dataset size: {len(ds)}")

    # Fetch one sample
    sample = ds[0]
    inputs = sample["inputs"]
    partner_indices = sample["partner_indices"]
    targets = sample["targets"]
    sample_id = sample["id"]

    print(f"Sample ID: {sample_id}")
    print(f"Inputs Shape: {inputs.shape}")  # Expected: (107, 18)
    print(f"Partner Indices Shape: {partner_indices.shape}")  # Expected: (107,)
    print(f"Targets Shape: {targets.shape}")  # Expected: (107, 5)

    # Assertions to verify data integrity
    assert inputs.shape == (107, 18), f"Expected inputs (107, 18), got {inputs.shape}"
    assert targets.shape == (107, 5), f"Expected targets (107, 5), got {targets.shape}"
    assert partner_indices.shape == (
        107,
    ), f"Expected partner_indices (107,), got {partner_indices.shape}"

    # Create a DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch Inputs Shape: {batch['inputs'].shape}")  # Expected: (4, 107, 18)

    return loader


def demo_model_forward_pass(loader):
    print("\n=== 2. Demonstrating Model Architecture & Forward Pass ===")

    device = torch.device(
        "cpu"
    )  # Use CPU for simple demo to avoid overhead/memory issues on small tasks
    model = DDFRN().to(device)

    # Get a batch
    batch = next(iter(loader))
    inputs = batch["inputs"].to(device)
    partner_indices = batch["partner_indices"].to(device)

    print("Running forward pass...")
    y1, y2 = model(inputs, partner_indices)

    print(f"Output y1 (Pass 1) Shape: {y1.shape}")
    print(f"Output y2 (Pass 2) Shape: {y2.shape}")

    # Verify output dimensions
    # Batch size 4, Seq Len 107, 5 Targets
    expected_shape = (4, 107, 5)
    assert y1.shape == expected_shape, f"y1 shape mismatch: {y1.shape}"
    assert y2.shape == expected_shape, f"y2 shape mismatch: {y2.shape}"

    print("Model forward pass successful.")
    return model


def demo_loss_function():
    print("\n=== 3. Demonstrating Masked MCRMSE Loss ===")

    criterion = MaskedMCRMSELoss()

    # Config parameters for reference
    seq_scored = Config.SEQ_SCORED  # 68
    scored_indices = (
        Config.SCORED_INDICES
    )  # [0, 1, 3] usually (reactivity, deg_Mg_pH10, deg_Mg_50C)

    print(f"Scored Sequence Length: {seq_scored}")
    print(f"Scored Column Indices: {scored_indices}")

    # Create synthetic data
    # Batch=1, Seq=107, Channels=5
    inputs = torch.zeros(1, 107, 5)
    targets = torch.zeros(1, 107, 5)

    # Case 1: Perfect prediction
    loss_zero = criterion(inputs, targets)
    print(f"Loss (Perfect Prediction): {loss_zero.item()}")
    assert np.isclose(loss_zero.item(), 0.0), "Loss should be 0 for perfect predictions"

    # Case 2: Error in UNSCORED sequence position (index 70)
    # Should be ignored by loss
    inputs_unscored_pos = inputs.clone()
    inputs_unscored_pos[:, 70, 0] = 100.0
    loss_unscored = criterion(inputs_unscored_pos, targets)
    print(f"Loss (Error in unscored position > 68): {loss_unscored.item()}")
    assert np.isclose(
        loss_unscored.item(), 0.0
    ), "Loss should ignore positions > seq_scored"

    # Case 3: Error in UNSCORED column (index 2, deg_pH10)
    # Should be ignored by loss if index 2 is not in scored_indices
    # scored_indices are typically 0, 1, 3. Index 2 is deg_pH10.
    inputs_unscored_col = inputs.clone()
    inputs_unscored_col[:, 0, 2] = 100.0
    loss_unscored_col = criterion(inputs_unscored_col, targets)
    print(f"Loss (Error in unscored column index 2): {loss_unscored_col.item()}")
    assert np.isclose(
        loss_unscored_col.item(), 0.0
    ), "Loss should ignore unscored columns"

    # Case 4: Error in SCORED position (index 0) and SCORED column (index 0)
    # Error = 1.0. Squared Error = 1.0.
    # MCRMSE calculation:
    # Col 0 RMSE: sqrt(1.0) = 1.0
    # Col 1 RMSE: 0.0
    # Col 3 RMSE: 0.0
    # Mean(RMSE) = (1+0+0)/3 = 0.3333...
    inputs_scored = inputs.clone()
    inputs_scored[:, 0, 0] = 1.0  # Error of 1.0 at one point

    # However, MSE is averaged over the sequence length (seq_scored) and batch.
    # MSE_col_0 = Sum(Errors) / (Batch * Seq_Scored) = 1.0 / 68
    # RMSE_col_0 = sqrt(1/68)
    # Loss = RMSE_col_0 / 3

    loss_scored = criterion(inputs_scored, targets)

    expected_mse = 1.0 / seq_scored
    expected_rmse = np.sqrt(expected_mse)
    expected_loss = expected_rmse / 3.0  # Averaging over 3 scored columns

    print(f"Loss (Error 1.0 at one scored point): {loss_scored.item():.6f}")
    print(f"Expected Loss: {expected_loss:.6f}")

    assert np.isclose(
        loss_scored.item(), expected_loss, atol=1e-5
    ), "Loss calculation mismatch"
    print("Loss function logic verified.")


def demo_training_loop():
    print("\n=== 4. Demonstrating Training Loop ===")

    # Override Config for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32

    # Setup
    device = torch.device("cpu")  # CPU for demo stability

    # Create datasets
    print("Initializing datasets...")
    train_dataset = RNADataset(
        mode="train", load_cached_data=False, debug_subset=Config.DEBUG_SUBSET_SIZE
    )
    val_dataset = RNADataset(
        mode="val", load_cached_data=False, debug_subset=Config.DEBUG_SUBSET_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Model & Opt
    model = DDFRN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MaskedMCRMSELoss()

    # Initialize Trainer
    trainer = Trainer(model, device, criterion, optimizer, scheduler=None)

    # Define a temporary save path
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, patience=1, save_path=save_path
    )

    # Check if model saved
    if os.path.exists(save_path):
        print("Training complete. Model checkpoint found.")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    return save_path


def demo_inference(model_path):
    print("\n=== 5. Demonstrating Inference ===")

    # Load Test Data
    test_subset = 10
    test_dataset = RNADataset(
        mode="test", load_cached_data=False, debug_subset=test_subset
    )
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # Load Model
    device = torch.device("cpu")
    model = DDFRN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Running inference on test subset...")

    ids = []
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            p_idx = batch["partner_indices"].to(device)
            batch_ids = batch["id"]

            # Forward
            _, y_final = model(inputs, p_idx)

            # Store
            preds.append(y_final.cpu().numpy())
            ids.extend(batch_ids)

    preds = np.concatenate(preds, axis=0)  # (N, 107, 5)

    print(f"Predictions shape: {preds.shape}")

    # Demonstrate formatting for one sample
    sample_idx = 0
    sample_id = ids[sample_idx]
    sample_pred = preds[sample_idx]  # (107, 5)

    print(f"\nExample Submission Format for {sample_id} (First 3 rows):")
    print("id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C")

    for i in range(3):
        row_id = f"{sample_id}_{i}"
        vals = sample_pred[i]
        # Columns in output: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Config.TARGET_COLS order matches this
        line = f"{row_id},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f},{vals[3]:.4f},{vals[4]:.4f}"
        print(line)


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # Ensure working dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data
    loader = demo_data_loading()

    # 3. Model
    demo_model_forward_pass(loader)

    # 4. Loss
    demo_loss_function()

    # 5. Training
    saved_model_path = demo_training_loop()

    # 6. Inference
    demo_inference(saved_model_path)

    print("\n=== Demonstration Complete ===")
