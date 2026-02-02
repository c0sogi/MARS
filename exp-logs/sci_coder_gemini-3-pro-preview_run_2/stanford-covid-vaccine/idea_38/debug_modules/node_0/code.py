import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import process_data, RNADataset
from library.model import DF_DCN
from library.train import train_one_epoch, validate

if __name__ == "__main__":
    # =========================================================================
    # 1. Setup & Configuration Overrides
    # =========================================================================
    print(">>> 1. Initializing Setup and Configuration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use a specific demo directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to the demo directory
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"    Working directory set to: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # =========================================================================
    # 2. Data Processing Verification
    # =========================================================================
    print("\n>>> 2. Verifying Data Processing...")

    # Process training data (this will use the debug subset size due to Config override logic in manual slicing below)
    # Note: process_data loads the full CSV, so we manually slice the result for the demo to match Config.DEBUG logic
    raw_train_data = process_data("train", load_cached_data=False)

    # Manually slice for debug speed (simulating what happens in run_training)
    train_data = {k: v[: Config.DEBUG_SUBSET_SIZE] for k, v in raw_train_data.items()}

    # Verify Keys
    expected_keys = {"inputs", "partner_indices", "ids", "targets"}
    assert expected_keys.issubset(
        train_data.keys()
    ), f"Missing keys in processed data. Found: {train_data.keys()}"

    # Verify Shapes
    # Inputs: (N, 107, 18)
    # Targets: (N, 107, 5)
    n_samples = train_data["inputs"].shape[0]
    seq_len = train_data["inputs"].shape[1]
    n_features = train_data["inputs"].shape[2]

    assert (
        n_samples == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {n_samples}"
    assert seq_len == 107, f"Expected sequence length 107, got {seq_len}"
    assert n_features == 18, f"Expected 18 input features, got {n_features}"
    assert train_data["targets"].shape == (
        n_samples,
        107,
        5,
    ), f"Target shape mismatch: {train_data['targets'].shape}"

    print("    Data processing shapes verified successfully.")

    # =========================================================================
    # 3. Dataset & DataLoader Verification
    # =========================================================================
    print("\n>>> 3. Verifying Dataset and DataLoader...")

    dataset = RNADataset(train_data, mode="train")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Fetch one batch
    inputs, partner_idx, targets = next(iter(loader))

    # Verify Tensor types and shapes
    assert isinstance(inputs, torch.Tensor)
    assert inputs.dtype == torch.float32
    assert inputs.shape == (Config.BATCH_SIZE, 107, 18)

    assert isinstance(partner_idx, torch.Tensor)
    assert partner_idx.dtype == torch.long
    assert partner_idx.shape == (Config.BATCH_SIZE, 107)

    assert isinstance(targets, torch.Tensor)
    assert targets.shape == (Config.BATCH_SIZE, 107, 5)

    print("    Dataset and DataLoader yield correct tensor shapes.")

    # =========================================================================
    # 4. Model Architecture Verification
    # =========================================================================
    print("\n>>> 4. Verifying Model Architecture...")

    model = DF_DCN().to(Config.DEVICE)

    # Move batch to device
    inputs = inputs.to(Config.DEVICE)
    partner_idx = partner_idx.to(Config.DEVICE)

    # Test Forward Backbone
    z = model.forward_backbone(inputs)
    assert z.shape == (
        Config.BATCH_SIZE,
        107,
        Config.MAIN_LATENT_DIM,
    ), f"Backbone output shape mismatch. Expected {(Config.BATCH_SIZE, 107, Config.MAIN_LATENT_DIM)}, got {z.shape}"

    # Test Full Forward (Pass 1 - No feedback)
    preds = model(inputs, partner_idx)
    assert preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {preds.shape}"

    print("    Model forward pass successful.")

    # =========================================================================
    # 5. Loss Function Verification
    # =========================================================================
    print("\n>>> 5. Verifying Loss Function Logic...")

    criterion = MCRMSELoss()

    # Create dummy predictions and targets
    # Scored targets are indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # Indices 2 and 4 are ignored.

    # Case: Preds = 0, Targets = 1.
    # Squared Error for scored columns = (0-1)^2 = 1.
    # MSE per column = 1.
    # RMSE per column = 1.
    # MCRMSE = Mean(1, 1, 1) = 1.

    dummy_preds = torch.zeros(2, 107, 5)
    dummy_targets = torch.ones(2, 107, 5)

    loss_val = criterion(dummy_preds, dummy_targets)

    assert (
        abs(loss_val.item() - 1.0) < 1e-6
    ), f"Expected loss 1.0, got {loss_val.item()}"

    # Case: Perfect prediction
    loss_zero = criterion(dummy_targets, dummy_targets)
    assert abs(loss_zero.item()) < 1e-6, "Loss should be 0 for perfect predictions"

    print("    MCRMSE Loss calculation verified.")

    # =========================================================================
    # 6. Training Loop Simulation
    # =========================================================================
    print("\n>>> 6. Simulating Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run for a few epochs
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, Config.DEVICE)
        print(f"    Epoch {epoch+1} Train Loss: {train_loss:.6f}")

        # Verify loss is a valid number
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert train_loss > 0, "Training loss should be positive"

    # Save the model to verify serialization
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print("    Training simulation complete and model saved.")

    # =========================================================================
    # 7. Inference & Submission Verification
    # =========================================================================
    print("\n>>> 7. Simulating Inference and Submission...")

    # Load Test Data (Subset)
    raw_test_data = process_data("test", load_cached_data=False)
    test_data = {k: v[: Config.DEBUG_SUBSET_SIZE] for k, v in raw_test_data.items()}

    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    predictions = []
    ids_list = []

    with torch.no_grad():
        for inputs, partner_idx, sample_ids in test_loader:
            inputs = inputs.to(Config.DEVICE)
            partner_idx = partner_idx.to(Config.DEVICE)

            # Inference: Backbone -> Pass 1 -> Pass 2
            z = model.forward_backbone(inputs)
            preds_1 = model.forward_head(z, partner_idx, prev_preds=None)
            preds_2 = model.forward_head(z, partner_idx, prev_preds=preds_1)

            predictions.append(preds_2.cpu().numpy())
            ids_list.extend(sample_ids)

    predictions = np.concatenate(predictions, axis=0)

    # Verify prediction shape
    assert predictions.shape == (Config.DEBUG_SUBSET_SIZE, 107, 5)

    # Create Submission DataFrame
    # Flattening predictions: (N, 107, 5) -> (N*107, 5)
    preds_flat = predictions.reshape(-1, 5)

    # Create ID column: id_seqpos
    id_seqpos = []
    for sample_id in ids_list:
        for i in range(107):
            id_seqpos.append(f"{sample_id}_{i}")

    submission_df = pd.DataFrame(preds_flat, columns=Config.ALL_TARGETS)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    # Verify Submission Format
    assert len(submission_df) == Config.DEBUG_SUBSET_SIZE * 107
    assert "id_seqpos" in submission_df.columns
    assert "reactivity" in submission_df.columns

    # Save submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"    Submission file generated at {sub_path}")
    print(f"    Shape: {submission_df.shape}")

    print("\n>>> All demonstrations completed successfully.")
