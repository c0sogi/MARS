import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.preprocessing import RNAPreprocessor
from library.dataset import RNADataset
from library.model import SDCGBiGRU
from library.loss import MCRMSELoss
from library.engine import train_fn, eval_fn, set_seed


def main():
    print("Starting RNA Degradation Prediction Demo...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Set fixed seed for reproducibility
    set_seed(42)

    # Modify Config for a small, fast run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure the directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Preprocessing Verification
    # ==========================================
    print("\n[2] Verifying RNAPreprocessor...")

    preprocessor = RNAPreprocessor()

    # Process training data (this will use the debug subset)
    # forcing load_cached_data=False to ensure we test the processing logic
    train_data = preprocessor.process_data(split="train", load_cached_data=False)

    # Verify keys
    expected_keys = ["ids", "inputs", "pair_indices", "targets"]
    for key in expected_keys:
        assert key in train_data, f"Missing key '{key}' in processed data."

    # Verify shapes
    n_samples = len(train_data["ids"])
    assert (
        n_samples == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {n_samples}"

    # Inputs: (N, 107, 14)
    assert train_data["inputs"].shape == (
        n_samples,
        107,
        14,
    ), f"Input shape mismatch. Expected ({n_samples}, 107, 14), got {train_data['inputs'].shape}"

    # Targets: (N, 68, 5)
    assert train_data["targets"].shape == (
        n_samples,
        68,
        5,
    ), f"Target shape mismatch. Expected ({n_samples}, 68, 5), got {train_data['targets'].shape}"

    # Pair Indices: (N, 107)
    assert train_data["pair_indices"].shape == (
        n_samples,
        107,
    ), f"Pair indices shape mismatch. Expected ({n_samples}, 107), got {train_data['pair_indices'].shape}"

    print("Preprocessing verification passed.")

    # ==========================================
    # 3. Dataset & DataLoader Verification
    # ==========================================
    print("\n[3] Verifying RNADataset and DataLoader...")

    # Initialize dataset (it will load the cache created in step 2)
    train_dataset = RNADataset(split="train", load_cached_data=True)

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "input" in batch
    assert "pair_indices" in batch
    assert "target" in batch
    assert "id" in batch

    inputs = batch["input"]
    pair_indices = batch["pair_indices"]
    targets = batch["target"]

    # Check tensor types and devices
    assert isinstance(inputs, torch.Tensor)
    assert inputs.dtype == torch.float32
    assert (
        pair_indices.dtype == torch.int64
    )  # LongTensor required for embedding/indexing

    print(f"Batch loaded. Input shape: {inputs.shape}, Target shape: {targets.shape}")
    print("Dataset verification passed.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Verifying SDCGBiGRU Model...")

    device = Config.DEVICE
    model = SDCGBiGRU().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)

    # Forward pass
    outputs = model(inputs, pair_indices)

    # Verify output shape: (Batch, SeqLen=107, Targets=5)
    expected_out_shape = (inputs.size(0), 107, 5)
    assert (
        outputs.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {outputs.shape}"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Loss Function Verification
    # ==========================================
    print("\n[5] Verifying MCRMSELoss...")

    criterion = MCRMSELoss()

    # Move targets to device
    targets = targets.to(device)

    # Slice outputs to match target length (68)
    outputs_sliced = outputs[:, : Config.PRED_LEN, :]

    # Compute loss
    loss = criterion(outputs_sliced, targets)

    # Verify loss
    assert loss.dim() == 0, "Loss should be a scalar."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"Calculated Loss: {loss.item():.6f}")
    print("Loss function verification passed.")

    # ==========================================
    # 6. Training Loop Simulation
    # ==========================================
    print("\n[6] Simulating Training Loop...")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Create validation set (small subset)
    val_dataset = RNADataset(split="val", load_cached_data=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Run training step
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        # Run validation step
        val_score = eval_fn(model, val_loader, device)

        print(
            f"  Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        # Verify values are finite
        assert np.isfinite(train_loss), "Training loss is NaN or Inf"
        assert np.isfinite(val_score), "Validation score is NaN or Inf"

    # Save model state
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Training simulation complete. Model saved.")

    # ==========================================
    # 7. Inference & Submission Verification
    # ==========================================
    print("\n[7] Simulating Inference and Submission Generation...")

    # Load Test Data
    test_dataset = RNADataset(split="test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    preds_map = {}

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass (Full length 107)
            outputs = model(inputs, pair_indices)
            outputs_np = outputs.cpu().numpy()

            for i, sample_id in enumerate(ids):
                preds_map[sample_id] = outputs_np[i]

    # Check if we have predictions for all test samples
    assert len(preds_map) == len(
        test_dataset
    ), "Mismatch in number of predictions vs test samples."

    # Generate submission rows
    submission_data = []
    target_cols = Config.TARGET_COLS

    # Process just the first few samples to verify logic without processing everything
    sample_ids_to_check = test_dataset.ids[:5]

    for sample_id in sample_ids_to_check:
        pred_matrix = preds_map[sample_id]

        # Check prediction matrix shape
        assert pred_matrix.shape == (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = pred_matrix[seqpos]

            row_dict = {"id_seqpos": row_id}
            for idx, col in enumerate(target_cols):
                row_dict[col] = float(row_preds[idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify submission columns
    expected_cols = ["id_seqpos"] + target_cols
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Verify row count for the checked samples
    expected_rows = len(sample_ids_to_check) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    print("Inference and submission logic verification passed.")

    # Save dummy submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Demo submission saved to {Config.SUBMISSION_PATH}")

    print("\nAll demonstrations and verifications completed successfully!")


if __name__ == "__main__":
    main()
