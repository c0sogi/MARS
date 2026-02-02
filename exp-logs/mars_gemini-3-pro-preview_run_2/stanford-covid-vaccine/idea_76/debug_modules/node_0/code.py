import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(".")

# 1. Configuration Override
from library.config import Config

# Modify Config for a fast demonstration
print("Configuring environment for demonstration...")
Config.SUBSET_SIZE = 50  # Use only 50 training samples for speed
Config.EPOCHS = 2  # Train for only 2 epochs
Config.BATCH_SIZE = 8  # Small batch size
Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
Config.WORKING_DIR = "./working/demo_execution"
Config.CACHE_NAME = "train_data_demo.npz"  # Custom cache name to avoid conflicts

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import library components after Config modification
from library.utils import set_seed
from library.data_processor import DataProcessor
from library.dataset import RNADataset
from library.model import AHCHIDN
from library.loss import AnchoredMCRMSELoss
from library.trainer import Trainer


def main():
    # Set reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Data Processing Verification
    # =========================================================================
    print("\n--- Verifying Data Processing ---")
    processor = DataProcessor()

    # Process training data (this will use the SUBSET_SIZE)
    # We force load_cached_data=False to ensure we process the subset now
    print("Processing training data...")
    train_data = processor.process_data(mode="train", load_cached_data=False)

    inputs = train_data["inputs"]
    targets = train_data["targets"]

    print(f"Train Inputs Shape: {inputs.shape}")
    print(f"Train Targets Shape: {targets.shape}")

    # Assertions
    assert (
        len(inputs) == Config.SUBSET_SIZE
    ), f"Expected {Config.SUBSET_SIZE} samples, got {len(inputs)}"
    assert inputs.shape[1] == Config.SEQ_LENGTH, "Incorrect sequence length in inputs"
    assert inputs.shape[2] == 18, "Incorrect channel dimension (expected 18)"
    assert targets.shape[2] == 5, "Incorrect target dimension (expected 5)"

    # Process val and test data (full size as per DataProcessor logic, but small enough for demo)
    print("Processing validation and test data...")
    processor.process_data(mode="val", load_cached_data=False)
    processor.process_data(mode="test", load_cached_data=False)

    # =========================================================================
    # 3. Dataset Verification
    # =========================================================================
    print("\n--- Verifying Dataset ---")
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    sample = train_dataset[0]

    print("Sample keys:", sample.keys())

    # Assertions
    assert "inputs" in sample
    assert "partner_indices" in sample
    assert "targets" in sample
    assert isinstance(sample["inputs"], torch.Tensor)
    assert sample["inputs"].dtype == torch.float32
    assert sample["partner_indices"].dtype == torch.long

    # =========================================================================
    # 4. Model Architecture Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = AHCHIDN().to(device)
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_inputs = sample["inputs"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    dummy_partners = (
        sample["partner_indices"].unsqueeze(0).repeat(batch_size, 1).to(device)
    )

    print(f"Dummy Input Shape: {dummy_inputs.shape}")

    # Forward Pass 1 (Initial)
    with torch.no_grad():
        preds_1 = model(dummy_inputs, dummy_partners, prev_preds=None)

    print(f"Output Shape (Pass 1): {preds_1.shape}")
    assert preds_1.shape == (batch_size, Config.SEQ_LENGTH, 5)

    # Forward Pass 2 (Feedback)
    with torch.no_grad():
        preds_2 = model(dummy_inputs, dummy_partners, prev_preds=preds_1)

    print(f"Output Shape (Pass 2): {preds_2.shape}")
    assert preds_2.shape == (batch_size, Config.SEQ_LENGTH, 5)

    # =========================================================================
    # 5. Loss Function Verification
    # =========================================================================
    print("\n--- Verifying Loss Function ---")
    criterion = AnchoredMCRMSELoss()
    dummy_targets = sample["targets"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    loss = criterion(preds_2, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() >= 0, "Loss cannot be negative"

    # =========================================================================
    # 6. Training Loop Verification
    # =========================================================================
    print("\n--- Verifying Training Loop ---")
    trainer = Trainer()

    print("Starting training (2 epochs)...")
    trainer.fit()

    assert os.path.exists(trainer.best_model_path), "Best model file was not created"
    print("Training complete. Model saved.")

    # =========================================================================
    # 7. Inference and Submission
    # =========================================================================
    print("\n--- Verifying Inference and Submission ---")

    # Load Test Dataset
    test_dataset = RNADataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Best Model
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["id"]

            # Two-pass inference
            p1 = model(inputs, partner_indices, prev_preds=None)
            p2 = model(inputs, partner_indices, prev_preds=p1)

            all_preds.append(p2.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"Inference shape: {all_preds.shape}")  # Should be (N_test, 107, 5)

    # Create Submission DataFrame
    print("Generating submission file...")
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_pred = all_preds[i]  # (107, 5)
        for seq_pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_data = {"id_seqpos": row_id}

            for col_idx, col_name in enumerate(target_cols):
                row_data[col_name] = float(sample_pred[seq_pos, col_idx])

            submission_rows.append(row_data)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("First 5 rows:")
    print(submission_df.head())


if __name__ == "__main__":
    main()
